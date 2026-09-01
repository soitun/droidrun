import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from mobilerun.agent.utils import tracing_setup
from mobilerun.config_manager.config_manager import TracingConfig
from mobilerun.telemetry import langfuse_processor
from mobilerun.telemetry.langfuse_processor import LangfuseSpanProcessor


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    monkeypatch.setattr(tracing_setup, "_tracing_initialized", False)
    monkeypatch.setattr(tracing_setup, "_tracing_provider", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_client", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_preprocessor", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_tracer_provider", None)
    monkeypatch.setattr(tracing_setup, "_langfuse_setup_attempted", False)
    monkeypatch.setattr(tracing_setup, "_session_id", "test-session")
    monkeypatch.setattr(tracing_setup, "_user_id", "anonymous")
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)

    agent_token = langfuse_processor._current_agent.set(None)
    root_token = langfuse_processor._root_span_context.set(None)
    step_token = langfuse_processor._last_step_span_context.set(None)
    yield
    langfuse_processor._current_agent.reset(agent_token)
    langfuse_processor._root_span_context.reset(root_token)
    langfuse_processor._last_step_span_context.reset(step_token)


class _FakeTracerProvider:
    def __init__(self):
        self.processors = []

    def add_span_processor(self, processor):
        self.processors.append(processor)


def _install_fake_setup(
    monkeypatch, *, auth_result=True, auth_error=None, attach_exporter=True
):
    import langfuse
    from openinference.instrumentation import llama_index as instrumentation
    from openinference.instrumentation.llama_index import _handler

    provider = _FakeTracerProvider()
    clients = []

    class FakeExporter:
        pass

    FakeExporter.__module__ = "langfuse._client.span_processor"
    exporter = FakeExporter()
    instrument_calls = []

    class FakeInstrumentor:
        is_instrumented_by_opentelemetry = False

        def instrument(self):
            instrument_calls.append(True)

    class FakeLangfuse:
        def __init__(self, **kwargs):
            time.sleep(0.01)
            self.kwargs = kwargs
            clients.append(self)
            if attach_exporter:
                kwargs["tracer_provider"].add_span_processor(exporter)

        def auth_check(self):
            if auth_error:
                raise auth_error
            return auth_result

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(instrumentation, "LlamaIndexInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(_handler, "_encoder", lambda obj: obj)
    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    return provider, clients, exporter, instrument_calls


def _langfuse_config(**kwargs):
    values = {
        "enabled": True,
        "provider": "langfuse",
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }
    values.update(kwargs)
    return TracingConfig(**values)


def test_setup_is_concurrent_safe_and_registers_preprocessor_first(monkeypatch):
    provider, clients, exporter, instrument_calls = _install_fake_setup(monkeypatch)
    config = _langfuse_config()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: tracing_setup.setup_tracing(config), range(16)))

    assert len(clients) == 1
    assert instrument_calls == [True]
    assert isinstance(provider.processors[0], LangfuseSpanProcessor)
    assert provider.processors[1:] == [exporter]
    assert clients[0].kwargs["should_export_span"](object()) is True
    assert tracing_setup._tracing_initialized is True
    assert tracing_setup._tracing_provider == "langfuse"


@pytest.mark.parametrize(
    ("auth_result", "auth_error"),
    [(False, None), (True, RuntimeError("sensitive-detail"))],
)
def test_auth_diagnostic_does_not_retry_or_leak_details(
    monkeypatch, caplog, auth_result, auth_error
):
    _provider, clients, _exporter, _calls = _install_fake_setup(
        monkeypatch, auth_result=auth_result, auth_error=auth_error
    )

    tracing_setup.setup_tracing(_langfuse_config())
    tracing_setup.setup_tracing(_langfuse_config())

    assert len(clients) == 1
    assert tracing_setup._tracing_initialized is True
    assert "sensitive-detail" not in caplog.text


def test_failed_client_construction_cannot_accumulate_processors(monkeypatch):
    import langfuse
    from openinference.instrumentation import llama_index as instrumentation
    from openinference.instrumentation.llama_index import _handler

    provider = _FakeTracerProvider()
    construction_count = 0

    class FakeInstrumentor:
        is_instrumented_by_opentelemetry = True

    class FailingLangfuse:
        def __init__(self, **_kwargs):
            nonlocal construction_count
            construction_count += 1
            raise ValueError("sensitive-detail")

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(instrumentation, "LlamaIndexInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(_handler, "_encoder", lambda obj: obj)
    monkeypatch.setattr(langfuse, "Langfuse", FailingLangfuse)

    tracing_setup.setup_tracing(_langfuse_config())
    tracing_setup.setup_tracing(_langfuse_config())

    assert construction_count == 1
    assert len(provider.processors) == 1
    assert isinstance(provider.processors[0], LangfuseSpanProcessor)
    assert tracing_setup._tracing_initialized is False


def test_setup_rejects_preexisting_langfuse_exporter(monkeypatch, caplog):
    provider, clients, _exporter, _calls = _install_fake_setup(monkeypatch)

    class ExistingLangfuseExporter:
        pass

    ExistingLangfuseExporter.__module__ = "langfuse._client.span_processor"
    provider._active_span_processor = SimpleNamespace(
        _span_processors=(ExistingLangfuseExporter(),)
    )

    tracing_setup.setup_tracing(_langfuse_config())

    assert clients == []
    assert provider.processors == []
    assert tracing_setup._tracing_initialized is False
    assert "pk-test" not in caplog.text
    assert "sk-test" not in caplog.text


def test_setup_rejects_same_key_client_owned_by_another_provider(monkeypatch, caplog):
    provider, clients, _exporter, _calls = _install_fake_setup(
        monkeypatch, attach_exporter=False
    )

    tracing_setup.setup_tracing(_langfuse_config())

    assert len(clients) == 1
    assert len(provider.processors) == 1
    assert isinstance(provider.processors[0], LangfuseSpanProcessor)
    assert tracing_setup._langfuse_client is None
    assert tracing_setup._tracing_initialized is False
    assert "pk-test" not in caplog.text
    assert "sk-test" not in caplog.text


@pytest.mark.parametrize(
    ("explicit", "base_env", "legacy_env", "expected"),
    [
        ("https://explicit", "https://base", "https://legacy", "https://explicit"),
        ("", "https://base", "https://legacy", "https://base"),
        ("", "", "https://legacy", "https://legacy"),
        ("", "", "", tracing_setup.DEFAULT_LANGFUSE_BASE_URL),
    ],
)
def test_langfuse_base_url_precedence(
    monkeypatch, explicit, base_env, legacy_env, expected
):
    if base_env:
        monkeypatch.setenv("LANGFUSE_BASE_URL", base_env)
    if legacy_env:
        monkeypatch.setenv("LANGFUSE_HOST", legacy_env)

    config = _langfuse_config(langfuse_host=explicit)
    assert tracing_setup._resolve_langfuse_base_url(config) == expected


def test_processor_normalizes_agent_and_llm_metadata():
    class LLM:
        model = "test-model"
        temperature = 0.2

        @staticmethod
        def class_name():
            return "TestLLM"

    agent = SimpleNamespace(
        shared_state=SimpleNamespace(
            instruction="Open Settings",
            agent_memory=["memory"],
            message_history=[],
            current_subgoal=None,
            error_flag_plan=False,
        ),
        config=SimpleNamespace(
            agent=SimpleNamespace(
                reasoning=False,
                after_sleep_action=1,
                manager=SimpleNamespace(vision=False),
                executor=SimpleNamespace(vision=False),
                fast_agent=SimpleNamespace(vision=True),
            ),
            device=SimpleNamespace(
                platform="android", serial="emulator", use_tcp=False
            ),
        ),
        output_model=None,
        fast_agent_llm=LLM(),
        app_opener_llm=None,
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(LangfuseSpanProcessor(agent))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with provider.get_tracer("test").start_as_current_span("MobileAgent.run"):
        pass

    attrs = exporter.get_finished_spans()[0].attributes
    metadata = json.loads(attrs["langfuse.observation.input"])
    assert metadata["goal"] == "Open Settings"
    assert metadata["vision_enabled"] is True
    assert metadata["llms"][0] == {
        "role": "fast_agent",
        "provider": "TestLLM",
        "vision": True,
        "model": "test-model",
        "temperature": 0.2,
    }
    assert attrs["langfuse.trace.tags"] == ("fast",)


class _CollectingExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def test_public_v4_client_exports_custom_span_once_and_uploads_native_media(
    monkeypatch,
):
    from langfuse import Langfuse
    from langfuse._task_manager.media_manager import MediaManager

    media_jobs = []

    def record_media(_self, **kwargs):
        media_jobs.append(kwargs)

    monkeypatch.setattr(MediaManager, "_process_media", record_media)
    provider = TracerProvider()
    exporter = _CollectingExporter()
    preprocessor = LangfuseSpanProcessor()
    provider.add_span_processor(preprocessor)
    client = Langfuse(
        public_key=f"pk-test-{uuid4()}",
        secret_key="sk-test",
        base_url="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
        should_export_span=lambda _span: True,
    )

    try:
        image = base64.b64encode(b"png-bytes").decode()
        with provider.get_tracer("mobilerun.custom").start_as_current_span(
            "droidrun.screenshot"
        ) as span:
            span.set_attribute("droidrun.screenshot.image_base64", image)
            span.set_attribute("droidrun.screenshot.mime_type", "image/png")
        client.flush()

        assert len(exporter.spans) == 1
        attrs = exporter.spans[0].attributes
        assert "droidrun.screenshot.image_base64" not in attrs
        assert "@@@langfuseMedia:type=image/png" in attrs["langfuse.observation.output"]
        assert len(media_jobs) == 1
        assert preprocessor.force_flush() is True
    finally:
        client.shutdown()


def test_native_image_size_and_error_handling(monkeypatch, caplog):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"x").decode()

    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"image": encoded, "image_mimetype": "image/png"}
        )
        is None
    )
    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"image": "not-base64", "image_mimetype": "image/png"}
        )
        is None
    )
    assert (
        LangfuseSpanProcessor._prepare_image_for_native_upload(
            {"url": f"data:image/png;base64,{encoded}"}
        )
        is None
    )
    assert "not-base64" not in caplog.text


@pytest.mark.parametrize("nested", [False, True])
def test_rejected_image_bytes_are_not_restored_to_observation_attributes(
    monkeypatch, nested
):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"oversized-image").decode()
    message = {
        "role": "user",
        "blocks": [
            {
                "block_type": "image",
                "image": encoded,
                "image_mimetype": "image/png",
            }
        ],
    }
    if nested:
        message = {"json": message}
    attrs = {"input.value": json.dumps({"messages": [message]})}

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert encoded not in attrs["langfuse.observation.input"]
    assert "input.value" not in attrs


def test_rejected_serialized_content_media_is_removed(monkeypatch):
    monkeypatch.setattr(langfuse_processor, "MAX_IMAGE_SIZE_KB", 0)
    encoded = base64.b64encode(b"oversized-image").decode()
    attrs = {
        "input.value": json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            }
                        ],
                    }
                ]
            }
        )
    }

    LangfuseSpanProcessor()._process_field(attrs, "input")

    assert encoded not in attrs["langfuse.observation.input"]
    assert "input.value" not in attrs


def test_apply_session_context_sets_langfuse_trace_attributes():
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.context import get_value

    tracing_setup._tracing_initialized = True
    tracing_setup._tracing_provider = "langfuse"
    tracing_setup._session_id = "session-123"
    tracing_setup._user_id = "user-123"

    def read_context():
        tracing_setup.apply_session_context()
        return (
            get_value(SpanAttributes.SESSION_ID),
            get_value(SpanAttributes.USER_ID),
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(read_context).result() == ("session-123", "user-123")
