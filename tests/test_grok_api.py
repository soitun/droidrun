from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from openai.types.responses import Response
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from mobilerun.agent.providers.grok import XAI_API_BASE
from mobilerun.agent.providers.registry import (
    VARIANT_ENV_KEY_SLOT,
    list_models_for_variant,
    normalize_model_id_for_variant,
    resolve_provider_variant,
)
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.llm_picker import load_llm, normalize_provider_name
from mobilerun.config_manager import env_keys


def _xai_completed_response(*, usage: ResponseUsage) -> Response:
    return Response(
        id="response-id",
        created_at=0,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model="grok-4.5",
        object="response",
        output=[
            ResponseFunctionToolCall(
                arguments='{"value":"ok"}',
                call_id="call-id",
                name="inspect",
                type="function_call",
                id="item-id",
                status="completed",
            )
        ],
        parallel_tool_calls=True,
        temperature=0.4,
        tool_choice="auto",
        tools=[],
        top_p=0.6,
        status="completed",
        usage=usage,
    )


def _xai_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=7,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=4,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=11,
    )


def test_grok_api_key_variant_is_first_class_xai_responses_provider() -> None:
    variant = resolve_provider_variant("grok", "api_key")

    assert variant.id == "XAI"
    assert variant.runtime_provider_name == "XAI"
    assert variant.default_model == "grok-4.5"
    assert list_models_for_variant("grok", "api_key") == ("grok-4.5",)
    assert variant.requires_api_key is True
    assert variant.base_url == XAI_API_BASE
    assert VARIANT_ENV_KEY_SLOT[variant.id] == "xai"
    assert env_keys.API_KEY_ENV_VARS["xai"] == "XAI_API_KEY"


def test_grok_oauth_variant_shares_the_canonical_model_catalog() -> None:
    variant = resolve_provider_variant("grok", "oauth")

    assert variant.id == "grok_oauth"
    assert variant.runtime_provider_name == "grok_oauth"
    assert variant.default_model == "grok-4.5"
    assert variant.models == ("grok-4.5",)
    assert variant.credential_path


@pytest.mark.parametrize("auth_mode", ("api_key", "oauth"))
@pytest.mark.parametrize(
    "model_alias",
    ("grok-4.5", "grok-4.5-latest", "grok-build-latest", "xai/grok-4.5"),
)
def test_grok_model_aliases_normalize_to_canonical_id(
    auth_mode: str, model_alias: str
) -> None:
    assert normalize_model_id_for_variant("grok", auth_mode, model_alias) == "grok-4.5"


@pytest.mark.parametrize("alias", ("grok", "xai", "x.ai", "XAI"))
def test_grok_runtime_aliases_select_xai(alias: str) -> None:
    assert normalize_provider_name(alias) == "XAI"


def test_grok_profile_wires_api_base_context_and_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    variant = resolve_provider_variant("grok", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="grok",
            variant_id="XAI",
            auth_mode="api_key",
            model="grok-build-latest",
            api_key_source="env",
        ),
        temperature=0.4,
    )

    assert profile.provider == "XAI"
    assert profile.provider_family == "grok"
    assert profile.model == "grok-4.5"
    assert profile.temperature == 0.4
    assert profile.base_url == XAI_API_BASE
    assert profile.api_base == XAI_API_BASE
    assert profile.kwargs == {"context_window": 500_000}

    load_kwargs = profile.to_load_llm_kwargs()
    assert load_kwargs["api_key"] == "xai-env-key"
    assert load_kwargs["api_base"] == XAI_API_BASE
    assert load_kwargs["context_window"] == 500_000


def test_grok_profile_resolves_saved_api_key(monkeypatch, tmp_path) -> None:
    credential_path = tmp_path / "auth-profiles.json"
    credential_path.write_text(
        json.dumps({"apiKeys": {"xai": "xai-saved-key"}}), encoding="utf-8"
    )
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", credential_path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    variant = resolve_provider_variant("grok", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="grok",
            variant_id="XAI",
            auth_mode="api_key",
            model="grok-4.5",
            api_key_source="file",
        ),
    )

    assert profile.to_load_llm_kwargs()["api_key"] == "xai-saved-key"


def test_xai_loader_uses_responses_metadata_and_forces_payload_contract() -> None:
    llm = load_llm(
        "XAI",
        model="grok-4.5-latest",
        api_key="stub",
        temperature=0.4,
        top_p=0.7,
        store=True,
        reasoning_options={"effort": "high"},
        additional_kwargs={
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "stop": ["done"],
        },
    )

    assert type(llm).__name__ == "MobilerunOpenAIResponses"
    assert llm.model == "grok-4.5"
    assert llm.api_base == XAI_API_BASE
    assert llm.metadata.context_window == 500_000
    assert llm.metadata.is_function_calling_model is True
    assert llm.reasoning_options is None

    payload = llm._get_model_kwargs(
        model="caller-selected-model",
        store=True,
        temperature=0.3,
        top_p=0.6,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        stop="stop",
        reasoning={"effort": "low"},
        extra_body={
            "model": "extra-body-model",
            "store": True,
            "temperature": 0.2,
            "top_p": 0.8,
            "presence_penalty": 0.9,
            "frequency_penalty": 0.9,
            "stop": "extra-stop",
            "reasoning": {"effort": "low"},
            "metadata": {"safe": "value"},
        },
    )
    assert payload["model"] == "grok-4.5"
    assert payload["store"] is False
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.6
    assert {
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "reasoning",
    }.isdisjoint(payload)
    assert payload["extra_body"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "metadata": {"safe": "value"},
    }


def test_xai_loader_pins_catalog_context_metadata() -> None:
    llm = load_llm(
        "XAI",
        model="grok-4.5",
        api_key="stub",
        context_window=1,
    )

    assert llm.metadata.context_window == 500_000


def test_xai_sync_chat_pins_final_sdk_wire_body() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_xai_completed_response(usage=_xai_usage()).model_dump(mode="json"),
        )

    llm = load_llm(
        "XAI",
        model="grok-4.5",
        api_key="stub",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        llm.chat(
            [ChatMessage(role=MessageRole.USER, content="inspect")],
            model="caller-selected-model",
            store=True,
            reasoning={"effort": "high"},
            presence_penalty=0.3,
            extra_body={
                "model": "extra-body-model",
                "store": True,
                "reasoning": {"effort": "low"},
                "presence_penalty": 0.9,
                "metadata": {"safe": "value"},
            },
        )
    finally:
        llm._client.close()
        asyncio.run(llm._aclient.close())

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{XAI_API_BASE}/responses"
    payload = json.loads(request.content)
    assert payload["model"] == "grok-4.5"
    assert payload["store"] is False
    assert payload["metadata"] == {"safe": "value"}
    assert {"reasoning", "presence_penalty"}.isdisjoint(payload)


def test_xai_sync_and_async_chat_send_sanitized_multimodal_tool_payloads() -> None:
    usage = _xai_usage()
    response = _xai_completed_response(usage=usage)
    sync_payload: dict[str, Any] = {}
    async_payload: dict[str, Any] = {}

    def create_sync(**kwargs: Any) -> Response:
        sync_payload.update(kwargs)
        return response

    async def create_async(**kwargs: Any) -> Response:
        async_payload.update(kwargs)
        return response

    llm = load_llm("XAI", model="grok-4.5", api_key="stub")
    llm._client = SimpleNamespace(responses=SimpleNamespace(create=create_sync))
    llm._aclient = SimpleNamespace(responses=SimpleNamespace(create=create_async))
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="inspect"),
                ImageBlock(image=b"png-bytes", image_mimetype="image/png"),
            ],
        )
    ]
    call_kwargs = {
        "temperature": 0.4,
        "top_p": 0.6,
        "tools": [
            {
                "type": "function",
                "name": "inspect",
                "description": "Inspect the image",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "stop": "done",
        "store": True,
        "reasoning": {"effort": "high"},
    }

    sync_result = llm.chat(messages, **dict(call_kwargs))
    async_result = asyncio.run(llm.achat(messages, **dict(call_kwargs)))

    for result in (sync_result, async_result):
        tool_calls = llm.get_tool_calls_from_response(result)
        assert [(call.tool_name, call.tool_kwargs) for call in tool_calls] == [
            ("inspect", {"value": "ok"})
        ]
        usage_result = get_usage_from_response("MobilerunOpenAIResponses", result)
        assert (
            usage_result.request_tokens,
            usage_result.response_tokens,
            usage_result.total_tokens,
        ) == (7, 4, 11)

    for payload in (sync_payload, async_payload):
        assert payload["stream"] is False
        assert payload["model"] == "grok-4.5"
        assert payload["temperature"] == 0.4
        assert payload["top_p"] == 0.6
        assert payload["store"] is False
        assert payload["tools"] == call_kwargs["tools"]
        assert payload["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "inspect",
        }
        assert payload["input"][0]["content"][1]["type"] == "input_image"
        assert payload["input"][0]["content"][1]["image_url"].startswith(
            "data:image/png;base64,"
        )
        assert {
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "reasoning",
        }.isdisjoint(payload)


def test_xai_sync_and_async_stream_preserve_completed_usage() -> None:
    event = ResponseCompletedEvent(
        response=_xai_completed_response(usage=_xai_usage()),
        sequence_number=1,
        type="response.completed",
    )
    sync_payload: dict[str, Any] = {}
    async_payload: dict[str, Any] = {}

    def create_sync(**kwargs: Any):  # type: ignore[no-untyped-def]
        sync_payload.update(kwargs)
        return iter((event,))

    async def event_stream():  # type: ignore[no-untyped-def]
        yield event

    async def create_async(**kwargs: Any):  # type: ignore[no-untyped-def]
        async_payload.update(kwargs)
        return event_stream()

    llm = load_llm("XAI", model="grok-4.5", api_key="stub")
    llm._client = SimpleNamespace(responses=SimpleNamespace(create=create_sync))
    llm._aclient = SimpleNamespace(responses=SimpleNamespace(create=create_async))
    messages = [ChatMessage(role=MessageRole.USER, content="inspect")]
    runtime_kwargs = {
        "temperature": 0.4,
        "top_p": 0.6,
        "presence_penalty": 0.1,
        "store": True,
    }

    sync_result = list(llm.stream_chat(messages, **dict(runtime_kwargs)))[-1]

    async def collect_async():  # type: ignore[no-untyped-def]
        return [
            item
            async for item in await llm.astream_chat(
                messages, **dict(runtime_kwargs)
            )
        ]

    async_result = asyncio.run(collect_async())[-1]

    for result in (sync_result, async_result):
        usage_result = get_usage_from_response("MobilerunOpenAIResponses", result)
        assert (
            usage_result.request_tokens,
            usage_result.response_tokens,
            usage_result.total_tokens,
        ) == (7, 4, 11)
        assert result.additional_kwargs["usage"].total_tokens == 11
    for payload in (sync_payload, async_payload):
        assert payload["stream"] is True
        assert payload["temperature"] == 0.4
        assert payload["top_p"] == 0.6
        assert payload["store"] is False
        assert "presence_penalty" not in payload


def test_xai_structured_predict_sanitizes_sync_and_async_call_kwargs() -> None:
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class StructuredResult(BaseModel):
        value: str

    sync_payload: dict[str, Any] = {}
    async_payload: dict[str, Any] = {}

    def parse_sync(**kwargs: Any) -> Any:
        sync_payload.update(kwargs)
        return SimpleNamespace(output_parsed=StructuredResult(value="OK"))

    async def parse_async(**kwargs: Any) -> Any:
        async_payload.update(kwargs)
        return SimpleNamespace(output_parsed=StructuredResult(value="OK"))

    llm = load_llm("XAI", model="grok-4.5", api_key="stub")
    llm._client = SimpleNamespace(responses=SimpleNamespace(parse=parse_sync))
    llm._aclient = SimpleNamespace(responses=SimpleNamespace(parse=parse_async))
    prompt = PromptTemplate("Return {value}")
    call_kwargs = {
        "temperature": 0.4,
        "top_p": 0.6,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "stop": "done",
        "store": True,
        "tool_choice": "none",
        "reasoning": {"effort": "high"},
        "model": "caller-selected-model",
        "extra_body": {
            "model": "extra-body-model",
            "store": True,
            "tool_choice": "required",
            "reasoning": {"effort": "low"},
            "presence_penalty": 0.9,
            "metadata": {"safe": "value"},
        },
    }

    assert (
        llm.structured_predict(
            StructuredResult,
            prompt,
            llm_kwargs=dict(call_kwargs),
            value="OK",
        ).value
        == "OK"
    )
    assert (
        asyncio.run(
            llm.astructured_predict(
                StructuredResult,
                prompt,
                llm_kwargs=dict(call_kwargs),
                value="OK",
            )
        ).value
        == "OK"
    )

    for payload in (sync_payload, async_payload):
        assert payload["temperature"] == 0.4
        assert payload["top_p"] == 0.6
        assert payload["store"] is False
        assert "tool_choice" not in payload
        assert "model" not in payload["extra_body"]
        assert payload["extra_body"] == {"metadata": {"safe": "value"}}
        assert {
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "reasoning",
        }.isdisjoint(payload)


def test_xai_loader_uses_environment_key_for_direct_runtime(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-runtime-key")

    llm = load_llm("grok", model="grok-4.5")

    assert llm.api_key == "xai-runtime-key"


@pytest.mark.parametrize("alias", ("grok", "xai", "x.ai", "XAI"))
def test_xai_runtime_aliases_default_to_canonical_model(
    alias: str, monkeypatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-runtime-key")

    llm = load_llm(alias)

    assert llm.model == "grok-4.5"


def test_xai_loader_does_not_fall_back_to_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        load_llm("XAI", model="grok-4.5")


@pytest.mark.parametrize(
    "endpoint_override",
    (
        {"api_base": "https://attacker.invalid/v1"},
        {"base_url": "https://attacker.invalid/v1"},
        {
            "api_base": "https://attacker.invalid/v1",
            "base_url": "https://another-attacker.invalid/v1",
        },
    ),
)
def test_xai_loader_pins_api_endpoint(endpoint_override: dict[str, str]) -> None:
    llm = load_llm(
        "XAI",
        model="grok-4.5",
        api_key="xai-secret",
        **endpoint_override,
    )

    assert llm.api_base == XAI_API_BASE
