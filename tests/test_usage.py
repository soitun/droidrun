from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from mobilerun.agent.usage import (
    TokenCountingHandler,
    create_tracker,
    get_usage_from_response,
    track_usage,
)
from mobilerun.agent.utils.llm_picker import load_llm


def _openai_responses_chat_response() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
            )
        ),
    )


def test_track_usage_supports_mobilerun_openai_responses_wrapper() -> None:
    llm = load_llm("OpenAIResponses", model="gpt-5.5", api_key="stub")

    tracker = track_usage(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "OpenAIResponses"


def test_create_tracker_supports_mobilerun_openai_responses_wrapper() -> None:
    llm = load_llm("OpenAIResponses", model="gpt-5.5", api_key="stub")

    tracker = create_tracker(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "OpenAIResponses"


def test_openai_responses_wrapper_name_extracts_usage_from_response() -> None:
    usage = get_usage_from_response(
        "MobilerunOpenAIResponses", _openai_responses_chat_response()
    )

    assert usage.request_tokens == 3
    assert usage.response_tokens == 2
    assert usage.total_tokens == 5
    assert usage.requests == 1


def test_openai_responses_class_name_extracts_usage_from_response() -> None:
    usage = get_usage_from_response(
        "openai_responses_llm", _openai_responses_chat_response()
    )

    assert usage.request_tokens == 3
    assert usage.response_tokens == 2
    assert usage.total_tokens == 5
    assert usage.requests == 1
