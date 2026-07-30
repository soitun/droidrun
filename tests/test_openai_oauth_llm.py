import json

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from mobilerun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth


def _offline_oauth_llm(tmp_path) -> OpenAIOAuth:
    return OpenAIOAuth(
        model="gpt-5.5",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )


def test_openai_oauth_constructs_with_updated_openai_adapter(tmp_path) -> None:
    llm = _offline_oauth_llm(tmp_path)

    assert llm.class_name() == "OpenAIOAuth"
    assert llm.model == "gpt-5.5"
    assert llm.metadata.model_name == "gpt-5.5"
    assert llm.metadata.context_window == 400_000


def test_openai_oauth_preserves_text_and_serializes_tool_arguments(tmp_path) -> None:
    llm = _offline_oauth_llm(tmp_path)
    payload = llm._build_responses_payload(
        [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[
                    TextBlock(text="I will open Settings."),
                    ToolCallBlock(
                        tool_call_id="call-1",
                        tool_name="start_app",
                        tool_kwargs={"package": "com.android.settings"},
                    ),
                ],
            )
        ]
    )

    text_item = next(item for item in payload if item.get("role") == "assistant")
    tool_item = next(item for item in payload if item.get("type") == "function_call")

    assert text_item["content"] == [
        {"type": "output_text", "text": "I will open Settings."}
    ]
    assert isinstance(tool_item["arguments"], str)
    assert json.loads(tool_item["arguments"]) == {"package": "com.android.settings"}
    assert tool_item["call_id"] == "call-1"
    assert tool_item["name"] == "start_app"
