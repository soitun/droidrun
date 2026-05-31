from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    DEFAULT_MAX_TOKENS,
    AnthropicOAuthLLM,
)


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "content": [{"type": "text", "text": "ok"}],
            "id": "msg_test",
            "usage": {},
            "stop_reason": "end_turn",
        }


class _CapturingSession:
    def __init__(self):
        self.payload = None

    def post(self, url, headers, json, timeout):
        self.payload = dict(json)
        return _FakeResponse()


def _payload_for(**kwargs):
    llm = AnthropicOAuthLLM(
        access_token="test-token",
        credential_path=None,
        **kwargs,
    )
    session = _CapturingSession()
    llm._session = session
    llm.chat([ChatMessage(role=MessageRole.USER, content="hello")])
    return session.payload


def test_default_max_tokens_is_8192():
    assert DEFAULT_MAX_TOKENS == 8192
    assert AnthropicOAuthLLM(credential_path=None).metadata.num_output == 8192


def test_default_opus_payload_sends_max_tokens_without_temperature():
    payload = _payload_for()

    assert payload["model"] == "claude-opus-4-7"
    assert payload["max_tokens"] == 8192
    assert "temperature" not in payload
