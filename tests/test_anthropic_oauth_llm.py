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


def test_opus_4_8_payload_sends_max_tokens_without_temperature():
    payload = _payload_for(model="claude-opus-4-8")

    assert payload["model"] == "claude-opus-4-8"
    assert payload["max_tokens"] == 8192
    assert "temperature" not in payload


def _chat_payload(messages):
    llm = AnthropicOAuthLLM(access_token="test-token", credential_path=None)
    session = _CapturingSession()
    llm._session = session
    llm.chat(messages)
    return session.payload


def _tiny_png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_block_is_sent_as_base64_image_content():
    import base64

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    png = _tiny_png()
    payload = _chat_payload(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="what is this?"), ImageBlock(image=png)],
            )
        ]
    )

    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["text", "image"]
    assert content[0]["text"] == "what is this?"
    source = content[1]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert base64.b64decode(source["data"]) == png


def test_image_only_message_is_sent_as_image_content():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _chat_payload(
        [ChatMessage(role=MessageRole.USER, blocks=[ImageBlock(image=_tiny_png())])]
    )

    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["image"]


def test_jpeg_media_type_is_detected():
    from io import BytesIO

    from llama_index.core.base.llms.types import ImageBlock
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="JPEG")
    payload = _chat_payload(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[ImageBlock(image=buf.getvalue(), image_mimetype="image/jpeg")],
            )
        ]
    )

    assert payload["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_text_only_messages_keep_plain_string_content():
    payload = _chat_payload(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
            ChatMessage(role=MessageRole.USER, content="bye"),
        ]
    )

    assert [m["content"] for m in payload["messages"]] == ["hello", "hi", "bye"]
    assert all(isinstance(m["content"], str) for m in payload["messages"])


def test_system_message_text_reaches_system_blocks():
    payload = _chat_payload(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ]
    )

    system_texts = [block["text"] for block in payload["system"]]
    assert "be terse" in system_texts
    assert all(m["role"] != "system" for m in payload["messages"])
