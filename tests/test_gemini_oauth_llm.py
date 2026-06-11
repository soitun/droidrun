from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.utils.llm_picker import load_llm


def test_gemini_oauth_profile_sends_gemini_flash_lite_verbatim(tmp_path) -> None:
    variant = resolve_provider_variant("gemini", "oauth")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="gemini",
            variant_id=variant.id,
            auth_mode="oauth",
            model="gemini-3.1-flash-lite",
            credential_path=str(tmp_path / "missing-auth-profiles.json"),
        ),
    )

    llm = load_llm(
        profile.provider,
        model=profile.model,
        credential_path=profile.credential_path,
        **profile.kwargs,
    )
    payload = llm._to_code_assist_request(
        [ChatMessage(role=MessageRole.USER, content="hello")]
    )

    assert profile.provider == "gemini_oauth_code_assist"
    assert profile.model == "gemini-3.1-flash-lite"
    assert payload["model"] == "gemini-3.1-flash-lite"


def _gemini_llm():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    return GeminiOAuthCodeAssistLLM(access_token="test-token", credential_path=None)


def _tiny_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


def test_image_block_is_sent_as_inline_data_part():
    import base64

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    png = _tiny_image("PNG")
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="what is this?"), ImageBlock(image=png)],
            )
        ]
    )

    parts = payload["request"]["contents"][0]["parts"]
    assert parts[0] == {"text": "what is this?"}
    inline = parts[1]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert base64.b64decode(inline["data"]) == png


def test_image_only_message_is_sent_as_inline_data():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _gemini_llm()._to_code_assist_request(
        [ChatMessage(role=MessageRole.USER, blocks=[ImageBlock(image=_tiny_image("PNG"))])]
    )

    parts = payload["request"]["contents"][0]["parts"]
    assert len(parts) == 1 and "inlineData" in parts[0]


def test_jpeg_mime_type_is_detected():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    ImageBlock(image=_tiny_image("JPEG"), image_mimetype="image/jpeg")
                ],
            )
        ]
    )

    inline = payload["request"]["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"


def test_text_only_messages_keep_single_text_part_shape():
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
        ]
    )

    contents = payload["request"]["contents"]
    assert contents[0] == {"role": "user", "parts": [{"text": "hello"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "hi"}]}


def test_system_message_text_reaches_system_instruction():
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ]
    )

    assert "be terse" in payload["request"]["systemInstruction"]["parts"][0]["text"]
    assert all(c["role"] != "system" for c in payload["request"]["contents"])
