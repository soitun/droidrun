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
        [
            ChatMessage(
                role=MessageRole.USER, blocks=[ImageBlock(image=_tiny_image("PNG"))]
            )
        ]
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


def test_oauth_default_model_resolves_to_antigravity_flash():
    from mobilerun.agent.utils.llm_picker import load_llm

    # no model arg -> the Antigravity consumer default
    assert load_llm("gemini_oauth_code_assist").model == "gemini-3.5-flash-low"


def test_oauth_explicit_default_model_is_honored_not_preset():
    from mobilerun.agent.utils.llm_picker import load_llm

    # explicit model equal to DEFAULT_MODEL must NOT fall through to a preset
    assert (
        load_llm("gemini_oauth_code_assist", model="gemini-3.5-flash-low").model
        == "gemini-3.5-flash-low"
    )


def test_oauth_explicit_agy_models_are_honored():
    from mobilerun.agent.utils.llm_picker import load_llm

    for m in ("gemini-3-flash", "gemini-pro-agent", "gemini-3.1-pro-low"):
        assert load_llm("gemini_oauth_code_assist", model=m).model == m


def test_oauth_preset_key_still_resolves():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    assert (
        GeminiOAuthCodeAssistLLM(model_preset="flash").model == "gemini-3.5-flash-low"
    )


def test_deprecated_models_are_rejected_with_reconfigure_error():
    import pytest

    from mobilerun.agent.utils.llm_picker import load_llm

    for m in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"):
        with pytest.raises(ValueError):
            load_llm("gemini_oauth_code_assist", model=m)


def test_never_sends_project_even_if_passed_as_kwarg():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    payload = llm._to_code_assist_request(
        [ChatMessage(role=MessageRole.USER, content="hi")], project="proj-123"
    )
    assert "project" not in payload


def test_consumer_default_credential_slot_is_antigravity():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        DEFAULT_CREDENTIAL_SLOT,
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    assert llm.credential_slot == DEFAULT_CREDENTIAL_SLOT == "geminiAntigravityOauth"


def test_consumer_mode_uses_antigravity_client_and_aicode_scope():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        DEFAULT_CLIENT_ID,
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    assert llm.client_id == DEFAULT_CLIENT_ID
    assert any("aicode" in s for s in llm.scopes)
    assert "daily-cloudcode-pa" in llm.code_assist_endpoint


def test_flash_lite_preset_resolves_to_picker_model():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    assert (
        GeminiOAuthCodeAssistLLM(model_preset="flash_lite").model
        == "gemini-3.5-flash-extra-low"
    )


def test_consumer_mode_ignores_bare_credential_file(tmp_path):
    import json

    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"access_token": "raw-old", "refresh_token": "raw-r"}))
    llm = GeminiOAuthCodeAssistLLM(credential_path=str(p))  # consumer mode default
    assert llm._cached_access_token is None
    assert llm._cached_refresh_token is None


def test_wizard_oauth_detection_handles_provider_field_names(tmp_path):
    import json

    from mobilerun.cli.configure_wizard import _oauth_credentials_present

    p = tmp_path / "auth.json"
    p.write_text(
        json.dumps(
            {
                "geminiAntigravityOauth": {"access_token": "a", "refresh_token": "b"},
                "claudeAiOauth": {"accessToken": "a", "refreshToken": "b"},
                "openaiOauth": {"access": "a", "refresh": "b"},
            }
        )
    )
    assert _oauth_credentials_present(str(p), "gemini_oauth_code_assist")
    assert _oauth_credentials_present(str(p), "anthropic_oauth")
    assert _oauth_credentials_present(str(p), "openai_oauth")

    p2 = tmp_path / "auth2.json"
    p2.write_text(json.dumps({"openaiOauth": {"access": "a"}}))
    # gemini slot absent -> not detected as present
    assert not _oauth_credentials_present(str(p2), "gemini_oauth_code_assist")
