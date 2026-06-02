from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.utils.llm_picker import load_llm


def test_gemini_oauth_profile_sends_gemini_3_5_flash_verbatim(tmp_path) -> None:
    variant = resolve_provider_variant("gemini", "oauth")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="gemini",
            variant_id=variant.id,
            auth_mode="oauth",
            model="gemini-3.5-flash",
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
    assert profile.model == "gemini-3.5-flash"
    assert payload["model"] == "gemini-3.5-flash"
