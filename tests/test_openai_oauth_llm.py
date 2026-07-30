import json

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from mobilerun.agent.providers import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
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


@pytest.mark.parametrize(
    "model_alias",
    (
        "gpt-5.6",
        "openai/gpt-5.6",
        "openai-codex/gpt-5.6",
    ),
)
def test_openai_oauth_normalizes_gpt_5_6_aliases(tmp_path, model_alias: str) -> None:
    llm = OpenAIOAuth(
        model=model_alias,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "gpt-5.6-sol"
    assert llm.metadata.model_name == "gpt-5.6-sol"
    assert llm.metadata.context_window == 400_000


def test_openai_oauth_normalizes_auth_model_alias(tmp_path) -> None:
    llm = OpenAIOAuth(
        auth_model="openai-codex/gpt-5.6",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "gpt-5.6-sol"


def test_openai_oauth_preserves_explicit_custom_model(tmp_path) -> None:
    llm = OpenAIOAuth(
        custom_model="acme/custom-reasoning-model",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "acme/custom-reasoning-model"


@pytest.mark.parametrize(
    ("auth_mode", "model_alias"),
    (
        ("api_key", "gpt-5.6"),
        ("api_key", "openai/gpt-5.6"),
        ("oauth", "gpt-5.6"),
        ("oauth", "openai/gpt-5.6"),
        ("oauth", "openai-codex/gpt-5.6"),
    ),
)
def test_openai_setup_profiles_normalize_gpt_5_6_aliases(
    auth_mode: str, model_alias: str
) -> None:
    variant = resolve_provider_variant("openai", auth_mode)
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="openai",
            variant_id=variant.id,
            auth_mode=auth_mode,
            model=model_alias,
            api_key_source="env",
        ),
    )

    assert profile.model == "gpt-5.6-sol"
    assert profile.provider == variant.runtime_provider_name


def test_openai_setup_profile_preserves_unknown_custom_model() -> None:
    variant = resolve_provider_variant("openai", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="openai",
            variant_id=variant.id,
            auth_mode="api_key",
            model="acme/custom-reasoning-model",
            api_key_source="env",
        ),
    )

    assert profile.model == "acme/custom-reasoning-model"


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
