import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import mobilerun.cli.configure_wizard as configure_wizard
import mobilerun.cli.main as cli_main
from mobilerun.cli.configure_wizard import ConfigureWizardCallbacks
from mobilerun.config_manager import MobileConfig


def test_grok_oauth_credentials_are_detected_by_nested_slot(tmp_path) -> None:
    credential_path = tmp_path / "auth-profiles.json"
    credential_path.write_text(
        json.dumps(
            {
                "openaiOauth": {"access": "unrelated"},
                "grokOauth": {"access_token": "grok-access"},
            }
        ),
        encoding="utf-8",
    )

    assert configure_wizard._oauth_credentials_present(
        str(credential_path), "grok_oauth"
    )

    credential_path.write_text(
        json.dumps({"openaiOauth": {"access": "unrelated"}}),
        encoding="utf-8",
    )
    assert not configure_wizard._oauth_credentials_present(
        str(credential_path), "grok_oauth"
    )


def test_wizard_prepares_grok_oauth_with_selected_model(tmp_path) -> None:
    calls: list[dict] = []
    callbacks = ConfigureWizardCallbacks(
        run_openai_oauth_login=lambda **kwargs: None,
        run_anthropic_oauth_login=lambda **kwargs: None,
        run_gemini_oauth_login=lambda **kwargs: None,
        run_grok_oauth_login=lambda **kwargs: calls.append(kwargs),
    )

    configure_wizard._prepare_variant_auth(
        callbacks=callbacks,
        variant=SimpleNamespace(id="grok_oauth"),
        credential_path=str(tmp_path / "auth-profiles.json"),
        selected_model="grok-4.5",
    )

    assert calls == [
        {
            "credential_path": str(tmp_path / "auth-profiles.json"),
            "model": "grok-4.5",
        }
    ]


def test_configure_grok_command_forwards_device_code_options(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        cli_main,
        "_run_grok_oauth_login",
        lambda **kwargs: calls.append(kwargs),
    )
    credential_path = tmp_path / "auth-profiles.json"

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "configure",
            "grok",
            "--credential-path",
            str(credential_path),
            "--model",
            "grok-4.5",
            "--timeout",
            "12",
            "--no-browser",
            "--device-code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "credential_path": str(credential_path),
            "model": "grok-4.5",
            "timeout": 12.0,
            "open_browser": False,
            "device_code": True,
        }
    ]


def test_configure_help_advertises_grok_provider_and_login_command() -> None:
    runner = CliRunner()

    configure_help = runner.invoke(cli_main.cli, ["configure", "--help"])
    grok_help = runner.invoke(cli_main.cli, ["configure", "grok", "--help"])

    assert configure_help.exit_code == 0
    assert "grok" in configure_help.output.lower()
    assert grok_help.exit_code == 0
    assert "--device-code" in grok_help.output
    assert "native xAI OAuth" in grok_help.output


@pytest.mark.parametrize(
    ("auth_mode", "expected_provider"),
    (("api_key", "XAI"), ("oauth", "grok_oauth")),
)
def test_exact_grok_configure_forms_keep_provider_and_auth_fixed(
    monkeypatch, auth_mode: str, expected_provider: str
) -> None:
    config = MobileConfig()
    saved_configs: list[MobileConfig] = []
    login_calls: list[dict] = []
    model_prompts: list[tuple[tuple[str, ...], str]] = []

    monkeypatch.setattr(configure_wizard.ConfigLoader, "load", lambda: config)
    monkeypatch.setattr(
        configure_wizard.ConfigLoader,
        "save",
        lambda saved: saved_configs.append(saved),
    )

    def choose_model(models, *, default_model, allow_back=True):  # type: ignore[no-untyped-def]
        model_prompts.append((tuple(models), default_model))
        return default_model

    monkeypatch.setattr(configure_wizard, "_prompt_model_choice", choose_model)
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_api_key_for_variant",
        lambda variant: ("xai-env-key", "env"),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_oauth_credentials_present",
        lambda credential_path, variant_id: True,
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_oauth_credential_action",
        lambda credential_path: "use_existing",
    )
    monkeypatch.setattr(
        configure_wizard,
        "select_prompt",
        lambda *args, **kwargs: pytest.fail(
            "fixed Grok configure flow unexpectedly reopened the top-level menu"
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "_run_grok_oauth_login",
        lambda **kwargs: login_calls.append(kwargs),
    )

    result = CliRunner().invoke(
        cli_main.cli,
        ["configure", "--provider", "grok", "--auth-mode", auth_mode],
    )

    assert result.exit_code == 0, result.output
    assert saved_configs == [config]
    assert login_calls == []
    assert model_prompts == [(("grok-4.5",), "grok-4.5")]
    assert {
        (profile.provider, profile.provider_family, profile.auth_mode, profile.model)
        for profile in config.llm_profiles.values()
    } == {(expected_provider, "grok", auth_mode, "grok-4.5")}
