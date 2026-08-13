from __future__ import annotations

from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    DEFAULT_SETUP_TOKEN_SCOPE,
    AnthropicOAuthLLM,
)
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    DEFAULT_MODEL as GEMINI_OAUTH_DEFAULT_MODEL,
)
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    GeminiOAuthCodeAssistLLM,
)
from mobilerun.agent.utils.oauth.grok_oauth_llm import (
    DEFAULT_GROK_MODEL,
    GrokOAuth,
)
from mobilerun.agent.utils.oauth.openai_oauth_llm import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH,
    OpenAIOAuth,
)
from mobilerun.config_manager.auth_profile_store import AuthProfileStore

SETUP_TOKEN_EXPIRES_IN_SECONDS = 365 * 24 * 60 * 60


def run_openai_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    callback_host: str = DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    callback_port: int = DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    callback_path: str = DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    open_browser: bool = True,
) -> None:
    llm = OpenAIOAuth(model=model, oauth_credential_path=credential_path)
    llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        redirect_host=callback_host,
    )


def run_gemini_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    callback_path: str = "/oauth2callback",
    open_browser: bool = True,
) -> None:
    llm = GeminiOAuthCodeAssistLLM(
        model=model or GEMINI_OAUTH_DEFAULT_MODEL,
        credential_path=credential_path,
    )
    llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
    )
    # Verify the Antigravity consumer entitlement resolves before declaring
    # success (catches scope / header / endpoint problems at login time). Raise
    # on failure so the caller does not print a misleading success message.
    try:
        models = llm.fetch_available_models()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Gemini OAuth login saved a token, but the Antigravity entitlement "
            f"check (fetchAvailableModels) failed: {exc}. The login is not "
            "usable; verify your Google One / AI access and retry."
        ) from exc
    print(f"✓ Gemini (Antigravity) login OK — {len(models)} models available.")


def run_grok_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    open_browser: bool = True,
    device_code: bool = False,
    no_browser: bool = False,
) -> None:
    """Authenticate directly with xAI and save Mobilerun-owned credentials."""
    llm = GrokOAuth(
        model=model or DEFAULT_GROK_MODEL,
        oauth_credential_path=credential_path,
    )
    llm.login(
        open_browser=open_browser and not no_browser,
        timeout_seconds=timeout,
        device_code=device_code,
    )


def run_anthropic_setup_token_oauth(
    *,
    timeout: float = 300.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    callback_path: str = "/callback",
    open_browser: bool = True,
) -> str:
    llm = AnthropicOAuthLLM(
        credential_path=None,
        authorize_url="https://claude.com/cai/oauth/authorize",
        login_scope=DEFAULT_SETUP_TOKEN_SCOPE,
    )
    return llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        expires_in=SETUP_TOKEN_EXPIRES_IN_SECONDS,
    )


def save_anthropic_setup_token(credential_path: str, token: str) -> None:
    AuthProfileStore(credential_path).update_slot(
        "claudeAiOauth",
        {
            "accessToken": token,
            "refreshToken": None,
            "expiresAt": None,
            "scopes": [],
        },
    )


def run_anthropic_oauth_setup(credential_path: str) -> None:
    token = run_anthropic_setup_token_oauth()
    save_anthropic_setup_token(credential_path, token)


def get_default_openai_credential_path() -> str:
    return str(DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH)
