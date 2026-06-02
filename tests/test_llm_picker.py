import pytest

from mobilerun.agent.utils.llm_picker import load_llm, normalize_provider_name


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("OpenAI", "OpenAIResponses"),
        ("openai", "OpenAIResponses"),
        ("GPT", "OpenAIResponses"),
        ("Gemini", "GoogleGenAI"),
        ("Google", "GoogleGenAI"),
        ("Claude", "Anthropic"),
        ("OpenAI Compatible", "OpenAILike"),
        ("OpenAI-like", "OpenAILike"),
        ("ZAI", "ZAI"),
        ("Z.AI", "ZAI"),
    ],
)
def test_normalize_provider_name_accepts_user_facing_aliases(
    alias: str, expected: str
) -> None:
    assert normalize_provider_name(alias) == expected


def test_openai_responses_omits_temperature_for_gpt_5_5() -> None:
    llm = load_llm(
        "OpenAIResponses",
        model="gpt-5.5",
        api_key="stub",
        temperature=0.4,
    )

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "gpt-5.5"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_openai_responses_keeps_temperature_for_gpt_5_4() -> None:
    llm = load_llm(
        "OpenAIResponses",
        model="gpt-5.4",
        api_key="stub",
        temperature=0.4,
    )

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 1.0


def test_openai_alias_loads_openai_responses_without_temperature_for_gpt_5_5() -> None:
    llm = load_llm(
        "OpenAI",
        model="gpt-5.5",
        api_key="stub",
        temperature=0.4,
    )

    assert type(llm).__name__ == "MobilerunOpenAIResponses"
    assert "temperature" not in llm._get_model_kwargs()
    assert "top_p" not in llm._get_model_kwargs()


def test_zai_alias_uses_openai_like_transport_defaults() -> None:
    llm = load_llm(
        "ZAI",
        model="glm-5",
        api_key="stub",
    )

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == "https://api.z.ai/api/paas/v4"
