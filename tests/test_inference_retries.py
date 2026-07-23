import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from mobilerun.agent.utils.inference import (
    _http_status_code,
    acall_with_retries,
    acomplete_with_retries,
    astructured_predict_with_retries,
)


class StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class StructuredResult(BaseModel):
    value: str


class FailingLLM:
    def __init__(self, status_code: int):
        self.error = StatusError(status_code)
        self.calls = 0

    async def achat(self, *, messages):
        self.calls += 1
        raise self.error

    async def acomplete(self, prompt):
        self.calls += 1
        raise self.error

    async def astructured_predict(self, output_cls, prompt, **prompt_args):
        self.calls += 1
        raise self.error


def _run_failing_helper(helper_name: str, status_code: int) -> int:
    llm = FailingLLM(status_code)

    with pytest.raises(StatusError):
        if helper_name == "chat":
            asyncio.run(
                acall_with_retries(
                    llm,
                    [{"role": "user", "content": "hello"}],
                    retries=3,
                    delay=0,
                )
            )
        elif helper_name == "completion":
            asyncio.run(
                acomplete_with_retries(
                    llm,
                    "hello",
                    retries=3,
                    delay=0,
                )
            )
        else:
            asyncio.run(
                astructured_predict_with_retries(
                    llm,
                    StructuredResult,
                    PromptTemplate("Return a value for {value}"),
                    retries=3,
                    delay=0,
                    value="hello",
                )
            )

    return llm.calls


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_permanent_http_client_errors_are_not_retried(
    helper_name: str, status_code: int
) -> None:
    assert _run_failing_helper(helper_name, status_code) == 1


@pytest.mark.parametrize("helper_name", ["chat", "completion", "structured"])
@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500])
def test_transient_http_errors_are_retried(helper_name: str, status_code: int) -> None:
    assert _run_failing_helper(helper_name, status_code) == 3


def test_http_status_code_falls_back_to_exception_response() -> None:
    error = Exception("request failed")
    error.response = SimpleNamespace(status_code=401)

    assert _http_status_code(error) == 401


def test_authentication_error_does_not_sleep_before_failing(monkeypatch) -> None:
    async def fail_if_called(delay):
        pytest.fail("permanent 401 errors must not enter retry backoff")

    monkeypatch.setattr(asyncio, "sleep", fail_if_called)

    assert _run_failing_helper("chat", 401) == 1
