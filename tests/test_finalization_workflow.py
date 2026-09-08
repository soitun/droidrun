"""Exercise finalization through the real workflow timer and event stream."""

import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.workflow import Context, StartEvent, Workflow, step
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError

import mobilerun.agent.droid.droid_agent as module
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import (
    FastAgentExecuteEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ResultEvent,
)


class FinishingAgent(MobileAgent):
    """Replace device/LLM setup, retaining MobileAgent.run and finalize."""

    def __init__(self, timeout):
        Workflow.__init__(self, timeout=timeout)
        self.timeout = timeout

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> FastAgentExecuteEvent | ManagerInputEvent | FinalizeEvent:
        await asyncio.sleep(ev.delay)
        return FinalizeEvent(success=ev.success, reason="decided")


def _agent(monkeypatch, timeout, screenshot):
    async def flush(**kwargs):
        pass

    async def state():
        return SimpleNamespace(elements=[])

    monkeypatch.setattr(module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(module, "flush", flush)
    monkeypatch.setattr(module, "_FINALIZE_SCHEDULING_RESERVE_SECONDS", 0.05)
    agent = FinishingAgent(timeout)
    agent.shared_state = SimpleNamespace(
        workflow_completed=False,
        step_number=1,
        visited_packages=set(),
        visited_activities=set(),
        telemetry_config_enabled=False,
    )
    agent.user_id = None
    agent.output_model = None
    agent.config = SimpleNamespace(
        agent=SimpleNamespace(
            manager=SimpleNamespace(vision=True),
            executor=SimpleNamespace(vision=False),
            fast_agent=SimpleNamespace(vision=False),
        ),
        logging=SimpleNamespace(save_trajectory="none", debug=False),
        tracing=SimpleNamespace(langfuse_screenshots=False),
    )
    agent._stream_screenshots = False
    agent.action_ctx = SimpleNamespace(driver=SimpleNamespace(screenshot=screenshot))
    agent.state_provider = SimpleNamespace(get_state=state)
    agent.mcp_manager = None
    return agent


@pytest.mark.parametrize("success", [True, False])
def test_near_deadline_preserves_decision_and_terminates_stream(monkeypatch, success):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        agent = _agent(monkeypatch, 0.6, screenshot)
        handler = agent.run(delay=0.3, success=success)
        events = [event async for event in handler.stream_events()]
        result = await handler
        assert entered.is_set(), "the hung epilog must actually run"
        await asyncio.wait_for(cancelled.wait(), 0.5)
        assert isinstance(result, ResultEvent)
        assert (result.success, result.reason) == (success, "decided")
        assert isinstance(events[-1], ResultEvent)
        assert sum(isinstance(e, FinalizeEvent) for e in events) == 1
        assert sum(isinstance(e, ResultEvent) for e in events) == 1

    asyncio.run(run())


def test_timeout_before_decision_still_fails(monkeypatch):
    async def run():
        async def screenshot():
            raise AssertionError("finalization must not start")

        agent = _agent(monkeypatch, 0.05, screenshot)
        with pytest.raises(WorkflowTimeoutError):
            await agent.run(delay=1, success=True)
        assert agent.shared_state.workflow_completed is False

    asyncio.run(run())


def test_epilog_ceiling_applies_with_plenty_of_workflow_time(monkeypatch):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.15)
        agent = _agent(monkeypatch, 60, screenshot)
        handler = agent.run(delay=0, success=True)

        async def collect():
            return [event async for event in handler.stream_events()], await handler

        events, result = await asyncio.wait_for(collect(), 1)
        assert entered.is_set()
        await asyncio.wait_for(cancelled.wait(), 0.5)
        assert result.success is True
        assert isinstance(events[-1], ResultEvent)

    asyncio.run(run())


def test_user_cancellation_during_epilog_still_cancels_workflow(monkeypatch):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        agent = _agent(monkeypatch, 5, screenshot)
        handler = agent.run(delay=0, success=True)
        await asyncio.wait_for(entered.wait(), 1)
        await handler.cancel_run()
        with pytest.raises(WorkflowCancelledByUser):
            await handler
        await asyncio.wait_for(cancelled.wait(), 1)

    asyncio.run(run())
