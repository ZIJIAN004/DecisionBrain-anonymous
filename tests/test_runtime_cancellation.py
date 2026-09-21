"""Runtime 取消测试。"""

import asyncio

from decisionbrain.core.models import AgentInput, Succeeded

from decisionbrain.runtime import AgentRuntime
from decisionbrain.run_storage import RunStatus
from helpers import MemoryEventSink


class SlowAgent:
    version = "slow-1"

    def __init__(self, services=None):
        pass

    def initialize(self, agent_input: AgentInput):
        self.state = {"stage": "start"}

    def resume(self, user_message):
        return

    async def outputs(self, control):
        await asyncio.sleep(30)
        yield Succeeded(message="完成", summary={})

    def snapshot_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


def test_cancellation_updates_manifest_and_emits_one_terminal(tmp_path, monkeypatch):
    async def scenario():
        sink = MemoryEventSink()
        monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
        runtime = AgentRuntime(agent_factory=SlowAgent)
        task = asyncio.create_task(
            runtime.run(
                workspace=tmp_path,
                command="cancel",
                initial_input="问题",
                sink=sink,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        await task
        return runtime.repository, sink

    repository, sink = asyncio.run(scenario())
    assert repository.read_record(sink.events[0].run_id).status is RunStatus.CANCELLED
    terminals = [event for event in sink.events if event.type == "run_finished"]
    assert len(terminals) == 1
    assert terminals[0].payload["status"] == "cancelled"
