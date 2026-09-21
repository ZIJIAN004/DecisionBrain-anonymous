"""通用 CoreAgent Runtime 集成测试。"""

import asyncio
import json

import pytest

from decisionbrain.config import Settings
from decisionbrain.core.models import (
    AgentInput,
    AgentStage,
    ArtifactDraft,
    ArtifactProduced,
    AssistantMessage,
    ClarificationQuestion,
    Failed,
    NeedsClarification,
    StageFinished,
    StageStarted,
    Succeeded,
)
from decisionbrain.events import EVENT_PROGRESS_REPORTED, EventLevel

from decisionbrain.exceptions import AgentRuntimeError
from decisionbrain.runtime import AgentRuntime
from decisionbrain.runtime.agent_runtime import generate_run_id
from decisionbrain.run_storage import RunRepository, RunStatus
from helpers import MemoryEventSink


class ScriptedAgent:
    version = "test-1"

    def __init__(self, services=None, *, mode="success"):
        self.mode = mode
        self.state = {}
        self._services = services

    def initialize(self, agent_input: AgentInput):
        self.state = {"prompt": agent_input.problem_description, "stage": "start"}

    def resume(self, user_message):
        self.state["answer"] = user_message
        self.state["stage"] = "done"

    def outcome(self):
        if self.mode == "failed":
            return Failed(message="确定性失败")
        if self.mode == "clarify" and self.state["stage"] == "start":
            return NeedsClarification(
                message="需要补充",
                questions=(ClarificationQuestion(id="q1", text="补充内容"),),
            )
        return Succeeded(message="完成", summary={"objective": 3})

    async def outputs(self, control):
        decision = self.outcome()
        if isinstance(decision, NeedsClarification):
            yield decision
            while self.state["stage"] == "start":
                if await control.is_cancelled():
                    raise asyncio.CancelledError
                await asyncio.sleep(0.01)
            yield self.outcome()
            return
        if isinstance(decision, Failed):
            yield decision
            return

        yield StageStarted(stage=AgentStage.SOLVING, message="开始")
        yield AssistantMessage(stage=AgentStage.SOLVING, message="求解中")
        yield ArtifactProduced(
            stage=AgentStage.SOLVING,
            message="结果",
            artifacts=(
                ArtifactDraft(
                    relative_path="result/solution.json",
                    content='{"objective":3}',
                ),
            ),
        )
        if self.mode == "artifact_revision":
            yield ArtifactProduced(
                stage=AgentStage.SOLVING,
                message="修复后的结果",
                artifacts=(
                    ArtifactDraft(
                        relative_path="result/solution.json",
                        content='{"objective":2}',
                    ),
                ),
            )
        self.state["stage"] = "done"
        yield self.outcome()

    def snapshot_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


class WorkspaceAwareAgent(ScriptedAgent):
    configured_roots: list[str] = []
    initialized_workspace_roots: list[str | None] = []

    def __init__(self, services=None):
        super().__init__(services)
        WorkspaceAwareAgent.configured_roots.append(services.config.workspace_root)

    def initialize(self, agent_input: AgentInput):
        super().initialize(agent_input)
        WorkspaceAwareAgent.initialized_workspace_roots.append(self._services.config.workspace_root)


def run(coro):
    return asyncio.run(coro)


def make_runtime(settings, monkeypatch, agent_factory):
    monkeypatch.setenv("DBN_RUNS_DIR", str(settings.runs_dir))
    return AgentRuntime(settings, agent_factory=agent_factory)


def test_success_persists_events_artifact_and_unique_terminal(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(settings, monkeypatch, ScriptedAgent).run(
            workspace=tmp_path,
            command="success",
            initial_input="问题",
            sink=sink,
        )
    )
    repository = RunRepository(settings)
    run_id = sink.events[0].run_id
    assert repository.read_record(run_id).status is RunStatus.SUCCEEDED
    report_path = repository.run_path(run_id) / "agent-run.html"
    assert report_path.is_file()
    assert "Agent started" in report_path.read_text(encoding="utf-8")
    assert not (repository.run_path(run_id) / "events.jsonl").exists()
    assert not (repository.run_path(run_id) / "state").exists()
    artifacts = repository.list_artifacts(run_id)
    assert artifacts[0].relative_path == "artifacts/result/solution.json"
    terminals = [event for event in sink.events if event.type == "run_finished"]
    assert len(terminals) == 1
    assert terminals[0].payload["status"] == "succeeded"


def test_success_persists_structured_timing_report(tmp_path, monkeypatch):
    class TimedAgent(ScriptedAgent):
        async def outputs(self, control):
            yield StageStarted(stage=AgentStage.SOLVING, message="开始")
            yield StageFinished(stage=AgentStage.SOLVING, message="完成")
            yield Succeeded(message="完成", summary={"objective": 3})

    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(settings, monkeypatch, TimedAgent).run(
            workspace=tmp_path,
            command="timing",
            initial_input="问题",
            sink=sink,
        )
    )

    timing = json.loads((settings.runs_dir / sink.events[0].run_id / "timing.json").read_text())
    assert timing["schema_version"] == "1.2"
    assert timing["stages"][0]["stage"] == "solving"
    assert timing["stages"][0]["duration_ms"] >= 0
    assert timing["summary"]["self_checker_execution_count"] == 0


def test_runtime_config_can_disable_timing_report(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(settings, monkeypatch, ScriptedAgent).run(
            workspace=tmp_path,
            command="no timing",
            config={"timing_enabled": False},
            initial_input="问题",
            sink=sink,
        )
    )

    run_path = settings.runs_dir / sink.events[0].run_id
    assert not (run_path / "timing.json").exists()
    assert RunRepository(settings).read_record(sink.events[0].run_id).status is RunStatus.SUCCEEDED


def test_runtime_writes_package_usage_report_when_enabled(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")

    run(
        make_runtime(settings, monkeypatch, ScriptedAgent).run(
            workspace=tmp_path,
            command="package usage",
            config={"algorithm_package_stats_enabled": True},
            initial_input="问题",
            sink=sink,
        )
    )

    report_path = settings.runs_dir / sink.events[0].run_id / "package-usage.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["attempts"] == []
    assert report["assessment"]["semantic_selection_judgment"] == (
        "requires_independent_review"
    )


def test_runtime_config_can_disable_package_usage_report(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")

    run(
        make_runtime(settings, monkeypatch, ScriptedAgent).run(
            workspace=tmp_path,
            command="no package usage",
            config={"algorithm_package_stats_enabled": False},
            initial_input="问题",
            sink=sink,
        )
    )

    run_path = settings.runs_dir / sink.events[0].run_id
    assert not (run_path / "package-usage.json").exists()


def test_runtime_configures_workspace_tools_with_execution_snapshot(tmp_path, monkeypatch):
    WorkspaceAwareAgent.configured_roots = []
    WorkspaceAwareAgent.initialized_workspace_roots = []
    (tmp_path / "source.txt").write_text("source data", encoding="utf-8")
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")

    run(
        make_runtime(settings, monkeypatch, WorkspaceAwareAgent).run(
            workspace=tmp_path,
            command="workspace tools",
            initial_input="问题",
            sink=sink,
        )
    )

    run_id = sink.events[0].run_id
    execution_workspace = tmp_path / "runs" / run_id / "workspace"
    assert WorkspaceAwareAgent.configured_roots == [str(execution_workspace)]
    assert WorkspaceAwareAgent.initialized_workspace_roots == [str(execution_workspace)]
    assert (execution_workspace / "source.txt").read_text(encoding="utf-8") == "source data"


def test_runtime_can_start_from_prebuilt_problem_contract_stage(tmp_path, monkeypatch):
    class StartStageAgent(ScriptedAgent):
        initialized = False
        restored = []

        def initialize(self, agent_input):
            StartStageAgent.initialized = True
            super().initialize(agent_input)

        def restore_state_for_stage(self, state, stage):
            StartStageAgent.restored.append((state, stage))
            self.state = {"stage": "done"}

    StartStageAgent.initialized = False
    StartStageAgent.restored = []
    (tmp_path / "problem.md").write_text("完整 benchmark 问题", encoding="utf-8")
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "stage_outputs" / "intake.json").write_text(
        json.dumps(
            {
                "decision": "proceed",
                "problem_definition": {"objective": {"direction": "minimize"}},
            }
        ),
        encoding="utf-8",
    )
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")

    run(
        make_runtime(settings, monkeypatch, StartStageAgent).run(
            workspace=tmp_path,
            command="benchmark",
            initial_input="不应触发 intake",
            start_stage=AgentStage.PROBLEM_CONTRACT,
            sink=sink,
        )
    )

    assert StartStageAgent.initialized is False
    assert len(StartStageAgent.restored) == 1
    state, stage = StartStageAgent.restored[0]
    assert stage is AgentStage.PROBLEM_CONTRACT
    assert state["problem_description"] == "完整 benchmark 问题"
    assert state["problem_definition"]["confirmed"] is True


def test_resume_reconstructs_upstream_state_without_checkpoint(tmp_path, monkeypatch):
    class ResumeAgent(ScriptedAgent):
        restored = []

        def restore_state_for_stage(self, state, stage):
            ResumeAgent.restored.append((state, stage))
            self.state = {"stage": "done"}

    ResumeAgent.restored = []
    source = tmp_path / "source"
    stage_outputs = source / "stage_outputs"
    stage_outputs.mkdir(parents=True)
    (source / "problem.md").write_text("配送问题\n", encoding="utf-8")
    (stage_outputs / "intake.json").write_text(
        json.dumps({"decision": "proceed", "problem_definition": {"problem_type": "routing"}}),
        encoding="utf-8",
    )
    (stage_outputs / "problem_contract.json").write_text(
        json.dumps({"constraints": [{"name": "capacity"}]}),
        encoding="utf-8",
    )
    (stage_outputs / "algorithm_design.json").write_text(
        json.dumps({"selected_algorithm": "pyvrp"}),
        encoding="utf-8",
    )

    settings = Settings(runs_dir=tmp_path / "runs")
    repository = RunRepository(settings)
    parent = repository.create_run_record(
        run_id=generate_run_id(),
        workspace=source,
        command="parent",
    )
    repository.update_record(parent.run_id, status=RunStatus.FAILED)
    sink = MemoryEventSink()
    runtime = AgentRuntime(settings, repository=repository, agent_factory=ResumeAgent)

    run(
        runtime.resume_run(
            sink=sink,
            parent_run_id=parent.run_id,
            resume_stage=AgentStage.SOLVING,
            command="resume solving",
        )
    )

    restored, stage = ResumeAgent.restored[0]
    assert stage is AgentStage.SOLVING
    assert restored["problem_definition"]["problem_type"] == "routing"
    assert restored["problem_contract"]["constraints"][0]["name"] == "capacity"
    assert restored["algorithm_design"]["selected_algorithm"] == "pyvrp"
    derived_run_id = sink.events[0].run_id
    assert not (repository.run_path(derived_run_id) / "state").exists()
    assert (repository.run_path(derived_run_id) / "agent-run.html").is_file()


def test_debug_runtime_publishes_llm_stream_events(tmp_path, monkeypatch):
    class DebugStreamLLM:
        def __init__(self):
            self.handler = None

        def set_debug_stream_handler(self, handler):
            self.handler = handler

        async def emit(self):
            assert self.handler is not None
            await self.handler(
                {
                    "kind": "started",
                    "request_id": "request-1",
                    "role": "intake",
                    "message_count": 1,
                }
            )
            await self.handler(
                {
                    "kind": "chunk",
                    "request_id": "request-1",
                    "role": "intake",
                    "channel": "reasoning",
                    "text": "先判断。",
                }
            )
            await self.handler(
                {
                    "kind": "chunk",
                    "request_id": "request-1",
                    "role": "intake",
                    "channel": "code",
                    "text": '{"decision":"proceed"}',
                }
            )
            await self.handler(
                {
                    "kind": "finished",
                    "request_id": "request-1",
                    "role": "intake",
                    "status": "ok",
                    "content_chars": 22,
                    "reasoning_chars": 4,
                }
            )

    class DebugStreamAgent(ScriptedAgent):
        def __init__(self, services=None):
            super().__init__(services)
            self.llm = DebugStreamLLM()
            self.llm.set_debug_stream_handler(services.debug_event_handler)

        async def outputs(self, control):
            await self.llm.emit()
            yield Succeeded(message="完成", summary={"objective": 3})

    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs", debug=True)
    run(
        AgentRuntime(settings, agent_factory=DebugStreamAgent).run(
            workspace=tmp_path,
            command="debug stream",
            initial_input="问题",
            sink=sink,
        )
    )

    stream_events = [event for event in sink.events if event.type.startswith("llm_stream_")]
    assert [event.type for event in stream_events] == [
        "llm_stream_started",
        "llm_stream_chunk",
        "llm_stream_chunk",
        "llm_stream_finished",
    ]
    assert all(event.level is EventLevel.DEBUG for event in stream_events)
    assert stream_events[1].message == "先判断。"
    assert stream_events[2].message == '{"decision":"proceed"}'
    assert stream_events[0].stage is AgentStage.INTAKE


def test_debug_runtime_publishes_compressed_workspace_tool_events(tmp_path, monkeypatch):
    class DebugToolLLM:
        def __init__(self):
            self.handler = None

        def set_debug_stream_handler(self, handler):
            self.handler = handler

    class DebugToolAgent(WorkspaceAwareAgent):
        def __init__(self, services=None):
            super().__init__(services)
            self.llm = DebugToolLLM()
            self.llm.set_debug_stream_handler(services.debug_event_handler)

        async def outputs(self, control):
            assert self.llm.handler is not None
            await self.llm.handler(
                {
                    "kind": "tool_call_started",
                    "role": "algorithm_design",
                    "tool_call_id": "call-1",
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "out.txt",
                        "content": "x" * 1000,
                    },
                }
            )
            await self.llm.handler(
                {
                    "kind": "tool_result",
                    "role": "algorithm_design",
                    "tool_call_id": "call-1",
                    "tool_name": "shell",
                    "result": "Exit code: 0\nWall time: 0.1 seconds\nOutput:\n" + ("y" * 1000),
                }
            )
            yield Succeeded(message="完成", summary={"objective": 3})

    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs", debug=True)
    run(
        AgentRuntime(settings, agent_factory=DebugToolAgent).run(
            workspace=tmp_path,
            command="debug tool",
            initial_input="问题",
            sink=sink,
        )
    )

    tool_call = next(event for event in sink.events if event.type == "tool_call")
    tool_result = next(event for event in sink.events if event.type == "tool_result")
    assert tool_call.stage is AgentStage.ALGORITHM_DESIGN
    assert tool_call.payload["arguments"]["content"]["chars"] == 1000
    assert len(tool_call.payload["arguments"]["content"]["preview"]) < 230
    assert "x" * 1000 not in str(tool_call.payload)
    assert tool_result.payload["exit_code"] == 0
    assert tool_result.payload["wall_time_seconds"] == 0.1
    assert tool_result.payload["result_chars"] > 500
    assert len(tool_result.message) <= 514
    assert "y" * 1000 not in tool_result.message


def test_runtime_publishes_stage_progress_events(tmp_path, monkeypatch):
    class StageEventAgent(ScriptedAgent):
        async def outputs(self, control):
            handler = getattr(self._services, "stage_event_handler", None)
            assert handler is not None
            await handler(
                {
                    "kind": EVENT_PROGRESS_REPORTED,
                    "agent_stage": "solving",
                    "payload": {
                        "stage": "solving",
                        "summary": "已经生成可行解",
                        "completed": ["运行 solver.py"],
                        "key_findings": ["硬约束自检通过"],
                        "assumptions": [],
                        "next_step": "生成解释",
                    },
                }
            )
            yield Succeeded(message="完成", summary={"objective": 3})

    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(settings, monkeypatch, StageEventAgent).run(
            workspace=tmp_path,
            command="stage events",
            initial_input="问题",
            sink=sink,
        )
    )

    progress = next(event for event in sink.events if event.type == EVENT_PROGRESS_REPORTED)
    assert progress.stage is AgentStage.SOLVING
    assert progress.level is EventLevel.INFO
    assert progress.message == "已经生成可行解"
    assert progress.payload["next_step"] == "生成解释"


def test_repair_artifact_uses_stable_revision_path(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(
            settings,
            monkeypatch,
            lambda services: ScriptedAgent(services, mode="artifact_revision"),
        ).run(
            workspace=tmp_path,
            command="repair",
            initial_input="问题",
            sink=sink,
        )
    )

    repository = RunRepository(settings)
    run_record = repository.list_runs()[0]
    assert run_record.status is RunStatus.SUCCEEDED
    assert {item.relative_path for item in repository.list_artifacts(run_record.run_id)} == {
        "artifacts/result/solution.json",
        "artifacts/result/solution-2.json",
    }


def test_failed_outcome_is_unique_terminal(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    run(
        make_runtime(
            settings, monkeypatch, lambda services: ScriptedAgent(services, mode="failed")
        ).run(
            workspace=tmp_path,
            command="failed",
            initial_input="问题",
            sink=sink,
        )
    )
    repository = RunRepository(settings)
    assert repository.read_record(sink.events[0].run_id).status is RunStatus.FAILED
    assert [event.type for event in sink.events[-2:]] == ["error", "run_finished"]


def test_clarification_continues_inside_live_runtime(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    runtime = make_runtime(
        settings, monkeypatch, lambda services: ScriptedAgent(services, mode="clarify")
    )

    async def scenario():
        task = asyncio.create_task(
            runtime.run(
                workspace=tmp_path,
                command="clarify",
                initial_input="排程",
                sink=sink,
            )
        )
        for _ in range(100):
            if any(event.type == "clarification_requested" for event in sink.events):
                break
            await asyncio.sleep(0.01)
        run_id = sink.events[0].run_id
        await runtime.submit_user_message(user_message="每天 8 小时")
        await task
        return run_id

    run_id = run(scenario())
    repository = RunRepository(settings)
    assert repository.read_record(run_id).status is RunStatus.SUCCEEDED
    assert not (tmp_path / "runs" / run_id / "state").exists()
    assert (tmp_path / "runs" / run_id / "agent-run.html").is_file()


def test_user_message_gate_uses_runtime_status_not_persisted_status(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    runtime = make_runtime(
        settings, monkeypatch, lambda services: ScriptedAgent(services, mode="clarify")
    )

    async def scenario():
        task = asyncio.create_task(
            runtime.run(
                workspace=tmp_path,
                command="clarify",
                initial_input="排程",
                sink=sink,
            )
        )
        for _ in range(100):
            if any(event.type == "clarification_requested" for event in sink.events):
                break
            await asyncio.sleep(0.01)
        run_id = sink.events[0].run_id
        runtime.repository.update_record(run_id, status=RunStatus.RUNNING)
        await runtime.submit_user_message(user_message="每天 8 小时")
        await task
        return run_id

    run_id = run(scenario())
    repository = RunRepository(settings)
    assert repository.read_record(run_id).status is RunStatus.SUCCEEDED


def test_clarification_can_pause_without_terminal_status(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    settings = Settings(runs_dir=tmp_path / "runs")
    runtime = make_runtime(
        settings, monkeypatch, lambda services: ScriptedAgent(services, mode="clarify")
    )

    async def scenario():
        task = asyncio.create_task(
            runtime.run(
                workspace=tmp_path,
                command="clarify",
                initial_input="排程",
                sink=sink,
            )
        )
        for _ in range(100):
            if any(event.type == "clarification_requested" for event in sink.events):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await task
        return sink.events[0].run_id

    run_id = run(scenario())
    repository = RunRepository(settings)
    assert repository.read_record(run_id).status is RunStatus.NEEDS_CLARIFICATION


def test_unknown_report_fails_loudly(tmp_path, monkeypatch):
    class BadAgent(ScriptedAgent):
        async def outputs(self, control):
            yield object()

    with pytest.raises(AgentRuntimeError, match="unknown output"):
        run(
            make_runtime(Settings(runs_dir=tmp_path / "runs"), monkeypatch, BadAgent).run(
                workspace=tmp_path,
                command="bad report",
                initial_input="问题",
                sink=MemoryEventSink(),
            )
        )


def test_missing_stage_result_fails_loudly(tmp_path, monkeypatch):
    class BadAgent(ScriptedAgent):
        async def outputs(self, control):
            raise AgentRuntimeError("Core Agent 未产生有效 StageResult")
            yield  # pragma: no cover

    with pytest.raises(AgentRuntimeError, match="StageResult"):
        run(
            make_runtime(Settings(runs_dir=tmp_path / "runs"), monkeypatch, BadAgent).run(
                workspace=tmp_path,
                command="missing result",
                initial_input="问题",
                sink=MemoryEventSink(),
            )
        )


def test_repository_rejects_schema_v1_run_record(tmp_path):
    repository = RunRepository(Settings(runs_dir=tmp_path / "runs"))
    run_record = repository.create_run_record(
        run_id=generate_run_id(), workspace=tmp_path, command="schema"
    )
    path = repository.run_path(run_record.run_id) / "run.json"
    text = path.read_text(encoding="utf-8").replace(
        '"schema_version": "2.0"', '"schema_version": "1.0"'
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match="schema_version"):
        repository.read_record(run_record.run_id)
