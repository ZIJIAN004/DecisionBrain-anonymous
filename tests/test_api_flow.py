import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from decisionbrain.api.app import app
from decisionbrain.api.execution import RunExecutionManager
from decisionbrain.exceptions import RunCapacityError
from decisionbrain.core.models import (
    AgentInput,
    AgentStage,
    ArtifactDraft,
    ArtifactProduced,
    AssistantMessage,
    ClarificationQuestion,
    Failed,
    NeedsClarification,
    StageStarted,
    StreamChannel,
    Succeeded,
)
from decisionbrain.runtime import AgentRuntime
from decisionbrain.runtime.agent_runtime import generate_run_id
from decisionbrain.run_storage import RunStatus


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        data = next(
            (line[5:].strip() for line in block.splitlines() if line.startswith("data:")), None
        )
        if data:
            events.append(json.loads(data))
    return events


class SuccessAgent:
    version = "1.0"

    def __init__(self, services=None):
        self.state = {}

    def initialize(self, agent_input: AgentInput):
        self.state = {"prompt": agent_input.problem_description, "stage": "run"}

    def resume(self, user_message):
        self.state = {**self.state, "answer": user_message, "stage": "done"}

    def outcome(self):
        return Succeeded(message="求解完成", summary={"objective": 3})

    async def outputs(self, control):
        decision = self.outcome()
        if isinstance(decision, NeedsClarification):
            yield decision
            while self.state.get("stage") == "waiting":
                if await control.is_cancelled():
                    raise asyncio.CancelledError
                await asyncio.sleep(0.01)
            yield self.outcome()
            return
        if isinstance(decision, Failed):
            yield decision
            return

        yield StageStarted(
            stage=AgentStage.SOLVING,
            message="开始求解",
        )
        yield AssistantMessage(
            stage=AgentStage.SOLVING,
            message="正在求解",
            stream=StreamChannel.REASONING,
        )
        yield ArtifactProduced(
            stage=AgentStage.EXPLANATION,
            message="结果已生成",
            artifacts=(
                ArtifactDraft(
                    relative_path="result/solution.json",
                    content='{"status":"optimal","objective":3}',
                    media_type="application/json",
                    category="result",
                ),
                ArtifactDraft(
                    relative_path="code/solver.py",
                    content='print("ok")',
                    media_type="text/x-python",
                    category="code",
                ),
            ),
        )
        self.state["stage"] = "done"
        yield self.outcome()

    def snapshot_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


class ErrorAgent(SuccessAgent):
    async def outputs(self, control):
        yield Failed(message="确定性失败")


class ClarificationAgent(SuccessAgent):
    def initialize(self, agent_input):
        self.state = {"stage": "waiting"}

    def outcome(self):
        if self.state["stage"] == "waiting":
            return NeedsClarification(
                message="需要补充信息",
                questions=(ClarificationQuestion(id="q1", text="产能上限是多少？"),),
            )
        return Succeeded(
            message="澄清后完成",
            summary={"answer": self.state["answer"]},
        )


def runtime_factory(agent_factory, monkeypatch):
    return lambda: AgentRuntime(agent_factory=agent_factory)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
    with TestClient(app) as client:
        app.state.run_manager.runtime_factory = runtime_factory(SuccessAgent, monkeypatch)
        yield client, app.state.run_manager


def test_openapi_is_run_centric(api_client):
    client, _manager = api_client
    paths = set(app.openapi()["paths"])
    assert "/api/runs" in paths
    assert "/api/runs/{run_id}/continue" in paths
    assert "/api/runs/{run_id}/cancel" in paths
    assert "/api/runs/{run_id}/report" in paths
    assert not any(path.startswith("/api/opt") for path in paths)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/api/opt/chat", json={"message": "x"}).status_code == 404


def test_create_streams_agent_events_and_generates_html_report(api_client):
    client, manager = api_client
    response = client.post(
        "/api/runs",
        data={"prompt": "安排生产任务", "title": "生产排程"},
        files=[
            ("files", ("orders.csv", b"job,duration\nA,3\n", "text/csv")),
            ("files", ("orders.csv", b"job,duration\nB,4\n", "text/csv")),
            ("files", ("capacity.json", b'{"A": 8}', "application/json")),
        ],
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run_created" in response.text

    streamed = parse_sse(response.text)
    run_id = streamed[0]["run_id"]
    assert [event["sequence"] for event in streamed] == list(range(1, len(streamed) + 1))
    assert manager.repository.read_events(run_id) == []
    assert not (manager.repository.run_path(run_id) / "events.jsonl").exists()
    assert streamed[-1]["type"] == "run_finished"
    assert streamed[-1]["payload"]["status"] == "succeeded"
    assert sum(event["type"] == "run_finished" for event in streamed) == 1

    run_record = manager.repository.read_record(run_id)
    assert run_record.status == RunStatus.SUCCEEDED
    report = client.get(f"/api/runs/{run_id}/report")
    assert report.status_code == 200
    assert "DecisionBrain Agent Run" in report.text
    run_workspace = manager.repository.run_path(run_id) / "workspace"
    assert (run_workspace / "problem.md").read_text(encoding="utf-8") == "安排生产任务\n"
    data_snapshot = run_workspace / "data"
    assert sorted(path.name for path in data_snapshot.iterdir()) == [
        "capacity.json",
        "orders-2.csv",
        "orders.csv",
    ]


def test_history_artifact_listing_and_download(api_client):
    client, _manager = api_client
    streamed = parse_sse(client.post("/api/runs", data={"prompt": "solve", "title": "demo"}).text)
    run_id = streamed[0]["run_id"]

    listing = client.get("/api/runs").json()["runs"]
    assert listing[0]["run_id"] == run_id
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "succeeded"
    assert client.get(f"/api/runs/{run_id}/events").json()["events"] == []
    assert client.get(f"/api/runs/{run_id}/report").status_code == 200

    artifacts = client.get(f"/api/runs/{run_id}/artifacts").json()["artifacts"]
    code = next(item for item in artifacts if item["category"] == "code")
    downloaded = client.get(f"/api/runs/{run_id}/artifacts/{code['artifact_id']}")
    assert downloaded.status_code == 200
    assert downloaded.text == 'print("ok")'
    assert client.get(f"/api/runs/{run_id}/artifacts/missing").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifacts/..%2Frun.json").status_code == 404
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


def test_display_view_filters_debug_and_stream_previews(api_client):
    client, _manager = api_client
    streamed = parse_sse(client.post("/api/runs", data={"prompt": "solve", "view": "display"}).text)
    run_id = streamed[0]["run_id"]

    assert "run_created" in [event["type"] for event in streamed]
    assert "stage_started" in [event["type"] for event in streamed]
    assert "artifact_created" in [event["type"] for event in streamed]
    assert streamed[-1]["type"] == "run_finished"
    assert all(event["level"] != "debug" for event in streamed)
    assert all(
        not (
            event["type"] == "assistant_message"
            and event.get("payload", {}).get("stream") in {"reasoning", "code"}
        )
        for event in streamed
    )

    assert client.get(f"/api/runs/{run_id}/events").json()["events"] == []
    assert client.get(f"/api/runs/{run_id}/events?view=display").json()["events"] == []
    assert client.get(f"/api/runs/{run_id}/report").status_code == 200


def test_error_action_is_a_unique_failed_terminal_event(api_client, monkeypatch):
    client, manager = api_client
    manager.runtime_factory = runtime_factory(ErrorAgent, monkeypatch)
    events = parse_sse(client.post("/api/runs", data={"prompt": "fail"}).text)
    assert [event["type"] for event in events[-2:]] == ["error", "run_finished"]
    assert events[-1]["payload"]["status"] == "failed"
    assert sum(event["type"] == "run_finished" for event in events) == 1
    assert manager.repository.read_record(events[0]["run_id"]).status == RunStatus.FAILED


def test_clarification_submits_message_to_live_runtime(api_client, monkeypatch):
    client, manager = api_client
    manager.runtime_factory = runtime_factory(ClarificationAgent, monkeypatch)
    first = parse_sse(client.post("/api/runs", data={"prompt": "need details"}).text)
    run_id = first[0]["run_id"]
    assert first[-1]["type"] == "clarification_requested"
    assert manager.repository.read_record(run_id).status == RunStatus.NEEDS_CLARIFICATION

    second = parse_sse(
        client.post(f"/api/runs/{run_id}/continue", json={"message": "每天 8 小时"}).text
    )
    assert second[0]["type"] == "user_message"
    assert second[-1]["type"] == "run_finished"
    assert second[-1]["payload"]["status"] == "succeeded"


def test_run_id_errors_have_http_contract(api_client):
    client, _manager = api_client
    assert client.get("/api/runs/not-a-run").status_code == 400
    missing = "20260101-000000Z-deadbeef"
    assert client.get(f"/api/runs/{missing}").status_code == 404
    assert client.post(f"/api/runs/{missing}/cancel").status_code == 404


def test_explicit_cancel_and_detached_stream_keep_persistence(tmp_path, monkeypatch):
    class SlowAgent(SuccessAgent):
        async def outputs(self, control):
            yield StageStarted(stage=AgentStage.INTAKE, message="启动")
            await asyncio.sleep(30)
            yield self.outcome()

    async def scenario():
        monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
        workspace = tmp_path / "workspace"
        (workspace / "data").mkdir(parents=True)
        manager = RunExecutionManager(
            runtime_factory=runtime_factory(SlowAgent, monkeypatch),
        )
        stream = manager.start(workspace=workspace, prompt="slow")
        iterator = stream.events()
        created = await iterator.__anext__()
        await iterator.__anext__()
        run_id = created.run_id
        stream.detach()
        assert not manager._tasks[run_id].done()
        await manager.cancel(run_id)
        assert manager.repository.read_record(run_id).status == RunStatus.CANCELLED
        assert manager.repository.read_events(run_id) == []
        assert (manager.repository.run_path(run_id) / "agent-run.html").is_file()
        await manager.shutdown()

    asyncio.run(scenario())


def test_detached_stream_does_not_cancel_background_run(tmp_path, monkeypatch):
    class BriefAgent(SuccessAgent):
        async def outputs(self, control):
            yield StageStarted(stage=AgentStage.INTAKE, message="启动")
            await asyncio.sleep(0.01)
            yield self.outcome()

    async def scenario():
        monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        manager = RunExecutionManager(
            runtime_factory=runtime_factory(BriefAgent, monkeypatch),
        )
        stream = manager.start(workspace=workspace, prompt="continue in background")
        iterator = stream.events()
        created = await iterator.__anext__()
        await iterator.__anext__()
        task = manager._tasks[created.run_id]
        stream.detach()
        await task
        assert manager.repository.read_record(created.run_id).status == RunStatus.SUCCEEDED
        assert manager.repository.read_events(created.run_id) == []
        assert (manager.repository.run_path(created.run_id) / "agent-run.html").is_file()

    asyncio.run(scenario())


def test_run_manager_enforces_max_active_runs(tmp_path, monkeypatch):
    class SlowAgent(SuccessAgent):
        async def outputs(self, control):
            yield StageStarted(stage=AgentStage.INTAKE, message="启动")
            await asyncio.sleep(30)
            yield self.outcome()

    async def scenario():
        monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
        monkeypatch.setenv("DBN_MAX_ACTIVE_RUNS", "2")
        manager = RunExecutionManager(
            runtime_factory=runtime_factory(SlowAgent, monkeypatch),
        )
        for index in range(2):
            workspace = tmp_path / f"workspace-{index}"
            workspace.mkdir()
            manager.start(workspace=workspace, prompt=f"slow {index}")

        assert manager.max_active_runs == 2
        assert manager.active_run_count() == 2

        blocked_workspace = tmp_path / "workspace-blocked"
        blocked_workspace.mkdir()
        with pytest.raises(RunCapacityError, match="limit 2"):
            manager.start(workspace=blocked_workspace, prompt="blocked")

        await manager.shutdown()
        assert manager.active_run_count() == 0

    asyncio.run(scenario())


def test_reconcile_marks_orphaned_running_run_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "runs"))
    manager = RunExecutionManager()
    run_record = manager.repository.create_run_record(
        run_id=generate_run_id(),
        workspace=tmp_path,
        command="orphan",
    )
    manager.repository.update_record(run_record.run_id, status=RunStatus.RUNNING)

    asyncio.run(manager.reconcile_orphans())

    assert manager.repository.read_record(run_record.run_id).status == RunStatus.CANCELLED
    assert manager.repository.read_events(run_record.run_id) == []
