import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from decisionbrain.config import Settings
from decisionbrain.core.models import AgentStage
from decisionbrain.events import RunEvent, EventLevel
from decisionbrain.exceptions import (
    ArtifactConflictError,
    CorruptEventLogError,
    InvalidRunPathError,
    RunStateError,
    UnsupportedSchemaVersionError,
)
from decisionbrain.run_storage import (
    ArtifactIndex,
    InputSnapshotReport,
    RunRecord,
    RunRepository,
    RunStatus,
)
from decisionbrain.runtime.agent_runtime import generate_run_id


FIXED_TIME = datetime(2026, 7, 3, 15, 30, 45, tzinfo=timezone(timedelta(hours=8)))


def make_repository(tmp_path: Path) -> RunRepository:
    return RunRepository(Settings(runs_dir=tmp_path / "runs"))


def create_run_record(repository: RunRepository, workspace: Path) -> RunRecord:
    return repository.create_run_record(
        run_id=generate_run_id(),
        workspace=workspace,
        command="dbn run",
        argv=["run", "排程.yaml"],
        config={"语言": "中文", "nested": {"enabled": True}},
    )


def make_event(
    run_id: str,
    sequence: int,
    event_type: str,
    message: str,
    *,
    stage: AgentStage | None = None,
) -> RunEvent:
    return RunEvent(
        event_id=f"event-{sequence}",
        run_id=run_id,
        sequence=sequence,
        timestamp=FIXED_TIME,
        type=event_type,
        stage=stage,
        message=message,
    )


def test_run_id_is_utc_recognizable_unique_and_has_no_global_counter():
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_TIME if tz is None else FIXED_TIME.astimezone(tz)

    with patch("decisionbrain.runtime.agent_runtime.datetime", FixedDateTime):
        values = {generate_run_id() for _ in range(500)}

    assert len(values) == 500
    assert all(re.fullmatch(r"20260703-153045Z-[0-9a-f]{8}", value) for value in values)


def test_run_id_is_strictly_argument_free():
    with pytest.raises(TypeError):
        generate_run_id(datetime(2026, 7, 3, 15, 30, 45))


def test_create_run_record_builds_layout_and_utf8_config(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    run_dir = repository.run_path(run.run_id)

    assert run.status is RunStatus.CREATED
    assert set(path.name for path in run_dir.iterdir()) == {
        "run.json",
        "input.snapshot.json",
        "workspace",
        "artifacts",
    }
    assert not (run_dir / "events.jsonl").exists()
    assert not (run_dir / "state").exists()
    assert repository.read_record(run.run_id).config["语言"] == "中文"
    assert (
        ArtifactIndex.model_validate_json(
            (run_dir / "artifacts" / ".index.json").read_bytes()
        ).artifacts
        == []
    )


def test_record_update_is_atomic_and_terminal_is_immutable(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    real_replace = os.replace

    with patch("decisionbrain.run_storage.io.os.replace", wraps=real_replace) as replace:
        running = repository.update_record(run.run_id, status=RunStatus.RUNNING)

    assert running.status is RunStatus.RUNNING
    assert replace.call_count == 1
    assert not list(repository.run_path(run.run_id).glob(".run.json.*.tmp"))

    finished = repository.update_record(
        run.run_id,
        status=RunStatus.SUCCEEDED,
        result_summary={"objective": 42, "说明": "完成"},
    )
    assert finished.result_summary == {"objective": 42, "说明": "完成"}
    with pytest.raises(RunStateError, match="terminal"):
        repository.update_record(run.run_id, error_summary="不应再修改")


def test_record_rejects_invalid_state_transition(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)

    with pytest.raises(RunStateError, match="not allowed"):
        repository.update_record(run.run_id, status=RunStatus.SUCCEEDED)


def test_record_allows_expected_lifecycle_transitions(tmp_path):
    transitions = [
        (RunStatus.RUNNING,),
        (RunStatus.RUNNING, RunStatus.NEEDS_CLARIFICATION),
        (RunStatus.RUNNING, RunStatus.NEEDS_CLARIFICATION, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.SUCCEEDED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLED),
    ]
    repository = make_repository(tmp_path)

    for path in transitions:
        run = create_run_record(repository, tmp_path)
        updated = run
        for status in path:
            updated = repository.update_record(updated.run_id, status=status)
        assert updated.status is path[-1]


def test_agent_event_json_roundtrip_requires_aware_time():
    event = RunEvent(
        event_id="event-1",
        run_id="20260703-073045Z-a1b2c3d4",
        sequence=1,
        timestamp=datetime.now(timezone.utc),
        type="stage_started",
        stage=AgentStage.ALGORITHM_DESIGN,
        level=EventLevel.INFO,
        message="开始算法设计",
        payload={"轮次": 1},
    )

    loaded = RunEvent.model_validate_json(event.model_dump_json())
    assert loaded == event
    with pytest.raises(ValidationError, match="timezone"):
        RunEvent(
            event_id="bad",
            run_id=event.run_id,
            sequence=1,
            timestamp=datetime(2026, 7, 3),
            type="error",
            message="bad",
        )


def test_event_log_is_ordered_one_complete_json_per_line(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    first = repository.append_event(make_event(run.run_id, 1, "run_created", "已创建"))
    second = repository.append_event(
        make_event(
            run.run_id,
            2,
            "user_message",
            "请优化中文排程",
            stage=AgentStage.INTAKE,
        )
    )

    assert (first.sequence, second.sequence) == (1, 2)
    raw_lines = (repository.run_path(run.run_id) / "events.jsonl").read_bytes().splitlines()
    assert len(raw_lines) == 2
    assert all(json.loads(line)["run_id"] == run.run_id for line in raw_lines)
    assert repository.read_events(run.run_id) == [first, second]


def test_event_log_truncated_final_line_is_strict_by_default_and_optionally_ignored(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    event = repository.append_event(make_event(run.run_id, 1, "run_created", "created"))
    path = repository.run_path(run.run_id) / "events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":"2.0","event_id":"truncated"')

    with pytest.raises(CorruptEventLogError, match="line 2"):
        repository.read_events(run.run_id)
    assert repository.read_events(run.run_id, tolerate_truncated_final_line=True) == [event]


def test_event_log_never_ignores_corrupt_completed_line(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    path = repository.run_path(run.run_id) / "events.jsonl"
    path.write_bytes(b"{broken}\n")

    with pytest.raises(CorruptEventLogError):
        repository.read_events(run.run_id, tolerate_truncated_final_line=True)


def test_artifact_text_json_bytes_copy_hash_and_lookup(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)

    text = repository.write_artifact_text(
        run.run_id, "报告/结果.md", "目标值：42", category="report"
    )
    structured = repository.write_artifact_json(
        run.run_id, "data/solution.json", {"任务": ["甲", "乙"]}
    )
    binary = repository.write_artifact_bytes(run.run_id, "raw/blob.bin", b"\x00\x01")
    source = tmp_path / "source.csv"
    source.write_text("名称,值\n甲,1\n", encoding="utf-8")
    copied = repository.copy_artifact_file(run.run_id, source, "tables/source.csv")

    assert text.sha256 == hashlib.sha256("目标值：42".encode()).hexdigest()
    assert text.media_type == "text/markdown"
    assert structured.media_type == "application/json"
    assert binary.media_type == "application/octet-stream"
    assert copied.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert repository.get_artifact_metadata(run.run_id, text.artifact_id) == text
    assert [item.relative_path for item in repository.list_artifacts(run.run_id)] == sorted(
        [text.relative_path, structured.relative_path, binary.relative_path, copied.relative_path]
    )
    assert (repository.run_path(run.run_id) / text.relative_path).read_text(
        encoding="utf-8"
    ) == "目标值：42"


@pytest.mark.parametrize(
    "path", ["../outside.txt", "/tmp/outside.txt", "C:\\outside.txt", ".index.json"]
)
def test_artifact_rejects_path_escape_and_reserved_path(tmp_path, path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)

    with pytest.raises(InvalidRunPathError):
        repository.write_artifact_text(run.run_id, path, "blocked")


def test_artifact_same_name_conflicts_unless_explicitly_overwritten(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    first = repository.write_artifact_text(run.run_id, "answer.txt", "first")

    with pytest.raises(ArtifactConflictError):
        repository.write_artifact_text(run.run_id, "answer.txt", "second")
    replaced = repository.write_artifact_text(run.run_id, "answer.txt", "second", overwrite=True)
    assert replaced.artifact_id == first.artifact_id
    assert replaced.sha256 != first.sha256
    assert len(repository.list_artifacts(run.run_id)) == 1


def test_snapshot_preserves_unicode_and_records_ignored_and_large_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "数据").mkdir()
    (workspace / "数据" / "订单.csv").write_text("订单,数量\n甲,2\n", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("secret", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "x.pyc").write_bytes(b"cache")
    (workspace / "large.bin").write_bytes(b"x" * 200)
    repository = make_repository(tmp_path)
    run = repository.create_run_record(
        run_id=generate_run_id(),
        workspace=workspace,
        command="dbn run",
        argv=["run", "排程.yaml"],
        config={"语言": "中文", "nested": {"enabled": True}},
        max_snapshot_file_size_bytes=100,
    )

    run_dir = repository.run_path(run.run_id)
    report = InputSnapshotReport.model_validate_json((run_dir / "input.snapshot.json").read_bytes())
    assert (run_dir / "workspace" / "数据" / "订单.csv").read_text(encoding="utf-8") == (
        "订单,数量\n甲,2\n"
    )
    assert not (run_dir / "workspace" / ".git").exists()
    reasons = {(item.relative_path, item.reason) for item in report.ignored}
    assert (".git", "ignored_directory") in reasons
    assert ("__pycache__", "ignored_directory") in reasons
    assert ("large.bin", "file_too_large") in reasons
    persisted = InputSnapshotReport.model_validate_json(
        (run_dir / "input.snapshot.json").read_bytes()
    )
    assert persisted == report
    assert repository.read_record(run.run_id).input_summary["ignored_files"] == 3


def test_snapshot_does_not_follow_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("当前平台不支持符号链接")
    repository = make_repository(tmp_path)
    run = create_run_record(repository, workspace)

    run_dir = repository.run_path(run.run_id)
    report = InputSnapshotReport.model_validate_json((run_dir / "input.snapshot.json").read_bytes())

    assert not (run_dir / "workspace" / "link.txt").exists()
    assert [(item.relative_path, item.reason) for item in report.ignored] == [
        ("link.txt", "symlink")
    ]


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            RunRecord,
            {
                "run_id": "20260703-073045Z-a1b2c3d4",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "workspace": "/tmp",
                "command": "dbn run",
            },
        ),
        (
            RunEvent,
            {
                "event_id": "event",
                "run_id": "20260703-073045Z-a1b2c3d4",
                "sequence": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "run_created",
                "message": "created",
            },
        ),
        (ArtifactIndex, {"artifacts": []}),
        (
            InputSnapshotReport,
            {
                "run_id": "20260703-073045Z-a1b2c3d4",
                "workspace": "/tmp",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "max_file_size_bytes": 1,
            },
        ),
    ],
)
def test_persisted_models_reject_unsupported_schema(model, payload):
    with pytest.raises(UnsupportedSchemaVersionError, match="9.9"):
        model.model_validate({"schema_version": "9.9", **payload})


def test_repository_read_rejects_unsupported_record_schema(tmp_path):
    repository = make_repository(tmp_path)
    run = create_run_record(repository, tmp_path)
    path = repository.run_path(run.run_id) / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="1.0"):
        repository.read_record(run.run_id)


def test_repository_list_skips_unsupported_record_schema(tmp_path):
    repository = make_repository(tmp_path)
    old_run = create_run_record(repository, tmp_path)
    current_run = create_run_record(repository, tmp_path)
    path = repository.run_path(old_run.run_id) / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    listed = repository.list_runs()

    assert [item.run_id for item in listed] == [current_run.run_id]


def test_repository_rejects_run_id_path_traversal(tmp_path):
    repository = make_repository(tmp_path)

    with pytest.raises(InvalidRunPathError):
        repository.run_path("../outside")
