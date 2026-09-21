from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from decisionbrain.cli.app import app
from decisionbrain.run_storage import RunRepository, RunStatus
from decisionbrain.config import Settings


runner = CliRunner()

# Current UTC+8 hour prefix used to validate debug timestamps.
_TIMESTAMP_HOUR = datetime.now(timezone(timedelta(hours=8))).strftime("[%H:")


@pytest.mark.parametrize(
    ("args", "present", "absent"),
    [
        ([], "Solving", "sequence="),
        (["--debug"], _TIMESTAMP_HOUR, None),
    ],
)
def test_demo_events_create_complete_run(tmp_path, args, present, absent):
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["dev", "demo-events", *args],
        env={"DBN_RUNS_DIR": str(runs_dir)},
    )

    assert result.exit_code == 0, result.output
    assert present in result.output
    if absent is not None:
        assert absent not in result.output
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1

    repository = RunRepository(Settings(runs_dir=runs_dir))
    run_id = run_dirs[0].name
    run_record = repository.read_record(run_id)
    artifacts = repository.list_artifacts(run_id)

    assert run_record.status is RunStatus.SUCCEEDED
    assert repository.read_events(run_id) == []
    assert not (run_dirs[0] / "events.jsonl").exists()
    assert len(artifacts) == 1
    assert Path(artifacts[0].relative_path).name == "report.md"
    assert run_id in result.output
