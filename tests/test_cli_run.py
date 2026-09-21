"""Contract tests for directory-based dbn run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from decisionbrain.cli.app import app
from decisionbrain.cli.commands._runtime import collect_clarification_answers

runner = CliRunner()
pytestmark = pytest.mark.usefixtures("fake_cli_agent")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "problem.md").write_text("# Minimize production cost", encoding="utf-8")
    (tmp_path / "dbn.yaml").write_text("backend: mock\ninvalid: [\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "input.csv").write_text("id\n1\n", encoding="utf-8")
    return tmp_path


def test_run_success_and_snapshot_execution(workspace: Path):
    result = runner.invoke(
        app,
        ["run", str(workspace)],
        env={"DBN_RUNS_DIR": str(workspace / "runs")},
    )
    assert result.exit_code == 0
    run_dir = next((workspace / "runs").iterdir())
    run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_data["status"] == "succeeded"
    assert run_data["config"]["_runtime_execute_from_snapshot"] is True
    assert run_data["config"]["source"] == "cli-folder"
    assert (run_dir / "workspace" / "problem.md").is_file()
    assert (run_dir / "workspace" / "data" / "input.csv").is_file()


def test_run_tty_clarification_continues_in_place(tmp_path: Path, monkeypatch):
    (tmp_path / "problem.md").write_text("NEED_CLARIFICATION scheduling", encoding="utf-8")
    monkeypatch.setattr("decisionbrain.cli.commands.run._stdin_is_tty", lambda: True)
    result = runner.invoke(
        app,
        ["run", str(tmp_path)],
        input="Minimize cost\n",
        env={"DBN_RUNS_DIR": str(tmp_path / "runs")},
    )
    assert result.exit_code == 0
    assert "Clarification answer" in result.stdout
    assert "succeeded" in result.stdout
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_clarification_questions_are_prompted_one_by_one():
    console = Console(record=True, width=80)
    questions = [
        {"id": "q1", "text": "目标函数是什么？", "answer_type": "text"},
        {"id": "q2", "text": "容量上限是多少？", "answer_type": "text"},
    ]
    snapshots: list[tuple[str, str]] = []
    answers = iter(["成本最低", "每天 8 小时"])

    def read_answer(prompt: str) -> str:
        snapshots.append((prompt, console.export_text()))
        return next(answers)

    message = collect_clarification_answers(
        questions,
        console=console,
        read_answer=read_answer,
    )

    assert snapshots[0][0] == "Clarification answer 1/2> "
    assert "目标函数是什么？" in snapshots[0][1]
    assert "容量上限是多少？" not in snapshots[0][1]
    assert snapshots[1][0] == "Clarification answer 2/2> "
    assert "容量上限是多少？" in snapshots[1][1]
    assert "成本最低" in message
    assert "每天 8 小时" in message


def test_run_non_tty_clarification_exits_two(tmp_path: Path):
    (tmp_path / "problem.md").write_text("NEED_CLARIFICATION scheduling", encoding="utf-8")
    result = runner.invoke(
        app,
        ["run", str(tmp_path)],
        env={"DBN_RUNS_DIR": str(tmp_path / "runs")},
    )
    assert result.exit_code == 2
    run_dir = next((tmp_path / "runs").iterdir())
    run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_data["status"] == "needs_clarification"


def test_run_json_stdout_is_machine_readable(workspace: Path):
    result = runner.invoke(app, ["run", str(workspace), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded"
    assert isinstance(payload["artifacts"], list)


def test_run_requires_nonempty_problem_md(tmp_path: Path):
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 3
    assert "problem.md" in result.stdout


def test_backend_option_is_rejected(tmp_path: Path):
    result = runner.invoke(app, ["run", str(tmp_path), "--backend", "mock"])
    assert result.exit_code != 0


def test_run_debug_shows_diagnostic(workspace: Path):
    result = runner.invoke(app, ["run", str(workspace), "--debug"])
    assert result.exit_code == 0
    assert "Runtime input and configuration diagnostics" in result.stdout
    assert "input.snapshot" not in result.stdout  # payload is structured, not a raw file dump
