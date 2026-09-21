"""Contract tests for workspace-free dbn chat."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from decisionbrain.cli.app import app

runner = CliRunner()
pytestmark = pytest.mark.usefixtures("fake_cli_agent")


def _interactive(monkeypatch) -> None:
    monkeypatch.setattr("decisionbrain.cli.commands.chat._stdin_is_tty", lambda: True)


def test_chat_rejects_non_tty():
    result = runner.invoke(app, ["chat"], input="Problem\n")
    assert result.exit_code == 3
    assert "requires an interactive terminal" in result.stdout


def test_chat_runs_from_first_message_without_workspace(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    result = runner.invoke(
        app,
        ["chat"],
        input="Minimize delivery cost\n",
        env={"DBN_RUNS_DIR": str(tmp_path / "runs")},
    )
    assert result.exit_code == 0
    assert "succeeded" in result.stdout
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "workspace" / "problem.md").read_text(encoding="utf-8") == "Minimize delivery cost\n"
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_chat_accepts_absolute_and_relative_files(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "data.csv").write_text("a\n1\n", encoding="utf-8")
    (second / "data.csv").write_text("a\n2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "chat",
            "--file",
            "first/data.csv",
            "--file",
            str(second / "data.csv"),
        ],
        input="Test problem\n",
        env={"DBN_RUNS_DIR": str(runs)},
    )

    assert result.exit_code == 0
    run_dir = next(runs.iterdir())
    assert (run_dir / "workspace" / "problem.md").read_text(encoding="utf-8") == "Test problem\n"
    assert (run_dir / "workspace" / "data" / "data.csv").read_text() == "a\n1\n"
    assert (run_dir / "workspace" / "data" / "data-2.csv").read_text() == "a\n2\n"


def test_chat_add_command_before_run(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    data = tmp_path / "items.csv"
    data.write_text("id\n1\n", encoding="utf-8")
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["chat"],
        input=f"/add {data}\n/files\nTest problem\n",
        env={"DBN_RUNS_DIR": str(runs)},
    )
    assert result.exit_code == 0
    assert "Added" in result.stdout
    run_dir = next(runs.iterdir())
    assert (run_dir / "workspace" / "data" / "items.csv").is_file()


def test_chat_rejects_directory_and_missing_file(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    directory = tmp_path / "data"
    directory.mkdir()
    directory_result = runner.invoke(app, ["chat", "--file", str(directory)], input="Problem\n")
    missing_result = runner.invoke(app, ["chat", "--file", str(tmp_path / "missing.csv")])
    assert directory_result.exit_code == 3
    assert missing_result.exit_code == 3


def test_chat_clarification_and_late_add_rejection(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    result = runner.invoke(
        app,
        ["chat"],
        input="NEED_CLARIFICATION scheduling\n/add later.csv\nMinimize cost\n",
        env={"DBN_RUNS_DIR": str(tmp_path / "runs")},
    )
    assert result.exit_code == 0
    assert "cannot be added" in result.stdout
    assert "succeeded" in result.stdout


def test_chat_help_and_quit_before_run(monkeypatch):
    _interactive(monkeypatch)
    result = runner.invoke(
        app,
        ["chat"],
        input="/help\n/quit\n",
    )
    assert result.exit_code == 0
    assert "/add PATH" in result.stdout


def test_chat_debug_shows_diagnostic(tmp_path: Path, monkeypatch):
    _interactive(monkeypatch)
    result = runner.invoke(
        app,
        ["chat", "--debug"],
        input="Test problem\n",
        env={"DBN_RUNS_DIR": str(tmp_path / "runs")},
    )
    assert result.exit_code == 0
    assert "Runtime input and configuration diagnostics" in result.stdout
    # Event traces are hidden from debug output.
