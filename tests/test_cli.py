import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from decisionbrain import __version__
from decisionbrain.cli.app import app
from decisionbrain.cli.doctor import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    DoctorReport,
    run_doctor,
)
from decisionbrain.config import Settings


runner = CliRunner()


def _clear_runtime_configuration(monkeypatch) -> None:
    for field in Settings.model_fields.values():
        aliases = field.validation_alias
        if isinstance(aliases, str):
            monkeypatch.delenv(aliases, raising=False)
    for name in ("DBN_RUNS_DIR", "DBN_DEBUG"):
        monkeypatch.delenv(name, raising=False)


def _healthy_report() -> DoctorReport:
    return DoctorReport(
        checks=[
            CheckResult(
                name="Python",
                level=CheckLevel.REQUIRED,
                status=CheckStatus.PASS,
                message="Python test",
            ),
            CheckResult(
                name="Gurobi Python",
                level=CheckLevel.OPTIONAL,
                status=CheckStatus.WARN,
                message="未安装",
            ),
        ]
    )


def test_help_describes_current_cli_scope():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Decision Brain OR Agent" in result.output
    assert "doctor" in result.output
    assert "init" in result.output
    assert "run" in result.output
    assert "chat" in result.output
    assert "DBN_RUNS_DIR" in result.output
    assert "DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES" in result.output
    assert "DBN_DEBUG" in result.output
    assert "dev" in result.output
    assert "algorithms" in result.output
    assert "--debug" in result.output


def test_algorithms_add_connects_manifest_to_catalog(tmp_path):
    source = Path(__file__).resolve().parents[1] / "algorithms" / "manifests" / "pyjobshop.yaml"
    catalog = tmp_path / "manifests"

    result = runner.invoke(
        app,
        ["algorithms", "add", str(source), "--manifests-dir", str(catalog)],
    )

    assert result.exit_code == 0
    assert "Added" in result.output
    assert "pyjobshop" in result.output
    assert (catalog / "pyjobshop.yaml").is_file()


def test_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"Decision Brain {__version__}"


def test_offline_commands_do_not_require_runtime_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _clear_runtime_configuration(monkeypatch)

    for args in (["--help"], ["--version"], ["algorithms", "list"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.exception


def test_algorithms_add_needs_only_catalog_configuration(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "algorithms" / "manifests" / "pyjobshop.yaml"
    catalog = tmp_path / "catalog"
    monkeypatch.chdir(tmp_path)
    _clear_runtime_configuration(monkeypatch)

    result = runner.invoke(
        app,
        ["algorithms", "add", str(source), "--manifests-dir", str(catalog)],
    )

    assert result.exit_code == 0, result.exception
    assert (catalog / "pyjobshop.yaml").is_file()


def test_doctor_renders_rich_table():
    with patch("decisionbrain.cli.commands.doctor.run_doctor", return_value=_healthy_report()):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Environment Diagnostics" in result.output
    assert "Python" in result.output
    assert "All required checks passed" in result.output


def test_doctor_json_is_machine_readable():
    with patch("decisionbrain.cli.commands.doctor.run_doctor", return_value=_healthy_report()):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["version"] == __version__
    assert payload["checks"][1]["status"] == "warn"


def test_doctor_returns_one_when_required_check_fails():
    report = DoctorReport(
        checks=[
            CheckResult(
                name="运行目录",
                level=CheckLevel.REQUIRED,
                status=CheckStatus.FAIL,
                message="不可写",
            )
        ]
    )

    with patch("decisionbrain.cli.commands.doctor.run_doctor", return_value=report):
        result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.output)["ok"] is False


def test_missing_optional_dependencies_do_not_fail_doctor(tmp_path):
    settings = Settings(runs_dir=tmp_path / "runs")

    with (
        patch("decisionbrain.cli.doctor.importlib.util.find_spec", return_value=None),
        patch("decisionbrain.cli.doctor.shutil.which", return_value=None),
    ):
        report = run_doctor(settings, cwd=tmp_path, environ={})

    assert report.ok is True
    optional = [check for check in report.checks if check.level is CheckLevel.OPTIONAL]
    assert optional
    assert all(check.status is CheckStatus.WARN for check in optional)


def test_settings_reads_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in (
        "DBN_RUNS_DIR",
        "DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES",
        "DBN_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("DBN_RUNS_DIR", str(tmp_path / "custom-runs"))
    monkeypatch.setenv("DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES", "1048576")
    monkeypatch.setenv("DBN_DEBUG", "true")
    configured = Settings()
    assert configured.runs_dir == tmp_path / "custom-runs"
    assert configured.snapshot_max_file_size_bytes == 1048576
    assert configured.debug is True


def test_python_module_entrypoint():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "decisionbrain", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Decision Brain OR Agent" in completed.stdout
