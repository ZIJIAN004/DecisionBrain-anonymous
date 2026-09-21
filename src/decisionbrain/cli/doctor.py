"""Reusable environment diagnostics independent of Typer rendering."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings


class CheckLevel(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    WARNING = "warning"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class CheckResult(BaseModel):
    name: str
    level: CheckLevel
    status: CheckStatus
    message: str


class DoctorReport(BaseModel):
    version: str = __version__
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            check.level is CheckLevel.REQUIRED and check.status is CheckStatus.FAIL
            for check in self.checks
        )

    def as_json_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": self.version,
            "checks": [check.model_dump(mode="json") for check in self.checks],
        }


def _result(
    name: str,
    level: CheckLevel,
    status: CheckStatus,
    message: str,
) -> CheckResult:
    return CheckResult(name=name, level=level, status=status, message=message)


def _python_check() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    status = CheckStatus.PASS if sys.version_info >= (3, 10) else CheckStatus.FAIL
    return _result("Python", CheckLevel.REQUIRED, status, f"Python {version}")


def _working_directory_check(cwd: Path) -> CheckResult:
    readable = cwd.is_dir() and os.access(cwd, os.R_OK)
    status = CheckStatus.PASS if readable else CheckStatus.FAIL
    message = str(cwd) if readable else f"Working directory is not readable: {cwd}"
    return _result("Working directory", CheckLevel.REQUIRED, status, message)


def _runs_directory_check(runs_dir: Path) -> CheckResult:
    path = runs_dir.expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".dbn-doctor-", dir=path):
            pass
    except OSError as exc:
        return _result(
            "Runs directory",
            CheckLevel.REQUIRED,
            CheckStatus.FAIL,
            f"Cannot create or write {path}: {exc}",
        )
    return _result("Runs directory", CheckLevel.REQUIRED, CheckStatus.PASS, str(path))


def _command_check(name: str, command: list[str], *, timeout_seconds: float) -> CheckResult:
    executable = shutil.which(command[0])
    if executable is None:
        return _result(name, CheckLevel.OPTIONAL, CheckStatus.WARN, "Not installed or not on PATH")
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _result(name, CheckLevel.OPTIONAL, CheckStatus.WARN, f"Check failed: {exc}")
    if completed.returncode != 0:
        error_output = (completed.stderr or completed.stdout).strip().splitlines()
        detail = error_output[-1] if error_output else f"Exit code {completed.returncode}"
        return _result(name, CheckLevel.OPTIONAL, CheckStatus.WARN, detail)
    output = completed.stdout.strip().splitlines()
    return _result(
        name,
        CheckLevel.OPTIONAL,
        CheckStatus.PASS,
        output[0] if output else "Available",
    )


def _package_check(module_name: str, display_name: str) -> CheckResult:
    installed = importlib.util.find_spec(module_name) is not None
    return _result(
        display_name,
        CheckLevel.OPTIONAL,
        CheckStatus.PASS if installed else CheckStatus.WARN,
        "Installed" if installed else "Not installed (optional)",
    )


def _environment_checks(environ: Mapping[str, str]) -> list[CheckResult]:
    names = ("LLM_MODEL_URL", "LLM_API_KEY", "GRB_LICENSE_FILE")
    return [
        _result(
            f"Environment variable {name}",
            CheckLevel.WARNING,
            CheckStatus.PASS if environ.get(name) else CheckStatus.WARN,
            "Set" if environ.get(name) else "Not set (not required by offline CLI commands)",
        )
        for name in names
    ]


def run_doctor(
    settings: Settings | None = None,
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Run development-environment diagnostics without business side effects."""

    current_settings = settings or Settings()
    current_cwd = cwd or Path.cwd()
    current_environ = environ if environ is not None else os.environ
    checks = [
        _python_check(),
        _working_directory_check(current_cwd),
        _result("Decision Brain", CheckLevel.REQUIRED, CheckStatus.PASS, __version__),
        _runs_directory_check(current_settings.runs_dir),
        _command_check(
            "Git",
            ["git", "--version"],
            timeout_seconds=current_settings.doctor_command_timeout_seconds,
        ),
        _package_check("gurobipy", "Gurobi Python"),
        _package_check("ortools", "OR-Tools"),
    ]
    checks.extend(_environment_checks(current_environ))
    return DoctorReport(checks=checks)
