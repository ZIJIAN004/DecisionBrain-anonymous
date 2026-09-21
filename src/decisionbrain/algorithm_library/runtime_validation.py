"""Fail-fast runtime validation for algorithm manifests and solver examples."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from .catalog import AlgorithmCatalog


DEFAULT_VALIDATION_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RuntimeValidationFailure:
    package_id: str
    check: str
    detail: str
    solver_id: str | None = None

    def render(self) -> str:
        target = self.package_id
        if self.solver_id is not None:
            target += f"/{self.solver_id}"
        return f"{target} [{self.check}]: {self.detail}"


class AlgorithmRuntimeValidationError(RuntimeError):
    def __init__(self, failures: list[RuntimeValidationFailure]) -> None:
        self.failures = tuple(failures)
        lines = "\n".join(f"- {failure.render()}" for failure in failures)
        super().__init__(f"algorithm library runtime validation failed:\n{lines}")


def _run_python(code: str, *, timeout_seconds: int) -> str | None:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout_seconds}s"
    if completed.returncode == 0:
        return None
    output = (completed.stderr or completed.stdout or "no process output").strip()
    return " ".join(output.split())[-2000:]


def validate_algorithm_library_runtime(
    catalog: AlgorithmCatalog,
    *,
    timeout_seconds: int = DEFAULT_VALIDATION_TIMEOUT_SECONDS,
) -> None:
    """Validate every enabled package version and every documented solver example."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    failures: list[RuntimeValidationFailure] = []
    for manifest in catalog.list_available():
        version_code = (
            "import importlib\n"
            "from importlib.metadata import version\n"
            f"module = importlib.import_module({json.dumps(manifest.distribution.import_name)})\n"
            f"actual = version({json.dumps(manifest.distribution.package)})\n"
            f"expected = {json.dumps(manifest.validation.availability_check.expected_version)}\n"
            "assert actual == expected, f'expected {expected}, got {actual}'\n"
        )
        error = _run_python(version_code, timeout_seconds=timeout_seconds)
        if error is not None:
            failures.append(
                RuntimeValidationFailure(
                    package_id=manifest.id,
                    check="version_and_import",
                    detail=error,
                )
            )
            continue
        for solver in manifest.solver_documentation:
            error = _run_python(
                solver.minimal_call_example,
                timeout_seconds=timeout_seconds,
            )
            if error is not None:
                failures.append(
                    RuntimeValidationFailure(
                        package_id=manifest.id,
                        solver_id=solver.id,
                        check="minimal_call_example",
                        detail=error,
                    )
                )
    if failures:
        raise AlgorithmRuntimeValidationError(failures)
