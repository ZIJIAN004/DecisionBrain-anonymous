import subprocess

import pytest

from decisionbrain.algorithm_library import LocalAlgorithmCatalog
from decisionbrain.algorithm_library.runtime_validation import (
    AlgorithmRuntimeValidationError,
    validate_algorithm_library_runtime,
)
from decisionbrain.paths import ALGORITHM_MANIFESTS_DIR


def test_runtime_validation_checks_every_package_and_solver(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    catalog = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR)

    validate_algorithm_library_runtime(catalog)

    package_count = len(catalog.list_available())
    solver_count = sum(len(item.solver_documentation) for item in catalog.list_available())
    assert len(calls) == package_count + solver_count
    assert all(call[0][1] == "-c" for call in calls)


def test_runtime_validation_skips_examples_when_package_check_fails(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "importlib.import_module" in command[2]:
            return subprocess.CompletedProcess(command, 1, "", "missing package")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    catalog = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR)

    with pytest.raises(AlgorithmRuntimeValidationError) as raised:
        validate_algorithm_library_runtime(catalog)

    assert len(calls) == len(catalog.list_available())
    assert all(failure.check == "version_and_import" for failure in raised.value.failures)


def test_runtime_validation_reports_the_failing_solver(monkeypatch):
    def fake_run(command, **kwargs):
        if 'Model("lp")' in command[2]:
            return subprocess.CompletedProcess(command, 1, "", "bad LP example")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AlgorithmRuntimeValidationError) as raised:
        validate_algorithm_library_runtime(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    assert raised.value.failures[0].package_id == "gurobipy"
    assert raised.value.failures[0].solver_id == "lp"
    assert "bad LP example" in str(raised.value)


def test_runtime_validation_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="greater than zero"):
        validate_algorithm_library_runtime(
            LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR), timeout_seconds=0
        )
