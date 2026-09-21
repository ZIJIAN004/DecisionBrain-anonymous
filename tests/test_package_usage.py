from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from decisionbrain.runtime.package_usage import PackageUsageObserver, inspect_solver_code


class FakeCatalog:
    def __init__(self) -> None:
        self.manifests = (
            SimpleNamespace(
                id="gurobipy",
                interface=SimpleNamespace(
                    imports=("import gurobipy as gp", "from gurobipy import GRB")
                ),
            ),
            SimpleNamespace(
                id="ortools",
                interface=SimpleNamespace(
                    imports=("from ortools.sat.python import cp_model",)
                ),
            ),
        )

    def list_available(self):
        return self.manifests


def test_inspect_solver_code_records_imports_and_statically_visible_api_calls():
    evidence = inspect_solver_code(
        "import gurobipy as gp\nmodel = gp.Model('x')\nmodel.optimize()\n",
        FakeCatalog(),
    )

    assert evidence["parse_status"] == "ok"
    assert evidence["package_ids"] == ["gurobipy"]
    assert evidence["imports"] == ["gurobipy"]
    assert evidence["api_call_families"] == {
        "gurobipy:gurobipy.Model": 1,
        "gurobipy:gurobipy.Model.optimize": 1,
    }


def test_observer_reads_single_algorithm_plan_and_execution(tmp_path: Path):
    workspace = tmp_path / "workspace"
    stage_outputs = workspace / "stage_outputs"
    stage_outputs.mkdir(parents=True)
    (stage_outputs / "algorithm_design.json").write_text(
        json.dumps(
            {
                "scale": {"risk": "low"},
                "algorithm": {
                    "source": "package",
                    "package": {"id": "gurobipy"},
                    "solver_id": "milp",
                    "method": "mixed_integer_linear_programming",
                    "method_class": "exact",
                    "reason": "fit",
                    "capability_mapping": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (workspace / "solver_result.json").write_text(
        json.dumps(
            {
                "execution": {
                    "source": "package",
                    "executed_package_id": "gurobipy",
                    "executed_solver_id": "milp",
                    "version": "1.0",
                }
            }
        ),
        encoding="utf-8",
    )

    report = PackageUsageObserver(workspace, FakeCatalog()).to_dict()

    assert report["plan"][0]["package_id"] == "gurobipy"
    assert "component_id" not in report["plan"][0]
    assert report["declared_executions"][0]["executed_solver_id"] == "milp"
    assert report["assessment"]["fallback_observed"] is False
    assert "model_declared_fallback" not in report["assessment"]["fallback_reasons"]


def test_observer_records_guide_order_runtime_error_and_failed_time(tmp_path: Path):
    workspace = tmp_path / "workspace"
    stage_outputs = workspace / "stage_outputs"
    stage_outputs.mkdir(parents=True)
    (workspace / "solver.py").write_text(
        "import gurobipy as gp\nmodel = gp.Model('x')\nmodel.optimize()\n",
        encoding="utf-8",
    )
    (stage_outputs / "algorithm_design.json").write_text(
        json.dumps(
            {
                "scale": {"size_class": "large"},
                "selection": {
                    "components": [
                        {
                            "component_id": "mip",
                            "role": "primary",
                            "source": "package",
                            "package": {"id": "gurobipy"},
                            "solver_id": "milp",
                            "method": "mixed_integer_linear_programming",
                            "method_class": "exact",
                            "reason": "fit",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    observer = PackageUsageObserver(workspace, FakeCatalog())

    asyncio.run(
        observer.record_event(
            {
                "kind": "tool_call",
                "stage": "algorithm_design",
                "tool_name": "get_algorithm_guide",
                "tool_arguments": {"package_id": "gurobipy", "solver_id": "milp"},
                "status": "ok",
            }
        )
    )
    error_text = (
        "prefix\nTraceback (most recent call last):\n"
        "  File 'solver.py', line 3\n"
        "gurobipy.GurobiError: License expired\n"
    )
    asyncio.run(
        observer.record_event(
            {
                "kind": "shell_process",
                "stage": "solving",
                "command": "python3 solver.py",
                "workdir": ".",
                "duration_ms": 1250,
                "exit_code": 1,
                "status": "error",
                "output_text": error_text,
                "output_tail": error_text,
                "solution_changed": False,
            }
        )
    )

    report = observer.to_dict()
    attempt = report["attempts"][0]
    assessment = report["assessment"]
    assert attempt["guide_consulted_before_use"] == {"gurobipy": True}
    assert attempt["attempt_failed"] is True
    assert attempt["error_observed"] is True
    assert attempt["error_category"] == "license_error"
    assert "License expired" in attempt["error_output_tail"]
    assert assessment["plan_alignment"] == "consistent_with_plan"
    assert assessment["failed_attempt_count"] == 1
    assert assessment["errored_attempt_count"] == 1
    assert assessment["failed_attempt_duration_ms"] == 1250


def test_observer_distinguishes_recovered_visible_error_from_failed_attempt(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solver.py").write_text(
        "from ortools.sat.python import cp_model\ncp_model.CpModel()\n",
        encoding="utf-8",
    )
    observer = PackageUsageObserver(workspace, FakeCatalog())
    output = "caught ValueError: invalid parameter; fallback completed\n"

    asyncio.run(
        observer.record_event(
            {
                "kind": "shell_process",
                "stage": "solving",
                "command": "python3 solver.py",
                "duration_ms": 400,
                "exit_code": 0,
                "status": "ok",
                "output_text": output,
                "output_tail": output,
                "solution_changed": True,
            }
        )
    )

    assessment = observer.to_dict()["assessment"]
    assert assessment["errored_attempt_count"] == 1
    assert assessment["failed_attempt_count"] == 0
    assert assessment["errored_attempt_duration_ms"] == 400
    assert assessment["failed_attempt_duration_ms"] == 0


def test_observer_marks_retry_success_as_recovery_fallback(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solver.py").write_text(
        "import gurobipy as gp\ngp.Model('x')\n",
        encoding="utf-8",
    )
    observer = PackageUsageObserver(workspace, FakeCatalog())

    for exit_code, status, output in (
        (1, "error", "GurobiError: invalid parameter\n"),
        (0, "ok", "solved\n"),
    ):
        asyncio.run(
            observer.record_event(
                {
                    "kind": "shell_process",
                    "stage": "solving",
                    "command": "python3 solver.py",
                    "duration_ms": 100,
                    "exit_code": exit_code,
                    "status": status,
                    "output_text": output,
                    "output_tail": output,
                }
            )
        )

    assessment = observer.to_dict()["assessment"]
    assert assessment["fallback_observed"] is True
    assert "recovery_after_error" in assessment["fallback_reasons"]


def test_observer_does_not_invent_suppressed_errors(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solver.py").write_text(
        "import gurobipy as gp\ntry:\n    gp.Model('x')\nexcept Exception:\n    pass\n",
        encoding="utf-8",
    )
    observer = PackageUsageObserver(workspace, FakeCatalog())

    asyncio.run(
        observer.record_event(
            {
                "kind": "shell_process",
                "stage": "solving",
                "command": "python3 solver.py",
                "duration_ms": 10,
                "exit_code": 0,
                "status": "ok",
                "output_text": "",
                "output_tail": "",
            }
        )
    )

    attempt = observer.to_dict()["attempts"][0]
    assert attempt["attempt_failed"] is False
    assert attempt["error_observed"] is False
    assert attempt["error_category"] is None
    assert attempt["error_output_tail"] == ""
