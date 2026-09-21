"""Benchmark report indexes and tabular output writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def build_case_index(report_payload: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    resolved_runs_dir = runs_dir.expanduser().resolve()
    for item in report_payload.get("tasks", []):
        task_id = str(item["task_id"])
        run_id = item.get("run_id")
        cases[task_id] = {
            "task_id": task_id,
            "problem_class": item.get("problem_class"),
            "formulation_type": item.get("formulation_type"),
            "category": item.get("category"),
            "run_id": run_id,
            "run_directory": str(resolved_runs_dir / run_id) if run_id else None,
            "run_status": item.get("run_status"),
            "executed": item.get("executed"),
            "feasible": item.get("feasible"),
            "declared_infeasible": item.get("declared_infeasible"),
            "gap": item.get("gap"),
            "error": item.get("error"),
            "checker_stderr": item.get("checker_stderr"),
            "error_diagnostics": item.get("error_diagnostics"),
            "timing": item.get("timing"),
            "successful_algorithm_packages": item.get("successful_algorithm_packages"),
            "successful_solver_execution": item.get("successful_solver_execution"),
            "package_usage": item.get("package_usage"),
        }
    return {
        "schema_version": "1.0",
        "suite": report_payload.get("suite"),
        "feasibility_review_enabled": report_payload.get("feasibility_review_enabled"),
        "input_schema_enabled": report_payload.get("input_schema_enabled"),
        "algorithm_library_enabled": report_payload.get("algorithm_library_enabled"),
        "algorithm_design_enabled": report_payload.get("algorithm_design_enabled"),
        "problem_contract_enabled": report_payload.get("problem_contract_enabled"),
        "timing_enabled": report_payload.get("timing_enabled"),
        "algorithm_package_stats_enabled": report_payload.get(
            "algorithm_package_stats_enabled"
        ),
        "components_enabled": report_payload.get("components_enabled", True),
        "package_pool": report_payload.get("package_pool", "full"),
        "cross_package_enabled": report_payload.get("cross_package_enabled", True),
        "workflow": report_payload.get("workflow", "standard"),
        "gurobi_formulator_enabled": report_payload.get("gurobi_formulator_enabled", False),
        "solving_mode": report_payload.get("solving_mode", "algorithm-guided"),
        "started_at": report_payload.get("started_at"),
        "finished_at": report_payload.get("finished_at"),
        "cases": cases,
    }


def write_package_usage_csv(path: Path, report_payload: dict[str, Any]) -> None:
    fieldnames = [
        "task_id", "run_id", "run_status", "hidden_checker_feasible", "gap",
        "plan_alignment", "observed_package_ids", "unplanned_package_ids",
        "missing_guide_before_use", "solver_attempt_count", "failed_attempt_count",
        "errored_attempt_count", "error_categories", "solver_duration_ms",
        "failed_attempt_duration_ms", "errored_attempt_duration_ms",
        "successful_solver_attempt_sequence", "successful_solver_duration_ms",
        "successful_solver_packages", "fallback_observed", "fallback_reasons",
        "package_usage_report",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in report_payload.get("tasks", []):
            usage = task.get("package_usage") if isinstance(task, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            successful_execution = task.get("successful_solver_execution") or {}
            writer.writerow(
                {
                    "task_id": task.get("task_id"),
                    "run_id": task.get("run_id"),
                    "run_status": task.get("run_status"),
                    "hidden_checker_feasible": task.get("feasible"),
                    "gap": task.get("gap"),
                    "plan_alignment": usage.get("plan_alignment"),
                    "observed_package_ids": ";".join(usage.get("observed_package_ids") or []),
                    "unplanned_package_ids": ";".join(usage.get("unplanned_package_ids") or []),
                    "missing_guide_before_use": ";".join(
                        usage.get("missing_guide_before_use") or []
                    ),
                    "solver_attempt_count": usage.get("solver_attempt_count"),
                    "failed_attempt_count": usage.get("failed_attempt_count"),
                    "errored_attempt_count": usage.get("errored_attempt_count"),
                    "error_categories": json.dumps(
                        usage.get("error_category_counts") or {}, ensure_ascii=False
                    ),
                    "solver_duration_ms": usage.get("solver_duration_ms"),
                    "failed_attempt_duration_ms": usage.get("failed_attempt_duration_ms"),
                    "errored_attempt_duration_ms": usage.get("errored_attempt_duration_ms"),
                    "successful_solver_attempt_sequence": successful_execution.get(
                        "attempt_sequence"
                    ),
                    "successful_solver_duration_ms": successful_execution.get("duration_ms"),
                    "successful_solver_packages": json.dumps(
                        successful_execution.get("packages") or [], ensure_ascii=False
                    ),
                    "fallback_observed": usage.get("fallback_observed"),
                    "fallback_reasons": ";".join(usage.get("fallback_reasons") or []),
                    "package_usage_report": usage.get("report_file"),
                }
            )
    temporary.replace(path)
