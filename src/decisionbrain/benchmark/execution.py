"""Benchmark task execution, aggregation, and legacy CLI implementation."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..config import Settings
from ..algorithm_library import (
    AlgorithmRuntimeValidationError,
    LocalAlgorithmCatalog,
    validate_algorithm_library_runtime,
)
from ..core import OptimizationAgent
from ..core.models import AgentStage
from ..core.package_policy import PACKAGE_POOLS, PackagePolicy
from ..events import RunEvent
from ..run_storage.io import atomic_write_json
from ..runtime import AgentRuntime
from .intake_bypass import materialize
from .adapter import BenchmarkSolutionAdapter, BenchmarkAdaptationError
from .cli import build_parser, validate_reported_ablation
from .solving_intake import (
    CheckerGroundedIntakeAnswerer,
    IntakeAnswererError,
    format_clarification_message,
)
from .output import build_case_index, write_package_usage_csv
from .scoring import BenchmarkReport, TaskScore, score_task
from .task import (
    BenchmarkTask,
    discover_tasks,
    load_large_max_tasks,
    normalize_large_suite_task_ids,
    load_selected_tasks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = "FrontierOR65-Fea"
DEFAULT_TASKS_ROOT = PROJECT_ROOT / "benchmarks" / DEFAULT_SUITE
DEFAULT_FRONTIEROR65_FEA_INDEX = DEFAULT_TASKS_ROOT / "index.json"
FRONTIEROR10_INF_ROOT = PROJECT_ROOT / "benchmarks" / "FrontierOR10-Inf"
FRONTIEROR10_INF_INDEX = FRONTIEROR10_INF_ROOT / "index.json"
HARD32_FEA_ROOT = PROJECT_ROOT / "benchmarks" / "Hard32-Fea"
HARD32_FEA_SELECTION = HARD32_FEA_ROOT / "selection.json"
LARGE_SUITE_INDEXES = {
    "FrontierOR65-Fea": DEFAULT_FRONTIEROR65_FEA_INDEX,
    "FrontierOR10-Inf": FRONTIEROR10_INF_INDEX,
}

_RESOURCE_EXHAUSTION_MARKERS = (
    "memoryerror",
    "cannot allocate memory",
    "not enough memory",
    "out of memory",
    "virtual memory",
    "page file too small",
    "paging file is too small",
    "winerror 1455",
    "errno 12",
    "paging file is too small",
    "virtual memory exhausted",
)
_NETWORK_FAILURE_MARKERS = (
    "all connection attempts failed",
    "connection attempts failed",
    "connecterror",
    "connection refused",
    "connection reset",
    "connection timed out",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "readtimeout",
)
TaskEventCallback = Callable[[dict[str, Any]], None]
IntakeMode = Literal["bypass", "full"]


def _single_line(value: object, *, max_chars: int = 1000) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _score_failure_detail(score: TaskScore) -> str:
    if score.error:
        return score.error
    if score.declared_infeasible:
        return "" if score.reference_feasible is False else "agent incorrectly declared infeasible"
    if score.feasible is False:
        constraints = ", ".join(score.violated_constraints[:5])
        if constraints:
            return f"checker reported infeasible: {constraints}"
        if score.violations:
            return f"checker reported infeasible: {_single_line(score.violations[0])}"
        return "checker reported infeasible"
    return ""


class _RunObserver:
    def __init__(
        self,
        resource_exhaustion_callback: Callable[[str], None] | None = None,
        task_event_callback: TaskEventCallback | None = None,
    ) -> None:
        self.run_id: str | None = None
        self.flow_section_counts: Counter[str] = Counter()
        self.error_kind_counts: Counter[str] = Counter()
        self.resource_exhaustion_error: str | None = None
        self.clarification_requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._resource_exhaustion_callback = resource_exhaustion_callback
        self._task_event_callback = task_event_callback

    async def handle(self, event: RunEvent) -> None:
        if self.run_id is None:
            self.run_id = event.run_id
            self._emit_task_event("run_created", run_id=event.run_id)
        if event.type == "agent_error":
            section = str(
                event.payload.get("flow_section")
                or (event.stage.value if event.stage else "runtime")
            )
            kind = str(event.payload.get("error_kind") or "agent_error")
            self.record_error(section, kind)
            event_error = str(event.payload.get("message") or event.message)
            self._emit_task_event(
                "agent_error",
                run_id=event.run_id,
                stage=section,
                error_kind=kind,
                error=event_error,
            )
            if _contains_marker(event_error.lower(), _RESOURCE_EXHAUSTION_MARKERS):
                self.resource_exhaustion_error = event_error
                if self._resource_exhaustion_callback is not None:
                    self._resource_exhaustion_callback(event_error)
        elif event.type == "error":
            section = event.stage.value if event.stage else "runtime"
            self.record_error(section, "terminal_error")
            self._emit_task_event(
                "terminal_error",
                run_id=event.run_id,
                stage=section,
                error_kind="terminal_error",
                error=event.message,
            )
        elif event.type == "clarification_requested":
            self.clarification_requests.put_nowait(
                {
                    "message": event.message,
                    "questions": list(event.payload.get("questions") or []),
                }
            )

    def _emit_task_event(self, event_type: str, **payload: Any) -> None:
        if self._task_event_callback is None:
            return
        self._task_event_callback({"event": event_type, **payload})

    def record_error(self, flow_section: str, error_kind: str, count: int = 1) -> None:
        if count <= 0:
            return
        self.flow_section_counts[flow_section] += count
        self.error_kind_counts[error_kind] += count

    def error_diagnostics(self) -> dict[str, Any]:
        section_counts = dict(sorted(self.flow_section_counts.items()))
        kind_counts = dict(sorted(self.error_kind_counts.items()))
        max_count = max(section_counts.values(), default=0)
        most_repeated = next(
            (name for name, count in section_counts.items() if count == max_count),
            None,
        )
        return {
            "total_errors": sum(section_counts.values()),
            "flow_section_counts": section_counts,
            "error_kind_counts": kind_counts,
            "most_repeated_flow_section": most_repeated,
            "most_repeated_flow_section_count": max_count,
            "resource_exhaustion_detected": self.resource_exhaustion_error is not None,
            "resource_exhaustion_error": self.resource_exhaustion_error,
        }


def _format_task_progress(event: dict[str, Any]) -> str:
    prefix = f"[{event['index']}/{event['total']}]"
    task_id = event["task_id"]
    event_type = event["event"]
    run_id = event.get("run_id") or "pending"
    if event_type == "started":
        return f"{prefix} START task={task_id}"
    if event_type == "run_created":
        return f"{prefix} RUN task={task_id} run_id={run_id} run_dir={event.get('run_directory')}"
    if event_type in {"agent_error", "terminal_error", "checker_error"}:
        return (
            f"{prefix} ERROR task={task_id} run_id={run_id} "
            f"stage={event.get('stage') or 'runtime'} "
            f"kind={event.get('error_kind') or event_type} "
            f"message={_single_line(event.get('error')) or '<none>'}"
        )
    if event_type == "completed":
        verdict = "PASS" if event.get("passed") else "FAIL"
        return (
            f"{prefix} {verdict} task={task_id} run_id={run_id} "
            f"status={event.get('run_status')} feasible={event.get('feasible')} "
            f"gap={event.get('gap')} error={_single_line(event.get('error')) or '<none>'}"
        )
    return f"{prefix} {event_type.upper()} task={task_id} run_id={run_id}"


@dataclass(frozen=True)
class TaskEvaluation:
    task: BenchmarkTask
    score: TaskScore
    run_id: str | None
    run_status: str
    duration_seconds: float
    error_diagnostics: dict[str, Any] | None = None
    timing_summary: dict[str, Any] | None = None
    successful_algorithm_packages: tuple[dict[str, str], ...] = ()
    successful_solver_execution: dict[str, Any] | None = None
    package_usage_summary: dict[str, Any] | None = None
    intake_mode: IntakeMode = "bypass"
    intake_dialogue: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.score.to_dict(),
            "paper_id": self.task.paper_id,
            "formulation_type": self.task.metadata.get("formulation_type"),
            "category": self.task.metadata.get("category"),
            "instance_variant": self.task.instance_variant,
            "instance_index": self.task.instance_index,
            "instance_bytes": self.task.metadata.get("instance_bytes"),
            "reference_feasible": self.task.reference_feasible,
            "run_id": self.run_id,
            "run_status": self.run_status,
            "duration_seconds": round(self.duration_seconds, 3),
            "contract_mode": "frontieror_direct",
            "intake_mode": self.intake_mode,
            "intake_dialogue": list(self.intake_dialogue),
            "error_diagnostics": self.error_diagnostics
            or {
                "total_errors": 0,
                "flow_section_counts": {},
                "error_kind_counts": {},
                "most_repeated_flow_section": None,
                "most_repeated_flow_section_count": 0,
            },
            "timing": self.timing_summary,
            "successful_algorithm_packages": list(self.successful_algorithm_packages),
            "successful_solver_execution": self.successful_solver_execution,
            "package_usage": self.package_usage_summary,
        }


def _read_successful_algorithm_packages(
    workspace: Path | None,
    *,
    hidden_checker_feasible: bool,
) -> tuple[dict[str, str], ...]:
    """Read declared package executions only for a hidden-checker-feasible final result."""

    if not hidden_checker_feasible or workspace is None:
        return ()
    try:
        payload = json.loads((workspace / "solver_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    executions = payload.get("executions") if isinstance(payload, dict) else None
    if not isinstance(executions, list):
        execution = payload.get("execution") if isinstance(payload, dict) else None
        executions = [execution] if isinstance(execution, dict) else []

    packages: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for execution in executions:
        if not isinstance(execution, dict) or execution.get("source") != "package":
            continue
        package_id = str(execution.get("executed_package_id") or "").strip()
        solver_id = str(execution.get("executed_solver_id") or "").strip()
        version = str(execution.get("version") or "").strip()
        if not package_id or not solver_id or not version:
            continue
        key = (package_id, solver_id, version)
        if key in seen:
            continue
        seen.add(key)
        packages.append(
            {"package_id": package_id, "solver_id": solver_id, "version": version}
        )
    return tuple(packages)


def _aggregate_successful_algorithm_packages(
    evaluations: Sequence[TaskEvaluation],
) -> dict[str, Any]:
    usages: dict[tuple[str, str, str], set[str]] = {}
    hidden_feasible_task_count = 0
    tasks_with_packages = 0
    for evaluation in evaluations:
        if evaluation.score.feasible is True:
            hidden_feasible_task_count += 1
        if evaluation.successful_algorithm_packages:
            tasks_with_packages += 1
        for package in evaluation.successful_algorithm_packages:
            key = (package["package_id"], package["solver_id"], package["version"])
            usages.setdefault(key, set()).add(evaluation.task.task_id)
    return {
        "hidden_feasible_task_count": hidden_feasible_task_count,
        "hidden_feasible_tasks_with_declared_packages": tasks_with_packages,
        "package_solver_usages": [
            {
                "package_id": package_id,
                "solver_id": solver_id,
                "version": version,
                "successful_task_count": len(task_ids),
                "successful_tasks": sorted(task_ids),
            }
            for (package_id, solver_id, version), task_ids in sorted(usages.items())
        ],
    }


def _read_package_usage_summary(run_path: Path | None) -> dict[str, Any] | None:
    if run_path is None:
        return None
    report_path = run_path / "package-usage.json"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    assessment = payload.get("assessment") if isinstance(payload, dict) else None
    attempts = payload.get("attempts") if isinstance(payload, dict) else None
    if not isinstance(assessment, dict) or not isinstance(attempts, list):
        return None
    return {
        "report_file": str(report_path.resolve()),
        **assessment,
        "attempts": [
            {
                "sequence": item.get("sequence"),
                "package_ids": item.get("package_ids") or [],
                "duration_ms": item.get("duration_ms"),
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
                "timed_out": item.get("timed_out"),
                "timeout_source": item.get("timeout_source"),
                "solution_changed": item.get("solution_changed"),
                "attempt_failed": item.get("attempt_failed"),
                "error_observed": item.get("error_observed"),
                "error_category": item.get("error_category"),
                "error_output_tail": item.get("error_output_tail"),
            }
            for item in attempts
            if isinstance(item, dict)
        ],
    }


def _select_successful_solver_execution(
    package_usage: dict[str, Any] | None,
    successful_packages: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    """Return the observed solver attempt that produced the checker-accepted result.

    The hidden checker evaluates the final solution artifact, so prefer the last
    non-failed attempt that changed that artifact.  If no fingerprint change was
    observed, fall back to the last non-failed attempt (a solver may rewrite
    identical bytes or update the result through an indirect process).
    """

    if not isinstance(package_usage, dict):
        return None
    attempts = [item for item in package_usage.get("attempts") or [] if isinstance(item, dict)]
    successful = [item for item in attempts if not item.get("attempt_failed")]
    if not successful:
        return None
    changed = [item for item in successful if item.get("solution_changed")]
    attempt = max(changed or successful, key=lambda item: int(item.get("sequence") or 0))
    return {
        "hidden_checker_feasible": True,
        "attempt_sequence": attempt.get("sequence"),
        "package_ids": list(attempt.get("package_ids") or []),
        "packages": [dict(item) for item in successful_packages],
        "duration_ms": attempt.get("duration_ms"),
        "status": attempt.get("status"),
        "exit_code": attempt.get("exit_code"),
        "solution_changed": bool(attempt.get("solution_changed")),
        "command": attempt.get("command"),
        "selection_basis": (
            "last_successful_solution_changing_attempt"
            if changed
            else "last_successful_attempt"
        ),
    }


def _aggregate_package_usage(evaluations: Sequence[TaskEvaluation]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    package_attempts: Counter[str] = Counter()
    plan_alignment: Counter[str] = Counter()
    tasks_with_usage = 0
    tasks_with_errors = 0
    tasks_with_failed_attempts = 0
    for evaluation in evaluations:
        usage = evaluation.package_usage_summary
        if not isinstance(usage, dict):
            continue
        tasks_with_usage += 1
        if int(usage.get("errored_attempt_count") or 0):
            tasks_with_errors += 1
        if int(usage.get("failed_attempt_count") or 0):
            tasks_with_failed_attempts += 1
        for name in (
            "solver_attempt_count",
            "failed_attempt_count",
            "errored_attempt_count",
            "solver_duration_ms",
            "failed_attempt_duration_ms",
            "errored_attempt_duration_ms",
        ):
            totals[name] += int(usage.get(name) or 0)
        for flag, total_name in (
            ("fallback_observed", "fallback_task_count"),
            ("missing_guide_before_use", "missing_guide_task_count"),
            ("unplanned_package_ids", "unplanned_package_task_count"),
        ):
            if usage.get(flag):
                totals[total_name] += 1
        error_categories.update(usage.get("error_category_counts") or {})
        plan_alignment[str(usage.get("plan_alignment") or "unknown")] += 1
        for attempt in usage.get("attempts") or []:
            if isinstance(attempt, dict):
                package_attempts.update(str(item) for item in attempt.get("package_ids") or [])
    return {
        "tasks_with_package_usage": tasks_with_usage,
        "tasks_missing_package_usage": len(evaluations) - tasks_with_usage,
        "tasks_with_package_errors": tasks_with_errors,
        "tasks_with_failed_package_attempts": tasks_with_failed_attempts,
        "solver_attempt_count": totals["solver_attempt_count"],
        "failed_attempt_count": totals["failed_attempt_count"],
        "errored_attempt_count": totals["errored_attempt_count"],
        "solver_duration_ms": totals["solver_duration_ms"],
        "failed_attempt_duration_ms": totals["failed_attempt_duration_ms"],
        "errored_attempt_duration_ms": totals["errored_attempt_duration_ms"],
        "fallback_task_count": totals["fallback_task_count"],
        "missing_guide_task_count": totals["missing_guide_task_count"],
        "unplanned_package_task_count": totals["unplanned_package_task_count"],
        "plan_alignment_counts": dict(sorted(plan_alignment.items())),
        "error_category_counts": dict(sorted(error_categories.items())),
        "package_attempt_counts": dict(sorted(package_attempts.items())),
    }


def _aggregate_timing(evaluations: Sequence[TaskEvaluation]) -> dict[str, Any]:
    """Sum Runtime timing only; hidden-checker execution is deliberately excluded."""

    stage_duration_ms: Counter[str] = Counter()
    llm_duration_ms: Counter[str] = Counter()
    tool_duration_ms: Counter[str] = Counter()
    totals = Counter[str]()
    for evaluation in evaluations:
        if evaluation.timing_summary is None:
            continue
        timing = evaluation.timing_summary
        totals["timing_available_task_count"] += 1
        for name, value in timing.get("stage_duration_ms", {}).items():
            if isinstance(value, int) and value >= 0:
                stage_duration_ms[str(name)] += value
        for name, value in timing.get("llm_duration_ms", {}).items():
            if isinstance(value, int) and value >= 0:
                llm_duration_ms[str(name)] += value
        for name, value in timing.get("tool_duration_ms", {}).items():
            if isinstance(value, int) and value >= 0:
                tool_duration_ms[str(name)] += value
        for name in (
            "wall_clock_duration_ms",
            "completed_stage_count",
            "incomplete_stage_count",
            "llm_call_count",
            "llm_retry_count",
            "llm_retry_backoff_ms",
            "tool_call_count",
            "self_checker_duration_ms",
            "self_checker_execution_count",
            "solver_execution_duration_ms",
            "solver_execution_count",
            "ambiguous_process_duration_ms",
            "ambiguous_process_count",
        ):
            value = timing.get(name)
            if isinstance(value, int) and value >= 0:
                totals[name] += value
    return {
        "timing_available_task_count": totals["timing_available_task_count"],
        "timing_missing_task_count": len(evaluations) - totals["timing_available_task_count"],
        "runtime_task_duration_sum_ms": totals["wall_clock_duration_ms"],
        "stage_duration_ms": dict(sorted(stage_duration_ms.items())),
        "completed_stage_count": totals["completed_stage_count"],
        "incomplete_stage_count": totals["incomplete_stage_count"],
        "llm_duration_ms": dict(sorted(llm_duration_ms.items())),
        "llm_call_count": totals["llm_call_count"],
        "llm_retry_count": totals["llm_retry_count"],
        "llm_retry_backoff_ms": totals["llm_retry_backoff_ms"],
        "tool_duration_ms": dict(sorted(tool_duration_ms.items())),
        "tool_call_count": totals["tool_call_count"],
        "self_checker_duration_ms": totals["self_checker_duration_ms"],
        "self_checker_execution_count": totals["self_checker_execution_count"],
        "solver_execution_duration_ms": totals["solver_execution_duration_ms"],
        "solver_execution_count": totals["solver_execution_count"],
        "ambiguous_process_duration_ms": totals["ambiguous_process_duration_ms"],
        "ambiguous_process_count": totals["ambiguous_process_count"],
    }


def _evaluation_error_text(evaluation: TaskEvaluation) -> str:
    parts = [evaluation.score.error or ""]
    if evaluation.error_diagnostics is not None:
        parts.append(str(evaluation.error_diagnostics.get("resource_exhaustion_error") or ""))
    return "\n".join(parts).lower()


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


@dataclass
class _CircuitBreaker:
    stop_on_resource_exhaustion: bool
    max_consecutive_network_failures: int
    consecutive_network_failures: int = 0
    kind: str | None = None
    reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self.kind is not None

    def observe(self, evaluation: TaskEvaluation) -> None:
        if self.tripped:
            return
        error_text = _evaluation_error_text(evaluation)
        if self.stop_on_resource_exhaustion and _contains_marker(
            error_text, _RESOURCE_EXHAUSTION_MARKERS
        ):
            self.trip_resource(evaluation.task.task_id, error_text)
            return
        if self.max_consecutive_network_failures <= 0:
            return
        if _contains_marker(error_text, _NETWORK_FAILURE_MARKERS):
            self.consecutive_network_failures += 1
            if self.consecutive_network_failures >= self.max_consecutive_network_failures:
                self.kind = "network_failure_threshold"
                self.reason = (
                    f"{self.consecutive_network_failures} consecutive tasks ended with "
                    "network connection failures"
                )
            return
        self.consecutive_network_failures = 0

    def trip_resource(self, task_id: str, detail: str) -> None:
        if self.tripped or not self.stop_on_resource_exhaustion:
            return
        detail = " ".join(detail.split())[:300]
        self.kind = "resource_exhaustion"
        self.reason = f"{task_id}: detected memory/resource exhaustion: {detail}"


@dataclass
class FrontierORReport:
    evaluations: list[TaskEvaluation]
    started_at: datetime
    finished_at: datetime
    model: str
    jobs: int
    feasibility_review_enabled: bool = True
    input_schema_enabled: bool = True
    algorithm_library_enabled: bool = True
    algorithm_design_enabled: bool = True
    problem_contract_enabled: bool = True
    timing_enabled: bool = True
    algorithm_package_stats_enabled: bool = True
    components_enabled: bool = True
    package_pool: str = "full"
    cross_package_enabled: bool = True
    workflow: str = "standard"
    gurobi_formulator_enabled: bool = False
    solving_mode: str = "algorithm-guided"
    suite: str = "large-max"
    llm_max_attempts: int = 1
    planned_tasks: int | None = None
    stop_on_resource_exhaustion: bool = False
    max_consecutive_network_failures: int = 0
    termination_kind: str | None = None
    termination_reason: str | None = None
    intake_mode: IntakeMode = "bypass"
    max_clarification_rounds: int = 5

    def to_dict(self) -> dict[str, Any]:
        aggregate = BenchmarkReport([item.score for item in self.evaluations]).to_dict()["summary"]
        aggregate["direct_contract"] = len(self.evaluations)
        section_counts: Counter[str] = Counter()
        tasks_with_errors = 0
        for item in self.evaluations:
            diagnostics = item.error_diagnostics or {}
            if int(diagnostics.get("total_errors") or 0) > 0:
                tasks_with_errors += 1
            section_counts.update(diagnostics.get("flow_section_counts") or {})
        max_section_count = max(section_counts.values(), default=0)
        aggregate.update(
            {
                "tasks_with_errors": tasks_with_errors,
                "error_count_total": sum(section_counts.values()),
                "error_flow_section_counts": dict(sorted(section_counts.items())),
                "most_error_prone_flow_section": next(
                    (
                        name
                        for name, count in sorted(section_counts.items())
                        if count == max_section_count
                    ),
                    None,
                ),
                "most_error_prone_flow_section_count": max_section_count,
            }
        )
        if self.timing_enabled:
            timing = _aggregate_timing(self.evaluations)
            timing["benchmark_wall_clock_duration_ms"] = int(
                (self.finished_at - self.started_at).total_seconds() * 1000
            )
            aggregate["timing"] = timing
        else:
            aggregate["timing"] = None
        aggregate["algorithm_package_success"] = (
            _aggregate_successful_algorithm_packages(self.evaluations)
            if self.algorithm_package_stats_enabled
            else None
        )
        aggregate["package_usage"] = (
            _aggregate_package_usage(self.evaluations)
            if self.algorithm_package_stats_enabled
            else None
        )
        planned_tasks = (
            self.planned_tasks if self.planned_tasks is not None else len(self.evaluations)
        )
        aggregate.update(
            {
                "planned_tasks": planned_tasks,
                "completed_tasks": len(self.evaluations),
                "skipped_tasks": max(0, planned_tasks - len(self.evaluations)),
                "terminated_early": self.termination_kind is not None,
            }
        )
        return {
            "benchmark": "FrontierOR",
            "suite": self.suite,
            "model": self.model,
            "intake_mode": self.intake_mode,
            "max_clarification_rounds": self.max_clarification_rounds,
            "feasibility_review_enabled": self.feasibility_review_enabled,
            "input_schema_enabled": self.input_schema_enabled,
            "algorithm_library_enabled": self.algorithm_library_enabled,
            "algorithm_design_enabled": self.algorithm_design_enabled,
            "problem_contract_enabled": self.problem_contract_enabled,
            "timing_enabled": self.timing_enabled,
            "algorithm_package_stats_enabled": self.algorithm_package_stats_enabled,
            "components_enabled": self.components_enabled,
            "package_pool": self.package_pool,
            "cross_package_enabled": self.cross_package_enabled,
            "workflow": self.workflow,
            "gurobi_formulator_enabled": self.gurobi_formulator_enabled,
            "solving_mode": self.solving_mode,
            "retry_policy": {
                "llm_max_attempts": self.llm_max_attempts,
            },
            "circuit_breaker": {
                "stop_on_resource_exhaustion": self.stop_on_resource_exhaustion,
                "max_consecutive_network_failures": self.max_consecutive_network_failures,
                "terminated_early": self.termination_kind is not None,
                "termination_kind": self.termination_kind,
                "termination_reason": self.termination_reason,
            },
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": round((self.finished_at - self.started_at).total_seconds(), 3),
            "jobs": self.jobs,
            "summary": aggregate,
            "tasks": [item.to_dict() for item in self.evaluations],
        }


async def evaluate_task(
    task: BenchmarkTask,
    settings: Settings,
    *,
    task_timeout_seconds: float,
    feasibility_review_enabled: bool = True,
    input_schema_enabled: bool = True,
    algorithm_library_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
    timing_enabled: bool = True,
    algorithm_package_stats_enabled: bool = True,
    components_enabled: bool = True,
    package_policy: PackagePolicy = PackagePolicy(),
    resource_exhaustion_callback: Callable[[str], None] | None = None,
    task_event_callback: TaskEventCallback | None = None,
    intake_mode: IntakeMode = "bypass",
    max_clarification_rounds: int = 5,
    solver_cpu_limit: int | None = None,
    workflow: str = "standard",
) -> TaskEvaluation:
    """Run one complete task, converting failures into results without stopping peers."""

    started = time.perf_counter()
    runtime = AgentRuntime(
        settings,
        solver_cpu_limit=solver_cpu_limit,
        input_schema_enabled=input_schema_enabled,
        algorithm_library_enabled=algorithm_library_enabled,
        feasibility_review_enabled=feasibility_review_enabled,
        algorithm_design_enabled=algorithm_design_enabled,
        problem_contract_enabled=problem_contract_enabled,
        components_enabled=components_enabled,
        package_policy=package_policy,
        agent_factory=lambda services: OptimizationAgent(
            services=services,
            feasibility_review_enabled=feasibility_review_enabled,
            input_schema_enabled=input_schema_enabled,
            algorithm_design_enabled=algorithm_design_enabled,
            problem_contract_enabled=problem_contract_enabled,
        ),
    )
    observer = _RunObserver(resource_exhaustion_callback, task_event_callback)
    runtime_error = ""

    with tempfile.TemporaryDirectory(prefix=f"dbn-frontieror-{task.task_id}-") as temp:
        temp_root = Path(temp)
        source_workspace = temp_root / "agent"
        answer_workspace = temp_root / "answerer"
        shutil.copytree(task.input_dir, source_workspace, dirs_exist_ok=True)
        problem_text = task.problem_md.read_text(encoding="utf-8")
        (source_workspace / "problem.md").write_text(problem_text, encoding="utf-8")
        workspace_instance = source_workspace / "data" / "instance.json"
        workspace_instance.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.instance_path, workspace_instance)
        if not problem_contract_enabled:
            # Runtime keeps its stable solver input filename, but this is the raw
            # benchmark instance, not a Core-generated input contract.
            shutil.copy2(task.instance_path, source_workspace / "input.json")
        if intake_mode == "bypass":
            materialize(
                task,
                source_workspace,
                input_schema_enabled=input_schema_enabled,
                problem_contract_enabled=problem_contract_enabled,
            )
        else:
            if problem_contract_enabled:
                target_schema = json.loads(task.target_solution_schema.read_text(encoding="utf-8"))
                (source_workspace / "fixed_solution_contract.json").write_text(
                    json.dumps(target_schema, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        if intake_mode == "full":
            shutil.copytree(source_workspace, answer_workspace, dirs_exist_ok=True)
            (answer_workspace / "hidden").mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                task.feasibility_checker,
                answer_workspace / "hidden" / "feasibility_check.py",
            )
        intake_dialogue: list[dict[str, Any]] = []
        try:
            async def execute_runtime() -> None:
                run_task = asyncio.create_task(runtime.run(
                    sink=observer,
                    workspace=source_workspace,
                    command=f"benchmark frontieror {task.task_id}",
                    argv=["--task", task.task_id],
                    config=settings.to_runtime_config()
                    | {
                        "source": "frontieror-benchmark",
                        "benchmark_task_id": task.task_id,
                        "benchmark_paper_id": task.paper_id,
                        "benchmark_instance_variant": task.instance_variant,
                        "benchmark_instance_index": task.instance_index,
                        "feasibility_review_enabled": feasibility_review_enabled,
                        "input_schema_enabled": input_schema_enabled,
                        "algorithm_library_enabled": algorithm_library_enabled,
                        "algorithm_design_enabled": algorithm_design_enabled,
                        "problem_contract_enabled": problem_contract_enabled,
                        "timing_enabled": timing_enabled,
                        "algorithm_package_stats_enabled": algorithm_package_stats_enabled,
                        "components_enabled": components_enabled,
                        "package_pool": package_policy.pool,
                        "cross_package_enabled": package_policy.cross_package_enabled,
                        "workflow": workflow,
                        "gurobi_formulator_enabled": workflow == "gurobi-formulator",
                        "solving_mode": (
                            "translation-only"
                            if workflow == "gurobi-formulator"
                            else "algorithm-guided"
                        ),
                        "benchmark_intake_mode": intake_mode,
                        "solver_cpu_limit": solver_cpu_limit,
                    },
                    initial_input=problem_text,
                    max_snapshot_file_size_bytes=settings.snapshot_max_file_size_bytes,
                    start_stage=(
                        AgentStage.INTAKE
                        if intake_mode != "bypass"
                        else (
                            AgentStage.PROBLEM_CONTRACT
                            if problem_contract_enabled
                            else (
                                AgentStage.ALGORITHM_DESIGN
                                if algorithm_design_enabled
                                else AgentStage.SOLVING
                            )
                        )
                    ),
                ))
                try:
                    if intake_mode == "bypass":
                        await run_task
                        return
                    answerer = CheckerGroundedIntakeAnswerer(settings, answer_workspace)
                    while not run_task.done():
                        request_task = asyncio.create_task(observer.clarification_requests.get())
                        done, _ = await asyncio.wait(
                            {run_task, request_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if run_task in done:
                            request_task.cancel()
                            await asyncio.gather(request_task, return_exceptions=True)
                            break
                        request = request_task.result()
                        if len(intake_dialogue) >= max_clarification_rounds:
                            raise RuntimeError(
                                f"Intake clarification exceeded {max_clarification_rounds} rounds"
                            )
                        try:
                            answers = await answerer.answer(
                                task_id=task.task_id,
                                questions=request["questions"],
                                previous_turns=intake_dialogue,
                            )
                        except IntakeAnswererError as exc:
                            observer.record_error("intake_answerer", exc.kind)
                            raise
                        intake_dialogue.append({**request, "answers": answers})
                        await runtime.submit_user_message(
                            user_message=format_clarification_message(answers)
                        )
                    await run_task
                finally:
                    if not run_task.done():
                        runtime.cancel()
                        run_task.cancel()
                        await asyncio.gather(run_task, return_exceptions=True)

            await asyncio.wait_for(execute_runtime(), timeout=task_timeout_seconds)
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                runtime.cancel()
                raise asyncio.CancelledError
        except asyncio.TimeoutError:
            runtime.cancel()
            runtime_error = f"Runtime timed out after {task_timeout_seconds:g}s"
            observer.record_error("runtime", "task_timeout")
        except IntakeAnswererError as exc:
            runtime_error = f"Intake answerer failed ({exc.kind}): {exc}"
        except Exception as exc:
            runtime_error = f"Runtime failed: {exc}"
            observer.record_error("runtime", "runtime_exception")

    run_id = observer.run_id
    run_status = "not_created"
    result_summary: dict[str, Any] = {}
    runtime_workspace: Path | None = None
    if run_id is not None:
        try:
            record = runtime.repository.read_record(run_id)
            run_status = record.status.value
            result_summary = getattr(record, "result_summary", None) or {}
            runtime_workspace = runtime.repository.run_path(run_id) / "workspace"
            runtime_error = runtime_error or record.error_summary or ""
        except Exception as exc:
            runtime_error = runtime_error or f"Failed to read Run record: {exc}"

    timing_summary: dict[str, Any] | None = None
    if timing_enabled and run_id is not None:
        try:
            timing_payload = json.loads(
                (runtime.repository.run_path(run_id) / "timing.json").read_text(encoding="utf-8")
            )
            candidate = timing_payload.get("summary")
            if isinstance(candidate, dict):
                timing_summary = candidate
        except (OSError, json.JSONDecodeError):
            pass

    if (
        run_status == "succeeded"
        and runtime_workspace is not None
        and (runtime_workspace / "solution.json").is_file()
    ):
        solution_path = runtime_workspace / "solution.json"
    elif run_status != "succeeded":
        solution_path = None
        score_error = runtime_error or f"Runtime did not succeed (status: {run_status})"
    else:
        solution_path = None
        score_error = "Runtime did not produce solution.json"

    # The no-Problem-Contract arm uses the baseline-style post-solver formatter.
    # It is intentionally outside Core: the Agent never sees contract artifacts.
    if (
        not problem_contract_enabled
        and run_status == "succeeded"
        and runtime_workspace is not None
        and runtime.llm_client is not None
    ):
        try:
            adapted = await BenchmarkSolutionAdapter(
                runtime.llm_client, settings
            ).adapt(task, runtime_workspace)
            solution_path = adapted.solution_path
            score_error = ""
        except BenchmarkAdaptationError as exc:
            solution_path = None
            score_error = str(exc)
            observer.record_error("external_formatter", "formatter_error")

    if solution_path is not None:
        score_error = ""
    if run_status == "succeeded" and result_summary.get("status") == "instance_infeasible":
        score = TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=True,
            feasible=False,
            declared_infeasible=True,
            reference_feasible=task.reference_feasible,
            error=(
                ""
                if task.reference_feasible is False
                else "Agent declared the instance infeasible, but the reference is feasible"
            ),
        )
    else:
        score = await asyncio.to_thread(
            score_task,
            task,
            solution_path,
            error=score_error,
        )
    if score.executed and score.feasible is None and score.error:
        observer.record_error("hidden_checker", "checker_execution_error")
        if task_event_callback is not None:
            task_event_callback(
                {
                    "event": "checker_error",
                    "run_id": run_id,
                    "stage": "hidden_checker",
                    "error_kind": "checker_execution_error",
                    "error": score.error,
                }
            )
    successful_algorithm_packages = _read_successful_algorithm_packages(
        runtime_workspace,
        hidden_checker_feasible=(
            algorithm_package_stats_enabled and score.feasible is True
        ),
    )
    package_usage_summary = _read_package_usage_summary(
        runtime.repository.run_path(run_id)
        if algorithm_package_stats_enabled and run_id is not None
        else None
    )
    successful_solver_execution = (
        _select_successful_solver_execution(
            package_usage_summary,
            successful_algorithm_packages,
        )
        if score.feasible is True
        else None
    )
    return TaskEvaluation(
        task=task,
        score=score,
        run_id=run_id,
        run_status=run_status,
        duration_seconds=time.perf_counter() - started,
        error_diagnostics=observer.error_diagnostics(),
        timing_summary=timing_summary,
        successful_algorithm_packages=successful_algorithm_packages,
        successful_solver_execution=successful_solver_execution,
        package_usage_summary=package_usage_summary,
        intake_mode=intake_mode,
        intake_dialogue=tuple(intake_dialogue),
    )


async def run_benchmark(
    tasks: Sequence[BenchmarkTask],
    settings: Settings,
    *,
    jobs: int = 1,
    task_timeout_seconds: float = 7200,
    feasibility_review_enabled: bool = True,
    input_schema_enabled: bool = True,
    algorithm_library_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
    timing_enabled: bool = True,
    algorithm_package_stats_enabled: bool = True,
    components_enabled: bool = True,
    package_policy: PackagePolicy = PackagePolicy(),
    workflow: str = "standard",
    suite: str = "large-max",
    stop_on_resource_exhaustion: bool = False,
    max_consecutive_network_failures: int = 0,
    progress: Callable[[str], None] | None = None,
    task_event: TaskEventCallback | None = None,
    intake_mode: IntakeMode = "bypass",
    max_clarification_rounds: int = 5,
) -> FrontierORReport:
    if jobs <= 0:
        raise ValueError("jobs must be greater than 0")
    if jobs > settings.solver_cpu_budget:
        raise ValueError(
            f"jobs cannot exceed SOLVER_CPU_BUDGET ({settings.solver_cpu_budget})"
        )
    if task_timeout_seconds <= 0:
        raise ValueError("task_timeout_seconds must be greater than 0")
    if max_consecutive_network_failures < 0:
        raise ValueError("max_consecutive_network_failures cannot be negative")
    if intake_mode not in {"bypass", "full"}:
        raise ValueError("intake_mode must be bypass or full")
    if max_clarification_rounds <= 0:
        raise ValueError("max_clarification_rounds must be greater than 0")
    if workflow not in {"standard", "gurobi-formulator"}:
        raise ValueError("workflow must be standard or gurobi-formulator")
    no_design_gurobi_arm = (
        not algorithm_design_enabled
        and not components_enabled
        and package_policy.pool == "gurobi-only"
        and package_policy.cross_package_enabled
        and input_schema_enabled
        and algorithm_library_enabled
        and feasibility_review_enabled
        and problem_contract_enabled
    )
    if workflow == "gurobi-formulator" and not no_design_gurobi_arm:
        raise ValueError(
            "gurobi-formulator workflow requires --package-pool gurobi-only, "
            "--no-algorithm-design, --no-components, Review and all contract/library inputs"
        )
    if no_design_gurobi_arm and workflow != "gurobi-formulator":
        raise ValueError("This configuration requires workflow=gurobi-formulator")
    no_review_gurobi_arm = (
        algorithm_design_enabled
        and components_enabled
        and package_policy.pool == "gurobi-only"
        and package_policy.cross_package_enabled
        and input_schema_enabled
        and algorithm_library_enabled
        and not feasibility_review_enabled
        and problem_contract_enabled
    )
    skeleton_arm = (
        not algorithm_design_enabled
        and not problem_contract_enabled
        and not input_schema_enabled
        and not algorithm_library_enabled
        and not feasibility_review_enabled
    )
    if not problem_contract_enabled and not skeleton_arm and not (
        input_schema_enabled
        and algorithm_library_enabled
        and feasibility_review_enabled
        and algorithm_design_enabled
        and components_enabled
        and package_policy.is_default
    ):
        raise ValueError("--no-problem-contract is a single-factor ablation and cannot be combined")
    if not components_enabled and not all(
        (
            input_schema_enabled,
            algorithm_library_enabled,
            feasibility_review_enabled,
            algorithm_design_enabled or no_design_gurobi_arm,
            problem_contract_enabled,
        )
    ):
        raise ValueError(
            "components ablation requires the complete workflow so it remains single-factor"
        )
    # Besides single-factor package-policy ablations, allow the two deliberate
    # Gurobi workflow endpoints: Formulator + Review and Design + no Review.
    # Other crossed configurations stay rejected because their meaning is undefined.
    if (
        not package_policy.is_default
        and not no_design_gurobi_arm
        and not no_review_gurobi_arm
    ):
        if not algorithm_library_enabled:
            raise ValueError(
                "package pool and cross-package ablations require the algorithm library"
            )
        if not all(
            (
                input_schema_enabled,
                feasibility_review_enabled,
                algorithm_design_enabled,
                problem_contract_enabled,
                components_enabled,
            )
        ):
            raise ValueError(
                "package pool and cross-package ablations require the complete workflow "
                "so they remain single-factor"
            )
    # Algorithm Design can be removed independently.  Solving then owns the
    # complete plan, while the normal algorithm-library/component interfaces
    # remain available unless explicitly disabled by their own flags.
    started_at = datetime.now(timezone.utc)
    breaker = _CircuitBreaker(
        stop_on_resource_exhaustion=stop_on_resource_exhaustion,
        max_consecutive_network_failures=max_consecutive_network_failures,
    )
    breaker_event = asyncio.Event()

    async def run_one(index: int, task: BenchmarkTask) -> tuple[int, TaskEvaluation]:
        task_started = time.perf_counter()

        def emit_task_event(event_type: str, **payload: Any) -> None:
            run_id = payload.get("run_id")
            if run_id and "run_directory" not in payload:
                payload["run_directory"] = str(settings.runs_dir.expanduser().resolve() / run_id)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event_type,
                "index": index + 1,
                "total": len(tasks),
                "task_id": task.task_id,
                **payload,
            }
            if task_event is not None:
                task_event(event)
            if progress is not None:
                progress(_format_task_progress(event))

        emit_task_event("started")

        def record_resource_exhaustion(detail: str) -> None:
            breaker.trip_resource(task.task_id, detail)
            if breaker.tripped:
                breaker_event.set()

        try:
            result = await evaluate_task(
                task,
                settings,
                task_timeout_seconds=task_timeout_seconds,
                feasibility_review_enabled=feasibility_review_enabled,
                input_schema_enabled=input_schema_enabled,
                algorithm_library_enabled=algorithm_library_enabled,
                algorithm_design_enabled=algorithm_design_enabled,
                problem_contract_enabled=problem_contract_enabled,
                timing_enabled=timing_enabled,
                algorithm_package_stats_enabled=algorithm_package_stats_enabled,
                components_enabled=components_enabled,
                package_policy=package_policy,
                resource_exhaustion_callback=record_resource_exhaustion,
                task_event_callback=lambda event: emit_task_event(
                    str(event.get("event") or "event"),
                    **{key: value for key, value in event.items() if key != "event"},
                ),
                intake_mode=intake_mode,
                max_clarification_rounds=max_clarification_rounds,
                solver_cpu_limit=settings.solver_cpu_budget // jobs,
                workflow=workflow,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            result = TaskEvaluation(
                task=task,
                score=TaskScore(
                    task_id=task.task_id,
                    problem_class=task.problem_class,
                    executed=False,
                    error=f"Runner failed: {detail}",
                ),
                run_id=None,
                run_status="runner_failed",
                duration_seconds=time.perf_counter() - task_started,
                error_diagnostics={
                    "total_errors": 1,
                    "flow_section_counts": {"runner": 1},
                    "error_kind_counts": {"runner_exception": 1},
                    "most_repeated_flow_section": "runner",
                    "most_repeated_flow_section_count": 1,
                },
            )
        breaker.observe(result)
        if breaker.tripped:
            breaker_event.set()
        emit_task_event(
            "completed",
            run_id=result.run_id,
            run_status=result.run_status,
            passed=result.score.passed,
            executed=result.score.executed,
            feasible=result.score.feasible,
            declared_infeasible=result.score.declared_infeasible,
            gap=result.score.gap,
            error=_score_failure_detail(result.score),
            duration_seconds=round(result.duration_seconds, 3),
        )
        return index, result

    next_index = 0
    active: set[asyncio.Task[tuple[int, TaskEvaluation]]] = set()
    completed: list[tuple[int, TaskEvaluation]] = []

    def schedule_available() -> None:
        nonlocal next_index
        while not breaker.tripped and next_index < len(tasks) and len(active) < jobs:
            active.add(asyncio.create_task(run_one(next_index, tasks[next_index])))
            next_index += 1

    schedule_available()
    breaker_waiter = asyncio.create_task(breaker_event.wait())
    while active:
        done, _ = await asyncio.wait(
            active | {breaker_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        completed_futures = done - {breaker_waiter}
        active.difference_update(completed_futures)
        for future in completed_futures:
            completed.append(future.result())
        if breaker.tripped:
            if progress is not None:
                progress(
                    "CIRCUIT BREAKER: "
                    f"{breaker.reason}; cancelling {len(active)} active task(s) and "
                    f"skipping {len(tasks) - next_index} unscheduled task(s)"
                )
            for future in active:
                future.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            active.clear()
            break
        schedule_available()

    breaker_waiter.cancel()
    await asyncio.gather(breaker_waiter, return_exceptions=True)

    evaluations = [item for _, item in sorted(completed, key=lambda pair: pair[0])]
    return FrontierORReport(
        evaluations=evaluations,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        model=settings.llm_chat_model,
        jobs=jobs,
        feasibility_review_enabled=feasibility_review_enabled,
        input_schema_enabled=input_schema_enabled,
        algorithm_library_enabled=algorithm_library_enabled,
        algorithm_design_enabled=algorithm_design_enabled,
        problem_contract_enabled=problem_contract_enabled,
        timing_enabled=timing_enabled,
        algorithm_package_stats_enabled=algorithm_package_stats_enabled,
        components_enabled=components_enabled,
        package_pool=package_policy.pool,
        cross_package_enabled=package_policy.cross_package_enabled,
        workflow=workflow,
        gurobi_formulator_enabled=workflow == "gurobi-formulator",
        solving_mode=(
            "translation-only" if workflow == "gurobi-formulator" else "algorithm-guided"
        ),
        suite=suite,
        llm_max_attempts=settings.opt_llm_max_attempts,
        planned_tasks=len(tasks),
        stop_on_resource_exhaustion=stop_on_resource_exhaustion,
        max_consecutive_network_failures=max_consecutive_network_failures,
        termination_kind=breaker.kind,
        termination_reason=breaker.reason,
        intake_mode=intake_mode,
        max_clarification_rounds=max_clarification_rounds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FrontierOR tasks through the Core Agent and hidden checker."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_TASKS_ROOT, help="Task collection directory")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Run only this task ID; repeatable",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N sorted tasks")
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent tasks (default: 1)")
    parser.add_argument(
        "--intake-mode",
        choices=("bypass", "full"),
        default="bypass",
        help="Use the historical bypass flow or full Intake without gold data",
    )
    parser.add_argument(
        "--max-clarification-rounds",
        type=int,
        default=5,
        help="Maximum automatic feedback rounds in full Intake mode (default: 5)",
    )
    parser.add_argument(
        "--feasibility-review",
        dest="feasibility_review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable the complete validation component (default: enabled). "
            "Disabling implements the paper's gurobipy-only No review arm."
        ),
    )
    parser.add_argument(
        "--algorithm-design",
        dest="algorithm_design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Algorithm Design; disabling is only for the paper's No design arm",
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Runtime and benchmark timing statistics (default: enabled)",
    )
    parser.add_argument(
        "--algorithm-package-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track source=package usage for hidden-checker passing solutions",
    )
    parser.add_argument(
        "--components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structured components; disabling is valid only for the No design arm",
    )
    parser.add_argument(
        "--package-pool",
        choices=tuple(sorted(PACKAGE_POOLS)),
        default="full",
        help=(
            "Algorithm package pool (default: full). The paper's library ablation uses "
            "gurobi-only and blocks all other package imports, including NumPy and SciPy."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=("standard", "gurobi-formulator"),
        default="standard",
        help="Explicit workflow; gurobi-formulator is only for the paper's No design arm",
    )
    # These capabilities stay enabled in every ablation reported in the paper.
    # Keep them in the runner contract without exposing unreported CLI arms.
    parser.set_defaults(
        input_schema=True,
        algorithm_library=True,
        problem_contract=True,
        cross_package=True,
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=7200,
        help="Timeout for each Core Runtime in seconds (default: 7200)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Evaluation report JSON path")
    parser.add_argument(
        "--suite",
        choices=("FrontierOR65-Fea", "FrontierOR10-Inf", "Hard32-Fea"),
        default=DEFAULT_SUITE,
        help="Evaluation suite; all instances are loaded from --large-root",
    )
    parser.add_argument(
        "--large-root",
        type=Path,
        default=None,
        help="External FrontierOR large-data root",
    )
    parser.add_argument(
        "--stop-on-resource-exhaustion",
        action="store_true",
        help="Stop the batch when memory exhaustion is detected",
    )
    parser.add_argument(
        "--max-consecutive-network-failures",
        type=int,
        default=0,
        help="Stop after this many consecutive task-level network failures; 0 disables",
    )
    return parser


def _validate_workflow_ablation_args(args: argparse.Namespace) -> None:
    """Accept only the full system and the three ablation arms reported in the paper."""

    configuration = (
        args.package_pool,
        args.workflow,
        args.algorithm_design,
        args.components,
        args.feasibility_review,
    )
    reported_arms = {
        ("full", "standard", True, True, True),
        ("gurobi-only", "standard", True, True, True),
        ("gurobi-only", "gurobi-formulator", False, False, True),
        ("gurobi-only", "standard", True, True, False),
    }
    if configuration not in reported_arms:
        raise ValueError(
            "Only paper-reported configurations are supported: Full, gurobipy-only, "
            "gurobipy-only/no-design, or gurobipy-only/no-review"
        )


# Stable private aliases retained for existing callers while ownership lives in cli.py.
_parser = lambda: build_parser(  # noqa: E731
    default_tasks_root=DEFAULT_TASKS_ROOT,
    default_suite=DEFAULT_SUITE,
)
_validate_workflow_ablation_args = validate_reported_ablation


def _build_case_index(report_payload: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Compatibility wrapper for the historical private helper."""

    return build_case_index(report_payload, runs_dir)


def _write_package_usage_csv(path: Path, report_payload: dict[str, Any]) -> None:
    """Compatibility wrapper for the historical private helper."""

    write_package_usage_csv(path, report_payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_workflow_ablation_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")
    if args.max_consecutive_network_failures < 0:
        raise SystemExit("--max-consecutive-network-failures cannot be negative")
    if args.large_root is None:
        raise SystemExit(f"--suite {args.suite} requires --large-root")
    settings = Settings()
    if args.algorithm_library:
        # Ablation 2 omits the catalog, so unused manifests must not block startup.
        try:
            validate_algorithm_library_runtime(
                LocalAlgorithmCatalog(settings.opt_algorithm_manifests_dir)
            )
        except AlgorithmRuntimeValidationError as exc:
            raise SystemExit(str(exc)) from exc
    only = set(args.task) if args.task else None
    if args.suite == "Hard32-Fea":
        root = args.root if args.root != DEFAULT_TASKS_ROOT else HARD32_FEA_ROOT
        tasks = load_selected_tasks(
            discover_tasks(root),
            args.large_root,
            HARD32_FEA_SELECTION,
            expected_count=32,
            suite=args.suite,
        )
        if only:
            tasks = [task for task in tasks if task.paper_id in only or task.task_id in only]
    else:
        suite_index = LARGE_SUITE_INDEXES[args.suite]
        suite_root = FRONTIEROR10_INF_ROOT if args.suite == "FrontierOR10-Inf" else DEFAULT_TASKS_ROOT
        root = args.root if args.root != DEFAULT_TASKS_ROOT else suite_root
        only = normalize_large_suite_task_ids(suite_index, suite=args.suite, requested=only)
        tasks = discover_tasks(root, only=only)
        tasks = load_large_max_tasks(
            tasks,
            args.large_root,
            suite_index,
            suite=args.suite,
        )
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No runnable FrontierOR tasks were found")

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"frontieror-{args.suite}"
        output = PROJECT_ROOT / "benchmark-results" / f"{prefix}-{stamp}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    case_events_path = output.with_name("case-events.jsonl")
    case_index_path = output.with_name("case-index.json")
    package_usage_csv_path = output.with_name("package-usage.csv")

    event_stream = case_events_path.open("w", encoding="utf-8", buffering=1)
    try:
        report = asyncio.run(
            run_benchmark(
                tasks,
                settings,
                jobs=args.jobs,
                task_timeout_seconds=args.task_timeout_seconds,
                feasibility_review_enabled=args.feasibility_review,
                input_schema_enabled=args.input_schema,
                algorithm_library_enabled=args.algorithm_library,
                algorithm_design_enabled=args.algorithm_design,
                problem_contract_enabled=args.problem_contract,
                timing_enabled=args.timing,
                algorithm_package_stats_enabled=args.algorithm_package_stats,
                components_enabled=args.components,
                package_policy=PackagePolicy(
                    pool=args.package_pool,
                    cross_package_enabled=args.cross_package,
                ),
                workflow=args.workflow,
                suite=args.suite,
                stop_on_resource_exhaustion=args.stop_on_resource_exhaustion,
                max_consecutive_network_failures=args.max_consecutive_network_failures,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                task_event=lambda event: event_stream.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                ),
                intake_mode=args.intake_mode,
                max_clarification_rounds=args.max_clarification_rounds,
            )
        )
    finally:
        event_stream.close()

    report_payload = report.to_dict()
    report_payload["case_events_file"] = str(case_events_path)
    report_payload["case_index_file"] = str(case_index_path)
    report_payload["package_usage_csv_file"] = (
        str(package_usage_csv_path) if args.algorithm_package_stats else None
    )
    atomic_write_json(output, report_payload)
    atomic_write_json(case_index_path, _build_case_index(report_payload, settings.runs_dir))
    if args.algorithm_package_stats:
        _write_package_usage_csv(package_usage_csv_path, report_payload)
    print(
        json.dumps(
            {
                "report": str(output),
                "case_events": str(case_events_path),
                "case_index": str(case_index_path),
                "package_usage_csv": (
                    str(package_usage_csv_path) if args.algorithm_package_stats else None
                ),
                "feasibility_review_enabled": args.feasibility_review,
                "input_schema_enabled": args.input_schema,
                "algorithm_library_enabled": args.algorithm_library,
                "timing_enabled": args.timing,
                "algorithm_package_stats_enabled": args.algorithm_package_stats,
                "components_enabled": args.components,
                "workflow": report_payload["workflow"],
                "gurobi_formulator_enabled": report_payload["gurobi_formulator_enabled"],
                "solving_mode": report_payload["solving_mode"],
                "intake_mode": args.intake_mode,
                **report_payload["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 2 if report.termination_kind is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
