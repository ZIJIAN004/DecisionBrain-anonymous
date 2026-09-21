"""Score candidate feasibility and objective gap against the Gurobi reference.

Only these two quantities are measured. Time and optimality proof are excluded:

- Reference timing is incomplete and not comparable with end-to-end Agent execution.
- Only 5 of 61 tasks provide a ``gap`` field supporting an optimality claim.

Task-specific hidden checkers determine feasibility through a uniform CLI. Three checkers lack
``check_feasibility()``, so all run in subprocesses rather than through imports.

The candidate reports ``objective_value``; the checker recomputes it and rejects discrepancies,
so the field is gated and suitable for scoring.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import BenchmarkTask

CHECKER_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class TaskScore:
    """Evaluation result for one task."""

    task_id: str
    problem_class: str
    executed: bool
    feasible: bool | None = None
    declared_infeasible: bool = False
    reference_feasible: bool = True
    objective: float | None = None
    reference_objective: float | None = None
    gap: float | None = None
    violated_constraints: tuple[str, ...] = ()
    violations: tuple[Any, ...] = ()
    error: str = ""
    checker_stderr: str = ""

    @property
    def passed(self) -> bool:
        """Return whether execution succeeded and produced a feasible result."""
        if self.declared_infeasible:
            return self.executed and self.reference_feasible is False
        return self.executed and self.feasible is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "problem_class": self.problem_class,
            "executed": self.executed,
            "feasible": self.feasible,
            "declared_infeasible": self.declared_infeasible,
            "objective": self.objective,
            "reference_objective": self.reference_objective,
            "gap": self.gap,
            "violated_constraints": list(self.violated_constraints),
            "violations": list(self.violations)[:20],
            "error": self.error,
            "checker_stderr": self.checker_stderr,
        }


@dataclass
class BenchmarkReport:
    """Summary of an evaluation run."""

    scores: list[TaskScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def executed(self) -> int:
        return sum(1 for s in self.scores if s.executed)

    @property
    def feasible(self) -> int:
        return sum(1 for s in self.scores if s.feasible is True)

    @property
    def declared_infeasible(self) -> int:
        return sum(1 for s in self.scores if s.declared_infeasible)

    def gaps(self) -> list[float]:
        """Return gaps only for feasible solutions."""
        return [s.gap for s in self.scores if s.feasible is True and s.gap is not None]

    def to_dict(self) -> dict[str, Any]:
        gaps = sorted(self.gaps())
        summary: dict[str, Any] = {
            "total": self.total,
            "executed": self.executed,
            "feasible": self.feasible,
            "declared_infeasible": self.declared_infeasible,
            "correct_infeasibility_declarations": sum(
                1
                for score in self.scores
                if score.declared_infeasible and score.reference_feasible is False
            ),
            "execution_rate": self.executed / self.total if self.total else 0.0,
            "feasibility_rate": self.feasible / self.total if self.total else 0.0,
            "gap_count": len(gaps),
        }
        if gaps:
            summary["gap_median"] = gaps[len(gaps) // 2]
            summary["gap_best"] = gaps[0]
            summary["gap_worst"] = gaps[-1]
            summary["matched_reference"] = sum(1 for g in gaps if g <= 1e-6)
        return {"summary": summary, "tasks": [s.to_dict() for s in self.scores]}


def relative_gap(
    objective: float | None,
    reference: float | None,
    *,
    direction: str,
) -> float | None:
    """Return candidate gap relative to the reference; positive values are worse.

    Normalize by the absolute reference value. Use absolute difference when the
    reference is zero.
    """
    if objective is None or reference is None:
        return None
    delta = objective - reference if direction == "minimize" else reference - objective
    scale = abs(reference)
    return delta / scale if scale > 1e-12 else delta


def run_checker(
    task: BenchmarkTask,
    instance_path: Path,
    solution_path: Path,
    *,
    timeout: int = CHECKER_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, str]:
    """Run the task's hidden checker and return ``(result, stderr)``.

    Use the subprocess CLI because some checkers expose no callable function and
    subprocesses isolate checker exceptions and ``sys.exit`` calls.
    """
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "feasibility_result.json"
        command = [
            sys.executable,
            str(task.feasibility_checker),
            "--instance_path",
            str(instance_path),
            "--solution_path",
            str(solution_path),
            "--result_path",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(task.hidden_dir),
            )
        except subprocess.TimeoutExpired:
            return None, f"Checker timed out after {timeout}s"
        if not result_path.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()
            return None, f"Checker did not produce a result file: {detail[:400]}"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"Checker result is not valid JSON: {exc.msg}"
        return payload, (completed.stderr or "").strip()


def score_task(
    task: BenchmarkTask,
    solution_path: Path | None,
    *,
    instance_path: Path | None = None,
    error: str = "",
) -> TaskScore:
    """Score a task candidate; ``None`` means no solution was produced."""

    reference = task.reference_objective()
    if solution_path is None or not solution_path.is_file():
        return TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=False,
            reference_feasible=task.reference_feasible,
            reference_objective=reference,
            error=error or "No candidate solution was produced",
        )

    instance = instance_path or task.instance_path
    result, stderr = run_checker(task, instance, solution_path)
    try:
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=False,
            reference_feasible=task.reference_feasible,
            reference_objective=reference,
            error=f"Candidate solution is not valid JSON: {exc.msg}",
        )

    raw_objective = solution.get("objective_value")
    objective = (
        float(raw_objective)
        if isinstance(raw_objective, (int, float)) and not isinstance(raw_objective, bool)
        else None
    )

    if result is None:
        return TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=True,
            reference_feasible=task.reference_feasible,
            objective=objective,
            reference_objective=reference,
            error=stderr,
            checker_stderr=stderr,
        )

    return TaskScore(
        task_id=task.task_id,
        problem_class=task.problem_class,
        executed=True,
        reference_feasible=task.reference_feasible,
        feasible=bool(result.get("feasible")),
        objective=objective,
        reference_objective=reference,
        gap=relative_gap(objective, reference, direction=task.objective_direction),
        violated_constraints=tuple(str(v) for v in (result.get("violated_constraints") or ())),
        violations=tuple(result.get("violations") or ()),
        checker_stderr=stderr,
    )
