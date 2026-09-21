"""Benchmark task discovery and loading.

A task-contract directory has this layout::

    <task_id>/
    |-- input/          Agent-visible problem.md
    |-- hidden/         evaluator-only checker and target schema
    `-- task.json       metadata

Large suites bind instances and references from an external data root. ``input`` and ``hidden``
must remain physically separate because Runtime snapshots almost all input content into the
Agent workspace; any reference material under ``input`` becomes visible to the Agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BenchmarkTaskError(ValueError):
    """The task directory is incomplete or violates its contract."""


HIDDEN_REQUIRED = (
    "feasibility_check.py",
    "solution_schema.json",
)


@dataclass(frozen=True)
class BenchmarkTask:
    """An executable benchmark task."""

    task_id: str
    root: Path
    metadata: dict[str, Any]
    instance_path_override: Path | None = None
    reference_solution_override: Path | None = None

    @property
    def input_dir(self) -> Path:
        """The only directory allowed as Runtime input."""
        return self.root / "input"

    @property
    def hidden_dir(self) -> Path:
        """Evaluator-only material that must never enter the workspace."""
        return self.root / "hidden"

    @property
    def problem_md(self) -> Path:
        return self.input_dir / "problem.md"

    @property
    def paper_id(self) -> str:
        return str(self.metadata.get("paper_id") or self.root.name)

    @property
    def instance_variant(self) -> str:
        return str(self.metadata.get("instance") or "external")

    @property
    def instance_index(self) -> int | None:
        value = self.metadata.get("instance_index")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def instance_path(self) -> Path:
        if self.instance_path_override is None:
            raise BenchmarkTaskError(
                f"{self.task_id}: no external instance is bound; select a large suite and "
                "provide --large-root"
            )
        return self.instance_path_override

    @property
    def reference_feasible(self) -> bool:
        return self.metadata.get("reference_feasible") is not False

    @property
    def reference_solution(self) -> Path:
        if self.reference_solution_override is None:
            raise BenchmarkTaskError(
                f"{self.task_id}: no external reference solution is bound; select a large "
                "suite and provide --large-root"
            )
        return self.reference_solution_override

    @property
    def feasibility_checker(self) -> Path:
        return self.hidden_dir / "feasibility_check.py"

    @property
    def target_solution_schema(self) -> Path:
        """Solution format injected into Core as the benchmark output contract."""
        return self.hidden_dir / "solution_schema.json"

    @property
    def objective_direction(self) -> str:
        """Return ``minimize`` or ``maximize`` from the abbreviated metadata value."""
        raw = str(self.metadata.get("direction") or "").strip().lower()
        if raw.startswith("max"):
            return "maximize"
        if raw.startswith("min"):
            return "minimize"
        raise BenchmarkTaskError(f"{self.task_id}: metadata has no recognized direction")

    @property
    def problem_class(self) -> str:
        return str(self.metadata.get("problem_class") or "unknown")

    def reference_objective(self) -> float | int | None:
        """Return the reference objective, or ``None`` when unavailable."""
        if not self.reference_feasible:
            return None
        payload = json.loads(self.reference_solution.read_text(encoding="utf-8"))
        value = payload.get("objective_value")
        if value is None:
            value = payload.get("objective")
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load_task(root: Path) -> BenchmarkTask:
    """Load one task directory and validate its structure."""

    root = root.expanduser().resolve()
    meta_path = root / "task.json"
    if not meta_path.is_file():
        raise BenchmarkTaskError(f"Missing task.json: {root}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    task = BenchmarkTask(task_id=root.name, root=root, metadata=metadata)
    if not task.problem_md.is_file() or not task.problem_md.read_text(encoding="utf-8").strip():
        raise BenchmarkTaskError(f"{task.task_id}: missing non-empty input/problem.md")
    for name in HIDDEN_REQUIRED:
        if not (task.hidden_dir / name).is_file():
            raise BenchmarkTaskError(f"{task.task_id}: missing hidden/{name}")

    leaked = sorted(
        path.relative_to(task.input_dir).as_posix()
        for path in task.input_dir.rglob("*")
        if path.is_file() and _looks_like_reference_material(path.name)
    )
    if leaked:
        raise BenchmarkTaskError(f"{task.task_id}: evaluator material leaked into input/: {', '.join(leaked)}")
    return task


def discover_tasks(root: Path, *, only: set[str] | None = None) -> list[BenchmarkTask]:
    """Scan a task collection and return tasks sorted by task ID."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise BenchmarkTaskError(f"Task collection directory does not exist: {root}")
    tasks: list[BenchmarkTask] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if only is not None and entry.name not in only:
            continue
        if not (entry / "task.json").is_file():
            continue
        tasks.append(load_task(entry))
    if only:
        missing = only - {t.task_id for t in tasks}
        if missing:
            raise BenchmarkTaskError(f"Tasks not found: {', '.join(sorted(missing))}")
    return tasks


def load_large_max_tasks(
    tasks: list[BenchmarkTask],
    large_root: Path,
    suite_index: Path,
    *,
    suite: str = "large-max",
) -> list[BenchmarkTask]:
    """Load each FrontierOR task's largest instance from a fixed suite manifest."""

    large_root = large_root.expanduser().resolve()
    suite_index = suite_index.expanduser().resolve()
    if not large_root.is_dir():
        raise BenchmarkTaskError(f"FrontierOR large directory does not exist: {large_root}")
    if not suite_index.is_file():
        raise BenchmarkTaskError(f"{suite} suite manifest does not exist: {suite_index}")
    payload = json.loads(suite_index.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "FrontierOR" or payload.get("suite") != suite:
        raise BenchmarkTaskError(f"{suite} suite manifest identifier is invalid: {suite_index}")
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise BenchmarkTaskError(f"{suite} suite manifest has no cases: {suite_index}")
    if payload.get("case_count") != len(cases):
        raise BenchmarkTaskError(f"{suite} suite case_count does not match cases: {suite_index}")

    selected: list[BenchmarkTask] = []
    for task in tasks:
        case = cases.get(task.task_id)
        if not isinstance(case, dict):
            raise BenchmarkTaskError(f"{suite} suite is missing task: {task.task_id}")
        index = case.get("instance_index")
        expected_bytes = case.get("instance_bytes")
        reference_feasible = case.get("reference_feasible")
        if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
            raise BenchmarkTaskError(f"{task.task_id}: invalid instance_index")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise BenchmarkTaskError(f"{task.task_id}: invalid instance_bytes")
        if not isinstance(reference_feasible, bool):
            raise BenchmarkTaskError(f"{task.task_id}: invalid reference_feasible")

        source_root = large_root / task.task_id
        instance_path = source_root / "instance" / f"large_instance_{index}.json"
        reference_solution = source_root / "gurobi_solution" / f"large_solution_{index}.json"
        if not instance_path.is_file():
            raise BenchmarkTaskError(f"{task.task_id}: missing large-max file: {instance_path}")
        if reference_feasible and not reference_solution.is_file():
            raise BenchmarkTaskError(f"{task.task_id}: missing large-max file: {reference_solution}")
        if instance_path.stat().st_size != expected_bytes:
            raise BenchmarkTaskError(
                f"{task.task_id}: large-max instance size changed; expected {expected_bytes}, "
                f"got {instance_path.stat().st_size}"
            )
        largest = max(
            (source_root / "instance").glob("large_instance_*.json"),
            key=lambda path: (path.stat().st_size, path.name),
        )
        if largest.resolve() != instance_path.resolve():
            raise BenchmarkTaskError(
                f"{task.task_id}: manifest instance is no longer the largest; current largest "
                f"is {largest.name}"
            )

        selected.append(
            BenchmarkTask(
                task_id=f"{task.task_id}-large-{index}",
                root=task.root,
                metadata=task.metadata
                | {
                    "paper_id": task.task_id,
                    "instance": "large-max",
                    "instance_index": index,
                    "instance_bytes": expected_bytes,
                    "reference_feasible": reference_feasible,
                },
                instance_path_override=instance_path,
                reference_solution_override=reference_solution if reference_solution.is_file() else None,
            )
        )
    return selected


def load_large_suite_task_ids(suite_index: Path, *, suite: str) -> set[str]:
    """Read and validate a large-suite manifest and return its paper IDs."""

    suite_index = suite_index.expanduser().resolve()
    if not suite_index.is_file():
        raise BenchmarkTaskError(f"{suite} suite manifest does not exist: {suite_index}")
    payload = json.loads(suite_index.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if payload.get("benchmark") != "FrontierOR" or payload.get("suite") != suite:
        raise BenchmarkTaskError(f"{suite} suite manifest identifier is invalid: {suite_index}")
    if not isinstance(cases, dict):
        raise BenchmarkTaskError(f"{suite} suite manifest has no cases: {suite_index}")
    if payload.get("case_count") != len(cases):
        raise BenchmarkTaskError(f"{suite} suite case_count does not match cases: {suite_index}")
    return set(cases)


def load_selected_tasks(
    tasks: list[BenchmarkTask],
    large_root: Path,
    selection_path: Path,
    *,
    expected_count: int = 166,
    suite: str = "Hard32-Fea",
) -> list[BenchmarkTask]:
    """Bind tasks from a repository-owned fixed case selection (list form).

    Unlike ``load_large_max_tasks`` the list form allows several instances of the
    same paper, which list-based release suites may rely on.
    """
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise BenchmarkTaskError(
            f"{suite} selection must contain exactly {expected_count} cases"
        )
    by_id = {task.task_id: task for task in tasks}
    selected = []
    for case in payload:
        paper_id, index = case.get("task"), case.get("instance_index")
        if paper_id not in by_id or not isinstance(index, int) or index <= 0:
            raise BenchmarkTaskError(f"invalid {suite} selection case: {case}")
        root = large_root / paper_id
        instance = root / "instance" / f"large_instance_{index}.json"
        reference = root / "gurobi_solution" / f"large_solution_{index}.json"
        if not instance.is_file() or not reference.is_file():
            raise BenchmarkTaskError(f"{paper_id}: selected large files are missing")
        selected.append(BenchmarkTask(
            task_id=f"{paper_id}-large-{index}", root=by_id[paper_id].root,
            metadata=by_id[paper_id].metadata | {"paper_id": paper_id, "instance": "large-max",
                "instance_index": index, "instance_bytes": instance.stat().st_size,
                "reference_feasible": bool(case.get("reference_feasible", True))},
            instance_path_override=instance, reference_solution_override=reference))
    return selected


def normalize_large_suite_task_ids(
    suite_index: Path,
    *,
    suite: str,
    requested: set[str] | None,
) -> set[str]:
    """Normalize paper IDs and emitted ``paper_id-large-index`` task IDs for a large suite."""

    paper_ids = load_large_suite_task_ids(suite_index, suite=suite)
    if requested is None:
        return paper_ids
    payload = json.loads(suite_index.expanduser().resolve().read_text(encoding="utf-8"))
    cases = payload["cases"]
    aliases: dict[str, str] = {}
    for paper_id in paper_ids:
        case = cases[paper_id]
        index = case.get("instance_index")
        aliases[paper_id] = paper_id
        if isinstance(index, int) and not isinstance(index, bool) and index > 0:
            aliases[f"{paper_id}-large-{index}"] = paper_id

    unknown = sorted(item for item in requested if item not in aliases)
    if unknown:
        examples = ", ".join(sorted(aliases)[:5])
        raise BenchmarkTaskError(
            f"{suite} suite cannot find tasks: {', '.join(unknown)}; "
            f"use paper_id or report task_id (for example: {examples})"
        )
    return {aliases[item] for item in requested}


def _looks_like_reference_material(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in ("reference", "gurobi", "feasibility_check", "formulation", "_schema")
    )
