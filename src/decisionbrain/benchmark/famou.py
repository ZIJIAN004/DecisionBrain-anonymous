"""Export and blind-score FrontierOR cases for an external solver platform.

The module deliberately has no dependency on a particular platform API.  The
public export can be uploaded through a UI or submitted through an API adapter;
the private scorer stays with the benchmark operator.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..run_storage.io import atomic_write_json
from .runner import DEFAULT_FRONTIEROR65_FEA_INDEX, DEFAULT_TASKS_ROOT
from .scoring import BenchmarkReport, TaskScore, score_task
from .task import (
    BenchmarkTask,
    discover_tasks,
    load_large_max_tasks,
    normalize_large_suite_task_ids,
)


DEFAULT_SUITE = "FrontierOR65-Fea"


FAMOU_PROMPT = """# FAMOU benchmark task prompt

Solve the optimization task using only the three files attached for the current task:

- `problem.md`: the authoritative problem statement and index semantics.
- `data/instance.json`: the complete instance data to solve.
- `submission_contract.json`: the exact JSON structure required for the answer.

Independently formulate and solve the optimization problem. Return one feasible solution with the
best objective value you can obtain. Interpret every array dimension and composite key exactly as
defined by `problem.md` and represented in `instance.json`; preserve index order and do not transpose,
rename, merge, or reinterpret indices. Check every constraint against the instance data before
returning the answer. Compute `objective_value` from the submitted decision values; do not estimate,
round without authorization, copy, or fabricate it.

Your final answer must be only one valid JSON object conforming exactly to
`submission_contract.json`. Include every required field, use the required JSON types, and do not add
undeclared fields. Do not wrap the JSON in Markdown fences and do not include explanations, formulas,
logs, source code, comments, or alternative solutions.

Prohibited:

- Do not request, infer, reconstruct, or use a reference solution, hidden checker, hidden test data,
  reference objective, reference implementation, solver log, or evaluation result.
- Do not replace the requested solution with model code, pseudocode, a mathematical formulation, a
  variable table, or instructions for solving it.
- Do not invent missing data, silently repair the instance, change units, relax constraints, drop
  constraints, or change the objective.
- Do not guess the meaning or order of ambiguous keys. Resolve them from `problem.md` and the actual
  structure of `instance.json`; if the supplied public files are genuinely inconsistent, report that
  conflict instead of silently choosing a different convention.
- Do not claim infeasibility merely because a solver times out, fails, or does not find a solution.

Only when you have established that the supplied instance is infeasible, return exactly:
`{"status":"infeasible"}`
"""


def _load_tasks(large_root: Path, only: set[str] | None = None) -> list[BenchmarkTask]:
    """Load exactly the fixed 65-case FrontierOR feasible suite."""
    selected_ids = normalize_large_suite_task_ids(
        DEFAULT_FRONTIEROR65_FEA_INDEX, suite=DEFAULT_SUITE, requested=only
    )
    base = discover_tasks(DEFAULT_TASKS_ROOT, only=selected_ids)
    return load_large_max_tasks(
        base, large_root, DEFAULT_FRONTIEROR65_FEA_INDEX, suite=DEFAULT_SUITE
    )


def _public_task_metadata(task: BenchmarkTask) -> dict[str, Any]:
    fields = (
        "paper_title",
        "year",
        "direction",
        "problem_class",
        "category",
        "formulation_type",
        "application_field",
        "avg_num_var",
        "avg_num_int_var",
        "avg_num_constr",
        "origin",
    )
    return {
        "task_id": task.task_id,
        "paper_id": task.paper_id,
        "instance_variant": task.instance_variant,
        "instance_index": task.instance_index,
        **{name: task.metadata.get(name) for name in fields if name in task.metadata},
    }


def _export_problem(source: Path, destination: Path) -> None:
    """Copy the benchmark problem statement without transforming its contents."""
    shutil.copy2(source, destination)


def export_suite(large_root: Path, output: Path, *, only: set[str] | None = None) -> dict[str, Any]:
    """Create the only material an external platform is permitted to receive."""
    tasks = _load_tasks(large_root, only)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for task in tasks:
        task_dir = output / "tasks" / task.task_id
        data_dir = task_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        _export_problem(task.problem_md, task_dir / "problem.md")
        shutil.copy2(task.instance_path, data_dir / "instance.json")
        # This is an interface contract, not a reference answer or checker.
        shutil.copy2(task.target_solution_schema, task_dir / "submission_contract.json")
        entries.append(_public_task_metadata(task))

    manifest = {
        "benchmark": "FrontierOR",
        "suite": DEFAULT_SUITE,
        "case_count": len(entries),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_layout": {
            "solved": "submissions/<task_id>/solution.json",
            "infeasible": "submissions/<task_id>/result.json containing {\"status\": \"infeasible\"}",
        },
        "tasks": entries,
    }
    atomic_write_json(output / "manifest.json", manifest)
    (output / "FAMOU_PROMPT.md").write_text(FAMOU_PROMPT, encoding="utf-8")
    (output / "SUBMISSION.md").write_text(
        "# External Solver Submission Protocol\n\n"
        "For solved tasks, provide `submissions/<task_id>/solution.json`. The file must be "
        "valid JSON and conform exactly to that task's `submission_contract.json`. "
        "`objective_value` must be included when the contract requires it. For an infeasibility "
        "claim, provide `submissions/<task_id>/result.json` with exactly `{\"status\": "
        "\"infeasible\"}`. Do not submit reference solutions, feasibility checkers, or generated "
        "model source as a substitute for the solution.\n",
        encoding="utf-8",
    )
    return manifest


def _json_objects_in_text(text: str) -> list[dict[str, Any]]:
    """Extract complete JSON objects from a platform's natural-language reply."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        identity = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if identity not in seen:
            seen.add(identity)
            objects.append(value)
    return objects


def _declares_infeasible(candidate: dict[str, Any]) -> bool:
    """Is this object the prompt's infeasibility declaration, however it was decorated?"""
    status = candidate.get("status")
    return isinstance(status, str) and status.strip().lower() == "infeasible"


def _python_literal_solution_objects(text: str) -> list[dict[str, Any]]:
    """Read literal solution assignments without executing platform-provided code."""
    try:
        module = ast.parse(text)
    except SyntaxError:
        return []

    names = {"solution", "SOLUTION", "result", "RESULT", "output", "OUTPUT"}
    objects: list[dict[str, Any]] = []
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id in names for target in statement.targets):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _python_literal_error(text: str) -> str | None:
    """Explain why a reply that resembles Python solution code was not usable."""
    if not any(f"{name} =" in text for name in ("solution", "SOLUTION", "result", "RESULT")):
        return None
    try:
        module = ast.parse(text)
    except SyntaxError as exc:
        return f"Python solution code is invalid or truncated ({exc.msg} at line {exc.lineno})"
    names = {"solution", "SOLUTION", "result", "RESULT", "output", "OUTPUT"}
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in statement.targets
        ):
            return "Python solution assignment is not a literal JSON-compatible object"
    return None


def _leading_json_error(text: str) -> str | None:
    """Report an invalid leading JSON object before considering nested objects."""
    start = text.find("{")
    if start == -1 or text[:start].strip():
        return None
    try:
        json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        return f"the leading JSON object is invalid ({exc.msg} at character {exc.pos + 1})"
    return None


def _write_import_failure(path: Path, task_id: str, reason: str) -> None:
    path.write_text(
        "# FAMOU reply could not be normalized\n\n"
        f"Task: `{task_id}`\n\n"
        f"Reason: {reason}\n",
        encoding="utf-8",
    )


def _reply_path(replies: Path, task_id: str) -> Path | None:
    """Find the task TXT, accepting one directory with an added platform suffix."""
    direct = replies / task_id / f"{task_id}.txt"
    if direct.is_file():
        return direct
    candidates = [
        path
        for path in replies.glob(f"{task_id}*/{task_id}.txt")
        if path.is_file() and path.parent.is_dir()
    ]
    return candidates[0] if len(candidates) == 1 else None


def import_replies(
    input_root: Path,
    replies: Path,
    submissions: Path | None = None,
    *,
    allow_python_literals: bool = False,
) -> dict[str, Any]:
    """Normalize per-task platform replies, preserving any existing solution JSON."""
    input_root = input_root.expanduser().resolve()
    replies = replies.expanduser().resolve()
    submissions = (submissions or replies).expanduser().resolve()
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8-sig"))
    if not replies.is_dir():
        raise ValueError(f"回复目录不存在：{replies}")

    imported: list[str] = []
    existing: list[str] = []
    unanswered: list[str] = []
    failed: list[str] = []
    for task in manifest["tasks"]:
        task_id = str(task["task_id"])
        target_dir = submissions / task_id
        existing_solution = target_dir / "solution.json"
        if existing_solution.is_file():
            existing.append(task_id)
            continue
        reply_path = _reply_path(replies, task_id)
        if reply_path is None:
            unanswered.append(task_id)
            continue
        reply_dir = reply_path.parent
        failure_path = reply_dir / "fail_to_match.md"
        try:
            text = reply_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            _write_import_failure(failure_path, task_id, "reply text is not valid UTF-8")
            failed.append(task_id)
            continue
        candidates = _json_objects_in_text(text)
        python_error = None
        if allow_python_literals:
            for candidate in _python_literal_solution_objects(text):
                if candidate not in candidates:
                    candidates.append(candidate)
            python_error = _python_literal_error(text)
        contract = json.loads(
            (input_root / "tasks" / task_id / "submission_contract.json").read_text(
                encoding="utf-8-sig"
            )
        )
        required_fields = set(contract)
        matching = [candidate for candidate in candidates if required_fields <= set(candidate)]
        # A declared infeasibility counts wherever it appears, not only as the reply's sole object:
        # `{"status":"infeasible","reason":...}`, or the declaration after a paragraph of prose, is
        # still an answer. A candidate that satisfies the contract wins, because a model that
        # produced an actual solution did not decline the instance.
        declaration = next((c for c in candidates if _declares_infeasible(c)), None)
        if not matching and declaration is not None:
            target = target_dir / "result.json"
            atomic_write_json(target, declaration)
            failure_path.unlink(missing_ok=True)
            imported.append(task_id)
            continue

        if len(matching) != 1:
            leading_error = _leading_json_error(text)
            if leading_error:
                reason = leading_error + "; the reply appears truncated or malformed"
            elif python_error:
                reason = python_error
            elif not candidates:
                reason = "no complete JSON object was found in the reply text"
            elif not matching:
                closest = max(candidates, key=lambda candidate: len(required_fields & set(candidate)))
                missing = ", ".join(sorted(required_fields - set(closest)))
                reason = (
                    "JSON was found, but the closest object is missing required top-level fields: "
                    + missing
                )
            else:
                reason = "multiple JSON objects match the submission contract, so the answer is ambiguous"
            _write_import_failure(failure_path, task_id, reason)
            failed.append(task_id)
            continue
        target = target_dir / "solution.json"
        atomic_write_json(target, matching[0])
        failure_path.unlink(missing_ok=True)
        imported.append(task_id)

    return {
        "imported": sorted(imported),
        "existing": sorted(existing),
        "unanswered": sorted(unanswered),
        "failed": sorted(failed),
    }


def evaluate_submissions(
    large_root: Path, submissions: Path, output: Path, *, only: set[str] | None = None
) -> dict[str, Any]:
    """Blind-score external answers using the benchmark's private checker."""
    tasks = _load_tasks(large_root, only)
    submissions = submissions.expanduser().resolve()
    scores = []
    for task in tasks:
        solution_path = submissions / task.task_id / "solution.json"
        result_path = submissions / task.task_id / "result.json"
        declared_infeasible = False
        if result_path.is_file():
            try:
                declared_infeasible = (
                    json.loads(result_path.read_text(encoding="utf-8")).get("status") == "infeasible"
                )
            except json.JSONDecodeError:
                pass
        if declared_infeasible:
            scores.append(
                TaskScore(
                    task_id=task.task_id,
                    problem_class=task.problem_class,
                    executed=True,
                    declared_infeasible=True,
                    reference_feasible=task.reference_feasible,
                    reference_objective=task.reference_objective(),
                    error="" if not task.reference_feasible else "platform incorrectly declared infeasible",
                )
            )
        else:
            scores.append(score_task(task, solution_path, instance_path=task.instance_path))

    report = BenchmarkReport(scores).to_dict()
    report.update(
        {
            "benchmark": "FrontierOR",
            "suite": DEFAULT_SUITE,
            "system": "famou",
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "tasks": [
                score.to_dict()
                | {
                    "paper_id": task.paper_id,
                    "formulation_type": task.metadata.get("formulation_type"),
                    "category": task.metadata.get("category"),
                    "instance_index": task.instance_index,
                }
                for task, score in zip(tasks, scores, strict=True)
            ],
        }
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FrontierOR external-platform benchmark pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--large-root", type=Path, required=True)
        command.add_argument("--only", action="append", default=[], help="paper_id or emitted task_id")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--submissions", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("import", help="normalize natural-language platform replies")
    command.add_argument("--input", type=Path, required=True, help="FAMOU export root containing manifest.json")
    command.add_argument("--replies", type=Path, required=True, help="per-task reply directories")
    command.add_argument(
        "--submissions",
        type=Path,
        help="output root; defaults to --replies and writes solution.json in each task directory",
    )
    command.add_argument(
        "--allow-python-literals",
        action="store_true",
        help="statically read literal SOLUTION/result assignments without executing reply code",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import":
        result = import_replies(
            args.input,
            args.replies,
            args.submissions,
            allow_python_literals=args.allow_python_literals,
        )
        print(
            "FAMOU replies: "
            f"imported={len(result['imported'])}, existing={len(result['existing'])}, "
            f"unanswered={len(result['unanswered'])}, failed={len(result['failed'])}"
        )
        return 0
    only = set(args.only) or None
    if args.command == "export":
        manifest = export_suite(args.large_root, args.output, only=only)
        print(f"Exported {manifest['case_count']} {DEFAULT_SUITE} tasks to {args.output}")
    else:
        report = evaluate_submissions(args.large_root, args.submissions, args.output, only=only)
        print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
