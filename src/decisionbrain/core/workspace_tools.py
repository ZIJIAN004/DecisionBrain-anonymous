"""Workspace-scoped tools for StageAgent tool-calling loops."""

from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import json
from contextlib import suppress
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

from .models import AgentStage
from .package_policy import (
    DEFAULT_PACKAGE_POLICY,
    PackagePolicy,
    blocked_modules_for,
    render_import_guard,
    render_module_stub,
    solving_units,
)
from .process_tree import ProcessTree
from .stage_agent import StageForcedCompletion, Tool
from .stage_output_models import (
    FeasibilityCheckReceipt,
    FeasibilityResult,
    RuntimeSolverTimeoutOutput,
    SolverExecutionReceipt,
    SolverResultOutput,
    SingleSolverResultOutput,
)
from .stage_outputs import (
    FEASIBILITY_HANDOFF_DIR,
    FEASIBILITY_REVIEW_HISTORY_FILE,
    PROBLEM_CONTRACT_CHECKER_FILE,
    PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
    PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
    FIXED_SOLUTION_CONTRACT_FILE,
    SOLVING_CODE_FILE,
    SOLVING_FEASIBILITY_RECEIPT_FILE,
    SOLVING_FEASIBILITY_RESULT_FILE,
    SOLVING_EXECUTION_RECEIPT_FILE,
    SOLVING_INPUT_BUILDER_FILE,
    SOLVING_INPUT_FILE,
    SOLVING_RESULT_FILE,
    SOLVING_RUNTIME_OUTCOME_FILE,
    SOLVING_SOLUTION_FILE,
    STAGE_OUTPUT_FILES,
    feasibility_handoff_dir,
)


SCRATCH_DIR_NAME = "scratch"


def file_sha256(path: Path) -> str | None:
    """Return the hex digest of a workspace file, or None when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None



def ensure_scratch_directory(root: Path) -> Path:
    """Create and return the run-scoped scratch directory."""
    scratch = root.resolve() / SCRATCH_DIR_NAME
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def clear_scratch_directory(root: Path) -> None:
    """Remove scratch contents while retaining the fixed directory itself."""
    scratch = ensure_scratch_directory(root)
    for child in tuple(scratch.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


TRUNCATION_NOTICE = "\n[truncated]\n"
SEARCH_MAX_FILES = 200
SEARCH_MAX_FILE_BYTES = 256 * 1024
SEARCH_DEFAULT_RESULTS = 20
SEARCH_MAX_RESULTS = 100
DIRECTORY_CHANGE_COMMAND = re.compile(
    r"(?:^|&&|\|\||[;|\n(]|\bthen\b|\bdo\b|\belse\b)"
    r"\s*(?:command\s+|builtin\s+)?"
    r"(?:cd|chdir|pushd|popd|set-location|push-location|pop-location)(?=\s|$)",
    re.IGNORECASE,
)


# ``python -E`` and ``-I`` ignore PYTHONPATH and defeat module/sitecustomize guards.
# Explicit PYTHONPATH replacement and env -i do the same, so reject them at command level.
IMPORT_GUARD_BYPASS_COMMAND = re.compile(
    r"(?:\bpython[0-9.]*(?:\.exe)?[\"']?\s+(?:-\w*[SEI])\b"
    r"|\bPYTHONPATH\s*="
    r"|\bPYTHONHOME\s*="
    r"|\benv\s+(?:-i\b|--ignore-environment\b|-u\s+PYTHONPATH\b))"
)


class WorkspaceToolError(ValueError):
    """A user-facing tool error that should be returned to the model."""


@dataclass(frozen=True)
class _LinuxSolverTrace:
    solver_pids: frozenset[int]
    active_solver_pids: frozenset[int]
    timeout_pids: frozenset[int]
    solver_killed_by: str | None
    gnu_timeout_observed: bool

    @property
    def observed(self) -> bool:
        return bool(self.solver_pids)


_STRACE_PID_PREFIX = re.compile(r"^(?:\[pid\s+)?(?P<pid>\d+)(?:\])?\s+")
_STRACE_EXEC = re.compile(r'^execve\("[^"]+",\s*(?P<argv>\[.*?\]),\s*')
_STRACE_EXIT = re.compile(r"\+\+\+ (?:exited with \d+|killed by (?P<signal>SIG\w+)) \+\+\+")
_STRACE_CHILD = re.compile(r"^(?:clone|clone3|fork|vfork)\(.*\)\s+=\s+(?P<child>\d+)$")
_STRACE_SIGNAL = re.compile(r"^--- (?P<signal>SIGTERM|SIGKILL) ")


def _python_script_from_argv(argv: Sequence[str]) -> str | None:
    if not argv:
        return None
    executable = Path(argv[0]).name.casefold()
    if not (executable.startswith("python") or executable == "py"):
        return None
    index = 1
    flags_without_values = {"-b", "-bb", "-e", "-i", "-o", "-oo", "-q", "-s", "-u", "-v", "-x"}
    while index < len(argv) and argv[index].startswith("-"):
        option = argv[index].casefold()
        if option in {"-c", "-m"}:
            return None
        if option in {"-w", "-x", "--check-hash-based-pycs"}:
            index += 2
            continue
        if option in flags_without_values or option.startswith("-x"):
            index += 1
            continue
        return None
    return argv[index] if index < len(argv) else None


def _parse_linux_solver_trace(trace_text: str, *, cwd: Path, solver: Path) -> _LinuxSolverTrace:
    solver_pids: set[int] = set()
    active_solver_pids: set[int] = set()
    timeout_pids: set[int] = set()
    killed_signals: dict[int, str] = {}
    received_timeout_signals: dict[int, str] = {}
    parents: dict[int, int] = {}
    for raw_line in trace_text.splitlines():
        prefix = _STRACE_PID_PREFIX.match(raw_line.strip())
        if prefix is None:
            continue
        pid = int(prefix.group("pid"))
        event = raw_line.strip()[prefix.end() :]
        exec_match = _STRACE_EXEC.match(event)
        if exec_match is not None:
            try:
                argv = ast.literal_eval(exec_match.group("argv"))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                continue
            if argv and Path(argv[0]).name == "timeout":
                timeout_pids.add(pid)
            script = _python_script_from_argv(argv)
            if script is None:
                continue
            target = Path(script)
            if not target.is_absolute():
                target = cwd / target
            if target.resolve(strict=False) == solver.resolve(strict=False):
                solver_pids.add(pid)
                active_solver_pids.add(pid)
            continue
        exit_match = _STRACE_EXIT.search(event)
        if exit_match is not None and pid in active_solver_pids:
            active_solver_pids.discard(pid)
            if exit_match.group("signal"):
                killed_signals[pid] = exit_match.group("signal")
            continue
        signal_match = _STRACE_SIGNAL.match(event)
        if signal_match is not None and pid in solver_pids:
            received_timeout_signals[pid] = signal_match.group("signal")
            continue
        child_match = _STRACE_CHILD.match(event)
        if child_match is not None:
            parents[int(child_match.group("child"))] = pid

    def has_timeout_ancestor(pid: int) -> bool:
        seen: set[int] = set()
        current = pid
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
            if current in timeout_pids:
                return True
        return False

    killed_by = next(
        (
            (killed_signals.get(pid) or received_timeout_signals.get(pid))
            for pid in solver_pids
            if pid in killed_signals or pid in received_timeout_signals
        ),
        None,
    )
    return _LinuxSolverTrace(
        solver_pids=frozenset(solver_pids),
        active_solver_pids=frozenset(active_solver_pids),
        timeout_pids=frozenset(timeout_pids),
        solver_killed_by=killed_by,
        gnu_timeout_observed=any(
            pid in received_timeout_signals and has_timeout_ancestor(pid)
            for pid in solver_pids
        ),
    )


@dataclass(frozen=True)
class WorkspaceAccessRule:
    path: str
    writer_stages: frozenset[AgentStage] | None = None
    reader_blacklist: frozenset[AgentStage] = frozenset()
    recursive: bool = False


class WorkspaceAccessPolicy:
    """Stage-aware read/write policy for protected workspace files."""

    def __init__(self, rules: tuple[WorkspaceAccessRule, ...] | None = None) -> None:
        self.rules = rules or DEFAULT_WORKSPACE_ACCESS_RULES

    def with_solving_feasibility_self_check(self) -> "WorkspaceAccessPolicy":
        """Restore branch-9 checker access for Solving when Reviewer is disabled."""

        adjusted: list[WorkspaceAccessRule] = []
        for rule in self.rules:
            if rule.path == PROBLEM_CONTRACT_CHECKER_FILE:
                adjusted.append(
                    replace(
                        rule,
                        reader_blacklist=rule.reader_blacklist - {AgentStage.SOLVING},
                    )
                )
            elif rule.path == SOLVING_FEASIBILITY_RESULT_FILE:
                adjusted.append(
                    replace(
                        rule,
                        writer_stages=frozenset(
                            {*rule.writer_stages, AgentStage.SOLVING}
                        ),
                        reader_blacklist=rule.reader_blacklist - {AgentStage.SOLVING},
                    )
                )
            else:
                adjusted.append(rule)
        return WorkspaceAccessPolicy(tuple(adjusted))

    def with_solving_outcome_submission(self) -> "WorkspaceAccessPolicy":
        """Require Solving to submit its final outcome through the dedicated tool."""

        adjusted: list[WorkspaceAccessRule] = []
        protected = {
            STAGE_OUTPUT_FILES[AgentStage.SOLVING],
            SOLVING_RESULT_FILE,
            SOLVING_SOLUTION_FILE,
        }
        for rule in self.rules:
            if rule.path in protected:
                adjusted.append(replace(rule, writer_stages=frozenset()))
            else:
                adjusted.append(rule)
        return WorkspaceAccessPolicy(tuple(adjusted))

    def ensure_can_read(self, path: Path, *, root: Path, stage: AgentStage | None) -> None:
        if stage is None:
            return
        rule = self._matching_rule(path, root=root)
        if rule is not None and stage in rule.reader_blacklist:
            raise WorkspaceToolError(
                f"stage {stage.value!r} is not allowed to read {self._display_path(path, root)}"
            )

    def ensure_can_write(self, path: Path, *, root: Path, stage: AgentStage | None) -> None:
        if stage is None:
            return
        rule = self._matching_rule(path, root=root)
        if rule is None or rule.writer_stages is None or stage in rule.writer_stages:
            return
        allowed = ", ".join(sorted(item.value for item in rule.writer_stages)) or "none"
        raise WorkspaceToolError(
            f"stage {stage.value!r} is not allowed to write "
            f"{self._display_path(path, root)}; allowed stages: {allowed}"
        )

    def can_read(self, path: Path, *, root: Path, stage: AgentStage | None) -> bool:
        try:
            self.ensure_can_read(path, root=root, stage=stage)
        except WorkspaceToolError:
            return False
        return True

    def unauthorized_write_targets(
        self, *, root: Path, stage: AgentStage | None
    ) -> tuple[Path, ...]:
        if stage is None:
            return ()
        targets: list[Path] = []
        for rule in self.rules:
            if rule.writer_stages is None or stage in rule.writer_stages or rule.recursive:
                continue
            targets.append((root / rule.path).resolve(strict=False))
        return tuple(targets)

    def unreadable_file_targets(
        self, *, root: Path, stage: AgentStage | None
    ) -> tuple[Path, ...]:
        """Return fixed files that must be absent from a stage shell view."""

        if stage is None:
            return ()
        targets: list[Path] = []
        for rule in self.rules:
            if rule.recursive or stage not in rule.reader_blacklist:
                continue
            target = (root / rule.path).resolve(strict=False)
            if target.is_file():
                targets.append(target)
        return tuple(targets)

    def _matching_rule(
        self,
        path: Path,
        *,
        root: Path,
    ) -> WorkspaceAccessRule | None:
        for rule in self.rules:
            base = (root / rule.path).resolve(strict=False)
            if rule.recursive:
                if path == base or base in path.parents:
                    return rule
            elif path == base:
                return rule
        return None

    @staticmethod
    def _display_path(path: Path, root: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(root).as_posix()
        except ValueError:
            return "<outside workspace>"


DEFAULT_WORKSPACE_ACCESS_RULES: tuple[WorkspaceAccessRule, ...] = (
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.INTAKE],
        writer_stages=frozenset({AgentStage.INTAKE}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT],
        writer_stages=frozenset({AgentStage.PROBLEM_CONTRACT}),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN],
        writer_stages=frozenset({AgentStage.ALGORITHM_DESIGN}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.GUROBI_FORMULATOR],
        writer_stages=frozenset({AgentStage.GUROBI_FORMULATOR}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.SOLVING],
        writer_stages=frozenset({AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        writer_stages=frozenset({AgentStage.FEASIBILITY_REVIEW}),
    ),
    WorkspaceAccessRule(
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
        writer_stages=frozenset({AgentStage.EXPLANATION}),
    ),
    WorkspaceAccessRule(
        PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
        writer_stages=frozenset({AgentStage.PROBLEM_CONTRACT}),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN}),
    ),
    WorkspaceAccessRule(
        PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
        writer_stages=frozenset({AgentStage.PROBLEM_CONTRACT}),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN}),
    ),
    # Benchmark injects this contract before Core starts; no stage may replace it.
    WorkspaceAccessRule(FIXED_SOLUTION_CONTRACT_FILE, writer_stages=frozenset()),
    WorkspaceAccessRule(
        PROBLEM_CONTRACT_CHECKER_FILE,
        writer_stages=frozenset({AgentStage.PROBLEM_CONTRACT}),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(SOLVING_INPUT_FILE, writer_stages=frozenset({AgentStage.SOLVING})),
    WorkspaceAccessRule(
        SOLVING_INPUT_BUILDER_FILE,
        writer_stages=frozenset({AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(SOLVING_CODE_FILE, writer_stages=frozenset({AgentStage.SOLVING})),
    # A candidate may only be written by the Runtime-launched solver process.
    WorkspaceAccessRule(SOLVING_SOLUTION_FILE, writer_stages=frozenset()),
    WorkspaceAccessRule(SOLVING_RESULT_FILE, writer_stages=frozenset({AgentStage.SOLVING})),
    WorkspaceAccessRule(SOLVING_RUNTIME_OUTCOME_FILE, writer_stages=frozenset()),
    WorkspaceAccessRule(
        SOLVING_EXECUTION_RECEIPT_FILE,
        writer_stages=frozenset(),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN}),
    ),
    WorkspaceAccessRule(
        # No stage may write the checker result directly: it is produced only by
        # run_feasibility_checker, which spawns the audited checker itself.  The
        # branch-9 self-check view re-adds SOLVING (see
        # with_solving_feasibility_self_check) because that path has no reviewer.
        SOLVING_FEASIBILITY_RESULT_FILE,
        writer_stages=frozenset(),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(
        # Runtime-owned execution receipt; never writable by a stage.
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        writer_stages=frozenset(),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(
        # Keep this non-recursive and file-specific so responsibility-stage shell calls isolate it.
        FEASIBILITY_REVIEW_HISTORY_FILE,
        writer_stages=frozenset(),
        reader_blacklist=frozenset({AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}),
    ),
    WorkspaceAccessRule(
        f"{FEASIBILITY_HANDOFF_DIR}/algorithm_design",
        writer_stages=frozenset(),
        reader_blacklist=frozenset(
            {
                AgentStage.INTAKE,
                AgentStage.PROBLEM_CONTRACT,
            }
        ),
        recursive=True,
    ),
    WorkspaceAccessRule(
        f"{FEASIBILITY_HANDOFF_DIR}/solving",
        writer_stages=frozenset(),
        reader_blacklist=frozenset(
            {
                AgentStage.INTAKE,
                AgentStage.PROBLEM_CONTRACT,
                AgentStage.ALGORITHM_DESIGN,
            }
        ),
        recursive=True,
    ),
    WorkspaceAccessRule(
        f"{FEASIBILITY_HANDOFF_DIR}/gurobi_formulator",
        writer_stages=frozenset(),
        reader_blacklist=frozenset(
            {AgentStage.INTAKE, AgentStage.PROBLEM_CONTRACT, AgentStage.ALGORITHM_DESIGN}
        ),
        recursive=True,
    ),
    WorkspaceAccessRule("problem.md"),
    WorkspaceAccessRule("data", recursive=True),
)

# -- Stage output files to delete for each resume stage -----------------------
_RESUME_CLEANUP_GLOBS: dict[AgentStage, tuple[str, ...]] = {
    AgentStage.INTAKE: (
        "stage_outputs/*.json",
        FEASIBILITY_REVIEW_HISTORY_FILE,
        PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
        PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
        PROBLEM_CONTRACT_CHECKER_FILE,
        SOLVING_INPUT_BUILDER_FILE,
        SOLVING_INPUT_FILE,
        SOLVING_CODE_FILE,
        SOLVING_SOLUTION_FILE,
        SOLVING_RESULT_FILE,
        SOLVING_RUNTIME_OUTCOME_FILE,
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
    ),
    AgentStage.PROBLEM_CONTRACT: (
        STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT],
        FEASIBILITY_REVIEW_HISTORY_FILE,
        PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
        PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
        PROBLEM_CONTRACT_CHECKER_FILE,
        STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN],
        STAGE_OUTPUT_FILES[AgentStage.SOLVING],
        SOLVING_INPUT_BUILDER_FILE,
        SOLVING_INPUT_FILE,
        SOLVING_CODE_FILE,
        SOLVING_SOLUTION_FILE,
        SOLVING_RESULT_FILE,
        SOLVING_RUNTIME_OUTCOME_FILE,
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
    ),
    AgentStage.GUROBI_FORMULATOR: (
        STAGE_OUTPUT_FILES[AgentStage.GUROBI_FORMULATOR],
        STAGE_OUTPUT_FILES[AgentStage.SOLVING],
        SOLVING_INPUT_BUILDER_FILE,
        SOLVING_INPUT_FILE,
        SOLVING_CODE_FILE,
        SOLVING_SOLUTION_FILE,
        SOLVING_RESULT_FILE,
        SOLVING_RUNTIME_OUTCOME_FILE,
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
    ),
    AgentStage.ALGORITHM_DESIGN: (
        STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN],
        STAGE_OUTPUT_FILES[AgentStage.SOLVING],
        SOLVING_INPUT_BUILDER_FILE,
        SOLVING_INPUT_FILE,
        SOLVING_CODE_FILE,
        SOLVING_SOLUTION_FILE,
        SOLVING_RESULT_FILE,
        SOLVING_RUNTIME_OUTCOME_FILE,
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
    ),
    AgentStage.SOLVING: (
        STAGE_OUTPUT_FILES[AgentStage.SOLVING],
        SOLVING_INPUT_BUILDER_FILE,
        SOLVING_CODE_FILE,
        SOLVING_INPUT_FILE,
        SOLVING_SOLUTION_FILE,
        SOLVING_RESULT_FILE,
        SOLVING_RUNTIME_OUTCOME_FILE,
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
    ),
    AgentStage.FEASIBILITY_REVIEW: (
        SOLVING_FEASIBILITY_RESULT_FILE,
        SOLVING_FEASIBILITY_RECEIPT_FILE,
        SOLVING_EXECUTION_RECEIPT_FILE,
        STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW],
        STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],
    ),
    AgentStage.EXPLANATION: (STAGE_OUTPUT_FILES[AgentStage.EXPLANATION],),
}


_ATTEMPT_DIR_PATTERN = re.compile(r"^attempt(\d+)$")

REVIEW_HISTORY_VIOLATION_SAMPLE = 20


def _summarize_checker_findings(items: Any) -> tuple[list[Any], int, dict[str, int]]:
    """Keep a bounded sample plus the constraint profile of one checker finding list.

    A single check can carry thousands of findings, and the whole file is read,
    unlinked and rewritten around every responsible-stage shell call. The counts and
    the constraint profile are what separate a converging repair from a stuck one,
    so they survive in full while the findings themselves are sampled.
    """

    if not isinstance(items, (list, tuple)):
        return [], 0, {}
    profile: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("constraint")
        if isinstance(name, str) and name:
            profile[name] = profile.get(name, 0) + 1
    return list(items[:REVIEW_HISTORY_VIOLATION_SAMPLE]), len(items), profile


def _history_check(check: Any) -> Any:
    if not isinstance(check, dict):
        return check
    trimmed = dict(check)
    for field in ("violations", "warnings"):
        if field not in check:
            continue
        sample, total, profile = _summarize_checker_findings(check.get(field))
        trimmed[field] = sample
        trimmed[f"{field}_count"] = total
        trimmed[f"{field}_truncated"] = total > len(sample)
        if profile:
            trimmed[f"{field}_constraints"] = profile
    return trimmed


def write_feasibility_review_history(root: Path, audits: Sequence[dict[str, Any]]) -> str:
    """Project the private feasibility audits into the reviewer-only workspace file.

    Core state stays the source of truth and the file is rewritten from it every time,
    so a shell call that is killed while the file is isolated loses nothing permanently.
    An empty selection removes the file rather than leaving one behind: while a
    consecutive instance run withholds the earlier instance verdicts, a stale copy from
    a parent run would otherwise still expose them.
    """

    if not audits:
        (root / FEASIBILITY_REVIEW_HISTORY_FILE).unlink(missing_ok=True)
        return FEASIBILITY_REVIEW_HISTORY_FILE

    records = [
        (
            {**audit, "feasibility_check": _history_check(audit["feasibility_check"])}
            if "feasibility_check" in audit
            # Instance decisions do not execute a checker, so there is no feasibility check to prune.
            else dict(audit)
        )
        for audit in audits
    ]
    target = root / FEASIBILITY_REVIEW_HISTORY_FILE
    target.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FEASIBILITY_REVIEW_HISTORY_FILE


def _retire_superseded_handoffs(root: Path, responsibility: str, attempt_name: str) -> None:
    """Keep only the attempt being prepared and delete every other handoff directory.

    Only the newest attempt is ever referenced: the active directive points at it and
    the compact ledger carries no paths. Earlier rounds stay recoverable because
    Solving publishes each round's solver and solution as immutable artifacts, which
    live outside the workspace. Keeping the directories would instead copy a whole
    previous solution per rewind, and solutions grow with the instance.

    Retiring everything else rather than only lower attempt numbers also clears the
    rounds orphaned when a problem contract revision resets the attempt counter; those
    baselines belong to a contract that no longer holds.
    """

    base = root / FEASIBILITY_HANDOFF_DIR
    if not base.is_dir():
        return
    for target_dir in sorted(base.iterdir()):
        if not target_dir.is_dir():
            continue
        for attempt_dir in sorted(target_dir.iterdir()):
            if not attempt_dir.is_dir():
                continue
            if _ATTEMPT_DIR_PATTERN.match(attempt_dir.name) is None:
                continue
            if target_dir.name == responsibility and attempt_dir.name == attempt_name:
                continue
            shutil.rmtree(attempt_dir, ignore_errors=True)


def prepare_feasibility_handoff(
    root: Path,
    attempt_index: int,
    directive: dict[str, Any],
) -> tuple[str, ...]:
    """Copy the checker-independent baselines a rejected candidate may be repaired from.

    The repair instruction itself stays in the stage prompt, so the directory holds
    only the large previous artifacts. Returns the baselines actually written, which
    is not a fixed list: a rejected solve with no candidate has no previous solution.
    Superseded attempts are retired first so a run keeps one baseline set, not one
    per rewind.
    """

    responsibility = str(directive.get("responsibility") or "")
    expected_handoff = feasibility_handoff_dir(responsibility, attempt_index)
    if directive.get("handoff_dir") != expected_handoff:
        raise WorkspaceToolError(
            "feasibility directive handoff_dir does not match responsibility"
        )
    _retire_superseded_handoffs(root, responsibility, f"attempt{attempt_index}")
    handoff_dir = root / expected_handoff
    handoff_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []

    handoff_files = (
        (STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN], "previous_algorithm_design.json"),
        (SOLVING_SOLUTION_FILE, "previous_solution.json"),
    )
    if responsibility == AgentStage.GUROBI_FORMULATOR.value:
        handoff_files = (
            (STAGE_OUTPUT_FILES[AgentStage.GUROBI_FORMULATOR], "previous_gurobi_formulation.json"),
            (SOLVING_SOLUTION_FILE, "previous_solution.json"),
        )
    if responsibility == AgentStage.SOLVING.value:
        handoff_files = (*handoff_files, (SOLVING_CODE_FILE, "previous_solver.py"))
    for source_relative, destination_name in handoff_files:
        source = root / source_relative
        if not source.is_file():
            continue
        destination = handoff_dir / destination_name
        destination.write_bytes(source.read_bytes())
        prepared.append(f"{expected_handoff}/{destination_name}")
    return tuple(prepared)


def reset_workspace_for_stage(root: Path, stage: AgentStage) -> None:
    """Delete target-stage and downstream artifacts from the workspace.

    Called during resume to prevent StageAgent from finding stale outputs from
    the parent run. This operates only on the derived Run's workspace copy.
    """
    globs = _RESUME_CLEANUP_GLOBS.get(stage, ())
    for pattern in globs:
        if "*" in pattern:
            for matched in sorted(root.glob(pattern)):
                if matched.is_file():
                    matched.unlink()
        else:
            target = root / pattern
            if target.is_file():
                target.unlink()


class WorkspaceToolset:
    """Lightweight file and shell tools restricted to a workspace root.

    The shell tool is intentionally simple: it runs with the process user's normal
    permissions, but forces the command's cwd to a path inside the workspace.
    """

    def __init__(
        self,
        root: Path,
        *,
        timeout_s: float,
        solver_cpu_limit: int = 1,
        max_output_chars: int,
        max_list_entries: int,
        default_list_entries: int,
        max_read_bytes: int,
        access_policy: WorkspaceAccessPolicy | None = None,
        process_timing_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        problem_contract_enabled: bool = True,
        components_enabled: bool = True,
        package_policy: PackagePolicy = DEFAULT_PACKAGE_POLICY,
    ) -> None:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceToolError(f"workspace root is not a directory: {resolved}")
        self.root = resolved
        if timeout_s <= 0:
            raise WorkspaceToolError("timeout_s must be greater than 0")
        if solver_cpu_limit <= 0:
            raise WorkspaceToolError("solver_cpu_limit must be greater than 0")
        if max_output_chars <= 0:
            raise WorkspaceToolError("max_output_chars must be greater than 0")
        if max_list_entries <= 0:
            raise WorkspaceToolError("max_list_entries must be greater than 0")
        if default_list_entries <= 0 or default_list_entries > max_list_entries:
            raise WorkspaceToolError(
                "default_list_entries must be greater than 0 and no larger than max_list_entries"
            )
        if max_read_bytes <= 0:
            raise WorkspaceToolError("max_read_bytes must be greater than 0")
        self.timeout_s = float(timeout_s)
        self.solver_cpu_limit = int(solver_cpu_limit)
        self.max_output_chars = int(max_output_chars)
        self.max_list_entries = int(max_list_entries)
        self.default_list_entries = int(default_list_entries)
        self.max_read_bytes = int(max_read_bytes)
        self.access_policy = access_policy or WorkspaceAccessPolicy()
        self.process_timing_handler = process_timing_handler
        self.problem_contract_enabled = bool(problem_contract_enabled)
        self.components_enabled = bool(components_enabled)
        self.package_policy = package_policy
        # Enforce package restrictions inside solver subprocesses. Arm A is fixed at construction;
        # arm B narrows after design. Shared state keeps copied tool views synchronized.
        self._guard_state: dict[str, Any] = {"units": None, "dir": None}
        self._rebind_import_guard(solving_units(package_policy, None))
        self.runtime_solver_timeout_routing_enabled = False
        self._linux_strace_probe_complete = False
        self._linux_strace_path: str | None = None

    def bind_solving_design(self, design: Any) -> None:
        """Restrict solving subprocess units after the algorithm design is known.

        Arm A's unit set is independent of the design, making this an idempotent no-op.
        """

        self._rebind_import_guard(solving_units(self.package_policy, design))

    def _rebind_import_guard(self, units: frozenset[str] | None) -> None:
        state = self._guard_state
        if state["units"] == units and (units is None or state["dir"]):
            return
        state["units"] = units
        state["dir"] = self._install_import_guard(units)

    @staticmethod
    def _install_import_guard(units: frozenset[str] | None) -> str | None:
        """Write the import guard to a temporary directory and return it.

        The directory contains ``sitecustomize.py`` plus shadow modules for every
        blocked import. Shadow modules still work under ``python -S`` because
        ``PYTHONPATH`` precedes site-packages. The shell tool rejects ``-E`` and ``-I``.
        """

        source = render_import_guard(units)
        if source is None:
            return None
        directory = Path(tempfile.mkdtemp(prefix="decisionbrain-package-guard-"))
        (directory / "sitecustomize.py").write_text(source, encoding="utf-8")
        for module in blocked_modules_for(units or frozenset()):
            (directory / f"{module}.py").write_text(
                render_module_stub(module), encoding="utf-8"
            )
        return str(directory)

    @property
    def _import_guard_dir(self) -> str | None:
        return self._guard_state["dir"]

    def _guarded_env(self) -> dict[str, str] | None:
        """Return a guarded subprocess environment, or ``None`` to inherit directly."""

        if self._import_guard_dir is None:
            return None
        env = os.environ.copy()
        self._apply_import_guard(env)
        return env

    def _apply_import_guard(self, env: dict[str, str]) -> None:
        guard = self._import_guard_dir
        if guard is None:
            return
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{guard}{os.pathsep}{existing}" if existing else guard
        # PYTHONHOME rewrites sys.path construction and would bypass the import guard.
        env.pop("PYTHONHOME", None)

    def _reject_import_guard_bypass(self, command: str) -> None:
        """Reject command forms that would bypass the import guard."""

        if self._import_guard_dir is None:
            return
        match = IMPORT_GUARD_BYPASS_COMMAND.search(command)
        if match is None:
            return
        raise WorkspaceToolError(
            f"命令包含会绕过本次运行算法包限制的形式（{match.group(0).strip()}）。"
            "请不要使用 python -S / -E / -I，也不要覆写 PYTHONPATH 或用 env -i "
            "清空环境；直接用普通的 python 调用即可。"
        )

    def _solution_schema_path(self) -> str:
        return (
            PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE
            if self.problem_contract_enabled
            else FIXED_SOLUTION_CONTRACT_FILE
        )

    def tools(
        self,
        *,
        stage: AgentStage | str | None = None,
        solving_feasibility_self_check: bool = False,
        solving_outcome_submission: bool = False,
        _include_solving_submission: bool = False,
    ) -> tuple[Tool, ...]:
        """Return StageAgent tools in stable registration order."""
        if solving_feasibility_self_check:
            if self._coerce_stage(stage) is not AgentStage.SOLVING:
                raise WorkspaceToolError(
                    "solving_feasibility_self_check is available only to the solving stage"
                )
            view = copy.copy(self)
            view.access_policy = self.access_policy.with_solving_feasibility_self_check()
            return view.tools(stage=stage)
        if solving_outcome_submission:
            if self._coerce_stage(stage) is not AgentStage.SOLVING:
                raise WorkspaceToolError(
                    "solving_outcome_submission is available only to the solving stage"
                )
            view = copy.copy(self)
            view.access_policy = self.access_policy.with_solving_outcome_submission()
            view.runtime_solver_timeout_routing_enabled = True
            return view.tools(stage=stage, _include_solving_submission=True)
        stage_value = self._coerce_stage(stage)
        tools: tuple[Tool, ...] = (
            Tool(
                name="shell",
                description=(
                    "Run a shell command from the workspace root by default and return exit "
                    "code, wall time, and truncated output. Use workspace-relative paths and "
                    "workdir; do not cd to guessed absolute workspace paths. Directory-changing "
                    "commands such as cd are rejected; use the relative workdir parameter for an "
                    "existing subdirectory."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run."},
                        "workdir": {
                            "type": "string",
                            "description": (
                                "Optional working directory relative to the workspace root. "
                                "Defaults to .; absolute paths are not allowed."
                            ),
                        },
                        "timeout_ms": {
                            "type": "number",
                            "description": "Maximum runtime in milliseconds; capped by policy.",
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.shell, stage_value),
            ),
            Tool(
                name="read_file",
                description=(
                    "Read a UTF-8 text file from the current working directory. Full-file reads "
                    f"larger than {self.max_read_bytes} bytes fail; do not use them for large files. "
                    f"Every response is capped at {self.max_output_chars} characters and max_chars "
                    "can only lower that cap. Past the cap the result is the head of the range, the "
                    f"marker {TRUNCATION_NOTICE.strip()}, then the tail: the middle is dropped and "
                    "no error is raised, so treat that marker as proof the range was not fully read. "
                    "For a multi-line large file, request start_line and line_count, and retry with "
                    "a smaller line_count while the marker keeps appearing. For a large or "
                    "single-line JSON file, use shell to extract only the relevant fields."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."},
                        "max_chars": {
                            "type": "number",
                            "description": (
                                "Maximum characters to return; only lowers the "
                                f"{self.max_output_chars} character cap, never raises it."
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "First line to return, using 1-based line numbers.",
                        },
                        "line_count": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Number of lines to return from start_line.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.read_file, stage_value),
            ),
            Tool(
                name="search_file",
                description=(
                    "Read-only recursive literal text search in UTF-8 workspace files. Returns "
                    "matching path, 1-based line number, and line text. Allowed arguments are only "
                    "path (optional workspace-relative file/directory; defaults to .), query "
                    "(required non-empty text), and max_results (optional integer, default 20, "
                    "maximum 100). Search is case-sensitive and does not support regex, glob, "
                    "case_sensitive, or limit parameters. It skips binary files, files larger than "
                    f"{SEARCH_MAX_FILE_BYTES} bytes, and files blocked for this stage; it scans at most "
                    f"{SEARCH_MAX_FILES} files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional workspace-relative file or directory; defaults to .",
                        },
                        "query": {
                            "type": "string",
                            "description": "Non-empty literal text to find, matched case-sensitively.",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum matching lines to return; capped by policy.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.search_file, stage_value),
            ),
            Tool(
                name="write_file",
                description="Write UTF-8 text to a file in the current working directory, creating parents as needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."},
                        "content": {"type": "string", "description": "Text content to write."},
                        "append": {
                            "type": "boolean",
                            "description": "Append instead of overwriting. Defaults to false.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.write_file, stage_value),
            ),
            Tool(
                name="list_files",
                description=(
                    "List files and directories inside the current working directory. "
                    "Each file is listed with its size, and a file too large for read_file "
                    "is marked as such."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative directory path. Defaults to .",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List recursively. Defaults to false.",
                        },
                        "max_entries": {
                            "type": "number",
                            "description": "Maximum entries to return; capped by policy.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._bind(self.list_files, stage_value),
            ),
            Tool(
                name="replace_in_file",
                description=(
                    "Find an exact string in a text file and replace it. "
                    "old_string must match exactly once (whitespace/indent sensitive). "
                    "Use this instead of sed or shell text editing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to current dir.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to find (include surrounding context to make it unique).",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text. Use empty string to delete the match.",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.replace_in_file, stage_value),
            ),
            Tool(
                name="delete_file",
                description="Delete a file from the current working directory. This is permanent.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to current dir.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self._bind(self.delete_file, stage_value),
            ),
        )
        if stage_value is AgentStage.SOLVING:
            tools = (*tools, self._solver_execution_tool(stage_value))
        if stage_value is AgentStage.SOLVING and _include_solving_submission:
            tools = (*tools, self._solving_submission_tool(stage_value))
        if stage_value is AgentStage.FEASIBILITY_REVIEW:
            tools = (*tools, self._feasibility_checker_tool(stage_value))
        return tools


    def _feasibility_checker_tool(self, stage: AgentStage) -> Tool:
        return Tool(
            name="run_feasibility_checker",
            description=(
                "Run the audited feasibility_checker.py against the current input.json and "
                "solution.json and record a Runtime-owned execution receipt. Pass only the CLI "
                "arguments the checker itself defines; the program, the interpreter and the "
                "working directory are fixed by Runtime and no shell is involved, so redirection "
                "and command chaining are not available. This is the only way to produce "
                "feasibility_result.json: it cannot be written with write_file or shell. A "
                "verdict may only rely on a run whose receipt still matches the current checker, "
                "input, and solution."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Arguments passed verbatim to feasibility_checker.py, for example "
                            "[\"--input\", \"input.json\", \"--solution\", \"solution.json\", "
                            "\"--output\", \"feasibility_result.json\"]."
                        ),
                    },
                    "timeout_ms": {
                        "type": "number",
                        "description": "Maximum runtime in milliseconds; capped by policy.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=self._bind(self.run_feasibility_checker, stage),
        )

    async def run_feasibility_checker(
        self,
        args: Sequence[Any] | None = None,
        timeout_ms: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        """Execute the audited checker directly and bind its result to a receipt."""

        try:
            if stage is not AgentStage.FEASIBILITY_REVIEW:
                raise WorkspaceToolError(
                    "run_feasibility_checker is available only to feasibility_review"
                )
            argv = self._checker_arguments(args)
            checker = self._resolve_existing(PROBLEM_CONTRACT_CHECKER_FILE, expected="file")
            if not checker.read_text(encoding="utf-8", errors="replace").strip():
                raise WorkspaceToolError(f"{PROBLEM_CONTRACT_CHECKER_FILE} is empty")
            effective_timeout = self._effective_timeout(timeout_ms)
            # Drop earlier evidence before starting a new run. The result must be
            # recreated by this checker invocation; otherwise a later receipt
            # could accidentally endorse a stale result from a prior attempt.
            (self.root / SOLVING_FEASIBILITY_RECEIPT_FILE).unlink(missing_ok=True)
            (self.root / SOLVING_FEASIBILITY_RESULT_FILE).unlink(missing_ok=True)
            before = self._protected_snapshot(stage)
            # The checker is expected to write exactly this file.
            before.pop((self.root / SOLVING_FEASIBILITY_RESULT_FILE).resolve(strict=False), None)
            hidden = self._hide_unreadable_files(stage)
        except WorkspaceToolError as exc:
            return self._error("run_feasibility_checker", exc)
        except OSError as exc:
            return self._error("run_feasibility_checker", WorkspaceToolError(str(exc)))

        start = time.monotonic()
        timed_out = False
        output = b""
        try:
            process_tree = await ProcessTree.create_exec(
                sys.executable,
                PROBLEM_CONTRACT_CHECKER_FILE,
                *argv,
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Do not guard checker imports; the checker is review evidence, not an ablated solver.
            )
        except (WorkspaceToolError, OSError) as exc:
            self._restore_hidden_files(hidden)
            self._restore_unauthorized_changes(before)
            return self._error(
                "run_feasibility_checker",
                WorkspaceToolError(f"could not start the checker: {exc}"),
            )

        process = process_tree.process
        communicate_task = asyncio.create_task(process.communicate())
        try:
            output, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            output = await self._terminate_process_tree(process_tree, communicate_task)
        except BaseException:
            await self._terminate_process_tree(process_tree, communicate_task)
            self._restore_hidden_files(hidden)
            self._restore_unauthorized_changes(before)
            raise
        finally:
            process_tree.close()
            exit_code = process.returncode if process.returncode is not None else -1

        duration = time.monotonic() - start
        text = output.decode("utf-8", errors="replace")
        if timed_out:
            text = f"checker timed out after {int(effective_timeout * 1000)} milliseconds\n" + text
        violations = [
            *self._restore_hidden_files(hidden),
            *self._restore_unauthorized_changes(before),
        ]
        if violations:
            names = ", ".join(dict.fromkeys(violations))
            return self._error(
                "run_feasibility_checker",
                WorkspaceToolError(
                    "the checker modified protected files and the changes were restored: "
                    f"{names}. Only {SOLVING_FEASIBILITY_RESULT_FILE} may be written."
                ),
            )

        receipt_note = self._publish_feasibility_receipt(
            argv=argv,
            exit_code=exit_code,
            elapsed_seconds=duration,
            timed_out=timed_out,
        )
        text, _truncated = self._truncate(text, self.max_output_chars)
        return "\n".join(
            [
                f"Exit code: {exit_code}",
                f"Wall time: {duration:.1f} seconds",
                f"Checker receipt: {receipt_note}",
                "Output:",
                text,
            ]
        )

    @staticmethod
    def _checker_arguments(args: Sequence[Any] | None) -> tuple[str, ...]:
        if args is None:
            return ()
        if isinstance(args, str) or not isinstance(args, Sequence):
            raise WorkspaceToolError("args must be an array of strings")
        argv: list[str] = []
        for item in args:
            if not isinstance(item, str):
                raise WorkspaceToolError("args must be an array of strings")
            if "\0" in item:
                raise WorkspaceToolError("args must not contain a NUL byte")
            argv.append(item)
        return tuple(argv)

    def _publish_feasibility_receipt(
        self,
        *,
        argv: tuple[str, ...],
        exit_code: int,
        elapsed_seconds: float,
        timed_out: bool,
    ) -> str:
        """Write the execution receipt, or explain why this run cannot be relied on."""

        if timed_out:
            return "not written: the checker was terminated by its time limit"
        # A non-zero exit is not a failure here: checkers routinely signal an
        # infeasible candidate that way. What matters is that this process ran to
        # completion and left a contract-valid result behind.
        try:
            result_data = self._read_workspace_json_object(SOLVING_FEASIBILITY_RESULT_FILE)
        except WorkspaceToolError as exc:
            return f"not written: {exc}"
        try:
            FeasibilityResult.model_validate(result_data)
        except ValidationError as exc:
            return (
                f"not written: {SOLVING_FEASIBILITY_RESULT_FILE} does not match its contract:\n"
                + self._format_validation_errors(exc.errors())
            )
        digests: dict[str, str] = {}
        for field, relative_path in (
            ("checker_sha256", PROBLEM_CONTRACT_CHECKER_FILE),
            ("input_sha256", SOLVING_INPUT_FILE),
            ("solution_sha256", SOLVING_SOLUTION_FILE),
            ("result_sha256", SOLVING_FEASIBILITY_RESULT_FILE),
        ):
            digest = file_sha256(self.root / relative_path)
            if digest is None:
                return f"not written: could not read {relative_path}"
            digests[field] = digest
        receipt = FeasibilityCheckReceipt.model_validate(
            {
                "schema_version": "1.0",
                "outcome_source": "runtime_checker_execution",
                "checker_file": PROBLEM_CONTRACT_CHECKER_FILE,
                "result_file": SOLVING_FEASIBILITY_RESULT_FILE,
                **digests,
                "argv": list(argv),
                "exit_code": exit_code,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        self._write_json_atomically(
            SOLVING_FEASIBILITY_RECEIPT_FILE,
            receipt.model_dump(mode="json"),
        )
        return (
            f"written to {SOLVING_FEASIBILITY_RECEIPT_FILE}; it binds this result to the "
            "current checker, input, and solution"
        )

    @staticmethod
    def _coerce_stage(stage: AgentStage | str | None) -> AgentStage | None:
        if stage is None or isinstance(stage, AgentStage):
            return stage
        return AgentStage(str(stage))

    @staticmethod
    def _bind(handler, stage: AgentStage | None):
        async def bound(**kwargs: Any) -> str:
            return await handler(**kwargs, stage=stage)

        return bound

    def _solver_execution_tool(self, stage: AgentStage) -> Tool:
        return Tool(
            name="run_solver",
            description=(
                "Run the current solver.py with the fixed Runtime interpreter and workspace. "
                "The solver process is the only permitted producer of solution.json. Runtime "
                "clears stale solution and solver receipts before execution, then records a "
                "receipt even when no candidate is produced. The result reports whether the "
                "candidate was bound, and lists the schema violations when it was not, so a "
                "malformed solution is visible here rather than at submission time. Pass only "
                "solver CLI arguments; do not use shell or write_file to create solution.json."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments passed verbatim to solver.py.",
                    },
                    "timeout_ms": {
                        "type": "number",
                        "description": "Maximum runtime in milliseconds; capped by policy.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=self._bind(self.run_solver, stage),
        )

    async def run_solver(
        self,
        args: Sequence[Any] | None = None,
        timeout_ms: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        """Execute solver.py and publish a Runtime-owned execution receipt."""

        try:
            if stage is not AgentStage.SOLVING:
                raise WorkspaceToolError("run_solver is available only to solving")
            argv = self._checker_arguments(args)
            solver = self._resolve_existing(SOLVING_CODE_FILE, expected="file")
            if not solver.read_text(encoding="utf-8", errors="replace").strip():
                raise WorkspaceToolError(f"{SOLVING_CODE_FILE} is empty")
            self._read_workspace_json_object(SOLVING_INPUT_FILE)
            effective_timeout = self._effective_timeout(timeout_ms)
            solution_before = self._file_fingerprint(self.root / SOLVING_SOLUTION_FILE)
            (self.root / SOLVING_SOLUTION_FILE).unlink(missing_ok=True)
            (self.root / SOLVING_EXECUTION_RECEIPT_FILE).unlink(missing_ok=True)
            before = self._protected_snapshot(stage)
            # The solver is the sole producer of its candidate file.
            before.pop((self.root / SOLVING_SOLUTION_FILE).resolve(strict=False), None)
            hidden = self._hide_unreadable_files(stage)
        except WorkspaceToolError as exc:
            return self._error("run_solver", exc)
        except OSError as exc:
            return self._error("run_solver", WorkspaceToolError(str(exc)))

        start = time.monotonic()
        timed_out = False
        output = b""
        try:
            program, launch_args = self._solver_launch_command(argv)
            solver_env = os.environ.copy()
            thread_limit = str(self.solver_cpu_limit)
            solver_env.update(
                {
                    "OMP_NUM_THREADS": thread_limit,
                    "MKL_NUM_THREADS": thread_limit,
                    "OPENBLAS_NUM_THREADS": thread_limit,
                    "NUMEXPR_NUM_THREADS": thread_limit,
                    "VECLIB_MAXIMUM_THREADS": thread_limit,
                    "BLIS_NUM_THREADS": thread_limit,
                }
            )
            self._apply_import_guard(solver_env)
            process_tree = await ProcessTree.create_exec(
                program,
                *launch_args,
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=solver_env,
            )
        except (WorkspaceToolError, OSError) as exc:
            self._restore_hidden_files(hidden)
            self._restore_unauthorized_changes(before)
            return self._error("run_solver", WorkspaceToolError(f"could not start solver: {exc}"))

        process = process_tree.process
        communicate_task = asyncio.create_task(process.communicate())
        try:
            output, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            output = await self._terminate_process_tree(process_tree, communicate_task)
        except BaseException:
            await self._terminate_process_tree(process_tree, communicate_task)
            self._restore_hidden_files(hidden)
            self._restore_unauthorized_changes(before)
            raise
        finally:
            process_tree.close()
            exit_code = process.returncode if process.returncode is not None else -1

        duration = time.monotonic() - start
        violations = [
            *self._restore_hidden_files(hidden),
            *self._restore_unauthorized_changes(before),
        ]
        if violations:
            names = ", ".join(dict.fromkeys(violations))
            return self._error(
                "run_solver",
                WorkspaceToolError(
                    "the solver modified protected files and the changes were restored: "
                    f"{names}. Only {SOLVING_SOLUTION_FILE} may be written."
                ),
            )

        text = output.decode("utf-8", errors="replace")
        if timed_out:
            # Keep the existing runtime timeout outcome as the stage transition
            # fact, while also recording the solver execution receipt.
            receipt_note = self._publish_solver_receipt(
                argv=argv,
                exit_code=exit_code,
                elapsed_seconds=duration,
                timed_out=True,
            )
            self._publish_runtime_solver_timeout(
                timeout_source="runtime_watchdog",
                timeout_seconds=effective_timeout,
                elapsed_seconds=duration,
                exit_code=exit_code,
                termination_signal="SIGTERM",
                stdout=text,
                solution_before=solution_before,
            )
            if self.runtime_solver_timeout_routing_enabled:
                # Same terminal rule as the shell path: an externally terminated
                # primary solver ends Solving here. Returning the timeout as an
                # ordinary tool result would leave the model in the loop, and it
                # would then try to submit an outcome the timeout contract cannot
                # satisfy.
                raise StageForcedCompletion(
                    "Primary solver was externally terminated by its time limit; "
                    "Solving ended and the runtime outcome was routed to feasibility review."
                )
            text, _truncated = self._truncate(text, self.max_output_chars)
            return "\n".join(
                [
                    f"Exit code: {exit_code}",
                    f"Wall time: {duration:.1f} seconds",
                    "Solver timed out; Runtime published runtime_solver_outcome.json.",
                    f"Solver receipt: {receipt_note}",
                    "Output:",
                    text,
                ]
            )

        receipt_note = self._publish_solver_receipt(
            argv=argv,
            exit_code=exit_code,
            elapsed_seconds=duration,
            timed_out=timed_out,
        )
        text, _truncated = self._truncate(text, self.max_output_chars)
        return "\n".join(
            [
                f"Exit code: {exit_code}",
                f"Wall time: {duration:.1f} seconds",
                f"Solver receipt: {receipt_note}",
                "Output:",
                text,
            ]
        )

    def _solver_launch_command(
        self,
        argv: Sequence[str],
        *,
        platform_name: str | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        solver_args = (SOLVING_CODE_FILE, *argv)
        if (platform_name or os.name) == "nt":
            return sys.executable, solver_args
        systemd_run = shutil.which("systemd-run")
        if systemd_run is None:
            raise WorkspaceToolError(
                "systemd-run is required to enforce the Linux solver CPU quota"
            )
        quota = self.solver_cpu_limit * 100
        args: list[str] = [
            "--user",
            "--scope",
            "--quiet",
            f"--property=CPUQuota={quota}%",
        ]
        # Make the licensed Gurobi installation explicit for Linux solver
        # children.  The path remains configurable for deployments, while the
        # read-only bind prevents a solver from replacing the license file.
        license_file = os.environ.get("GRB_LICENSE_FILE")
        if license_file:
            license_path = Path(license_file).expanduser().resolve()
        else:
            license_path = None
        if license_path is not None and license_path.is_file():
            args.extend(
                [
                    f"--property=BindReadOnlyPaths={license_path}:{license_path}",
                    f"--setenv=GRB_LICENSE_FILE={license_path}",
                ]
            )
        args.extend(["--", sys.executable, *solver_args])
        return systemd_run, tuple(args)

    def _publish_solver_receipt(
        self,
        *,
        argv: tuple[str, ...],
        exit_code: int,
        elapsed_seconds: float,
        timed_out: bool,
    ) -> str:
        try:
            self._read_workspace_json_object(SOLVING_INPUT_FILE)
        except WorkspaceToolError as exc:
            return f"not written: {exc}"
        # An invalid partial candidate is not bound as a solution, but the
        # execution receipt still records that the solver ran.
        solution_exists, solution_note = self._inspect_solver_candidate()

        digests: dict[str, str] = {}
        for field, relative_path in (
            ("input_sha256", SOLVING_INPUT_FILE),
            ("solver_sha256", SOLVING_CODE_FILE),
        ):
            digest = file_sha256(self.root / relative_path)
            if digest is None:
                return f"not written: could not read {relative_path}"
            digests[field] = digest
        solution_sha256 = file_sha256(self.root / SOLVING_SOLUTION_FILE) if solution_exists else None
        receipt = SolverExecutionReceipt.model_validate(
            {
                "schema_version": "1.0",
                "outcome_source": "runtime_solver_execution",
                "solver_file": SOLVING_CODE_FILE,
                "input_file": SOLVING_INPUT_FILE,
                **digests,
                "solution_file": SOLVING_SOLUTION_FILE if solution_exists else None,
                "solution_sha256": solution_sha256,
                "argv": list(argv),
                "exit_code": exit_code,
                "elapsed_seconds": elapsed_seconds,
                "timed_out": timed_out,
            }
        )
        self._write_json_atomically(
            SOLVING_EXECUTION_RECEIPT_FILE,
            receipt.model_dump(mode="json"),
        )
        return f"written to {SOLVING_EXECUTION_RECEIPT_FILE}; {solution_note}"

    def _inspect_solver_candidate(self) -> tuple[bool, str]:
        """Decide whether the candidate can be bound, and say why when it cannot.

        run_feasibility_checker reports a contract-invalid result the moment it is
        produced. The solver must do the same: silently declining to bind a
        malformed candidate makes a broken run look clean here and surfaces only
        much later, inside submit_solving_outcome, by which point the model has
        already moved on.
        """

        if not (self.root / SOLVING_SOLUTION_FILE).is_file():
            return False, "no candidate was produced"
        try:
            solution = self._read_workspace_json_object(SOLVING_SOLUTION_FILE)
        except WorkspaceToolError as exc:
            return False, f"the candidate was not bound: {exc}"
        try:
            schema_path = self._solution_schema_path()
            schema = self._read_workspace_json_object(schema_path)
            Draft202012Validator.check_schema(schema)
        except (WorkspaceToolError, SchemaError) as exc:
            detail = exc.message if isinstance(exc, SchemaError) else str(exc)
            return False, (
                f"the candidate was not bound because "
                f"{schema_path} could not be used to validate it: "
                f"{detail}"
            )
        errors = sorted(
            Draft202012Validator(schema).iter_errors(solution),
            key=lambda error: list(error.absolute_path),
        )
        if not errors:
            return True, f"candidate bound to {SOLVING_SOLUTION_FILE}"
        return False, (
            f"{SOLVING_SOLUTION_FILE} does not match {schema_path} "
            "and was not bound as a candidate:\n"
            + self._format_validation_errors(
                [
                    {"loc": tuple(error.absolute_path), "msg": error.message}
                    for error in errors[:12]
                ]
            )
            + f"\nFix {SOLVING_CODE_FILE} so it writes a schema-valid solution, then run it "
            "again; do not edit the candidate by hand."
        )

    def _solving_submission_tool(self, stage: AgentStage) -> Tool:
        schema_path = self._solution_schema_path()
        return Tool(
            name="submit_solving_outcome",
            description=(
                "Atomically submit the normal Solving outcome. Write solver.py, input.json, and, "
                "when a candidate exists, solution.json first. Then provide the small structured "
                "solver_result object and a short message. The tool validates solver_result, the "
                "required workspace files, and the candidate solution against the current "
                f"schema before writing solver_result.json and stage_outputs/solving.json. "
                f"The candidate is checked against {schema_path}. "
                "Do not use write_file or shell to write those two "
                "final files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Short factual completion summary.",
                    },
                    "result": {
                        "type": "object",
                        "description": (
                            "Complete solver_result object. Its runtime contract is validated by "
                            "the tool; do not include solution contents here."
                        ),
                    },
                },
                "required": ["message", "result"],
                "additionalProperties": False,
            },
            handler=self._bind(self.submit_solving_outcome, stage),
        )

    async def submit_solving_outcome(
        self,
        message: str,
        result: dict[str, Any],
        stage: AgentStage | None = None,
    ) -> str:
        """Validate the Solver's metadata and candidate file before publishing both outputs."""

        try:
            if stage is not AgentStage.SOLVING:
                raise WorkspaceToolError("submit_solving_outcome is available only to solving")
            if not isinstance(message, str) or not message.strip():
                raise WorkspaceToolError("message must be a non-empty string")
            if not isinstance(result, dict):
                raise WorkspaceToolError("result must be a JSON object")
            try:
                model = (
                    SolverResultOutput
                    if self.components_enabled
                    else SingleSolverResultOutput
                )
                solver_result = model.model_validate(result)
            except ValidationError as exc:
                raise WorkspaceToolError(
                    "solver_result does not match its contract:\n"
                    + self._format_validation_errors(exc.errors())
                ) from exc

            self._validate_solving_submission_files(solver_result)
            result_data = solver_result.model_dump(mode="json", exclude_none=False)
            self._write_json_atomically(SOLVING_RESULT_FILE, result_data)
            self._write_json_atomically(
                STAGE_OUTPUT_FILES[AgentStage.SOLVING],
                {
                    "decision": "solved",
                    "message": message.strip(),
                    "result": {"solver_result_file": SOLVING_RESULT_FILE},
                },
            )
        except WorkspaceToolError as exc:
            return self._error("submit_solving_outcome", exc)
        return (
            "submitted solving outcome: solver_result.json and "
            "stage_outputs/solving.json passed validation"
        )

    def _validate_solving_submission_files(
        self, result: SolverResultOutput | SingleSolverResultOutput
    ) -> None:
        """Report every mismatch at once.

        Raising on the first problem made the model resubmit once per defect: a bad
        solver_result hid the schema violations, which hid the receipt mismatch. The
        stage-output gate already reports its issues as one list; this does the same.
        """

        problems: list[str] = []
        if not (self._read_workspace_file_text(SOLVING_CODE_FILE) or "").strip():
            problems.append(f"missing or empty required file: {SOLVING_CODE_FILE}")
        try:
            self._read_workspace_json_object(SOLVING_INPUT_FILE)
        except WorkspaceToolError as exc:
            problems.append(str(exc))
        if result.solution_file is not None:
            problems.extend(self._solution_schema_problems())
        problems.extend(self._solver_receipt_problems(result))
        if problems:
            raise WorkspaceToolError(
                "the submitted outcome does not match the workspace:\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )

    @staticmethod
    def _error_location(parts: Any) -> str:
        return ".".join(str(part) for part in parts) or "<root>"

    def _solution_schema_problems(self) -> list[str]:
        try:
            solution = self._read_workspace_json_object(SOLVING_SOLUTION_FILE)
        except WorkspaceToolError as exc:
            return [
                f"{exc}; solver_result declares a candidate, so run_solver must have produced "
                f"{SOLVING_SOLUTION_FILE}"
            ]
        try:
            schema_path = self._solution_schema_path()
            schema = self._read_workspace_json_object(schema_path)
            Draft202012Validator.check_schema(schema)
        except WorkspaceToolError as exc:
            return [str(exc)]
        except SchemaError as exc:
            return [
                f"{schema_path} is not a valid JSON Schema: {exc.message}"
            ]
        errors = sorted(
            Draft202012Validator(schema).iter_errors(solution),
            key=lambda error: list(error.absolute_path),
        )
        return [
            f"{SOLVING_SOLUTION_FILE} does not match {schema_path} at "
            f"{self._error_location(error.absolute_path)}: {error.message}"
            for error in errors[:12]
        ]

    def _solver_receipt_problems(
        self, result: SolverResultOutput | SingleSolverResultOutput
    ) -> list[str]:
        if not (self.root / SOLVING_EXECUTION_RECEIPT_FILE).is_file():
            # The bare "path does not exist" this used to produce sent the model
            # looking for a file it is never allowed to create.
            return [
                f"no solver execution receipt: call run_solver to execute {SOLVING_CODE_FILE} "
                f"before submitting ({SOLVING_EXECUTION_RECEIPT_FILE} is missing and no stage "
                "may write it)"
            ]
        try:
            receipt_data = self._read_workspace_json_object(SOLVING_EXECUTION_RECEIPT_FILE)
            receipt = SolverExecutionReceipt.model_validate(receipt_data)
        except WorkspaceToolError as exc:
            return [str(exc)]
        except ValidationError as exc:
            return [
                f"{SOLVING_EXECUTION_RECEIPT_FILE} does not match its contract at "
                f"{self._error_location(error.get('loc', ()))}: "
                f"{error.get('msg', 'invalid value')}"
                for error in exc.errors()
            ]

        problems: list[str] = []
        expected = {
            SOLVING_CODE_FILE: receipt.solver_sha256,
            SOLVING_INPUT_FILE: receipt.input_sha256,
        }
        if result.solution_file is not None:
            if receipt.solution_file != SOLVING_SOLUTION_FILE or receipt.solution_sha256 is None:
                problems.append(
                    f"solver_result declares solution_file={SOLVING_SOLUTION_FILE!r} but the last "
                    "run_solver execution bound no candidate; run it again until it writes a "
                    f"schema-valid {SOLVING_SOLUTION_FILE}, or submit the no-candidate outcome it "
                    "actually produced"
                )
            else:
                expected[SOLVING_SOLUTION_FILE] = receipt.solution_sha256
        elif receipt.solution_file is not None or receipt.solution_sha256 is not None:
            problems.append(
                "solver_result declares no candidate but the last run_solver execution bound "
                f"{SOLVING_SOLUTION_FILE}; submit that candidate, or run the solver again if it "
                "should not have produced one"
            )
        stale = [
            relative_path
            for relative_path, digest in expected.items()
            if file_sha256(self.root / relative_path) != digest
        ]
        if stale:
            # Report every changed file, not just the first one found.
            problems.append(
                f"{', '.join(stale)} no longer match the digests recorded by the last run_solver "
                "execution; run run_solver again so the receipt describes the files you submit"
            )
        return problems

    def _read_workspace_json_object(self, relative_path: str) -> dict[str, Any]:
        target = self._resolve_existing(relative_path, expected="file")
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceToolError(f"{relative_path} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise WorkspaceToolError(f"{relative_path} must contain a JSON object")
        return value

    def _read_workspace_file_text(self, relative_path: str) -> str | None:
        target = self._resolve(relative_path)
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceToolError(f"could not read {relative_path}: {exc}") from exc

    @staticmethod
    def _format_validation_errors(errors: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for error in errors:
            raw_path = error.get("loc", ())
            path = ".".join(str(part) for part in raw_path) if raw_path else "<root>"
            lines.append(f"- {path}: {error.get('msg', 'invalid value')}")
        return "\n".join(lines)

    def _write_json_atomically(self, relative_path: str, value: dict[str, Any]) -> None:
        target = self._resolve_for_write(relative_path, stage=None)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_path, target)
        except OSError as exc:
            raise WorkspaceToolError(f"could not publish {relative_path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()

    async def shell(
        self,
        command: str,
        workdir: str | None = None,
        timeout_ms: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        try:
            if not isinstance(command, str) or not command.strip():
                raise WorkspaceToolError("command must be a non-empty string")
            if DIRECTORY_CHANGE_COMMAND.search(command):
                raise WorkspaceToolError(
                    "directory-changing commands are not allowed; shell already starts in the "
                    "workspace root. Use relative workdir for an existing subdirectory, and "
                    "create the directory in a separate shell call first"
                )
            # Restrict packages only during Solving so data inspection remains outside the ablation.
            shell_env = (
                self._guarded_env() if self._coerce_stage(stage) is AgentStage.SOLVING else None
            )
            if shell_env is not None:
                self._reject_import_guard_bypass(command)
            cwd = self._resolve_existing(workdir or ".", expected="dir")
            effective_timeout = self._effective_timeout(timeout_ms)
            # Every stage, the feasibility reviewer included, keeps protected-file
            # snapshots: a reviewer that can rewrite solution.json, input.json,
            # feasibility_checker.py or feasibility_result.json through shell could
            # manufacture the very evidence it is judging.  The reviewer still has
            # full read access, so its data analysis is unaffected.
            before: dict[Path, bytes | None] = self._protected_snapshot(stage)
            hidden: dict[Path, bytes] = self._hide_unreadable_files(stage)
        except WorkspaceToolError as exc:
            return self._error("shell", exc)

        start = time.monotonic()
        timed_out = False
        output = b""
        exit_code = -1
        trace_path: Path | None = None
        trace_before_timeout = ""
        trace_unavailable = False
        solution_before = (
            self._file_fingerprint(self.root / SOLVING_SOLUTION_FILE)
            if stage is AgentStage.SOLVING and self.runtime_solver_timeout_routing_enabled
            else None
        )
        try:
            if (
                os.name != "nt"
                and stage is AgentStage.SOLVING
                and self.runtime_solver_timeout_routing_enabled
            ):
                strace = await self._usable_linux_strace()
                if strace is not None:
                    trace_handle = tempfile.NamedTemporaryFile(
                        prefix="decisionbrain-solver-trace-",
                        suffix=".log",
                        delete=False,
                    )
                    trace_path = Path(trace_handle.name)
                    trace_handle.close()
                    process_tree = await ProcessTree.create_exec(
                        strace,
                        "-f",
                        "-qq",
                        "-e",
                        "trace=process,signal",
                        "-o",
                        str(trace_path),
                        "/bin/sh",
                        "-c",
                        command,
                        cwd=cwd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env=shell_env,
                    )
                else:
                    trace_unavailable = True
                    process_tree = await ProcessTree.create_shell(
                        command,
                        cwd=cwd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env=shell_env,
                    )
            else:
                process_tree = await ProcessTree.create_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=shell_env,
                )
        except WorkspaceToolError as exc:
            self._restore_hidden_files(hidden)
            return self._error("shell", exc)
        except OSError as exc:
            if trace_path is not None:
                trace_path.unlink(missing_ok=True)
            self._restore_hidden_files(hidden)
            return self._error("shell", WorkspaceToolError(f"could not start command: {exc}"))
        process = process_tree.process
        communicate_task = asyncio.create_task(process.communicate())
        try:
            output, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            trace_before_timeout = self._read_trace_file(trace_path)
            output = await self._terminate_process_tree(process_tree, communicate_task)
        except BaseException:
            await self._terminate_process_tree(process_tree, communicate_task)
            await self._emit_process_timing(
                command=command,
                stage=stage,
                started_at=start,
                exit_code=process.returncode,
                timed_out=False,
                status="interrupted",
            )
            self._restore_hidden_files(hidden)
            self._restore_unauthorized_changes(before)
            if trace_path is not None:
                trace_path.unlink(missing_ok=True)
            raise
        finally:
            process_tree.close()
            exit_code = process.returncode if process.returncode is not None else -1

        trace_text = self._read_trace_file(trace_path)
        if trace_path is not None:
            trace_path.unlink(missing_ok=True)
        trace = _parse_linux_solver_trace(
            trace_before_timeout or trace_text,
            cwd=cwd,
            solver=self.root / SOLVING_CODE_FILE,
        )
        runtime_solver_timeout = timed_out and bool(trace.active_solver_pids)
        gnu_solver_timeout = (
            not timed_out
            and exit_code == 124
            and trace.observed
            and trace.gnu_timeout_observed
        )

        primary_solver_timed_out = runtime_solver_timeout or gnu_solver_timeout
        timeout_source = (
            "runtime_watchdog"
            if runtime_solver_timeout
            else ("gnu_timeout" if gnu_solver_timeout else None)
        )
        duration = time.monotonic() - start
        text = output.decode("utf-8", errors="replace")
        solution_after = (
            self._file_fingerprint(self.root / SOLVING_SOLUTION_FILE)
            if solution_before is not None
            or (stage is AgentStage.SOLVING and self.runtime_solver_timeout_routing_enabled)
            else None
        )
        await self._emit_process_timing(
            command=command,
            stage=stage,
            started_at=start,
            exit_code=exit_code,
            timed_out=timed_out,
            status=(
                "primary_solver_timeout"
                if primary_solver_timed_out
                else ("timeout" if timed_out else ("ok" if exit_code == 0 else "error"))
            ),
            primary_solver_timed_out=primary_solver_timed_out,
            timeout_source=timeout_source,
            workdir=self._display_path(cwd),
            output_text=text,
            output_tail=text[-4000:],
            output_truncated=len(text) > 4000,
            solution_changed=solution_after is not None and solution_after != solution_before,
            solver_process_observed=trace.observed,
        )
        hidden_collisions = self._restore_hidden_files(hidden)
        violations = self._restore_unauthorized_changes(before)
        violations = [*hidden_collisions, *violations]
        if violations:
            names = ", ".join(dict.fromkeys(violations))
            return self._error(
                "shell",
                WorkspaceToolError(f"unauthorized protected file modification restored: {names}"),
            )

        if (
            stage is AgentStage.SOLVING
            and self.runtime_solver_timeout_routing_enabled
            and (runtime_solver_timeout or gnu_solver_timeout)
        ):
            source = timeout_source or "runtime_watchdog"
            signal_name = trace.solver_killed_by
            if signal_name not in {"SIGTERM", "SIGKILL"}:
                signal_name = "SIGTERM" if runtime_solver_timeout else None
            # The timeout terminal contract requires the execution receipt next to
            # the runtime outcome. run_solver publishes one; a solver launched
            # through shell must publish it here too, or the forced completion
            # would be rejected for an artifact the model never controlled.
            self._publish_solver_receipt(
                argv=(),
                exit_code=exit_code,
                elapsed_seconds=duration,
                timed_out=True,
            )
            self._publish_runtime_solver_timeout(
                timeout_source=source,
                timeout_seconds=effective_timeout if runtime_solver_timeout else None,
                elapsed_seconds=duration,
                exit_code=exit_code,
                termination_signal=signal_name,
                stdout=text,
                solution_before=solution_before,
            )
            raise StageForcedCompletion(
                "Primary solver was externally terminated by its time limit; "
                "Solving ended and the runtime outcome was routed to feasibility review."
            )
        if timed_out:
            text = f"command timed out after {int(effective_timeout * 1000)} milliseconds\n{text}"
            if trace_unavailable:
                text = (
                    "Primary-solver timeout routing was unavailable because a usable strace "
                    "installation was not found; the command ran with normal shell timeout "
                    "handling.\n"
                    + text
                )
        text, _truncated = self._truncate(text, self.max_output_chars)
        return "\n".join(
            [
                f"Exit code: {exit_code}",
                f"Wall time: {duration:.1f} seconds",
                "Output:",
                text,
            ]
        )

    @staticmethod
    def _read_trace_file(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _publish_runtime_solver_timeout(
        self,
        *,
        timeout_source: str,
        timeout_seconds: float | None,
        elapsed_seconds: float,
        exit_code: int,
        termination_signal: str | None,
        stdout: str,
        solution_before: tuple[int, int, int] | None,
    ) -> None:
        solution_path = self.root / SOLVING_SOLUTION_FILE
        solution_after = self._file_fingerprint(solution_path)
        solution_detected = solution_after is not None and solution_after != solution_before
        solution_schema_valid: bool | None = None
        solution_file: str | None = None
        if solution_detected:
            solution_schema_valid = self._solution_matches_schema()
            if solution_schema_valid:
                solution_file = SOLVING_SOLUTION_FILE
        outcome = RuntimeSolverTimeoutOutput.model_validate(
            {
                "schema_version": "1.0",
                "outcome_source": "runtime_external_timeout",
                "status": "runtime_timeout",
                "timeout_source": timeout_source,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "exit_code": exit_code,
                "termination_signal": termination_signal,
                "solver_file": SOLVING_CODE_FILE,
                "solver_execution_observed": True,
                "solution_detected": solution_detected,
                "solution_schema_valid": solution_schema_valid,
                "solution_file": solution_file,
                "stdout_tail": stdout[-4000:],
            }
        )
        self._write_json_atomically(
            SOLVING_RUNTIME_OUTCOME_FILE,
            outcome.model_dump(mode="json", exclude_none=False),
        )
        self._write_json_atomically(
            STAGE_OUTPUT_FILES[AgentStage.SOLVING],
            {
                "decision": "solved",
                "message": "Primary solver was externally terminated by its time limit.",
                "result": {"runtime_solver_outcome_file": SOLVING_RUNTIME_OUTCOME_FILE},
            },
        )

    async def _usable_linux_strace(self) -> str | None:
        if self._linux_strace_probe_complete:
            return self._linux_strace_path
        self._linux_strace_probe_complete = True
        strace = shutil.which("strace")
        if strace is None:
            return None
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                strace,
                "-qq",
                "-e",
                "trace=process",
                "-o",
                os.devnull,
                "/bin/true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return_code = await asyncio.wait_for(process.wait(), timeout=5.0)
        except (OSError, asyncio.TimeoutError):
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        if return_code == 0:
            self._linux_strace_path = strace
        return self._linux_strace_path

    @staticmethod
    def _file_fingerprint(path: Path) -> tuple[int, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _solution_matches_schema(self) -> bool:
        try:
            solution = self._read_workspace_json_object(SOLVING_SOLUTION_FILE)
            if not self.problem_contract_enabled:
                # The external formatter owns benchmark-shape validation in the
                # no-Problem-Contract arm; Runtime only requires a JSON object.
                return isinstance(solution, dict)
            schema = self._read_workspace_json_object(self._solution_schema_path())
            Draft202012Validator.check_schema(schema)
            return not any(Draft202012Validator(schema).iter_errors(solution))
        except (WorkspaceToolError, SchemaError):
            return False

    async def _emit_process_timing(
        self,
        *,
        command: str,
        stage: AgentStage | None,
        started_at: float,
        exit_code: int | None,
        timed_out: bool,
        status: str,
        primary_solver_timed_out: bool = False,
        timeout_source: str | None = None,
        workdir: str = ".",
        output_text: str = "",
        output_tail: str = "",
        output_truncated: bool = False,
        solution_changed: bool = False,
        solver_process_observed: bool = False,
    ) -> None:
        if self.process_timing_handler is None:
            return
        try:
            await self.process_timing_handler(
                {
                    "kind": "shell_process",
                    "stage": stage.value if stage is not None else "unknown",
                    "command": command,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "primary_solver_timed_out": primary_solver_timed_out,
                    "timeout_source": timeout_source,
                    "workdir": workdir,
                    "output_text": output_text,
                    "output_tail": output_tail,
                    "output_truncated": output_truncated,
                    "solution_changed": solution_changed,
                    "solver_process_observed": solver_process_observed,
                    "status": status,
                }
            )
        except Exception:
            # Telemetry must never change the command's observable behavior.
            pass

    @staticmethod
    async def _terminate_process_tree(
        process_tree: ProcessTree,
        communicate_task: asyncio.Task[tuple[bytes, bytes | None]],
    ) -> bytes:
        process_tree.terminate()
        try:
            output, _ = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=1.0)
            return output
        except asyncio.TimeoutError:
            process_tree.kill()

        try:
            output, _ = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=1.0)
            return output
        except asyncio.TimeoutError:
            communicate_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await communicate_task
            return b""

    async def read_file(
        self,
        path: str,
        max_chars: int | float | None = None,
        start_line: int | float | None = None,
        line_count: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        try:
            target = self._resolve_existing(path, expected="file")
            self.access_policy.ensure_can_read(target, root=self.root, stage=stage)
            first_line = self._effective_positive_int(start_line, name="start_line", default=1)
            count = self._effective_positive_int(line_count, name="line_count")
            limit = self._effective_char_limit(max_chars)
            if start_line is not None or line_count is not None:
                return self._read_text_line_range(
                    target,
                    first_line=first_line or 1,
                    line_count=count,
                    limit=limit,
                )
            text = self._read_text_target(target)
        except WorkspaceToolError as exc:
            return self._error("read_file", exc)
        text, _truncated = self._truncate(text, limit)
        return text

    async def search_file(
        self,
        query: str,
        path: str = ".",
        max_results: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        """Search stage-readable UTF-8 text files without exposing unrestricted shell search."""
        try:
            if not isinstance(query, str) or not query:
                raise WorkspaceToolError("query must be a non-empty string")
            target = self._resolve_existing(path or ".", expected="file_or_dir")
            self.access_policy.ensure_can_read(target, root=self.root, stage=stage)
            limit = self._effective_search_result_limit(max_results)
            candidates = [target] if target.is_file() else self._search_candidates(target, stage=stage)
            matches: list[str] = []
            for candidate in candidates:
                if len(matches) >= limit:
                    break
                if candidate.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                relative = self._display_path(candidate)
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query not in line:
                        continue
                    clipped, _ = self._truncate(line, 500, notice="...")
                    matches.append(f"{relative}:{line_number}: {clipped}")
                    if len(matches) >= limit:
                        break
        except WorkspaceToolError as exc:
            return self._error("search_file", exc)

        if not matches:
            return "(no matches)"
        result, truncated = self._truncate("\n".join(matches), self.max_output_chars)
        if truncated:
            return f"{result}\n[output truncated; narrow path or query]"
        return result

    async def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
        stage: AgentStage | None = None,
    ) -> str:
        try:
            if not isinstance(content, str):
                raise WorkspaceToolError("content must be a string")
            target = self._resolve_for_write(path, stage=stage)
            if target.exists() and target.is_dir():
                raise WorkspaceToolError("target is a directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8") as handle:
                handle.write(content)
            action = "appended" if append else "wrote"
            rel = self._display_path(target)
        except WorkspaceToolError as exc:
            return self._error("write_file", exc)
        return f"{action} {len(content.encode('utf-8'))} bytes to {rel}"

    async def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int | float | None = None,
        stage: AgentStage | None = None,
    ) -> str:
        try:
            root = self._resolve_existing(path or ".", expected="dir")
            limit = self._effective_entry_limit(max_entries)
            entries = self._collect_entries(
                root,
                recursive=bool(recursive),
                limit=limit,
                stage=stage,
            )
        except WorkspaceToolError as exc:
            return self._error("list_files", exc)

        if not entries:
            return "(empty)"
        lines = entries[:limit]
        if len(entries) > limit:
            lines.append(f"[truncated after {limit} entries]")
        return "\n".join(lines)

    async def replace_in_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        stage: AgentStage | None = None,
    ) -> str:
        """Find old_string in a text file and replace it with new_string.

        old_string must match exactly once in the file (including whitespace and
        indentation).  If it matches zero or more than one time the tool returns
        an error with line numbers so the caller can correct the match.
        """
        try:
            target = self._replace_resolve(path, stage=stage)
            text = target.read_text(encoding="utf-8")
        except WorkspaceToolError as exc:
            return self._error("replace_in_file", exc)
        except UnicodeDecodeError:
            return self._error("replace_in_file", WorkspaceToolError("file is not valid UTF-8"))

        if old_string == new_string:
            return self._error(
                "replace_in_file",
                WorkspaceToolError("old_string and new_string are identical"),
            )

        count = text.count(old_string)
        if count == 0:
            # Try to be helpful — show surrounding context
            lines = text.splitlines()
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                old_stripped = old_string.strip()
                if stripped == old_stripped or old_stripped in stripped:
                    return self._error(
                        "replace_in_file",
                        WorkspaceToolError(
                            f"old_string not found (whitespace/indent mismatch?). "
                            f"Line {idx} has similar content but differs by whitespace:\n"
                            f"  found: {line!r}\n"
                            f"  expected: {old_string!r}\n"
                            f"Tip: include the exact indentation and trailing spaces."
                        ),
                    )
            return self._error(
                "replace_in_file",
                WorkspaceToolError("old_string not found anywhere in the file"),
            )
        if count > 1:
            return self._error(
                "replace_in_file",
                WorkspaceToolError(
                    f"old_string matches {count} times — must match exactly once. "
                    f"Include more surrounding context lines to make it unique."
                ),
            )

        replaced = text.replace(old_string, new_string, 1)
        before = self._protected_snapshot(stage)
        target.write_text(replaced, encoding="utf-8")
        violations = self._restore_unauthorized_changes(before)
        if violations:
            names = ", ".join(violations)
            return self._error(
                "replace_in_file",
                WorkspaceToolError(f"unauthorized protected file modification restored: {names}"),
            )
        return (
            f"replace_in_file: {self._display_path(target)} — "
            f"{len(old_string)} chars replaced with {len(new_string)} chars"
        )

    async def delete_file(
        self,
        path: str,
        stage: AgentStage | None = None,
    ) -> str:
        """Delete a file from the working directory."""
        try:
            target = self._replace_resolve(path, stage=stage)
            if not target.is_file():
                raise WorkspaceToolError("path is not a file")
            target.unlink()
        except WorkspaceToolError as exc:
            return self._error("delete_file", exc)
        except OSError as exc:
            return self._error("delete_file", WorkspaceToolError(str(exc)))
        return f"delete_file: {self._display_path(target)} — deleted"

    def _replace_resolve(self, raw_path: str, *, stage: AgentStage | None) -> Path:
        """Resolve and validate a path for replace_in_file / delete_file."""
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkspaceToolError("path must be a non-empty string")
        if raw_path.startswith("/"):
            raise WorkspaceToolError("absolute paths are not allowed")
        if ".." in Path(raw_path).parts:
            raise WorkspaceToolError("path escapes working directory")
        target = (self.root / raw_path).resolve(strict=False)
        if target != self.root and self.root not in target.parents:
            raise WorkspaceToolError("path escapes working directory")
        if not target.exists():
            raise WorkspaceToolError(f"file does not exist: {self._display_path(target)}")
        self.access_policy.ensure_can_write(target, root=self.root, stage=stage)
        return target

    def _collect_entries(
        self,
        root: Path,
        *,
        recursive: bool,
        limit: int,
        stage: AgentStage | None,
    ) -> list[str]:
        entries: list[str] = []
        if recursive:
            for current, directories, files in os.walk(root, topdown=True, followlinks=False):
                directories.sort()
                files.sort()
                current_path = Path(current)
                for directory in directories:
                    target = current_path / directory
                    if not self.access_policy.can_read(target, root=self.root, stage=stage):
                        continue
                    entries.append(f"{self._display_path(target)}/")
                    if len(entries) > limit:
                        return entries
                for filename in files:
                    target = current_path / filename
                    if not self.access_policy.can_read(target, root=self.root, stage=stage):
                        continue
                    entries.append(self._entry_label(target, is_dir=False))
                    if len(entries) > limit:
                        return entries
            return entries

        for item in sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
            if not self.access_policy.can_read(item, root=self.root, stage=stage):
                continue
            entries.append(self._entry_label(item, is_dir=item.is_dir()))
            if len(entries) > limit:
                break
        return entries

    def _entry_label(self, target: Path, *, is_dir: bool) -> str:
        """Annotate a listed file with its size, and flag it when read_file will refuse.

        Instance files in the large suites reach hundreds of megabytes, so a plain
        name invites a read_file call that can only fail. Saying the size here, and
        naming the tool that still works, turns a wasted turn into a decision.
        """

        display = self._display_path(target)
        if is_dir:
            return f"{display}/"
        try:
            size = target.stat().st_size
        except OSError:
            return display
        label = f"{display}  ({self._format_size(size)})"
        if size > self.max_read_bytes:
            label += " [too large for read_file; use shell to extract what you need]"
        return label

    @staticmethod
    def _format_size(size: int) -> str:
        for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
            if size >= scale:
                return f"{size / scale:.1f} {unit}"
        return f"{size} B"

    def _read_text_target(self, target: Path) -> str:
        if target.stat().st_size > self.max_read_bytes:
            raise WorkspaceToolError(
                f"file too large: {target.stat().st_size} bytes; limit is {self.max_read_bytes}"
            )
        raw = target.read_bytes()
        if b"\0" in raw[:4096]:
            raise WorkspaceToolError("file appears to be binary")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceToolError("file is not valid UTF-8 text") from exc

    def _read_text_line_range(
        self,
        target: Path,
        *,
        first_line: int,
        line_count: int | None,
        limit: int,
    ) -> str:
        """Read a requested text-line range without loading a large file in full."""
        try:
            with target.open("rb") as handle:
                if b"\0" in handle.read(4096):
                    raise WorkspaceToolError("file appears to be binary")

            selected: list[str] = []
            last_line = 0
            with target.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    last_line = line_number
                    if line_number < first_line:
                        continue
                    if line_count is not None and line_number >= first_line + line_count:
                        break
                    selected.append(line)
        except UnicodeDecodeError as exc:
            raise WorkspaceToolError("file is not valid UTF-8 text") from exc
        except OSError as exc:
            raise WorkspaceToolError(str(exc)) from exc

        if first_line > last_line + 1:
            raise WorkspaceToolError(
                f"start_line {first_line} exceeds file length of {last_line} lines"
            )
        text, _truncated = self._truncate("".join(selected), limit)
        return text

    def _search_candidates(self, root: Path, *, stage: AgentStage | None) -> list[Path]:
        candidates: list[Path] = []
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            directories.sort()
            files.sort()
            current_path = Path(current)
            directories[:] = [
                directory
                for directory in directories
                if self.access_policy.can_read(
                    (current_path / directory).resolve(strict=False), root=self.root, stage=stage
                )
            ]
            for filename in files:
                candidate = (current_path / filename).resolve(strict=False)
                if candidate == self.root or self.root not in candidate.parents:
                    continue
                if not candidate.is_file() or not self.access_policy.can_read(
                    candidate, root=self.root, stage=stage
                ):
                    continue
                candidates.append(candidate)
                if len(candidates) >= SEARCH_MAX_FILES:
                    return candidates
        return candidates

    def _resolve_existing(self, raw_path: str, *, expected: str) -> Path:
        target = self._resolve(raw_path)
        if not target.exists():
            raise WorkspaceToolError(f"path does not exist: {raw_path}")
        if expected == "file" and not target.is_file():
            raise WorkspaceToolError(f"path is not a file: {raw_path}")
        if expected == "dir" and not target.is_dir():
            raise WorkspaceToolError(f"path is not a directory: {raw_path}")
        if expected == "file_or_dir" and not (target.is_file() or target.is_dir()):
            raise WorkspaceToolError(f"path is not a file or directory: {raw_path}")
        return target

    def _resolve_for_write(self, raw_path: str, *, stage: AgentStage | None = None) -> Path:
        target = self._resolve(raw_path)
        self.access_policy.ensure_can_write(target, root=self.root, stage=stage)
        return target

    def _resolve(self, raw_path: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkspaceToolError("path must be a non-empty string")
        if "\0" in raw_path:
            raise WorkspaceToolError("path contains a NUL byte")
        path = Path(raw_path)
        if path.is_absolute():
            raise WorkspaceToolError("absolute paths are not allowed")
        target = (self.root / path).resolve(strict=False)
        if target != self.root and self.root not in target.parents:
            raise WorkspaceToolError("path escapes workspace root")
        return target

    def _display_path(self, path: Path) -> str:
        try:
            rel = path.resolve(strict=False).relative_to(self.root)
        except ValueError:
            return "<outside workspace>"
        return rel.as_posix() or "."

    def _protected_snapshot(self, stage: AgentStage | None) -> dict[Path, bytes | None]:
        snapshot: dict[Path, bytes | None] = {}
        for target in self.access_policy.unauthorized_write_targets(root=self.root, stage=stage):
            if target.is_file():
                snapshot[target] = target.read_bytes()
            else:
                snapshot[target] = None
        return snapshot

    def _hide_unreadable_files(self, stage: AgentStage | None) -> dict[Path, bytes]:
        hidden: dict[Path, bytes] = {}
        for target in self.access_policy.unreadable_file_targets(root=self.root, stage=stage):
            try:
                hidden[target] = target.read_bytes()
                target.unlink()
            except OSError as exc:
                self._restore_hidden_files(hidden)
                raise WorkspaceToolError(
                    f"could not isolate unreadable file {self._display_path(target)}: {exc}"
                ) from exc
        return hidden

    def _restore_hidden_files(self, hidden: dict[Path, bytes]) -> list[str]:
        collisions: list[str] = []
        for target, original in hidden.items():
            if target.is_file() and target.read_bytes() != original:
                collisions.append(self._display_path(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original)
        return collisions

    def _restore_unauthorized_changes(self, before: dict[Path, bytes | None]) -> list[str]:
        changed: list[str] = []
        for target, original in before.items():
            if target.is_file():
                current = target.read_bytes()
            else:
                current = None
            if current == original:
                continue
            changed.append(self._display_path(target))
            if original is None:
                if target.exists() and target.is_file():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original)
        return changed

    def _effective_timeout(self, timeout_ms: int | float | None) -> float:
        max_timeout = self.timeout_s
        if timeout_ms is None:
            return max_timeout
        try:
            requested = float(timeout_ms) / 1000.0
        except (TypeError, ValueError) as exc:
            raise WorkspaceToolError("timeout_ms must be a number") from exc
        if requested <= 0:
            raise WorkspaceToolError("timeout_ms must be greater than 0")
        return min(requested, max_timeout)

    def _effective_char_limit(self, max_chars: int | float | None) -> int:
        if max_chars is None:
            return self.max_output_chars
        try:
            requested = int(max_chars)
        except (TypeError, ValueError) as exc:
            raise WorkspaceToolError("max_chars must be a number") from exc
        if requested <= 0:
            raise WorkspaceToolError("max_chars must be greater than 0")
        return min(requested, self.max_output_chars)

    @staticmethod
    def _effective_positive_int(
        value: int | float | None,
        *,
        name: str,
        default: int | None = None,
    ) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            raise WorkspaceToolError(f"{name} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise WorkspaceToolError(f"{name} must be a positive integer") from exc
        if parsed != value or parsed <= 0:
            raise WorkspaceToolError(f"{name} must be a positive integer")
        return parsed

    def _effective_entry_limit(self, max_entries: int | float | None) -> int:
        if max_entries is None:
            return self.default_list_entries
        try:
            requested = int(max_entries)
        except (TypeError, ValueError) as exc:
            raise WorkspaceToolError("max_entries must be a number") from exc
        if requested <= 0:
            raise WorkspaceToolError("max_entries must be greater than 0")
        return min(requested, self.max_list_entries)

    @staticmethod
    def _effective_search_result_limit(max_results: int | float | None) -> int:
        if max_results is None:
            return SEARCH_DEFAULT_RESULTS
        parsed = WorkspaceToolset._effective_positive_int(
            max_results,
            name="max_results",
        )
        assert parsed is not None
        return min(parsed, SEARCH_MAX_RESULTS)

    @staticmethod
    def _truncate(
        text: str,
        limit: int,
        *,
        notice: str = TRUNCATION_NOTICE,
    ) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        if limit <= len(notice) + 20:
            return text[:limit], True
        head_len = (limit - len(notice)) // 2
        tail_len = limit - len(notice) - head_len
        return f"{text[:head_len]}{notice}{text[-tail_len:]}", True

    @staticmethod
    def _error(tool_name: str, exc: Exception) -> str:
        return f"{tool_name} error: {exc}"
