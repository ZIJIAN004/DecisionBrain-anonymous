"""Structured timing collected for one Agent Runtime execution."""

from __future__ import annotations

import re
import shlex
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any


_PYTHON_EXECUTABLE = re.compile(r"python(?:3(?:\.\d+)?)?(?:\.exe)?|py(?:\.exe)?", re.I)
_TARGET_SCRIPTS = {"solver.py": "solver", "feasibility_checker.py": "self_checker"}
_SHELL_COMPOSITION = re.compile(r"[;&|\r\n]")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _basename(value: str) -> str:
    return value.strip("'\"").replace("\\", "/").rsplit("/", 1)[-1].lower()


def classify_shell_process(command: str) -> tuple[str, bool]:
    """Classify one standalone Python script invocation.

    Composite commands are deliberately ambiguous because one shell duration cannot be
    apportioned reliably across child programs.
    """

    mentioned = {name for name in _TARGET_SCRIPTS if name in command.lower()}
    if not mentioned:
        return "other", False
    if _SHELL_COMPOSITION.search(command):
        return "ambiguous", True
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "ambiguous", True
    if not tokens or not _PYTHON_EXECUTABLE.fullmatch(_basename(tokens[0])):
        return "ambiguous", True
    targets = [_TARGET_SCRIPTS[name] for token in tokens[1:] if (name := _basename(token)) in _TARGET_SCRIPTS]
    if len(targets) != 1 or "-c" in tokens[1:] or "-m" in tokens[1:]:
        return "ambiguous", True
    return targets[0], False


class RunTiming:
    """Collect wall-clock spans and monotonic durations without affecting execution."""

    def __init__(self) -> None:
        self._started_at = _utc_now()
        self._started_at_monotonic = perf_counter()
        self._stages: list[dict[str, Any]] = []
        self._open_stages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._llm_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._process_calls: list[dict[str, Any]] = []

    def stage_started(self, stage: str) -> None:
        record = {
            "stage": stage,
            "occurrence": sum(item["stage"] == stage for item in self._stages) + 1,
            "started_at": _utc_now().isoformat(),
            "_started_monotonic": perf_counter(),
            "status": "running",
        }
        self._stages.append(record)
        self._open_stages[stage].append(record)

    def stage_finished(self, stage: str) -> None:
        pending = self._open_stages.get(stage)
        if not pending:
            return
        record = pending.pop()
        finished = perf_counter()
        record["finished_at"] = _utc_now().isoformat()
        record["duration_ms"] = int((finished - record["_started_monotonic"]) * 1000)
        record["status"] = "completed"

    async def record(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        stage = str(event.get("stage") or "unknown")
        duration_ms = int(event.get("duration_ms") or 0)
        if duration_ms < 0:
            return
        finished_at = _utc_now()
        common = {
            "stage": stage,
            "started_at": (finished_at - timedelta(milliseconds=duration_ms)).isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "status": str(event.get("status") or "ok"),
        }
        if kind == "llm_call":
            self._llm_calls.append(
                {
                    "sequence": len(self._llm_calls) + 1,
                    **common,
                    "retry_count": int(event.get("retry_count") or 0),
                    "retry_backoff_ms": int(event.get("retry_backoff_ms") or 0),
                    "input_tokens": int(event.get("input_tokens") or 0),
                    "output_tokens": int(event.get("output_tokens") or 0),
                    "cached_tokens": int(event.get("cached_tokens") or 0),
                    "total_tokens": int(event.get("total_tokens") or 0),
                }
            )
            return
        if kind == "tool_call":
            self._tool_calls.append(
                {
                    "sequence": len(self._tool_calls) + 1,
                    **common,
                    "tool_name": str(event.get("tool_name") or ""),
                }
            )
            return
        if kind == "shell_process":
            command = str(event.get("command") or "")
            execution_kind, ambiguous = classify_shell_process(command)
            self._process_calls.append(
                {
                    "sequence": len(self._process_calls) + 1,
                    **common,
                    "execution_kind": execution_kind,
                    "ambiguous": ambiguous,
                    "command": command,
                    "workdir": str(event.get("workdir") or "."),
                    "exit_code": event.get("exit_code"),
                    "timed_out": bool(event.get("timed_out")),
                    "timeout_source": event.get("timeout_source"),
                    "solver_process_observed": bool(event.get("solver_process_observed")),
                    "primary_solver_timed_out": bool(event.get("primary_solver_timed_out")),
                }
            )

    def to_dict(self) -> dict[str, Any]:
        finished_at = _utc_now()
        finished_monotonic = perf_counter()
        duration_ms = int((finished_monotonic - self._started_at_monotonic) * 1000)
        stages = []
        for item in self._stages:
            value = dict(item)
            started_monotonic = value.pop("_started_monotonic")
            if value["status"] == "running":
                value["finished_at"] = finished_at.isoformat()
                value["duration_ms"] = int((finished_monotonic - started_monotonic) * 1000)
                value["status"] = "incomplete"
            stages.append(value)
        return {
            "schema_version": "1.2",
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "stages": stages,
            "llm_calls": list(self._llm_calls),
            "tool_calls": list(self._tool_calls),
            "process_calls": list(self._process_calls),
            "summary": self._summary(stages, duration_ms),
        }

    def _summary(self, stages: list[dict[str, Any]], duration_ms: int) -> dict[str, Any]:
        stage_duration_ms: dict[str, int] = defaultdict(int)
        for item in stages:
            stage_duration_ms[str(item["stage"])] += int(item["duration_ms"])
        llm_duration_ms: dict[str, int] = defaultdict(int)
        for item in self._llm_calls:
            llm_duration_ms[item["stage"]] += item["duration_ms"]
        token_totals = {
            name: sum(item[name] for item in self._llm_calls)
            for name in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens")
        }
        tool_duration_ms: dict[str, int] = defaultdict(int)
        for item in self._tool_calls:
            tool_duration_ms[item["tool_name"]] += item["duration_ms"]
        checker_calls = [item for item in self._process_calls if item["execution_kind"] == "self_checker"]
        solver_calls = [item for item in self._process_calls if item["execution_kind"] == "solver"]
        ambiguous_calls = [item for item in self._process_calls if item["ambiguous"]]
        package_shell_calls = [
            item
            for item in self._process_calls
            if item["execution_kind"] in {"solver", "self_checker"}
            or item["solver_process_observed"]
        ]
        return {
            "wall_clock_duration_ms": duration_ms,
            "stage_duration_ms": dict(sorted(stage_duration_ms.items())),
            "completed_stage_count": sum(item["status"] == "completed" for item in stages),
            "incomplete_stage_count": sum(item["status"] == "incomplete" for item in stages),
            "llm_duration_ms": dict(sorted(llm_duration_ms.items())),
            "llm_call_count": len(self._llm_calls),
            "llm_retry_count": sum(item["retry_count"] for item in self._llm_calls),
            "llm_retry_backoff_ms": sum(item["retry_backoff_ms"] for item in self._llm_calls),
            "llm_tokens": token_totals,
            "tool_duration_ms": dict(sorted(tool_duration_ms.items())),
            "tool_call_count": len(self._tool_calls),
            "self_checker_duration_ms": sum(item["duration_ms"] for item in checker_calls),
            "self_checker_execution_count": len(checker_calls),
            "solver_execution_duration_ms": sum(item["duration_ms"] for item in solver_calls),
            "solver_execution_count": len(solver_calls),
            "ambiguous_process_duration_ms": sum(item["duration_ms"] for item in ambiguous_calls),
            "ambiguous_process_count": len(ambiguous_calls),
            "algorithm_shell_duration_ms": sum(item["duration_ms"] for item in package_shell_calls),
            "algorithm_shell_count": len(package_shell_calls),
        }
