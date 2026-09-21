"""Evidence-based package usage records for one Agent run."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from ..algorithm_library.catalog import AlgorithmCatalog
from .timing import classify_shell_process


_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("module_not_found", re.compile(r"ModuleNotFoundError|ImportError", re.I)),
    ("license_error", re.compile(r"license.*(?:error|fail|invalid|expired|not found)", re.I)),
    ("memory_error", re.compile(r"MemoryError|out of memory|bad_alloc", re.I)),
    ("api_error", re.compile(r"AttributeError|TypeError|unexpected keyword|has no attribute", re.I)),
    ("data_error", re.compile(r"KeyError|IndexError|ValueError|JSONDecodeError", re.I)),
    ("solver_error", re.compile(r"GurobiError|SCIP Error|CpSolver.*error", re.I)),
    ("python_exception", re.compile(r"Traceback \(most recent call last\)", re.I)),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_import_roots(catalog: AlgorithmCatalog) -> dict[str, set[str]]:
    roots: dict[str, set[str]] = {}
    for manifest in catalog.list_available():
        manifest_roots: set[str] = set()
        for statement in manifest.interface.imports:
            try:
                tree = ast.parse(statement)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.Import):
                    manifest_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    manifest_roots.add(node.module.split(".", 1)[0])
        roots[manifest.id] = manifest_roots
    return roots


class _CodeUsageVisitor(ast.NodeVisitor):
    def __init__(self, root_to_package: Mapping[str, str]) -> None:
        self.root_to_package = root_to_package
        self.aliases: dict[str, tuple[str, str]] = {}
        self.package_objects: dict[str, tuple[str, str]] = {}
        self.imports: set[str] = set()
        self.calls: Counter[str] = Counter()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            package = self.root_to_package.get(root)
            if package:
                local = alias.asname or root
                self.aliases[local] = (package, alias.name)
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".", 1)[0]
            package = self.root_to_package.get(root)
            if package:
                self.imports.add(node.module)
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.aliases[local] = (package, f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = self._attribute_chain(node.func)
        resolved = self._resolve_chain(chain)
        if resolved:
            package, api = resolved
            self.calls[f"{package}:{api}"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        origin = self._call_origin(node.value)
        if origin:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.package_objects[target.id] = origin
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        origin = self._call_origin(node.value) if node.value is not None else None
        if origin and isinstance(node.target, ast.Name):
            self.package_objects[node.target.id] = origin
        self.generic_visit(node)

    def _call_origin(self, node: ast.AST) -> tuple[str, str] | None:
        if not isinstance(node, ast.Call):
            return None
        return self._resolve_chain(self._attribute_chain(node.func))

    def _resolve_chain(self, chain: list[str] | None) -> tuple[str, str] | None:
        if not chain:
            return None
        root = chain[0]
        origin = self.aliases.get(root) or self.package_objects.get(root)
        if origin is None:
            return None
        package, imported = origin
        suffix = ".".join(chain[1:])
        return package, imported if not suffix else f"{imported}.{suffix}"

    @staticmethod
    def _attribute_chain(node: ast.AST) -> list[str] | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        parts.append(current.id)
        return list(reversed(parts))


def inspect_solver_code(source: str, catalog: AlgorithmCatalog) -> dict[str, Any]:
    roots_by_package = _manifest_import_roots(catalog)
    root_to_package = {
        root: package_id
        for package_id, roots in roots_by_package.items()
        for root in roots
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "parse_status": "syntax_error",
            "parse_error": f"{exc.msg} (line {exc.lineno})",
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "imports": [],
            "package_ids": [],
            "api_call_families": {},
        }
    visitor = _CodeUsageVisitor(root_to_package)
    visitor.visit(tree)
    package_ids = sorted({key.split(":", 1)[0] for key in visitor.calls} | {
        root_to_package[item.split(".", 1)[0]]
        for item in visitor.imports
        if item.split(".", 1)[0] in root_to_package
    })
    return {
        "parse_status": "ok",
        "parse_error": None,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "imports": sorted(visitor.imports),
        "package_ids": package_ids,
        "api_call_families": dict(sorted(visitor.calls.items())),
    }


def _classify_error(event: Mapping[str, Any]) -> tuple[str | None, str]:
    output = str(event.get("output_text") or event.get("output_tail") or "")
    if event.get("primary_solver_timed_out"):
        return "runtime_timeout", output[-4000:]
    if event.get("timed_out"):
        return "time_limit", output[-4000:]
    exit_code = event.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        for category, pattern in _ERROR_PATTERNS:
            match = pattern.search(output)
            if match:
                return category, _error_excerpt(output, match.start())
        return "nonzero_exit", output[-4000:]
    for category, pattern in _ERROR_PATTERNS:
        match = pattern.search(output)
        if match:
            return category, _error_excerpt(output, match.start())
    return None, ""


def _error_excerpt(output: str, match_start: int) -> str:
    start = max(0, match_start - 1000)
    return output[start : start + 12000]


def _attempt_failed(event: Mapping[str, Any]) -> bool:
    if event.get("timed_out") or event.get("primary_solver_timed_out"):
        return True
    exit_code = event.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    return str(event.get("status") or "").lower() in {
        "error",
        "interrupted",
        "primary_solver_timeout",
        "timeout",
    }


class PackageUsageObserver:
    """Collect package-use evidence without feeding it back to the Agent."""

    def __init__(self, workspace: Path, catalog: AlgorithmCatalog) -> None:
        self.workspace = workspace
        self.catalog = catalog
        self._manifest_roots = _manifest_import_roots(catalog)
        self._guide_calls: list[dict[str, Any]] = []
        self._availability_checks: list[dict[str, Any]] = []
        self._attempts: list[dict[str, Any]] = []
        self._event_sequence = 0

    async def record_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "tool_call":
            self._record_tool_call(event)
            return
        if kind != "shell_process":
            return
        self._event_sequence += 1
        command = str(event.get("command") or "")
        execution_kind, ambiguous = classify_shell_process(command)
        if event.get("solver_process_observed"):
            execution_kind = "solver"
            ambiguous = False
        if execution_kind == "solver":
            self._record_solver_attempt(event, command, ambiguous)
            return
        referenced = self._packages_referenced_by_text(command)
        if referenced:
            error_category, error_output = _classify_error(event)
            self._availability_checks.append(
                {
                    "sequence": self._event_sequence,
                    "stage": str(event.get("stage") or ""),
                    "command": command,
                    "package_ids": referenced,
                    "duration_ms": int(event.get("duration_ms") or 0),
                    "exit_code": event.get("exit_code"),
                    "status": str(event.get("status") or ""),
                    "error_category": error_category,
                    "error_output_tail": error_output,
                }
            )

    def _record_tool_call(self, event: Mapping[str, Any]) -> None:
        if str(event.get("tool_name") or "") != "get_algorithm_guide":
            return
        arguments = event.get("tool_arguments")
        if not isinstance(arguments, Mapping):
            return
        self._event_sequence += 1
        status = str(event.get("status") or "")
        self._guide_calls.append(
            {
                "sequence": self._event_sequence,
                "stage": str(event.get("stage") or ""),
                "package_id": str(arguments.get("package_id") or "").strip(),
                "solver_id": str(arguments.get("solver_id") or "").strip(),
                "status": "ok" if status == "ok" else "error",
                "error": None if status == "ok" else status,
            }
        )

    def _record_solver_attempt(
        self, event: Mapping[str, Any], command: str, ambiguous: bool
    ) -> None:
        try:
            source = (self.workspace / "solver.py").read_text(encoding="utf-8")
        except OSError:
            source = ""
        code = inspect_solver_code(source, self.catalog)
        error_category, error_output = _classify_error(event)
        attempt_failed = _attempt_failed(event)
        package_ids = list(code["package_ids"])
        guide_ids = {
            item["package_id"]
            for item in self._guide_calls
            if item["status"] == "ok" and item["sequence"] < self._event_sequence
        }
        self._attempts.append(
            {
                "sequence": len(self._attempts) + 1,
                "event_sequence": self._event_sequence,
                "stage": str(event.get("stage") or ""),
                "command": command,
                "workdir": str(event.get("workdir") or "."),
                "ambiguous_command": ambiguous,
                "started_at": (
                    _utc_now() - timedelta(milliseconds=int(event.get("duration_ms") or 0))
                ).isoformat(),
                "duration_ms": int(event.get("duration_ms") or 0),
                "exit_code": event.get("exit_code"),
                "status": str(event.get("status") or ""),
                "timed_out": bool(event.get("timed_out")),
                "timeout_source": event.get("timeout_source"),
                "solution_changed": bool(event.get("solution_changed")),
                "code": code,
                "package_ids": package_ids,
                "guide_consulted_before_use": {
                    package_id: package_id in guide_ids for package_id in package_ids
                },
                "attempt_failed": attempt_failed,
                "error_observed": error_category is not None,
                "error_category": error_category,
                "error_output_tail": error_output,
                "output_tail": str(event.get("output_tail") or ""),
                "output_truncated": bool(event.get("output_truncated")),
            }
        )

    def _packages_referenced_by_text(self, text: str) -> list[str]:
        lowered = text.lower()
        found = []
        for package_id, roots in self._manifest_roots.items():
            names = {package_id, *roots}
            if any(re.search(rf"(?<![\w.]){re.escape(name.lower())}(?![\w.])", lowered) for name in names):
                found.append(package_id)
        return sorted(found)

    def to_dict(self) -> dict[str, Any]:
        design = _read_json_object(self.workspace / "stage_outputs" / "algorithm_design.json")
        solver_result = _read_json_object(self.workspace / "solver_result.json")
        planned = self._planned_components(design)
        declared = self._declared_executions(solver_result)
        planned_ids = {item["package_id"] for item in planned if item["package_id"]}
        observed_ids = {
            package_id for attempt in self._attempts for package_id in attempt["package_ids"]
        }
        unplanned = sorted(observed_ids - planned_ids)
        missing_guides = sorted(
            {
                package_id
                for attempt in self._attempts
                for package_id, consulted in attempt["guide_consulted_before_use"].items()
                if not consulted
            }
        )
        failed = [attempt for attempt in self._attempts if attempt["attempt_failed"]]
        errored = [attempt for attempt in self._attempts if attempt["error_observed"]]
        total_duration = sum(attempt["duration_ms"] for attempt in self._attempts)
        failed_duration = sum(attempt["duration_ms"] for attempt in failed)
        errored_duration = sum(attempt["duration_ms"] for attempt in errored)
        package_sequences = [tuple(attempt["package_ids"]) for attempt in self._attempts]
        recovery_after_error = any(
            attempt["error_observed"] or attempt["attempt_failed"]
            for attempt in self._attempts[:-1]
        ) and any(not attempt["attempt_failed"] for attempt in self._attempts[1:])
        declared_fallback = any(bool(item.get("fallback_used")) for item in declared)
        package_switch = len(set(package_sequences)) > 1
        fallback_reasons = [
            reason
            for reason, present in (
                ("package_switch", package_switch),
                ("unplanned_package", bool(unplanned)),
                ("recovery_after_error", recovery_after_error),
                ("model_declared_fallback", declared_fallback),
            )
            if present
        ]
        fallback_observed = bool(fallback_reasons)
        if not observed_ids:
            plan_alignment = "no_package_execution_observed"
        elif unplanned:
            plan_alignment = "deviates_from_plan"
        elif observed_ids <= planned_ids:
            plan_alignment = "consistent_with_plan"
        else:
            plan_alignment = "insufficient_evidence"
        return {
            "schema_version": "1.0",
            "generated_at": _utc_now().isoformat(),
            "evidence_policy": {
                "runtime_observed_fields": [
                    "guide_calls",
                    "availability_checks",
                    "attempts.command",
                    "attempts.duration_ms",
                    "attempts.exit_code",
                    "attempts.attempt_failed",
                    "attempts.error_observed",
                    "attempts.error_output_tail",
                    "attempts.output_tail",
                    "attempts.code",
                ],
                "agent_declared_fields": ["plan", "declared_executions"],
                "limitation": (
                    "Exceptions caught and suppressed inside generated solver code are not "
                    "observable unless the code prints or re-raises them. API calls are "
                    "identified statically; runtime outcomes apply to each solver.py process "
                    "attempt, not to every internal Python function call."
                ),
            },
            "plan": planned,
            "guide_calls": self._guide_calls,
            "availability_checks": self._availability_checks,
            "attempts": self._attempts,
            "declared_executions": declared,
            "assessment": {
                "plan_alignment": plan_alignment,
                "semantic_selection_judgment": "requires_independent_review",
                "observed_package_ids": sorted(observed_ids),
                "unplanned_package_ids": unplanned,
                "missing_guide_before_use": missing_guides,
                "solver_attempt_count": len(self._attempts),
                "failed_attempt_count": len(failed),
                "errored_attempt_count": len(errored),
                "error_category_counts": dict(
                    sorted(Counter(item["error_category"] for item in errored).items())
                ),
                "solver_duration_ms": total_duration,
                "failed_attempt_duration_ms": failed_duration,
                "errored_attempt_duration_ms": errored_duration,
                "fallback_observed": fallback_observed,
                "fallback_reasons": fallback_reasons,
            },
        }

    @staticmethod
    def _planned_components(design: dict[str, Any] | None) -> list[dict[str, Any]]:
        algorithm = design.get("algorithm") if isinstance(design, dict) else None
        if isinstance(algorithm, dict):
            package = algorithm.get("package")
            return [
                {
                    "source": algorithm.get("source"),
                    "package_id": package.get("id") if isinstance(package, dict) else None,
                    "solver_id": algorithm.get("solver_id"),
                    "method": algorithm.get("method"),
                    "method_class": algorithm.get("method_class"),
                    "reason": algorithm.get("reason"),
                    "capability_mapping": algorithm.get("capability_mapping") or [],
                    "scale": design.get("scale"),
                }
            ]
        selection = design.get("selection") if isinstance(design, dict) else None
        components = selection.get("components") if isinstance(selection, dict) else None
        if not isinstance(components, list):
            return []
        result = []
        for component in components:
            if not isinstance(component, dict):
                continue
            package = component.get("package")
            result.append(
                {
                    "component_id": component.get("component_id"),
                    "role": component.get("role"),
                    "source": component.get("source"),
                    "package_id": package.get("id") if isinstance(package, dict) else None,
                    "solver_id": component.get("solver_id"),
                    "method": component.get("method"),
                    "method_class": component.get("method_class"),
                    "reason": component.get("reason"),
                    "capability_mapping": component.get("capability_mapping") or [],
                    "scale": design.get("scale") if isinstance(design, dict) else None,
                }
            )
        return result

    @staticmethod
    def _declared_executions(solver_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        execution = solver_result.get("execution") if isinstance(solver_result, dict) else None
        if isinstance(execution, dict):
            return [dict(execution)]
        executions = solver_result.get("executions") if isinstance(solver_result, dict) else None
        return [dict(item) for item in executions if isinstance(item, dict)] if isinstance(executions, list) else []
