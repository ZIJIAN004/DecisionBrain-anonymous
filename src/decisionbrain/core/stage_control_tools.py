"""Shared non-terminal control tools for StageAgent loops."""

from __future__ import annotations

from typing import Any


REPORT_PROGRESS_TOOL_NAME = "report_progress"

STAGE_EVENT_PROGRESS_REPORTED = "progress_reported"


class StageControlToolError(ValueError):
    """A user-facing validation error for stage control tool arguments."""


REPORT_PROGRESS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage": {
            "type": "string",
            "description": (
                "Current agent stage, such as intake, problem_contract, "
                "algorithm_design, solving, explanation."
            ),
        },
        "summary": {
            "type": "string",
            "description": "A concise useful progress summary for the user.",
        },
        "completed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important work already completed.",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important facts discovered so far.",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Current assumptions that may affect modeling or solving.",
        },
        "next_step": {
            "type": "string",
            "description": "The next intended action.",
        },
    },
    "required": [
        "stage",
        "summary",
        "completed",
        "key_findings",
        "assumptions",
        "next_step",
    ],
    "additionalProperties": False,
}


def normalize_report_progress_args(args: dict[str, Any]) -> dict[str, Any]:
    allowed = set(REPORT_PROGRESS_PARAMETERS["properties"])
    _reject_unknown(args, allowed)
    return {
        "stage": _require_string(args, "stage"),
        "summary": _require_string(args, "summary"),
        "completed": _require_string_list(args, "completed"),
        "key_findings": _require_string_list(args, "key_findings"),
        "assumptions": _require_string_list(args, "assumptions"),
        "next_step": _require_string(args, "next_step"),
    }


def _reject_unknown(args: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise StageControlToolError(f"unexpected field(s): {names}")


def _require_string(args: dict[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise StageControlToolError(f"{name} must be a non-empty string")
    return value.strip()


def _require_string_list(args: dict[str, Any], name: str) -> list[str]:
    value = args.get(name)
    if not isinstance(value, list):
        raise StageControlToolError(f"{name} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise StageControlToolError(f"{name}[{index}] must be a string")
        result.append(item)
    return result
