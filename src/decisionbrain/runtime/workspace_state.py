"""Reconstruct resumable AgentState from canonical workspace outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.models import AgentStage, AgentState
from ..core.stage_outputs import (
    SOLVING_RESULT_FILE,
    SOLVING_RUNTIME_OUTCOME_FILE,
    STAGE_OUTPUT_FILES,
)


class WorkspaceStateError(ValueError):
    """Workspace files cannot provide the upstream state required for resume."""


def _read_required_text(workspace: Path, relative_path: str) -> str:
    path = workspace / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkspaceStateError(f"Missing or unreadable `{relative_path}`: {exc}") from exc


def _read_required_json(workspace: Path, relative_path: str) -> dict[str, Any]:
    text = _read_required_text(workspace, relative_path)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkspaceStateError(f"`{relative_path}` is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise WorkspaceStateError(f"`{relative_path}` must contain a top-level JSON object")
    return value


def reconstruct_agent_state(
    workspace: Path,
    resume_stage: AgentStage,
    *,
    problem_contract_enabled: bool = True,
    algorithm_design_enabled: bool = True,
) -> dict[str, Any]:
    """Build the minimal upstream AgentState needed to restart ``resume_stage``."""

    root = workspace.expanduser().resolve()
    problem_description = _read_required_text(root, "problem.md").strip()
    state = AgentState(problem_description=problem_description)

    if resume_stage is AgentStage.INTAKE:
        return state.to_dict()

    intake = _read_required_json(root, STAGE_OUTPUT_FILES[AgentStage.INTAKE])
    if str(intake.get("decision") or "") != "proceed":
        raise WorkspaceStateError(
            "`stage_outputs/intake.json` has not proceeded; cannot resume after Intake"
        )
    problem_definition = intake.get("problem_definition")
    if not isinstance(problem_definition, dict):
        raise WorkspaceStateError("`stage_outputs/intake.json` is missing problem_definition")
    confirmed_definition = dict(problem_definition)
    confirmed_definition["confirmed"] = True
    state = state.with_updates(problem_definition=confirmed_definition)

    if resume_stage is AgentStage.PROBLEM_CONTRACT:
        return state.to_dict()

    if problem_contract_enabled:
        problem_contract = _read_required_json(
            root, STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT]
        )
        state = state.with_updates(problem_contract=problem_contract)

    if resume_stage in {AgentStage.ALGORITHM_DESIGN, AgentStage.GUROBI_FORMULATOR}:
        return state.to_dict()

    if algorithm_design_enabled:
        algorithm_design = _read_required_json(
            root, STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN]
        )
        state = state.with_updates(algorithm_design=algorithm_design)

    if resume_stage is AgentStage.SOLVING:
        return state.to_dict()

    solving = _read_required_json(root, STAGE_OUTPUT_FILES[AgentStage.SOLVING])
    if isinstance(solving.get("result"), dict):
        result = dict(solving["result"])
    else:
        result = {
            key: value
            for key, value in solving.items()
            if key not in {"decision", "message", "questions"}
        }
    runtime_result_path = result.get("runtime_solver_outcome_file")
    if runtime_result_path == SOLVING_RUNTIME_OUTCOME_FILE:
        solver_result_path = SOLVING_RUNTIME_OUTCOME_FILE
    else:
        solver_result_path = str(result.get("solver_result_file") or SOLVING_RESULT_FILE)
    solver_result = _read_required_json(root, solver_result_path)
    merged_result = dict(solver_result)
    merged_result.update(result)
    generated_code = ""
    code_path = str(merged_result.get("code_file") or "solver.py")
    code_file = root / code_path
    if code_file.is_file():
        generated_code = code_file.read_text(encoding="utf-8")
    state = state.with_updates(
        generated_code=generated_code,
        solver_result=merged_result,
    )
    if resume_stage is AgentStage.FEASIBILITY_REVIEW:
        return state.to_dict()

    if merged_result.get("status") != "infeasible":
        feasibility_result_path = str(
            merged_result.get("feasibility_result_file") or "feasibility_result.json"
        )
        merged_result["feasibility_result_file"] = feasibility_result_path
        merged_result["feasibility_check"] = _read_required_json(root, feasibility_result_path)
    state = state.with_updates(solver_result=merged_result)
    feasibility_review = _read_required_json(
        root, STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]
    )
    persisted_instance_reviews = feasibility_review.pop("instance_infeasibility_reviews", [])
    instance_confirmed = (
        feasibility_review.get("decision") == "reject"
        and feasibility_review.get("responsibility") == "instance"
        and isinstance(persisted_instance_reviews, list)
        and len(persisted_instance_reviews) >= 3
        and all(
            isinstance(item, dict)
            and item.get("decision") == "reject"
            and item.get("responsibility") == "instance"
            and item.get("confidence") in {"medium", "high"}
            for item in persisted_instance_reviews[-3:]
        )
    )
    if str(feasibility_review.get("decision") or "") != "accept" and not instance_confirmed:
        raise WorkspaceStateError(
            "`stage_outputs/feasibility_review.json` is not accepted and lacks three trusted "
            "instance-infeasibility confirmations; cannot resume at Explanation"
        )
    state = state.with_updates(
        feasibility_review=feasibility_review,
        instance_infeasibility_reviews=(
            tuple(persisted_instance_reviews) if instance_confirmed else ()
        ),
    )
    return state.to_dict()
