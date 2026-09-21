"""Canonical workspace files produced by each StageAgent stage."""

from __future__ import annotations

from .models import AgentStage


STAGE_OUTPUT_FILES: dict[AgentStage, str] = {
    AgentStage.INTAKE: "stage_outputs/intake.json",
    AgentStage.PROBLEM_CONTRACT: "stage_outputs/problem_contract.json",
    AgentStage.ALGORITHM_DESIGN: "stage_outputs/algorithm_design.json",
    AgentStage.GUROBI_FORMULATOR: "stage_outputs/gurobi_formulation.json",
    AgentStage.SOLVING: "stage_outputs/solving.json",
    AgentStage.FEASIBILITY_REVIEW: "stage_outputs/feasibility_review.json",
    AgentStage.EXPLANATION: "stage_outputs/explanation.json",
}

FEASIBILITY_HANDOFF_DIR = "feasibility_handoffs"
FEASIBILITY_REVIEW_HISTORY_FILE = "feasibility_review_history.json"


def feasibility_handoff_dir(target: str, attempt_index: int) -> str:
    """Workspace-relative directory exposing a stage-safe rejected-candidate view."""

    if target not in {"algorithm_design", "gurobi_formulator", "solving"}:
        raise ValueError(f"unsupported feasibility handoff target: {target!r}")
    return f"{FEASIBILITY_HANDOFF_DIR}/{target}/attempt{attempt_index}"


PROBLEM_CONTRACT_INPUT_SCHEMA_FILE = "input_schema.json"
PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE = "solution_schema.json"
PROBLEM_CONTRACT_CHECKER_FILE = "feasibility_checker.py"
FIXED_SOLUTION_CONTRACT_FILE = "fixed_solution_contract.json"
SOLVING_INPUT_FILE = "input.json"
SOLVING_INPUT_BUILDER_FILE = "from_data_to_input.py"
SOLVING_CODE_FILE = "solver.py"
SOLVING_SOLUTION_FILE = "solution.json"
SOLVING_RESULT_FILE = "solver_result.json"
SOLVING_RUNTIME_OUTCOME_FILE = "runtime_solver_outcome.json"
SOLVING_FEASIBILITY_RESULT_FILE = "feasibility_result.json"
# Runtime-owned receipt binding feasibility_result.json to one checker execution.
SOLVING_FEASIBILITY_RECEIPT_FILE = "feasibility_check_receipt.json"
# Runtime-owned receipt proving one solver process execution.
SOLVING_EXECUTION_RECEIPT_FILE = "solver_execution_receipt.json"
