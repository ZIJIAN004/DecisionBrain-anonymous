import json

import pytest

from decisionbrain.core.models import AgentStage, AgentState
from decisionbrain.runtime.workspace_state import WorkspaceStateError, reconstruct_agent_state


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_reconstructs_explanation_state_from_stage_outputs(tmp_path):
    (tmp_path / "problem.md").write_text("配送问题\n", encoding="utf-8")
    write_json(
        tmp_path / "stage_outputs/intake.json",
        {"decision": "proceed", "problem_definition": {"problem_type": "routing"}},
    )
    write_json(
        tmp_path / "stage_outputs/problem_contract.json",
        {"constraints": [{"name": "capacity"}]},
    )
    write_json(
        tmp_path / "stage_outputs/algorithm_design.json",
        {"selected_algorithm": "pyvrp"},
    )
    write_json(
        tmp_path / "stage_outputs/solving.json",
        {
            "decision": "solved",
            "result": {
                "solver_result_file": "solver_result.json",
                "code_file": "solver.py",
            },
        },
    )
    write_json(tmp_path / "solver_result.json", {"status": "feasible", "objective": 42})
    write_json(
        tmp_path / "feasibility_result.json",
        {
            "schema_valid": True,
            "feasible": True,
            "objective_value": 42,
            "violations": [],
            "warnings": [],
        },
    )
    write_json(
        tmp_path / "stage_outputs/feasibility_review.json",
        {
            "schema_version": "1.0",
            "decision": "accept",
            "responsibility": None,
            "summary": "验收通过",
            "evidence": [],
            "required_changes": [],
            "confidence": "high",
            "infeasibility_proof": "",
        },
    )
    (tmp_path / "solver.py").write_text("print(42)\n", encoding="utf-8")

    state = AgentState.from_dict(reconstruct_agent_state(tmp_path, AgentStage.EXPLANATION))

    assert state.problem_description == "配送问题"
    assert state.problem_definition == {"problem_type": "routing", "confirmed": True}
    assert state.problem_contract == {"constraints": [{"name": "capacity"}]}
    assert state.algorithm_design == {"selected_algorithm": "pyvrp"}
    assert state.solver_result["objective"] == 42
    assert state.feasibility_review["decision"] == "accept"
    assert state.generated_code == "print(42)\n"
    assert state.conversation == ()


def test_reconstruct_only_requires_outputs_upstream_of_resume_stage(tmp_path):
    (tmp_path / "problem.md").write_text("配送问题", encoding="utf-8")
    write_json(
        tmp_path / "stage_outputs/intake.json",
        {"decision": "proceed", "problem_definition": {"problem_type": "routing"}},
    )
    write_json(tmp_path / "stage_outputs/problem_contract.json", {"constraints": []})

    state = AgentState.from_dict(reconstruct_agent_state(tmp_path, AgentStage.ALGORITHM_DESIGN))

    assert state.problem_contract == {"constraints": []}
    assert state.algorithm_design is None


def test_reconstructs_runtime_timeout_for_feasibility_review_resume(tmp_path):
    (tmp_path / "problem.md").write_text("排程问题", encoding="utf-8")
    write_json(
        tmp_path / "stage_outputs/intake.json",
        {"decision": "proceed", "problem_definition": {"problem_type": "scheduling"}},
    )
    write_json(tmp_path / "stage_outputs/problem_contract.json", {"constraints": []})
    write_json(tmp_path / "stage_outputs/algorithm_design.json", {"selected_algorithm": "mip"})
    write_json(
        tmp_path / "stage_outputs/solving.json",
        {
            "decision": "solved",
            "result": {"runtime_solver_outcome_file": "runtime_solver_outcome.json"},
        },
    )
    write_json(
        tmp_path / "runtime_solver_outcome.json",
        {
            "schema_version": "1.0",
            "outcome_source": "runtime_external_timeout",
            "status": "runtime_timeout",
            "timeout_source": "runtime_watchdog",
            "timeout_seconds": 600,
            "elapsed_seconds": 600.1,
            "exit_code": -15,
            "termination_signal": "SIGTERM",
            "solver_file": "solver.py",
            "solver_execution_observed": True,
            "solution_detected": False,
            "solution_schema_valid": None,
            "solution_file": None,
            "stdout_tail": "",
        },
    )
    (tmp_path / "solver.py").write_text("# timed out\n", encoding="utf-8")

    state = AgentState.from_dict(
        reconstruct_agent_state(tmp_path, AgentStage.FEASIBILITY_REVIEW)
    )

    assert state.solver_result["outcome_source"] == "runtime_external_timeout"
    assert state.solver_result["runtime_solver_outcome_file"] == (
        "runtime_solver_outcome.json"
    )
    assert state.generated_code == "# timed out\n"


def test_reconstructs_confirmed_instance_infeasible_explanation_state(tmp_path):
    (tmp_path / "problem.md").write_text("排程问题", encoding="utf-8")
    write_json(
        tmp_path / "stage_outputs/intake.json",
        {"decision": "proceed", "problem_definition": {"problem_type": "scheduling"}},
    )
    write_json(tmp_path / "stage_outputs/problem_contract.json", {"constraints": []})
    write_json(tmp_path / "stage_outputs/algorithm_design.json", {"selected_algorithm": "mip"})
    write_json(
        tmp_path / "stage_outputs/solving.json",
        {
            "decision": "solved",
            "result": {
                "status": "infeasible",
                "solver_result_file": "solver_result.json",
                "code_file": "solver.py",
            },
        },
    )
    write_json(
        tmp_path / "solver_result.json",
        {"status": "infeasible", "infeasibility_proof": {"solver": "exact"}},
    )
    reviews = [
        {
            "decision": "reject",
            "responsibility": "instance",
            "confidence": "high",
            "summary": f"confirmation {index}",
            "infeasibility_proof": "complete exact model",
        }
        for index in range(3)
    ]
    write_json(
        tmp_path / "stage_outputs/feasibility_review.json",
        {**reviews[-1], "instance_infeasibility_reviews": reviews},
    )
    (tmp_path / "solver.py").write_text("# exact model\n", encoding="utf-8")

    state = AgentState.from_dict(reconstruct_agent_state(tmp_path, AgentStage.EXPLANATION))

    assert state.solver_result["status"] == "infeasible"
    assert state.feasibility_review["responsibility"] == "instance"
    assert len(state.instance_infeasibility_reviews) == 3


def test_reconstruct_reports_missing_required_stage_output(tmp_path):
    (tmp_path / "problem.md").write_text("配送问题", encoding="utf-8")

    with pytest.raises(WorkspaceStateError, match="intake.json"):
        reconstruct_agent_state(tmp_path, AgentStage.SOLVING)
