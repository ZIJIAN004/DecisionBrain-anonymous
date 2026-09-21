"""Tests for CLI ``dbn resume`` command and resume-related core logic."""

import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from decisionbrain.cli.app import app
from decisionbrain.cli.commands.resume import _validate_stage_for_parent
from decisionbrain.core.agent import AgentProtocolError, OptimizationAgent
from decisionbrain.core.models import AgentStage
from decisionbrain.core.models import AgentState
from decisionbrain.core.stage_agents.solving import SolvingAgent
from decisionbrain.core.stage_agents.algorithm_design import AlgorithmDesignAgent
from decisionbrain.core.workspace_tools import reset_workspace_for_stage
from decisionbrain.exceptions import UserInputError
from decisionbrain.runtime.workspace_state import reconstruct_agent_state
from helpers import fake_core_services, fake_workspace_toolset

runner = CliRunner()


def test_formulator_resume_requires_parent_workflow_marker():
    with pytest.raises(UserInputError, match="did not enable the gurobi-formulator"):
        _validate_stage_for_parent("gurobi_formulator", {"workflow": "standard"})
    _validate_stage_for_parent(
        "gurobi_formulator",
        {"workflow": "gurobi-formulator", "gurobi_formulator_enabled": True},
    )

# ============================================================================
# Core: restore_state_for_stage()
# ============================================================================


class TestRestoreStateForStage:
    def test_disabled_upstream_stages_are_not_required_for_solving_resume(self, tmp_path):
        (tmp_path / "problem.md").write_text("test problem", encoding="utf-8")
        outputs = tmp_path / "stage_outputs"
        outputs.mkdir()
        (outputs / "intake.json").write_text(
            '{"decision":"proceed","problem_definition":{"objective":"minimize"}}',
            encoding="utf-8",
        )
        restored = reconstruct_agent_state(
            tmp_path,
            AgentStage.SOLVING,
            problem_contract_enabled=False,
            algorithm_design_enabled=False,
        )
        assert restored["problem_contract"] is None
        assert restored["algorithm_design"] is None

    def test_solving_prompt_omits_fixed_schema_when_contract_is_disabled(self, tmp_path):
        services = fake_core_services(workspace_toolset=fake_workspace_toolset(tmp_path))
        agent = SolvingAgent(services, problem_contract_enabled=False)
        request = agent.build_messages(AgentState(problem_description="test")).build()
        prompt_text = "\n".join(
            str(message.get("content") or "") for message in request.messages
        )
        assert "fixed_solution_contract.json" not in prompt_text
        assert "raw solution format" in prompt_text
        assert "solution_schema.json" not in prompt_text

    def test_algorithm_design_prompt_names_fixed_schema_when_contract_is_disabled(self):
        agent = AlgorithmDesignAgent(
            fake_core_services(), problem_contract_enabled=False
        )
        request = agent.build_messages(AgentState(problem_description="test")).build()
        prompt_text = "\n".join(
            str(message.get("content") or "") for message in request.messages
        )
        assert "fixed_solution_contract.json" in prompt_text
        assert "solution_schema.json" not in prompt_text

    def test_restore_from_intake_clears_all_downstream(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "algorithm_design": {"approach": "greedy"},
            "generated_code": "print(1)",
            "solver_result": {"status": "optimal"},
            "explanation": {"summary": "done"},
        }
        agent.restore_state_for_stage(state, AgentStage.INTAKE)
        current = agent._state  # type: ignore[attr-defined]
        assert current.current_stage is AgentStage.INTAKE
        assert current.problem_definition is None
        assert current.problem_contract is None
        assert current.algorithm_design is None
        assert current.generated_code == ""
        assert current.solver_result is None
        assert current.explanation is None

    def test_restore_from_algorithm_design_keeps_upstream(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "algorithm_design": {"approach": "greedy"},
            "generated_code": "print(1)",
            "solver_result": {"status": "optimal"},
            "explanation": {"summary": "done"},
        }
        agent.restore_state_for_stage(state, AgentStage.ALGORITHM_DESIGN)
        current = agent._state
        assert current.current_stage is AgentStage.ALGORITHM_DESIGN
        assert current.problem_definition is not None
        assert current.problem_contract is not None
        assert current.algorithm_design is None  # cleared (target)
        assert current.generated_code == ""  # cleared (downstream)
        assert current.solver_result is None  # cleared (downstream)

    def test_restore_from_solving_keeps_contract_and_design(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "algorithm_design": {"approach": "greedy"},
            "generated_code": "print(1)",
            "solver_result": {"status": "optimal"},
        }
        agent.restore_state_for_stage(state, AgentStage.SOLVING)
        current = agent._state
        assert current.problem_definition is not None
        assert current.problem_contract is not None
        assert current.algorithm_design is not None

    def test_restore_from_solving_missing_contract_fails(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "algorithm_design": {"approach": "greedy"},
        }
        with pytest.raises(AgentProtocolError, match="missing problem_contract"):
            agent.restore_state_for_stage(state, AgentStage.SOLVING)

    def test_restore_from_solving_missing_algorithm_design_fails(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
        }
        with pytest.raises(AgentProtocolError, match="missing algorithm_design"):
            agent.restore_state_for_stage(state, AgentStage.SOLVING)

    def test_restore_from_explanation_missing_solver_result_fails(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "algorithm_design": {"approach": "greedy"},
        }
        with pytest.raises(AgentProtocolError, match="missing solver_result"):
            agent.restore_state_for_stage(state, AgentStage.EXPLANATION)

    def test_restore_from_problem_contract_needs_problem_definition(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {"problem_description": "测试"}
        with pytest.raises(AgentProtocolError, match="missing problem_definition"):
            agent.restore_state_for_stage(state, AgentStage.PROBLEM_CONTRACT)

    def test_restore_clears_clarification_state(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "pending_clarification_questions": [{"id": "q1", "text": "?"}],
            "current_stage": None,
        }
        agent.restore_state_for_stage(state, AgentStage.ALGORITHM_DESIGN)
        current = agent._state
        assert current.current_stage is AgentStage.ALGORITHM_DESIGN
        assert current.pending_clarification_questions == ()

    def test_restore_from_explanation_with_all_deps_succeeds(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
            "algorithm_design": {"approach": "greedy"},
            "solver_result": {"status": "optimal"},
            "feasibility_review": {"decision": "accept"},
            "explanation": {"summary": "old"},
        }
        agent.restore_state_for_stage(state, AgentStage.EXPLANATION)
        current = agent._state
        assert current.current_stage is AgentStage.EXPLANATION
        assert current.explanation is None  # cleared

    def test_stage_str_input(self):
        agent = OptimizationAgent(services=fake_core_services())
        state = {
            "problem_description": "测试",
            "problem_definition": {"confirmed": True},
            "problem_contract": {"constraints": []},
        }
        agent.restore_state_for_stage(state, "algorithm_design")
        current = agent._state
        assert current.current_stage is AgentStage.ALGORITHM_DESIGN


# ============================================================================
# Core: reset_workspace_for_stage()
# ============================================================================


class TestResetWorkspaceForStage:
    def test_intake_deletes_all_stage_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # create stage outputs
            (root / "stage_outputs").mkdir()
            for fname in (
                "intake.json",
                "problem_contract.json",
                "algorithm_design.json",
                "solving.json",
                "feasibility_review.json",
                "explanation.json",
            ):
                (root / "stage_outputs" / fname).write_text("{}")
            (root / "input_schema.json").write_text("{}")
            (root / "solution_schema.json").write_text("{}")
            (root / "feasibility_checker.py").write_text("# check")
            (root / "from_data_to_input.py").write_text("# stale mapping")
            (root / "solver.py").write_text("# code")
            (root / "solution.json").write_text("{}")
            (root / "solver_result.json").write_text("{}")

            reset_workspace_for_stage(root, AgentStage.INTAKE)

            for fname in (
                "intake.json",
                "problem_contract.json",
                "algorithm_design.json",
                "solving.json",
                "feasibility_review.json",
                "explanation.json",
            ):
                assert not (root / "stage_outputs" / fname).exists(), f"{fname} should be deleted"
            assert not (root / "input_schema.json").exists()
            assert not (root / "solution_schema.json").exists()
            assert not (root / "feasibility_checker.py").exists()
            assert not (root / "from_data_to_input.py").exists()
            assert not (root / "solver.py").exists()
            assert not (root / "solution.json").exists()
            assert not (root / "solver_result.json").exists()

    def test_problem_contract_deletes_all_downstream_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage_outputs").mkdir()
            for fname in (
                "intake.json",
                "problem_contract.json",
                "algorithm_design.json",
                "solving.json",
                "feasibility_review.json",
                "explanation.json",
            ):
                (root / "stage_outputs" / fname).write_text("{}")
            for fname in (
                "input_schema.json",
                "solution_schema.json",
                "feasibility_checker.py",
                "from_data_to_input.py",
                "input.json",
                "solver.py",
                "solution.json",
                "solver_result.json",
                "feasibility_result.json",
            ):
                (root / fname).write_text("{}")

            reset_workspace_for_stage(root, AgentStage.PROBLEM_CONTRACT)

            assert (root / "stage_outputs" / "intake.json").exists()
            for fname in (
                "problem_contract.json",
                "algorithm_design.json",
                "solving.json",
                "feasibility_review.json",
                "explanation.json",
            ):
                assert not (root / "stage_outputs" / fname).exists()
            for fname in (
                "input_schema.json",
                "solution_schema.json",
                "feasibility_checker.py",
                "from_data_to_input.py",
                "input.json",
                "solver.py",
                "solution.json",
                "solver_result.json",
                "feasibility_result.json",
            ):
                assert not (root / fname).exists()

    def test_algorithm_design_preserves_problem_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage_outputs").mkdir()
            (root / "stage_outputs" / "problem_contract.json").write_text("{}")
            (root / "stage_outputs" / "algorithm_design.json").write_text("{}")
            (root / "stage_outputs" / "explanation.json").write_text("{}")
            (root / "from_data_to_input.py").write_text("# stale mapping")
            (root / "solver.py").write_text("# code")

            reset_workspace_for_stage(root, AgentStage.ALGORITHM_DESIGN)

            assert (root / "stage_outputs" / "problem_contract.json").exists()
            assert not (root / "stage_outputs" / "algorithm_design.json").exists()
            assert not (root / "stage_outputs" / "explanation.json").exists()
            assert not (root / "from_data_to_input.py").exists()
            assert not (root / "solver.py").exists()

    def test_solving_deletes_input_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "from_data_to_input.py").write_text("# stale mapping")
            (root / "input.json").write_text("{}")

            reset_workspace_for_stage(root, AgentStage.SOLVING)

            assert not (root / "from_data_to_input.py").exists()
            assert not (root / "input.json").exists()

    def test_explanation_only_deletes_explanation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stage_outputs").mkdir()
            (root / "stage_outputs" / "algorithm_design.json").write_text("{}")
            (root / "stage_outputs" / "explanation.json").write_text("{}")
            (root / "solver_result.json").write_text("{}")

            reset_workspace_for_stage(root, AgentStage.EXPLANATION)

            assert (root / "stage_outputs" / "algorithm_design.json").exists()
            assert (root / "solver_result.json").exists()
            assert not (root / "stage_outputs" / "explanation.json").exists()

    def test_noop_on_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # should not raise
            reset_workspace_for_stage(root, AgentStage.INTAKE)


# ============================================================================
# CLI: run_id resolution
# ============================================================================


class TestRunIdResolution:
    def test_resolve_cwd_as_run_dir(self, tmp_path: Path):
        """When cwd name matches run_id and contains run.json, resolve to cwd."""
        run_dir = tmp_path / "20250101-120000Z-aaaaaaaa"
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}")

        from decisionbrain.cli.commands.resume import _resolve_run_id

        old_cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            result = _resolve_run_id("20250101-120000Z-aaaaaaaa", tmp_path)
            assert result.resolve() == run_dir.resolve()
        finally:
            os.chdir(old_cwd)

    def test_resolve_cwd_contains_run_id(self, tmp_path: Path):
        """When cwd contains RUN_ID/ directory, resolve to it."""
        cwd = tmp_path / "repo"
        cwd.mkdir()
        run_dir = cwd / "20250101-120000Z-bbbbbbbb"
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}")

        from decisionbrain.cli.commands.resume import _resolve_run_id

        old_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            result = _resolve_run_id("20250101-120000Z-bbbbbbbb", tmp_path)
            assert result.resolve() == run_dir.resolve()
        finally:
            os.chdir(old_cwd)

    def test_resolve_cwd_contains_runs_dir(self, tmp_path: Path):
        """When cwd contains runs/RUN_ID/ directory, resolve to it."""
        cwd = tmp_path / "repo"
        cwd.mkdir()
        run_dir = cwd / "runs" / "20250101-120000Z-cccccccc"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}")

        from decisionbrain.cli.commands.resume import _resolve_run_id

        old_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            result = _resolve_run_id("20250101-120000Z-cccccccc", tmp_path)
            assert result.resolve() == run_dir.resolve()
        finally:
            os.chdir(old_cwd)

    def test_resolve_fallback_to_runs_dir(self, tmp_path: Path):
        """Fall back to DBN_RUNS_DIR/RUN_ID."""
        run_dir = tmp_path / "20250101-120000Z-dddddddd"
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}")

        from decisionbrain.cli.commands.resume import _resolve_run_id

        result = _resolve_run_id("20250101-120000Z-dddddddd", tmp_path)
        assert result.resolve() == run_dir.resolve()

    def test_resolve_not_found_raises(self, tmp_path: Path):
        from decisionbrain.cli.commands.resume import _resolve_run_id
        from decisionbrain.exceptions import UserInputError

        with pytest.raises(UserInputError, match="Run not found"):
            _resolve_run_id("nonexistent-run", tmp_path)


# ============================================================================
# CLI: command integration (basic smoke)
# ============================================================================


class TestResumeCommandSmoke:
    def test_help_shows_resume(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "resume" in result.stdout

    def test_invalid_stage_rejected(self):
        result = runner.invoke(
            app,
            ["resume", "some-run-id", "invalid_stage"],
        )
        assert result.exit_code != 0

    def test_missing_run_rejected(self):
        result = runner.invoke(
            app,
            ["resume", "nonexistent-run-id", "algorithm_design"],
        )
        assert result.exit_code != 0

    def test_json_flag_in_help(self):
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.stdout
