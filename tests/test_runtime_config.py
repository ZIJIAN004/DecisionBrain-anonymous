import json
from pathlib import Path

import pytest

from decisionbrain.config import Settings
from decisionbrain.core.models import AgentStage, AgentState, CoreConfig
from decisionbrain.core.package_policy import PackagePolicy
from decisionbrain.core.stage_agent import StageOutputValidationError
from decisionbrain.core.stage_agents import (
    AlgorithmDesignAgent,
    FeasibilityReviewAgent,
    SolvingAgent,
)
from decisionbrain.core.stage_agents.algorithm_design import _single_algorithm_catalog_summary
from decisionbrain.core.prompts import load_prompt_bundle
from decisionbrain.core.stage_output_models import (
    AlgorithmExecution,
    SingleAlgorithmDesignOutput,
    SingleSolverResultOutput,
)
from decisionbrain.runtime.agent_runtime import build_core_services


PROMPT_FILES = {
    "stage_common_system.txt": "stage common system",
    "algorithm_library_system.txt": "algorithm library system",
    "intake_system.txt": "intake system",
    "problem_contract_system.txt": "problem contract system",
    "algorithm_design_system.txt": "algorithm design system",
    "solving_system.txt": "model system",
    "feasibility_review_system.txt": "feasibility review system",
    "explain_system.txt": "explain system",
}
RUNTIME_PROMPT_FILES = {
    "intake_contract.txt": "intake contract",
    "problem_contract_contract.txt": "problem contract contract",
    "algorithm_design_contract.txt": "algorithm design contract",
    "solving_contract.txt": "model contract __TIME_BUDGET__",
    "feasibility_review_contract.txt": "feasibility review contract",
    "explain_contract.txt": "explain contract",
}


def write_prompt_tree(root: Path) -> None:
    sys_dir = root / "system"
    dev_dir = root / "developer"
    sys_dir.mkdir(parents=True)
    dev_dir.mkdir(parents=True)
    for name, content in PROMPT_FILES.items():
        (sys_dir / name).write_text(content, encoding="utf-8")
    for name, content in RUNTIME_PROMPT_FILES.items():
        (dev_dir / name).write_text(content, encoding="utf-8")


def test_build_core_services_maps_settings_to_core_config_and_prompts(tmp_path):
    prompts = tmp_path / "prompts"
    write_prompt_tree(prompts)
    settings = Settings(opt_prompts_dir=str(prompts))

    services = build_core_services(settings, tmp_path)

    assert services.config == CoreConfig(
        solver_timeout=settings.solver_timeout,
        prompts_dir=str(prompts),
        workspace_root=str(tmp_path),
        resume_poll_interval_seconds=settings.opt_agent_resume_poll_interval_seconds,
    )
    assert services.workspace_toolset.max_output_chars == settings.opt_workspace_max_output_chars
    assert services.workspace_toolset.max_list_entries == settings.opt_workspace_max_list_entries
    assert services.workspace_toolset.default_list_entries == (
        settings.opt_workspace_default_list_entries
    )
    assert services.workspace_toolset.max_read_bytes == settings.opt_workspace_max_read_bytes
    assert services.prompts.stage_common_system == "stage common system"
    assert services.prompts.algorithm_library_system == "algorithm library system"
    assert services.prompts.intake_system == "intake system"
    assert services.prompts.solving_contract == "model contract __TIME_BUDGET__"
    assert [tool.name for tool in services.algorithm_tools_for(AgentStage.ALGORITHM_DESIGN)] == [
        "get_algorithm_guide",
    ]
    messages = AlgorithmDesignAgent(services).build_messages(object()).messages
    assert "pyvrp.ils" in messages[-1]["content"]
    assert "rsome" in messages[-1]["content"]
    assert "alns" in messages[-1]["content"]
    assert "pyscipopt" in messages[-1]["content"]
    assert "上述内容是每个 package 的一句话摘要" in messages[-1]["content"]
    assert f'"solver_timeout_seconds": {settings.solver_timeout}' in messages[-1]["content"]
    assert '"source": "CoreConfig.solver_timeout"' in messages[-1]["content"]


def test_single_algorithm_prompts_and_schemas_remove_multi_solver_terms(tmp_path):
    prompts = load_prompt_bundle(
        Path(__file__).parents[1] / "prompts",
        components_enabled=False,
    )
    services = build_core_services(Settings(), tmp_path, components_enabled=False)
    state = AgentState(problem_description="test")
    assembled = []
    for agent in (
        AlgorithmDesignAgent(services),
        SolvingAgent(services),
        FeasibilityReviewAgent(services),
    ):
        messages = agent.build_messages(state)
        agent._inject_common_system_prompt(messages)
        assembled.extend(
            str(message.get("content") or "")
            for message in messages.messages
        )
    visible = "\n".join(
        (
            prompts.algorithm_design_system,
            prompts.algorithm_design_contract,
            prompts.solving_system,
            prompts.solving_contract,
            prompts.feasibility_review_contract,
            *assembled,
            str(SingleAlgorithmDesignOutput.model_json_schema()),
            str(SingleSolverResultOutput.model_json_schema()),
        )
    ).lower()
    for forbidden in (
        "component",
        "component_id",
        '"selection"',
        "selection.kind",
        "组件",
        "executions",
        '"role"',
        "subproblem",
        "relaxation",
        "repair_model",
        "子问题",
        "fallback",
    ):
        assert forbidden not in visible
    design_schema = str(SingleAlgorithmDesignOutput.model_json_schema()).lower()
    result_schema = str(SingleSolverResultOutput.model_json_schema()).lower()
    assert "fallback" not in design_schema
    assert "fallback_used" not in result_schema
    assert "deviation_reason" not in result_schema
    assert "每个 package 的一句话摘要" in prompts.algorithm_library_system
    assert '"scope": "full_problem"' in visible
    assert "不得使用任何分解策略" in visible
    assert "不得使用 warm start 或 mip start" in visible
    assert "启发式失败只能标记 unknown" in visible
    assert "shell timeout" in visible

    design_user_prompt = AlgorithmDesignAgent(services).build_messages(state).messages[-1][
        "content"
    ]
    injected_catalog = json.loads(
        design_user_prompt.split("【算法库目录】\n", 1)[1].split("\n- 上述内容是", 1)[0]
    )
    baseline_catalog = json.loads(services.algorithm_catalog_summary())
    assert injected_catalog["count"] == baseline_catalog["count"]
    assert [item["id"] for item in injected_catalog["algorithms"]] == [
        item["id"] for item in baseline_catalog["algorithms"]
    ]
    assert all(item.get("summary") for item in injected_catalog["algorithms"])
    assert all(
        solver.get("summary")
        for item in injected_catalog["algorithms"]
        for solver in item.get("solvers", [])
    )
    design = SingleAlgorithmDesignOutput.model_validate(
        {
            "schema_version": "1.0",
            "problem_family": "routing",
            "diagnosis": {
                "summary": "routing",
                "decision_type": "route",
                "key_difficulty": "capacity",
            },
            "scale": {
                "primary_items": 2,
                "resources": 1,
                "estimated_variables": 2,
                "estimated_constraints": 2,
                "risk": "low",
                "confidence": "high",
                "reason": "counted from data",
            },
            "candidates": [{
                "method": "greedy",
                "method_class": "constructive_heuristic",
                "source": "generated",
                "package_id": None,
                "solver_id": None,
                "fit_reason": "small input",
                "main_risk": "quality",
                "recommendation": "primary",
            }],
            "algorithm": {
                "source": "generated",
                "package": None,
                "solver_id": None,
                "method": "greedy",
                "method_class": "constructive_heuristic",
                "optimality": "heuristic_or_incumbent",
                "reason": "small input",
                "capability_mapping": [],
                "integration": None,
                "formulation": None,
            },
            "strategy": {
                "representation": "routes",
                "feasibility": "capacity checks",
                "objective": "distance",
                "construction": "greedy insertion",
                "improvement": "accept improving moves",
                "stopping": "time budget",
                "output": "routes",
            },
            "risks": [],
        }
    )
    assert design.algorithm.method == "greedy"

    formal_design_payload = design.model_dump(mode="json")
    formal_design_payload["algorithm"].update(
        {
            "method": "mixed_integer_linear_programming",
            "method_class": "exact",
            "optimality": "exact_or_incumbent",
            "formulation": {
                "method": "mixed_integer_linear_programming",
                "scope": "subproblem",
                "summary": "partial model",
                "variables": ["x"],
                "objective": "minimize x",
                "constraints": ["x >= 0"],
                "big_m_notes": "",
            },
        }
    )
    with pytest.raises(ValueError, match="full_problem"):
        SingleAlgorithmDesignOutput.model_validate(formal_design_payload)

    package_design_payload = design.model_dump(mode="json")
    package_design_payload["algorithm"] = {
        "source": "package",
        "package": {"id": "example", "version": "1.2.3"},
        "solver_id": "greedy_solver",
        "method": "greedy",
        "method_class": "constructive_heuristic",
        "optimality": "heuristic_or_incumbent",
        "reason": "guide-backed fit",
        "capability_mapping": [
            {
                "business_requirement": "capacity",
                "manifest_capability": "capacitated routing",
                "evidence": "solver guide",
            }
        ],
        "integration": {
            "primary_api": "example.solve",
            "input_mapping": "route input",
            "result_mapping": "route output",
            "runtime_control": "one second",
            "random_seed": 0,
            "availability_check": "import and version checked",
        },
        "formulation": None,
    }
    package_design = SingleAlgorithmDesignOutput.model_validate(package_design_payload)
    assert package_design.algorithm.package is not None
    assert package_design.algorithm.package.version == "1.2.3"

    result = SingleSolverResultOutput.model_validate(
        {
            "status": "feasible",
            "objective": 10,
            "mip_gap": None,
            "optimality": "heuristic_or_incumbent",
            "input_file": "input.json",
            "solution_file": "solution.json",
            "code_file": "solver.py",
            "feasibility_result_file": None,
            "execution": {
                "source": "generated",
                "planned_package_id": None,
                "planned_solver_id": None,
                "executed_package_id": None,
                "executed_solver_id": None,
                "method": "greedy",
                "method_class": "constructive_heuristic",
                "version": None,
                "guide_consulted": False,
                "availability_check": {"ok": True, "detail": "not applicable"},
                "seed": None,
                "runtime_limit_seconds": 1,
            },
            "solution_summary": {},
            "constraint_check": [],
            "validation": {
                "schema_check": {"ok": True, "issues": []},
                "business_semantic_check": {"ok": True, "issues": []},
                "algorithm_mapping_check": {
                    "ok": True,
                    "issues": [],
                    "mapped_capabilities": [],
                },
                "implementation_check": {"ok": True, "issues": []},
            },
            "diagnosis": "",
            "infeasibility_proof": None,
        }
    )
    assert result.execution.method == "greedy"


def test_no_review_solving_guide_rule_uses_package_components(tmp_path):
    services = build_core_services(Settings(), tmp_path, feasibility_review_enabled=False)
    messages = SolvingAgent(services, feasibility_review_enabled=False).build_messages(
        AgentState(problem_description="test")
    )
    visible = "\n".join(str(message["content"]) for message in messages.messages)

    assert "selection.components 和 fallback.components" in visible
    assert "source=package" in visible
    assert "selection.kind=library" not in visible


def test_gurobi_formulator_arm_prompts_require_complete_model(tmp_path):
    policy = PackagePolicy(pool="gurobi-only")
    services = build_core_services(
        Settings(),
        tmp_path,
        algorithm_design_enabled=False,
        components_enabled=False,
        package_policy=policy,
    )
    solving = SolvingAgent(
        services,
        algorithm_design_enabled=False,
    ).build_messages(AgentState(problem_description="test"))
    review = FeasibilityReviewAgent(
        services,
        algorithm_design_enabled=False,
    ).build_messages(
        AgentState(
            problem_description="test",
            solver_result={"status": "time_limit", "solution_file": None},
        )
    )
    visible_solving = "\n".join(str(item["content"]) for item in solving.messages)
    visible_review = "\n".join(str(item["content"]) for item in review.messages)

    assert "覆盖全部业务变量、硬约束、完整目标和完整业务结果映射" in visible_solving
    assert "Gurobi Formulator" in visible_solving
    assert "完整业务结果映射" in visible_solving
    assert "stage_outputs/algorithm_design.json" not in visible_solving
    assert "Algorithm Design" not in visible_solving
    assert "algorithm_design" not in visible_solving
    assert "算法设计" not in visible_solving

    assert "Gurobi Formulator 的完整模型" in visible_review
    assert "responsibility 只能是 instance、gurobi_formulator 或 solving" in visible_review
    assert "stage_outputs/algorithm_design.json" not in visible_review
    assert "Algorithm Design" not in visible_review
    assert "algorithm_design" not in visible_review
    assert "算法设计" not in visible_review


def test_gurobi_formulator_review_rejects_design_responsibility(tmp_path):
    services = build_core_services(
        Settings(),
        tmp_path,
        algorithm_design_enabled=False,
        components_enabled=False,
        package_policy=PackagePolicy(pool="gurobi-only"),
    )
    agent = FeasibilityReviewAgent(services, algorithm_design_enabled=False)
    verdict = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "algorithm_design",
        "summary": "model defect",
        "evidence": [{"source": "solver.py", "finding": "missing constraint"}],
        "required_changes": ["add the missing business constraint"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["constraint_coverage"],
            "change_ids": ["add_constraint"],
            "problem_requirement": "all hard constraints must be modeled",
            "observed_solution_behavior": "candidate violates a hard constraint",
            "responsibility_reason": "the complete model omitted a constraint",
            "required_changes": ["add the missing business constraint"],
        },
    }
    state = AgentState(
        problem_description="test",
        solver_result={"status": "time_limit", "solution_file": None},
    )

    with pytest.raises(StageOutputValidationError) as exc_info:
        agent.parse_output_file(verdict, state)
    assert any(
        issue.message == "该工作流不存在 algorithm_design；应归 gurobi_formulator"
        for issue in exc_info.value.issues
    )


@pytest.mark.parametrize(
    "disabled",
    (
        {"input_schema_enabled": False},
        {"algorithm_library_enabled": False},
        {"feasibility_review_enabled": False},
        {"algorithm_design_enabled": False},
        {"problem_contract_enabled": False},
    ),
)
def test_single_algorithm_ablation_rejects_crossed_workflow_ablations(disabled):
    if "algorithm_design_enabled" in disabled:
        # The no-design arm intentionally combines with no-components; the
        # benchmark runner additionally requires gurobi-only for that arm.
        return
    with pytest.raises(ValueError, match="single-factor experiment"):
        load_prompt_bundle(
            Path(__file__).parents[1] / "prompts",
            components_enabled=False,
            **disabled,
        )


def test_single_algorithm_execution_forbids_fallback_and_enforces_alignment():
    base = {
        "source": "package",
        "planned_package_id": "example",
        "planned_solver_id": "primary",
        "executed_package_id": "example",
        "executed_solver_id": "primary",
        "method": "greedy",
        "method_class": "constructive_heuristic",
        "version": "1.2.3",
        "guide_consulted": True,
        "availability_check": {"ok": True, "detail": "checked"},
        "seed": 0,
        "runtime_limit_seconds": 1,
    }
    execution = AlgorithmExecution.model_validate(base)
    assert execution.executed_solver_id == "primary"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AlgorithmExecution.model_validate({**base, "fallback_used": True})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AlgorithmExecution.model_validate({**base, "deviation_reason": "changed"})
    with pytest.raises(ValueError, match="package must match planned package"):
        AlgorithmExecution.model_validate({**base, "executed_package_id": "other"})
    with pytest.raises(ValueError, match="solver must match planned solver"):
        AlgorithmExecution.model_validate({**base, "executed_solver_id": "other"})


def test_single_algorithm_catalog_sanitizes_package_and_solver_summaries():
    catalog = {
        "count": 1,
        "algorithms": [
            {
                "id": "component_example",
                "summary": "Package component summary.",
                "solvers": [
                    {"id": "solver", "summary": "Uses two components internally."}
                ],
            }
        ],
    }

    rendered = _single_algorithm_catalog_summary(json.dumps(catalog))

    sanitized = json.loads(rendered)
    assert sanitized["algorithms"][0]["id"] == "component_example"
    assert "component" not in sanitized["algorithms"][0]["summary"].lower()
    assert "component" not in sanitized["algorithms"][0]["solvers"][0]["summary"].lower()
    assert sanitized["algorithms"][0]["solvers"][0]["id"] == "solver"
