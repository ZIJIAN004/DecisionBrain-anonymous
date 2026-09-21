from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from decisionbrain.benchmark.runner import FrontierORReport
from decisionbrain.benchmark.runner import run_benchmark
from decisionbrain.config import Settings
from decisionbrain.core.package_policy import PackagePolicy
from decisionbrain.core.models import AgentStage, AgentState, StageCompleted
from decisionbrain.core.stage_flow import apply_stage_result
from decisionbrain.core.stage_output_models import GurobiFormulationOutput


def _formulation(**updates):
    payload = {
        "formulation_type": "milp",
        "variables": [{"name": "x", "type": "binary"}],
        "parameters": [],
        "constraints": [{"name": "cover", "expression": "x >= 1"}],
        "objective": {"sense": "min", "expression": "x"},
        "data_mapping": [{"source": "data/items", "target": "x index"}],
        "solution_mapping": [{"source": "x", "target": "selected"}],
        "completeness_checklist": ["all hard constraints covered"],
    }
    payload.update(updates)
    return payload


def test_formulator_contract_accepts_complete_model_and_rejects_search_strategy():
    assert GurobiFormulationOutput.model_validate(_formulation()).formulation_type == "milp"
    with pytest.raises(ValidationError, match="启发式"):
        GurobiFormulationOutput.model_validate(
            _formulation(objective={"sense": "min", "expression": "x", "fallback": "greedy"})
        )


def test_problem_contract_advances_to_formulator_only_in_formulator_workflow():
    state = AgentState(problem_description="x", current_stage=AgentStage.PROBLEM_CONTRACT)
    result = StageCompleted(
        stage=AgentStage.PROBLEM_CONTRACT,
        data={"problem_contract": {"constraints": []}},
    )
    arm_a = apply_stage_result(
        state,
        result,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=True,
    )
    ordinary_no_design = apply_stage_result(
        state,
        result,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=False,
    )
    assert arm_a.current_stage is AgentStage.GUROBI_FORMULATOR
    assert ordinary_no_design.current_stage is AgentStage.SOLVING


def test_review_model_responsibility_rewinds_and_clears_formulation():
    state = AgentState(
        problem_description="x",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        gurobi_formulation=_formulation(),
        solver_result={"status": "unknown", "solution_file": None},
    )
    result = StageCompleted(
        stage=AgentStage.FEASIBILITY_REVIEW,
        data={
            "feasibility_review": {
                "decision": "reject",
                "responsibility": "gurobi_formulator",
                "summary": "missing capacity constraint",
                "required_changes": ["rebuild the complete model with capacity"],
                "remediation_handoff": {"summary": "repair modeling"},
            },
            "feasibility_check": None,
        },
    )
    updated = apply_stage_result(
        state,
        result,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=True,
    )
    assert updated.current_stage is AgentStage.GUROBI_FORMULATOR
    assert updated.gurobi_formulation is None
    assert updated.solver_result is None


def test_report_marks_formulator_workflow():
    now = datetime.now(timezone.utc)
    payload = FrontierORReport(
        evaluations=[],
        started_at=now,
        finished_at=now,
        model="test",
        jobs=8,
        workflow="gurobi-formulator",
        gurobi_formulator_enabled=True,
        solving_mode="translation-only",
    ).to_dict()
    assert payload["workflow"] == "gurobi-formulator"
    assert payload["gurobi_formulator_enabled"] is True
    assert payload["solving_mode"] == "translation-only"


def test_formulator_workflow_requires_explicit_marker_and_gurobi_only_pool():
    import asyncio

    kwargs = {"algorithm_design_enabled": False, "components_enabled": False}
    with pytest.raises(ValueError, match="requires workflow"):
        asyncio.run(
            run_benchmark([], Settings(), package_policy=PackagePolicy(pool="gurobi-only"), **kwargs)
        )
    with pytest.raises(ValueError, match="package-pool gurobi-only"):
        asyncio.run(
            run_benchmark([], Settings(), workflow="gurobi-formulator", **kwargs)
        )


def test_translation_contract_uses_runtime_time_budget_placeholder():
    from decisionbrain.paths import PROMPTS_DIR

    contract = (
        PROMPTS_DIR / "developer" / "solving_contract_gurobi_translation_only.txt"
    ).read_text(encoding="utf-8")
    assert "__TIME_BUDGET__" in contract
    assert "30 秒" not in contract


@pytest.mark.parametrize(
    "field, value",
    [
        ("completeness_checklist", ["先用启发式构造初始解"]),
        ("completeness_checklist", ["热启动后再求解"]),
        ("completeness_checklist", ["按分解求解各子问题"]),
        ("objective", {"sense": "min", "expression": "x", "note": "贪心排序后求解"}),
    ],
)
def test_formulator_contract_rejects_chinese_search_strategy_terms(field, value):
    """提示词与产物都是中文的，只匹配英文禁用词会被绕过。"""
    with pytest.raises(ValidationError) as excinfo:
        GurobiFormulationOutput.model_validate(_formulation(**{field: value}))

    assert "命中禁用词" in str(excinfo.value)


def test_formulator_stage_receives_feasibility_repair_directive():
    """Review 归责回退到 Formulator 后，重跑必须看到上一轮被拒的原因。"""
    from decisionbrain.core.feasibility_routing import (
        build_feasibility_directive,
        render_for_executor,
    )

    directive = build_feasibility_directive(
        attempt_index=1,
        review={
            "decision": "reject",
            "responsibility": "gurobi_formulator",
            "summary": "模型漏了容量约束",
            "evidence": [{"source": "feasibility_result.json", "finding": "容量超限"}],
            "required_changes": ["补齐容量约束"],
            "confidence": "high",
            "remediation_handoff": {
                "requirement_ids": ["vehicle_capacity"],
                "change_ids": ["capacity_constraint"],
                "problem_requirement": "每条路线累计需求不得超过车辆容量",
                "observed_solution_behavior": "旧 solution 存在超容量路线",
                "responsibility_reason": "完整模型缺少容量约束",
                "required_changes": ["在完整模型中补齐容量约束"],
            },
        },
    )

    text = render_for_executor(
        (directive,),
        stage="gurobi_formulator",
        components_enabled=False,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=True,
    )

    assert text
    assert "补齐容量约束" in text
    assert "全量完整 Gurobi 模型" in text
