import json

from decisionbrain.core.feasibility_routing import (
    annotate_feasibility_outcome,
    build_feasibility_directive,
    has_confirmed_instance_reviews,
    render_for_executor,
    render_for_reviewer,
    visible_review_history,
)
from decisionbrain.core.models import AgentStage, AgentState, StageCompleted
from decisionbrain.core.stage_flow import apply_stage_result
from decisionbrain.core.workspace_tools import prepare_feasibility_handoff


def rejection(responsibility="solving"):
    return {
        "decision": "reject",
        "responsibility": responsibility,
        "summary": "候选未覆盖容量约束",
        "evidence": [{"source": "feasibility_result.json", "finding": "容量超限"}],
        "required_changes": ["修复容量聚合"],
        "confidence": "high",
        "remediation_handoff": {
            "requirement_ids": ["vehicle_capacity"],
            "change_ids": ["capacity_aggregation"],
            "problem_requirement": "每条路线累计需求不得超过车辆容量",
            "observed_solution_behavior": "旧 solution 存在超出车辆容量的路线",
            "responsibility_reason": "当前实现没有正确维护路线累计载重",
            "required_changes": ["修复路线容量聚合并在导出前进行业务约束自检"],
        },
    }


def test_directive_is_replayable_and_core_owned():
    directive = build_feasibility_directive(attempt_index=2, review=rejection())

    assert directive["responsibility"] == "solving"
    assert "directive_id" not in directive
    assert "source" not in directive
    assert "target" not in directive
    assert directive["handoff_dir"] == "feasibility_handoffs/solving/attempt2"
    assert directive["handoff"]["problem_requirement"] == "每条路线累计需求不得超过车辆容量"
    serialized = json.dumps(directive, ensure_ascii=False)
    assert "feasibility_result" not in serialized
    assert "容量超限" not in serialized  # 原始 evidence 只留在 Core 私有审计历史


def test_pending_directive_is_injected_into_executor_and_reviewer():
    directive = build_feasibility_directive(attempt_index=1, review=rejection())

    solving = render_for_executor((directive,), stage="solving")
    design = render_for_executor((directive,), stage="algorithm_design")
    reviewer = render_for_reviewer((directive,))

    assert "修复路线容量聚合" in solving
    assert "Algorithm Design 未变" in solving
    assert design == ""
    assert "重新独立裁决" in reviewer


def test_executor_is_told_which_baselines_this_attempt_actually_has():
    directive = build_feasibility_directive(attempt_index=1, review=rejection())
    prepared = {
        **directive,
        "baseline_files": ["previous_algorithm_design.json", "previous_solver.py"],
    }

    rendered = render_for_executor((prepared,), stage="solving")

    assert "`feasibility_handoffs/solving/attempt1/`" in rendered
    assert "`previous_algorithm_design.json`" in rendered
    assert "`previous_solver.py`" in rendered
    assert "必须先全部读取再修改" in rendered
    # This round has no prior solution; do not reference a nonexistent file.
    assert "previous_solution.json" not in rendered


def test_executor_falls_back_to_listing_when_baselines_are_unknown():
    directive = build_feasibility_directive(attempt_index=1, review=rejection())

    rendered = render_for_executor((directive,), stage="solving")

    assert "`feasibility_handoffs/solving/attempt1/`" in rendered
    assert "list_files" in rendered


def test_active_directive_carries_solver_summary_for_the_current_attempt():
    directive = build_feasibility_directive(
        attempt_index=1,
        review=rejection(),
        solver_result={
            "status": "time_limit",
            "diagnosis": "search exhausted without a feasible candidate",
            "executions": [{"method": "large_neighborhood_search"}],
        },
    )

    rendered = render_for_executor((directive,), stage="solving")

    # The ledger exposes prior solver summaries; the current attempt must be no less transparent.
    assert '"status": "time_limit"' in rendered
    assert "search exhausted without a feasible candidate" in rendered
    assert "large_neighborhood_search" in rendered


def test_active_directive_reads_single_algorithm_execution_summary():
    directive = build_feasibility_directive(
        attempt_index=1,
        review=rejection(),
        solver_result={
            "status": "time_limit",
            "diagnosis": "search exhausted without a feasible candidate",
            "execution": {"method": "large_neighborhood_search"},
        },
    )

    rendered = render_for_executor((directive,), stage="solving")

    assert '"methods": [' in rendered
    assert "large_neighborhood_search" in rendered


def test_design_repair_is_visible_only_to_design():
    directive = build_feasibility_directive(
        attempt_index=1,
        review=rejection("algorithm_design"),
    )

    assert "旧设计和旧 solution 是失败反例" in render_for_executor(
        (directive,), stage="algorithm_design"
    )
    assert render_for_executor((directive,), stage="solving") == ""


def test_single_algorithm_design_repair_does_not_reintroduce_fallback():
    directive = build_feasibility_directive(
        attempt_index=1,
        review=rejection("algorithm_design"),
    )

    rendered = render_for_executor(
        (directive,),
        stage="algorithm_design",
        components_enabled=False,
    )

    assert "fallback" not in rendered.lower()
    assert "算法选择、formulation、硬约束覆盖或策略设计" in rendered


def test_design_rejection_routes_to_solving_when_design_stage_is_disabled():
    state = AgentState(
        problem_description="test",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        solver_result={"status": "time_limit"},
    )
    review = rejection("algorithm_design")

    routed = apply_stage_result(
        state,
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={"feasibility_review": review, "feasibility_check": None},
        ),
        algorithm_design_enabled=False,
    )

    assert routed.current_stage is AgentStage.SOLVING
    assert routed.algorithm_design is None
    assert routed.feasibility_directives[-1]["responsibility"] == "solving"
    assert routed.feasibility_audits[-1]["responsibility"] == "solving"
    assert routed.feasibility_audits[-1]["reported_responsibility"] == "algorithm_design"
    assert routed.feasibility_audits[-1]["feasibility_review"] is review

    rendered = render_for_executor(
        routed.feasibility_directives,
        stage="solving",
        components_enabled=False,
        algorithm_design_enabled=False,
    )
    assert "同一个 Gurobi Formulator 修复" in rendered
    assert "Formulator 的修复范围" in rendered
    assert "业务结果映射" in rendered
    assert "允许调整算法选择、建模范式" not in rendered
    assert "Algorithm Design 未变" not in rendered


def test_next_verdict_closes_pending_directive():
    directive = build_feasibility_directive(attempt_index=1, review=rejection())
    annotated = annotate_feasibility_outcome(
        (directive,),
        review={"decision": "accept", "summary": "修复后通过"},
    )

    assert annotated[-1]["outcome"] == "accept"
    assert "outcome_summary" not in annotated[-1]
    assert render_for_executor(annotated, stage="solving") == ""


def test_executor_history_excludes_private_review_summary():
    first = build_feasibility_directive(attempt_index=1, review=rejection())
    completed = annotate_feasibility_outcome(
        (first,),
        review={"decision": "reject", "responsibility": "solving", "summary": "私有细节"},
    )
    second = build_feasibility_directive(attempt_index=2, review=rejection())

    rendered = render_for_executor((*completed, second), stage="solving")

    assert '"outcome": "reject"' in rendered
    assert '"requirement_ids": [' in rendered
    assert '"change_ids": [' in rendered
    assert '"next_responsibility": "solving"' in rendered
    assert '"responsibility": "solving"' in rendered
    assert "私有细节" not in rendered


def test_reviewer_history_uses_compact_ledger_instead_of_prior_handoffs():
    first = build_feasibility_directive(
        attempt_index=1,
        review=rejection(),
        solver_result={
            "status": "time_limit",
            "executions": [
                {
                    "method": "large_neighborhood_search",
                }
            ],
            "diagnosis": "search exhausted without a feasible candidate",
        },
    )
    completed = annotate_feasibility_outcome(
        (first,),
        review={"decision": "reject", "responsibility": "algorithm_design", "summary": "私有"},
    )
    second = build_feasibility_directive(attempt_index=2, review=rejection())

    rendered = render_for_reviewer((*completed, second))

    assert '"attempt": 1' in rendered
    assert '"next_responsibility": "algorithm_design"' in rendered
    assert '"status": "time_limit"' in rendered
    assert '"methods": [' in rendered
    assert "search exhausted without a feasible candidate" in rendered
    assert rendered.count("problem_requirement") == 1
    assert "私有" not in rendered
    assert "feasibility_review_history.json" in rendered


def test_instance_verdicts_are_withheld_while_the_run_is_still_open():
    audits = (
        {"kind": "instance_infeasibility", "infeasibility_proof": "第一次证明"},
        {"kind": "repair_attempt", "attempt": 1},
        {"kind": "instance_infeasibility", "infeasibility_proof": "第二次证明"},
    )

    during = visible_review_history(audits, instance_run=({"x": 1},))
    after = visible_review_history(audits, instance_run=())
    confirmed = visible_review_history(audits, instance_run=({"x": 1},) * 3)

    # An unfinished segment hides only itself; completed instance decisions remain visible.
    assert [item.get("kind") for item in during] == [
        "instance_infeasibility",
        "repair_attempt",
    ]
    # Full history is visible after leaving instance responsibility.
    assert [item.get("kind") for item in after] == [
        "instance_infeasibility",
        "repair_attempt",
        "instance_infeasibility",
    ]
    # Explanation must see the decisions that completed confirmation.
    assert len(confirmed) == 3
    assert after[0] is not audits[0]  # 投影出的是副本，不会被下游改写


def test_three_instance_verdicts_confirm_regardless_of_confidence():
    assert not has_confirmed_instance_reviews(())
    assert not has_confirmed_instance_reviews(({"confidence": "high"},) * 2)
    # Resetting on a low-confidence decision previously caused an unbounded loop.
    assert has_confirmed_instance_reviews(({"confidence": "low"},) * 3)


def test_rejected_attempt_handoff_exposes_only_responsibility_safe_baseline(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text("{}", encoding="utf-8")
    (stage_outputs / "solving.json").write_text("{}", encoding="utf-8")
    (stage_outputs / "feasibility_review.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')", encoding="utf-8")
    (tmp_path / "solver_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feasibility_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "input.json").write_text('{"input": true}', encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"solution": true}', encoding="utf-8")
    directive = build_feasibility_directive(attempt_index=1, review=rejection())

    prepared = prepare_feasibility_handoff(tmp_path, 1, directive)

    handoff_dir = tmp_path / "feasibility_handoffs" / "solving" / "attempt1"
    # Remediation instructions stay in stage messages; handoff directories hold large baselines.
    assert not (handoff_dir / "remediation.json").exists()
    assert prepared == (
        "feasibility_handoffs/solving/attempt1/previous_algorithm_design.json",
        "feasibility_handoffs/solving/attempt1/previous_solution.json",
        "feasibility_handoffs/solving/attempt1/previous_solver.py",
    )
    assert (handoff_dir / "previous_solver.py").read_text(encoding="utf-8") == "print('solve')"
    assert json.loads((handoff_dir / "previous_solution.json").read_text(encoding="utf-8")) == {
        "solution": True
    }
    assert not (handoff_dir / "feasibility_result.json").exists()
    assert not (handoff_dir / "feasibility_review.json").exists()


def test_handoff_omits_baselines_the_rejected_solve_never_produced(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text('{"design": "old"}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')", encoding="utf-8")
    # A rejected no-candidate outcome never wrote solution.json.
    directive = build_feasibility_directive(attempt_index=1, review=rejection())

    prepared = prepare_feasibility_handoff(tmp_path, 1, directive)

    handoff_dir = tmp_path / "feasibility_handoffs" / "solving" / "attempt1"
    assert prepared == (
        "feasibility_handoffs/solving/attempt1/previous_algorithm_design.json",
        "feasibility_handoffs/solving/attempt1/previous_solver.py",
    )
    assert not (handoff_dir / "previous_solution.json").exists()


def test_preparing_an_attempt_retires_the_earlier_handoffs(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text('{"design": "new"}', encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"solution": "new"}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('new')", encoding="utf-8")
    base = tmp_path / "feasibility_handoffs"
    for stale in (
        base / "solving" / "attempt1",
        base / "solving" / "attempt2",
        base / "solving" / "attempt7",  # 契约修订重置计数后遗留的孤儿轮次
        base / "algorithm_design" / "attempt2",
    ):
        stale.mkdir(parents=True)
        (stale / "previous_solution.json").write_text("旧解", encoding="utf-8")
    keep_unrelated = base / "solving" / "notes"
    keep_unrelated.mkdir()
    (keep_unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    directive = build_feasibility_directive(attempt_index=3, review=rejection())

    prepare_feasibility_handoff(tmp_path, 3, directive)

    # Solving archives each round immutably; the workspace retains only the current baseline.
    assert not (base / "solving" / "attempt1").exists()
    assert not (base / "solving" / "attempt2").exists()
    assert not (base / "solving" / "attempt7").exists()
    assert not (base / "algorithm_design" / "attempt2").exists()
    assert (base / "solving" / "attempt3" / "previous_solver.py").is_file()
    assert (keep_unrelated / "keep.txt").is_file()


def test_first_attempt_prepares_cleanly_without_existing_handoffs(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text('{"design": "old"}', encoding="utf-8")
    directive = build_feasibility_directive(attempt_index=1, review=rejection())

    prepared = prepare_feasibility_handoff(tmp_path, 1, directive)

    assert prepared == ("feasibility_handoffs/solving/attempt1/previous_algorithm_design.json",)


def test_design_handoff_gets_old_solution_but_not_solver(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text('{"design": "old"}', encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"solution": "old"}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('private implementation')", encoding="utf-8")
    directive = build_feasibility_directive(
        attempt_index=1,
        review=rejection("algorithm_design"),
    )

    prepare_feasibility_handoff(tmp_path, 1, directive)

    handoff_dir = tmp_path / "feasibility_handoffs" / "algorithm_design" / "attempt1"
    assert json.loads(
        (handoff_dir / "previous_algorithm_design.json").read_text(encoding="utf-8")
    ) == {"design": "old"}
    assert json.loads((handoff_dir / "previous_solution.json").read_text(encoding="utf-8")) == {
        "solution": "old"
    }
    assert not (handoff_dir / "previous_solver.py").exists()


def test_translation_only_solving_repair_does_not_authorise_model_changes():
    """臂 A 的 Solving 是 translation-only，修复指令不得授权它改模型语义。"""
    directive = build_feasibility_directive(attempt_index=1, review=rejection("solving"))

    text = render_for_executor(
        (directive,),
        stage="solving",
        components_enabled=False,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=True,
    )

    assert "translation-only" in text
    assert "不得修改变量、约束或目标语义" in text
    assert "由同一个 Gurobi Formulator 修复完整模型" not in text


def test_legacy_no_design_arm_repair_text_is_unchanged():
    """没有独立 Formulator 阶段的旧无设计臂行为保持不变。"""
    directive = build_feasibility_directive(attempt_index=1, review=rejection("solving"))

    text = render_for_executor(
        (directive,),
        stage="solving",
        components_enabled=False,
        algorithm_design_enabled=False,
        gurobi_formulator_enabled=False,
    )

    assert "由同一个 Gurobi Formulator 修复完整模型" in text
    assert "translation-only" not in text
