from decisionbrain.paths import PROMPTS_DIR
from decisionbrain.core.prompts import load_prompt_bundle


def _prompt(group: str, name: str) -> str:
    return (PROMPTS_DIR / group / name).read_text(encoding="utf-8")


def test_prompt_markdown_fences_are_balanced() -> None:
    for group in ("system", "developer"):
        for path in (PROMPTS_DIR / group).glob("*.txt"):
            assert path.read_text(encoding="utf-8").count("```") % 2 == 0, path


def test_solving_prompts_require_deterministic_input_conversion():
    prompt_pairs = (
        ("solving_system.txt", "solving_contract.txt"),
        ("solving_self_check_system.txt", "solving_self_check_contract.txt"),
    )

    for system_name, contract_name in prompt_pairs:
        combined = _prompt("system", system_name) + _prompt("developer", contract_name)
        assert "from_data_to_input.py" in combined
        assert "python3 from_data_to_input.py" in combined
        assert "禁止使用 write_file" in combined
        assert "直接创建、重写或修补 input.json" in combined


def test_no_review_algorithm_design_prompt_removes_reviewer_semantics() -> None:
    prompts = load_prompt_bundle(PROMPTS_DIR, feasibility_review_enabled=False)
    combined = prompts.algorithm_design_system + prompts.algorithm_design_contract

    assert "Feasibility Reviewer" not in combined
    assert "Reviewer handoff" not in combined
    assert "下游独立验收" not in combined
    assert "为结果转换和文件输出保留安全余量" in combined


def test_no_review_self_check_guide_rule_uses_component_source_schema() -> None:
    prompt = _prompt("system", "solving_self_check_system_no_feasible_review.txt")

    assert "selection.components 和 fallback.components" in prompt
    assert "source=package" in prompt
    assert "selection.kind 为 library" not in prompt


def test_solving_shell_timeout_is_a_terminal_review_outcome() -> None:
    system = _prompt("system", "solving_system.txt")
    contract = _prompt("developer", "solving_contract.txt")

    assert "run_solver` 明确超时" in system
    assert "不得在当前 Solving 轮次修改代码后再次运行 solver.py" in system
    assert "Runtime 会发布既有 `runtime_solver_outcome.json` 外部超时终态" in contract
    assert "shell timeout 不属于允许留在本阶段修复重试的基础执行错误" in contract


def test_solving_contract_keeps_execution_schema_inside_its_section() -> None:
    contract = _prompt("developer", "solving_contract.txt")

    execution_heading = contract.index("executions 中每项结构：")
    execution_schema = contract.index('"component_id"', execution_heading)
    execution_rules = contract.index("execution 规则：", execution_schema)
    stage_output_heading = contract.index(
        "stage_outputs/solving.json 三种格式：", execution_rules
    )
    solved_output = contract.index('"decision":"solved"', stage_output_heading)

    assert execution_heading < execution_schema < execution_rules < stage_output_heading
    assert stage_output_heading < solved_output


def test_solving_receives_only_solving_repair_directives() -> None:
    system = _prompt("system", "solving_system.txt")

    assert "责任为 solving 时保持当前 Algorithm Design 不变并定点修复实现" in system
    assert "责任为 algorithm_design 时" not in system


def test_common_prompt_lists_every_fixed_stage_output() -> None:
    common = _prompt("system", "stage_common_system.txt")

    for stage in (
        "intake",
        "problem_contract",
        "algorithm_design",
        "solving",
        "feasibility_review",
        "explanation",
    ):
        assert f"- {stage}: `stage_outputs/{stage}.json`" in common


def test_reviewer_does_not_default_uncertain_responsibility_to_solving() -> None:
    system = _prompt("system", "feasibility_review_system.txt")
    contract = _prompt("developer", "feasibility_review_contract.txt")

    assert "不是把证据不足默认归为 solving" in system
    assert "没有新的具体实现缺陷时归为 algorithm_design" in system
    assert "不得以“尚不能证明 algorithm_design”为由默认归责 solving" in contract
    assert "本轮要求的修复动作与既往实质相同，则必须归责 algorithm_design" in contract


def test_repeated_repair_is_judged_by_meaning_not_by_change_id_string() -> None:
    system = _prompt("system", "feasibility_review_system.txt")
    contract = _prompt("developer", "feasibility_review_contract.txt")

    # change_id is model-authored free text; validation must inspect the remediation action itself.
    assert "不看 change_ids 字符串" in contract
    assert "它们只是标识，不构成任何判定依据" in contract
    for prompt in (system, contract):
        assert "换一个 change_id 写法" in prompt
        # Top-level required_changes is private audit data; validation must name the handoff field.
        assert "remediation_handoff.required_changes 语义" in prompt
        assert "feasibility_review_history.json" in prompt


def test_repeated_non_proof_failure_escalates_to_conditional_exact_design() -> None:
    review_system = _prompt("system", "feasibility_review_system.txt")
    review_contract = _prompt("developer", "feasibility_review_contract.txt")
    design_system = _prompt("system", "algorithm_design_system.txt")
    design_contract = _prompt("developer", "algorithm_design_contract.txt")

    assert "非证明型搜索多次无法产生完整可行候选" in review_system
    assert "不得把精确尝试写成无条件要求" in review_system
    assert "不得要求无条件使用精确法" in review_contract
    assert "不得只更换随机种子或原样延长同一搜索" in design_system
    assert "精确尝试返回 time_limit、unknown 或未找到解仍不证明实例不可行" in design_system
    assert "至少在 candidates、selection 或 fallback 中评估一个" in design_contract


def test_repeated_timeout_escalation_targets_runtime_timeout_not_native_time_limit() -> None:
    system = _prompt("system", "feasibility_review_system.txt")
    contract = _prompt("developer", "feasibility_review_contract.txt")

    assert "若相同 runtime_timeout 或失败模式重复出现" in system
    assert "同一 runtime_timeout 或失败模式仍然出现" in contract
    assert "time_limit 若携带完整候选，必须走候选解检查" in system
    assert "time_limit 不适用这条机械升级规则" in system
    assert "携带完整候选时必须检查候选" in contract
    assert "不适用重复 runtime_timeout 的机械升级规则" in contract
    assert "单次 time_limit 本身不自动证明设计错误" not in system
