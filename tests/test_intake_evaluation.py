from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisionbrain.benchmark.intake_evaluation import (
    ComponentScores,
    IntakeAnswerer,
    IntakeEvaluator,
    IntakeEvaluationError,
    IntakeEvaluationMaterials,
    IntakeGoldStandard,
    IntakeJudgeReport,
    IntakeDialogueTurn,
    VisibleEvidence,
    apply_score_policy,
    build_intake_evaluation_tools,
    normalize_report_context,
)
from decisionbrain.core.models import ChatResponse
from decisionbrain.core.stage_agent import Tool


def gold(*, expected: str = "clarify") -> IntakeGoldStandard:
    return IntakeGoldStandard(
        task_id="sample",
        expected_decision=expected,
        issue_severity="clarification_required" if expected == "clarify" else "none",
        benchmark_category="visible_ambiguity" if expected == "clarify" else "aligned_contract",
        issue_summary="setup_cost has two reasonable modeling meanings"
        if expected == "clarify"
        else None,
        visible_evidence=[
            VisibleEvidence(source="data", location="/setup_cost", finding="field is present")
        ]
        if expected == "clarify"
        else [],
        checker_impact="the objective changes" if expected == "clarify" else None,
        acceptable_questions=["Does setup_cost apply once or per transition?"]
        if expected == "clarify"
        else [],
    )


def report(
    *, observed: str, valid: bool = True, raw: int = 100, scope: str = "first_turn"
) -> IntakeJudgeReport:
    scores = ComponentScores(
        routing=min(raw, 40),
        critical_issue=min(max(raw - 40, 0), 25),
        action_quality=min(max(raw - 65, 0), 20),
        visible_grounding=min(max(raw - 85, 0), 10),
        protocol=min(max(raw - 95, 0), 5),
    )
    return IntakeJudgeReport(
        task_id="sample",
        evaluation_scope=scope,
        protocol_valid=valid,
        observed_decision=observed,
        decision_correct=False,
        issue_identified=False,
        issue_explanation="test explanation",
        question_evaluations=[],
        problem_definition_adequate=None,
        unsupported_assumptions=[],
        component_scores=scores,
        total_score=100,
        applied_cap=100,
        verdict="pass",
        rationale="test rationale",
    )


def test_missed_clarification_is_capped_at_25() -> None:
    judged = apply_score_policy(report(observed="proceed"), gold())
    assert judged.total_score == 25
    assert judged.applied_cap == 25
    assert judged.verdict == "fail"


def test_false_positive_clarification_is_capped_at_40() -> None:
    judged = apply_score_policy(report(observed="clarify"), gold(expected="proceed"))
    assert judged.total_score == 40
    assert judged.applied_cap == 40
    assert judged.verdict == "fail"


def test_protocol_failure_scores_zero() -> None:
    judged = apply_score_policy(report(observed="invalid", valid=False), gold())
    assert judged.total_score == 0
    assert judged.applied_cap == 0
    assert judged.verdict == "protocol_failure"


def test_correct_route_uses_component_total() -> None:
    judged = apply_score_policy(report(observed="clarify", raw=82), gold())
    assert judged.total_score == 82
    assert judged.applied_cap == 100
    assert judged.verdict == "pass"


def test_proceed_gold_rejects_clarification_fields() -> None:
    with pytest.raises(ValueError, match="must not contain clarification fields"):
        IntakeGoldStandard(
            task_id="negative",
            expected_decision="proceed",
            issue_severity="none",
            benchmark_category="aligned_contract",
            issue_summary="an issue that should not exist",
            checker_impact="not applicable",
        )


def test_gold_rejects_category_decision_mismatch() -> None:
    with pytest.raises(ValueError, match="aligned or checker-only"):
        IntakeGoldStandard(
            task_id="bad-negative",
            expected_decision="proceed",
            issue_severity="none",
            benchmark_category="visible_ambiguity",
        )


def test_material_paths_cannot_escape_workspace() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        IntakeEvaluationMaterials(
            workspace_root=Path("evaluation-workspace"),
            hidden_checker_path="../hidden.py",
            intake_output={},
            gold=gold(),
        )


def test_full_dialogue_scores_terminal_proceed_after_answer() -> None:
    judged = apply_score_policy(report(observed="proceed", scope="full_dialogue"), gold())
    assert judged.total_score == 100
    assert judged.applied_cap == 100
    assert judged.verdict == "pass"


def test_answerer_dynamically_answers_the_actual_question_with_full_context() -> None:
    class FakeLLM:
        request = None

        async def complete(self, request):
            self.request = request
            return ChatResponse(
                content=(
                    '{"answers":[{"question_id":"unexpected_q",'
                    '"status":"answered","answer":"Use zero-based indexing.",'
                    '"reason":"This is the intended indexing convention."}]}'
                )
            )

    llm = FakeLLM()
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"),
        intake_output={"decision": "clarify"},
        gold=gold(),
    )
    answers = asyncio.run(
        IntakeAnswerer(llm, (), require_source_reads=False).answer(
            materials=materials,
            questions=[
                {
                    "id": "unexpected_q",
                    "text": "Should periods start at zero or one?",
                    "answer_type": "choice",
                    "options": ["zero", "one"],
                }
            ],
            previous_turns=[],
        )
    )

    assert answers[0].answer == "Use zero-based indexing."
    prompt = llm.request.messages[-1]["content"]
    assert "Should periods start at zero or one?" in prompt
    assert "hidden/feasibility_check.py" in prompt
    assert "assert first_period == 0" not in prompt


def test_answerer_retries_schema_errors_with_validation_feedback() -> None:
    class RetryingLLM:
        calls = 0
        requests = []

        async def complete(self, request):
            self.calls += 1
            self.requests.append(request)
            outputs = [
                '{"response": []}',
                (
                    '{"answers":[{"question_id":"q1","status":"answered",'
                    '"answer":"Use scalars.","reason":"They are authoritative.",'
                    '"answer_type_choice_used":null}]}'
                ),
                (
                    '{"answers":[{"question_id":"q1","status":"answered",'
                    '"answer":"Use scalars.","reason":"They are authoritative."}]}'
                ),
            ]
            return ChatResponse(content=outputs[self.calls - 1])

    llm = RetryingLLM()
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"),
        intake_output={"decision": "clarify"},
        gold=gold(),
    )
    answers = asyncio.run(
        IntakeAnswerer(llm, (), require_source_reads=False).answer(
            materials=materials,
            questions=[{"id": "q1", "text": "Which fields?", "answer_type": "text"}],
            previous_turns=[],
        )
    )

    assert llm.calls == 3
    assert answers[0].answer == "Use scalars."
    retry_feedback = llm.requests[2].messages[-1]["content"]
    assert "answer_type_choice_used" in retry_feedback
    assert "Do not call tools again" in retry_feedback


def test_answerer_raises_only_after_protocol_attempts_are_exhausted() -> None:
    class InvalidLLM:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            return ChatResponse(content='{"not_answers": []}')

    llm = InvalidLLM()
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"), intake_output={}, gold=gold()
    )
    with pytest.raises(IntakeEvaluationError, match="answerer_protocol_exhausted after 3"):
        asyncio.run(
            IntakeAnswerer(llm, (), require_source_reads=False).answer(
                materials=materials,
                questions=[],
                previous_turns=[],
            )
        )
    assert llm.calls == 3


def test_evaluator_retries_invalid_report_schema() -> None:
    class RetryingJudgeLLM:
        calls = 0
        last_request = None

        async def complete(self, request):
            self.calls += 1
            self.last_request = request
            if self.calls == 1:
                return ChatResponse(content='{"task_id":"sample"}')
            return ChatResponse(content=report(observed="clarify").model_dump_json())

    llm = RetryingJudgeLLM()
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"),
        intake_output={"decision": "clarify"},
        gold=gold(),
    )
    judged = asyncio.run(IntakeEvaluator(llm, (), require_source_reads=False).evaluate(materials))

    assert llm.calls == 2
    assert judged.observed_decision == "clarify"
    assert "failed the required JSON protocol" in llm.last_request.messages[-1]["content"]


def test_evaluation_toolset_is_read_only_subset(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        solver_timeout=30,
        opt_workspace_max_output_chars=20_000,
        opt_workspace_max_list_entries=100,
        opt_workspace_default_list_entries=50,
        opt_workspace_max_read_bytes=100_000,
    )
    tools = build_intake_evaluation_tools(tmp_path, settings)
    assert {tool.name for tool in tools} == {
        "list_files",
        "read_file",
        "search_file",
        "data_get",
        "data_keys",
        "data_slice",
    }


def test_answerer_cannot_finalize_without_required_source_reads() -> None:
    class FinalOnlyLLM:
        async def complete(self, request):
            return ChatResponse(content='{"answers": []}')

    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"),
        intake_output={"decision": "clarify"},
        gold=gold(),
    )
    try:
        asyncio.run(
            IntakeAnswerer(FinalOnlyLLM(), ()).answer(
                materials=materials,
                questions=[],
                previous_turns=[],
            )
        )
    except IntakeEvaluationError as exc:
        assert "required sources" in str(exc)
    else:
        raise AssertionError("answerer should require source reads")


def test_failed_read_file_calls_do_not_satisfy_required_reads() -> None:
    class ReadThenFinishLLM:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                paths = [
                    "problem.md",
                    "hidden/feasibility_check.py",
                    "gold_standard.json",
                    "data/profile.json",
                ]
                return ChatResponse(
                    tool_calls=[
                        {
                            "id": f"read-{index}",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": f'{{"path":"{path}"}}',
                            },
                        }
                        for index, path in enumerate(paths)
                    ]
                )
            return ChatResponse(content='{"answers": []}')

    async def failed_read(**kwargs):
        return "read_file error: missing"

    read_tool = Tool(name="read_file", description="read", parameters={}, handler=failed_read)
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"), intake_output={}, gold=gold()
    )
    with pytest.raises(IntakeEvaluationError, match="required sources"):
        asyncio.run(
            IntakeAnswerer(ReadThenFinishLLM(), (read_tool,)).answer(
                materials=materials,
                questions=[],
                previous_turns=[],
            )
        )


def test_report_observables_are_derived_from_dialogue_and_output() -> None:
    materials = IntakeEvaluationMaterials(
        workspace_root=Path("evaluation-workspace"),
        intake_output={"decision": "proceed"},
        gold=gold(),
        dialogue=[
            IntakeDialogueTurn(
                round_index=1,
                intake_output={"decision": "clarify"},
            ),
            IntakeDialogueTurn(
                round_index=2,
                intake_output={"decision": "proceed"},
            ),
        ],
    )
    normalized = normalize_report_context(report(observed="clarify"), materials)
    assert normalized.evaluation_scope == "full_dialogue"
    assert normalized.observed_decision == "proceed"
    assert normalized.decision_correct is True
    assert normalized.clarification_rounds == 1
