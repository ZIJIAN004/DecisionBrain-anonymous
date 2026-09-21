import asyncio

import pytest

from decisionbrain.benchmark.intake_evaluation import IntakeEvaluationError
from decisionbrain.benchmark.solving_intake import (
    CheckerGroundedIntakeAnswerer,
    IntakeAnswererError,
    _is_overbroad_question,
    _reject_internal_source_disclosure,
)


def test_checker_grounded_answer_allows_domain_rule() -> None:
    _reject_internal_source_disclosure("请使用实例级容量；该容量覆盖类型模板中的默认值。")


@pytest.mark.parametrize(
    "answer",
    (
        "hidden checker requires this interpretation",
        "feasibility_check.py line 80 uses zero-based periods",
        "根据隐藏文件，应当采用 0 基索引",
        "违反约束 11，因此必须这样回答",
    ),
)
def test_checker_grounded_answer_rejects_internal_source_disclosure(answer: str) -> None:
    with pytest.raises(IntakeEvaluationError, match="internal"):
        _reject_internal_source_disclosure(answer)


def test_checker_grounded_answerer_retries_invalid_final_output(monkeypatch) -> None:
    responses = iter(
        (
            '{"answers":[{"question_id":"q1","answer":"the checker requires period 0"}]}',
            '{"answers":[{"question_id":"q1","answer":"第一期按 period 0 编号。"}]}',
        )
    )

    async def fake_run_reading_agent(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(
        "decisionbrain.benchmark.solving_intake._run_reading_agent",
        fake_run_reading_agent,
    )
    answerer = object.__new__(CheckerGroundedIntakeAnswerer)
    answerer._llm = object()
    answerer._tools = ()
    answerer._max_turns = 20
    answerer._max_protocol_attempts = 3

    answers = asyncio.run(
        answerer.answer(task_id="sample", questions=[{"id": "q1"}], previous_turns=[])
    )

    assert answers == [{"question_id": "q1", "answer": "第一期按 period 0 编号。"}]


@pytest.mark.parametrize(
    "text",
    (
        "请列出所有约束和输出规则。",
        "Provide the complete contract and every requirement.",
    ),
)
def test_overbroad_question_is_detected(text: str) -> None:
    assert _is_overbroad_question({"text": text}) is True


def test_overbroad_question_is_refused_without_reading_checker(monkeypatch) -> None:
    async def should_not_run(*args, **kwargs):
        raise AssertionError("overbroad questions must be refused before reading sources")

    monkeypatch.setattr(
        "decisionbrain.benchmark.solving_intake._run_reading_agent", should_not_run
    )
    answerer = object.__new__(CheckerGroundedIntakeAnswerer)
    answerer._llm = object()
    answerer._tools = ()
    answerer._max_turns = 20
    answerer._max_protocol_attempts = 3

    answers = asyncio.run(
        answerer.answer(
            task_id="sample",
            questions=[{"id": "q1", "text": "请列出所有约束。"}],
            previous_turns=[],
        )
    )

    assert answers[0]["question_id"] == "q1"
    assert "范围过宽" in answers[0]["answer"]


def test_answerer_classifies_reading_exhaustion(monkeypatch) -> None:
    async def fail_reading(*args, **kwargs):
        raise IntakeEvaluationError("reading agent exceeded 20 turns")

    monkeypatch.setattr(
        "decisionbrain.benchmark.solving_intake._run_reading_agent", fail_reading
    )
    answerer = object.__new__(CheckerGroundedIntakeAnswerer)
    answerer._llm = object()
    answerer._tools = ()
    answerer._max_turns = 20
    answerer._max_protocol_attempts = 3

    with pytest.raises(IntakeAnswererError) as captured:
        asyncio.run(
            answerer.answer(
                task_id="sample",
                questions=[{"id": "q1", "text": "第一期如何编号？"}],
                previous_turns=[],
            )
        )

    assert captured.value.kind == "tool_or_reading_exhausted"
