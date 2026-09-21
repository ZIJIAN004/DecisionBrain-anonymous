"""Gold-free, checker-grounded clarification answers for solving benchmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from ..core.message_builder import MessageBuilder
from ..infrastructure.llm_client import LLMClient
from .intake_evaluation import (
    IntakeEvaluationError,
    _append_protocol_retry,
    _parse_json_object,
    _register_tools,
    _run_reading_agent,
    build_intake_evaluation_tools,
)


_SYSTEM_PROMPT = """\
You are the benchmark author answering an optimization Intake agent's clarification questions.
Use problem.md, data/, and hidden/feasibility_check.py to identify the official intended contract.
The checker is authoritative when visible materials are ambiguous, incomplete, or conflicting.

Answer each question directly as the user or benchmark author. State the intended business rule,
parameter meaning, indexing convention, or interpretation needed to continue. Never mention a
checker, hidden file, source code, implementation, constraint number, gold label, reference
solution, evaluation process, or benchmark history. Do not quote or summarize internal code.
Translate the authoritative rule into ordinary domain language, disclose only what the question
requires, do not volunteer unrelated requirements, and do not solve the optimization problem.
Return JSON only.
"""

_FORBIDDEN_SOURCE_DISCLOSURES = (
    "checker",
    "hidden file",
    "hidden checker",
    "feasibility_check",
    "reference solution",
    "gold label",
    "检查器",
    "隐藏文件",
    "隐藏验收",
    "参考解",
)
_INTERNAL_LOCATION_PATTERN = re.compile(
    r"(?:constraint|line|约束|第)\s*#?\d+|\.py\b|hidden[/\\]", re.IGNORECASE
)
_OVERBROAD_QUESTION_PATTERN = re.compile(
    r"(?:all|every|complete|entire|full)\s+(?:constraints?|rules?|contract|requirements?)|"
    r"(?:list|describe|provide|explain)\s+(?:all|every|the complete|the entire)|"
    r"(?:所有|全部|完整|整个)(?:约束|规则|合同|契约|要求|需求)|"
    r"(?:列出|说明|提供|解释)(?:所有|全部|完整)(?:约束|规则|要求|需求)",
    re.IGNORECASE,
)


class IntakeAnswererError(RuntimeError):
    """A classified failure in the benchmark-side clarification answerer."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _is_overbroad_question(question: dict[str, Any]) -> bool:
    text = str(question.get("text") or "")
    return bool(_OVERBROAD_QUESTION_PATTERN.search(text))


def _scope_refusal(question_id: str) -> dict[str, str]:
    return {
        "question_id": question_id,
        "answer": (
            "这个问题范围过宽。请一次指出一个具体歧义、参数含义、索引约定或业务规则，"
            "我会针对该事项确认正式口径。"
        ),
    }


def _reject_internal_source_disclosure(answer: str) -> None:
    lowered = answer.lower()
    if any(marker in lowered for marker in _FORBIDDEN_SOURCE_DISCLOSURES):
        raise IntakeEvaluationError(
            "checker-grounded answer disclosed an internal evidence source"
        )
    if _INTERNAL_LOCATION_PATTERN.search(answer):
        raise IntakeEvaluationError(
            "checker-grounded answer disclosed an internal code location"
        )


class CheckerGroundedIntakeAnswerer:
    """Answer as the task author while keeping checker evidence isolated from Intake."""

    def __init__(
        self,
        settings: Settings,
        workspace: Path,
        *,
        max_turns: int = 20,
        max_protocol_attempts: int = 3,
    ) -> None:
        if max_protocol_attempts <= 0:
            raise ValueError("max_protocol_attempts must be positive")
        self._llm = LLMClient(settings)
        self._tools = build_intake_evaluation_tools(workspace, settings)
        self._max_turns = max_turns
        self._max_protocol_attempts = max_protocol_attempts

    async def answer(
        self,
        *,
        task_id: str,
        questions: list[dict[str, Any]],
        previous_turns: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        question_ids = [str(item.get("id") or "") for item in questions]
        if len(set(question_ids)) != len(question_ids) or any(not item for item in question_ids):
            raise IntakeAnswererError(
                "invalid_questions", "Intake clarification questions have invalid IDs"
            )
        if any(_is_overbroad_question(item) for item in questions):
            return [
                _scope_refusal(question_id) if _is_overbroad_question(question) else {
                    "question_id": question_id,
                    "answer": "请将本轮问题拆分为单一、具体且可确认的事项后再询问。",
                }
                for question_id, question in zip(question_ids, questions, strict=True)
            ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answers"],
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["question_id", "answer"],
                        "properties": {
                            "question_id": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                    },
                }
            },
        }
        messages = MessageBuilder.from_prompts(_SYSTEM_PROMPT)
        messages.add_user(
            f"Task: {task_id}\n"
            "Read problem.md, relevant data/, and hidden/feasibility_check.py before answering.\n"
            f"Previous clarification turns: {json.dumps(previous_turns, ensure_ascii=False)}\n"
            f"Current questions: {json.dumps(questions, ensure_ascii=False)}\n"
            f"Required schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        _register_tools(messages, self._tools)
        last_error: IntakeEvaluationError | None = None
        for attempt in range(1, self._max_protocol_attempts + 1):
            try:
                content = await _run_reading_agent(
                    self._llm,
                    messages,
                    self._tools,
                    max_turns=self._max_turns,
                    required_read_paths=(
                        ("problem.md", "hidden/feasibility_check.py") if attempt == 1 else ()
                    ),
                    required_read_prefixes=(("data/",) if attempt == 1 else ()),
                )
            except Exception as exc:
                text = str(exc).lower()
                kind = (
                    "network_failure"
                    if any(
                        marker in text
                        for marker in (
                            "connect",
                            "network",
                            "timeout",
                            "rate limit",
                            "429",
                            "server disconnected",
                        )
                    )
                    else "tool_or_reading_exhausted"
                )
                raise IntakeAnswererError(kind, str(exc) or type(exc).__name__) from exc
            try:
                return self._validate_answers(content, question_ids)
            except IntakeEvaluationError as exc:
                last_error = exc
                if attempt == self._max_protocol_attempts:
                    break
                _append_protocol_retry(messages, content, exc, attempt)
        kind = (
            "disclosure_rejected"
            if last_error is not None and "disclosed an internal" in str(last_error)
            else "protocol_exhausted"
        )
        raise IntakeAnswererError(
            kind,
            "checker_grounded_answerer_protocol_exhausted after "
            f"{self._max_protocol_attempts} attempts: {last_error}",
        ) from last_error

    @staticmethod
    def _validate_answers(content: str, question_ids: list[str]) -> list[dict[str, str]]:
        payload = _parse_json_object(content)
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, list):
            raise IntakeEvaluationError("checker-grounded answerer must return an answers array")
        answers: list[dict[str, str]] = []
        for item in raw_answers:
            if not isinstance(item, dict):
                raise IntakeEvaluationError("checker-grounded answerer entries must be objects")
            question_id = str(item.get("question_id") or "")
            answer = str(item.get("answer") or "").strip()
            if not question_id or not answer:
                raise IntakeEvaluationError(
                    "checker-grounded answerer entries require question_id and answer"
                )
            _reject_internal_source_disclosure(answer)
            answers.append({"question_id": question_id, "answer": answer})
        if [item["question_id"] for item in answers] != question_ids:
            raise IntakeEvaluationError(
                "checker-grounded answerer question IDs do not match the Intake questions"
            )
        return answers


def format_clarification_message(answers: list[dict[str, str]]) -> str:
    return json.dumps({"clarification_answers": answers}, ensure_ascii=False)
