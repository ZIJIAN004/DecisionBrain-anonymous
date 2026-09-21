"""LLM-based evaluation contract for the Intake benchmark.

The Intake agent sees only the task's visible problem and data. The evaluator is
separate and may additionally inspect the hidden checker and a human-approved gold
standard. Deterministic caps prevent a fluent rationale from hiding a routing failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import Settings
from ..core.message_builder import MessageBuilder
from ..core.models import ChatRequest, ChatResponse
from ..core.stage_agent import Tool, _tool_argument_contract, _tool_argument_issues
from ..core.workspace_tools import WorkspaceToolset
from .intake_data_tools import build_intake_data_tools


class IntakeEvaluationError(RuntimeError):
    """The evaluator could not produce a valid, scoreable report."""


class CompletionClient(Protocol):
    async def complete(self, request: ChatRequest) -> ChatResponse: ...


class VisibleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["problem", "data"]
    location: str = Field(min_length=1)
    finding: str = Field(min_length=1)


class IntakeGoldStandard(BaseModel):
    """Human-approved expected behavior for one frozen Intake case."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    expected_decision: Literal["clarify", "proceed"]
    issue_severity: Literal["none", "clarification_required", "blocking_conflict"]
    benchmark_category: Literal[
        "aligned_contract",
        "checker_only_difference",
        "visible_ambiguity",
        "problem_data_conflict",
    ]
    issue_summary: str | None = None
    visible_evidence: list[VisibleEvidence] = Field(default_factory=list)
    checker_impact: str | None = None
    acceptable_questions: list[str] = Field(default_factory=list)
    forbidden_assumptions: list[str] = Field(default_factory=list)
    expected_terminal_decision: Literal["proceed", "clarify"] = "proceed"

    @model_validator(mode="after")
    def validate_label_consistency(self) -> IntakeGoldStandard:
        if self.issue_severity == "none" and self.expected_decision != "proceed":
            raise ValueError("issue_severity=none requires expected_decision=proceed")
        if self.issue_severity != "none" and self.expected_decision != "clarify":
            raise ValueError("material Intake issues require expected_decision=clarify")
        if self.expected_decision == "proceed":
            if self.expected_terminal_decision != "proceed":
                raise ValueError("proceed gold labels must terminate with proceed")
            if self.benchmark_category not in {"aligned_contract", "checker_only_difference"}:
                raise ValueError("proceed requires an aligned or checker-only category")
            if any(
                (
                    self.issue_summary,
                    self.visible_evidence,
                    self.checker_impact,
                    self.acceptable_questions,
                    self.forbidden_assumptions,
                )
            ):
                raise ValueError("proceed gold labels must not contain clarification fields")
        else:
            if self.benchmark_category not in {"visible_ambiguity", "problem_data_conflict"}:
                raise ValueError("clarify requires an ambiguity or conflict category")
            if (
                self.benchmark_category == "problem_data_conflict"
                and self.issue_severity != "blocking_conflict"
            ):
                raise ValueError("problem_data_conflict requires blocking_conflict severity")
            if (
                self.benchmark_category == "visible_ambiguity"
                and self.issue_severity != "clarification_required"
            ):
                raise ValueError("visible_ambiguity requires clarification_required severity")
            if not all(
                (
                    self.issue_summary,
                    self.visible_evidence,
                    self.checker_impact,
                    self.acceptable_questions,
                )
            ):
                raise ValueError(
                    "clarify gold labels require complete issue evidence and questions"
                )
        return self


class ClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    status: Literal["answered", "cannot_determine", "ambiguous_question"]
    answer: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class IntakeDialogueTurn(BaseModel):
    """One Intake decision and the answers returned before the next round."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    intake_output: dict[str, Any]
    answers: list[ClarificationAnswer] = Field(default_factory=list)


class IntakeEvaluationMaterials(BaseModel):
    """Evaluator-visible material. Never pass hidden fields to the Intake agent."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: Path
    problem_path: str = "problem.md"
    data_path: str = "data"
    hidden_checker_path: str = "hidden/feasibility_check.py"
    gold_path: str = "gold_standard.json"
    intake_output: dict[str, Any]
    gold: IntakeGoldStandard
    dialogue: list[IntakeDialogueTurn] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relative_paths(self) -> IntakeEvaluationMaterials:
        for field_name in ("problem_path", "data_path", "hidden_checker_path", "gold_path"):
            raw = str(getattr(self, field_name)).replace("\\", "/")
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or raw in {"", "."}:
                raise ValueError(f"{field_name} must be a non-empty workspace-relative path")
        return self


class QuestionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    answer_type: Literal["choice", "boolean", "text", "invalid"]
    targets_gold_issue: bool
    grounded_in_visible_input: bool
    would_resolve_modeling_ambiguity: bool
    directly_answerable: bool
    options_cover_reasonable_interpretations: bool | None = None
    contains_unsupported_assumption: bool
    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=1)


class ComponentScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing: int = Field(ge=0, le=40)
    critical_issue: int = Field(ge=0, le=25)
    action_quality: int = Field(ge=0, le=20)
    visible_grounding: int = Field(ge=0, le=10)
    protocol: int = Field(ge=0, le=5)

    @property
    def raw_total(self) -> int:
        return (
            self.routing
            + self.critical_issue
            + self.action_quality
            + self.visible_grounding
            + self.protocol
        )


class IntakeJudgeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    evaluation_scope: Literal["first_turn", "full_dialogue"] = "first_turn"
    protocol_valid: bool
    observed_decision: Literal["clarify", "proceed", "invalid"]
    decision_correct: bool
    issue_identified: bool
    issue_explanation: str = Field(min_length=1)
    question_evaluations: list[QuestionEvaluation]
    problem_definition_adequate: bool | None
    unsupported_assumptions: list[str]
    clarification_rounds: int = Field(default=0, ge=0)
    answers_used_correctly: bool | None = None
    repeated_or_redundant_questions: list[str] = Field(default_factory=list)
    terminal_problem_definition_adequate: bool | None = None
    component_scores: ComponentScores
    total_score: int = Field(ge=0, le=100)
    applied_cap: int = Field(ge=0, le=100)
    verdict: Literal["pass", "partial", "fail", "protocol_failure"]
    rationale: str = Field(min_length=1)


INTAKE_EVALUATOR_SYSTEM_PROMPT = """\
You are the independent judge for a DecisionBrain Intake benchmark.

The tested Intake agent was allowed to see only problem.md and the task data. You receive
those same visible inputs, the hidden feasibility checker, a human-approved gold standard,
the agent's stage_outputs/intake.json, and, when present, the complete clarification dialogue.

Judge the Intake behavior, not whether the optimization problem was solved. The hidden checker
may verify that a visible omission or conflict affects evaluation, but checker-only unpublished
requirements are never evidence that Intake should have asked a question. If problem plus data
are complete and self-consistent, the correct decision is proceed even when the checker implements
a different contract.

The runtime route is binary: clarify or proceed. A gold blocking_conflict still expects clarify,
with a question that explicitly exposes the conflict and asks which visible interpretation is
authoritative. Clarification questions may use choice, boolean, or text answer types.

Score exactly five components: routing 0-40, critical_issue 0-25, action_quality 0-20,
visible_grounding 0-10, and protocol 0-5. For clarify, action_quality measures whether at least one
question is concrete, answerable, and sufficient to remove the material ambiguity. For proceed,
it measures whether problem_definition faithfully captures the visible contract without inventing
requirements. On proceed gold cases, critical_issue measures correctly recognizing that no material
visible issue exists rather than inventing one. Evaluate every emitted question separately. Do not
reward verbosity or keyword overlap without correct semantics.

For a full dialogue, separately judge whether each answer was used correctly in the next round,
whether the agent repeated an already answered question, and whether the terminal problem_definition
implements the clarified contract. Do not erase a weak first question merely because a later round
eventually converged.

Return one JSON object only. It must follow the requested report shape. Provide the uncapped
component scores; deterministic code will recompute total_score, applied_cap, and verdict.
"""


class IntakeEvaluator:
    """Read-only tool agent with deterministic score post-processing."""

    def __init__(
        self,
        llm: CompletionClient,
        tools: tuple[Tool, ...],
        *,
        max_turns: int = 20,
        require_source_reads: bool = True,
        max_protocol_attempts: int = 3,
    ) -> None:
        if max_protocol_attempts <= 0:
            raise ValueError("max_protocol_attempts must be positive")
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns
        self.require_source_reads = require_source_reads
        self.max_protocol_attempts = max_protocol_attempts

    async def evaluate(self, materials: IntakeEvaluationMaterials) -> IntakeJudgeReport:
        messages = self.build_messages(materials)
        last_error: IntakeEvaluationError | None = None
        for attempt in range(1, self.max_protocol_attempts + 1):
            content = await _run_reading_agent(
                self.llm,
                messages,
                self.tools,
                max_turns=self.max_turns,
                required_read_paths=(
                    _required_read_paths(materials)
                    if self.require_source_reads and attempt == 1
                    else ()
                ),
                required_read_prefixes=(
                    (f"{materials.data_path.rstrip('/')}/",)
                    if self.require_source_reads and attempt == 1
                    else ()
                ),
            )
            try:
                payload = _parse_json_object(content)
                report = IntakeJudgeReport.model_validate(payload)
                if report.task_id != materials.gold.task_id:
                    raise IntakeEvaluationError(
                        f"evaluator task_id mismatch: {report.task_id} != {materials.gold.task_id}"
                    )
            except (IntakeEvaluationError, ValueError) as exc:
                last_error = (
                    exc
                    if isinstance(exc, IntakeEvaluationError)
                    else IntakeEvaluationError(f"Intake evaluator report is invalid: {exc}")
                )
                if attempt == self.max_protocol_attempts:
                    break
                _append_protocol_retry(messages, content, last_error, attempt)
                continue
            report = normalize_report_context(report, materials)
            return apply_score_policy(report, materials.gold)
        raise IntakeEvaluationError(
            f"evaluator_protocol_exhausted after {self.max_protocol_attempts} attempts: "
            f"{last_error}"
        ) from last_error

    def build_messages(self, materials: IntakeEvaluationMaterials) -> MessageBuilder:
        messages = MessageBuilder.from_prompts(INTAKE_EVALUATOR_SYSTEM_PROMPT)
        messages.add_user(
            "Use the read-only tools to inspect all relevant source material before evaluating.\n"
            f"Problem: {materials.problem_path}\n"
            f"Visible data/profile: {materials.data_path}\n"
            f"Hidden checker: {materials.hidden_checker_path}\n"
            f"Approved gold audit: {materials.gold_path}\n\n"
            "<intake_output>\n"
            f"{json.dumps(materials.intake_output, ensure_ascii=False, indent=2)}\n"
            "</intake_output>\n\n"
            "<clarification_dialogue>\n"
            f"{json.dumps([turn.model_dump(mode='json') for turn in materials.dialogue], ensure_ascii=False, indent=2)}\n"
            "</clarification_dialogue>\n\n"
            f"Required report JSON schema:\n{json.dumps(IntakeJudgeReport.model_json_schema(), ensure_ascii=False)}"
        )
        _register_tools(messages, self.tools)
        return messages


INTAKE_ANSWERER_SYSTEM_PROMPT = """\
You simulate the benchmark user answering DecisionBrain Intake clarification questions.
You receive the visible problem and data, hidden checker, human-approved audit result, all previous
dialogue turns, and the questions from the current Intake round. Answer every actual question
dynamically and concretely according to the intended contract established by those materials.

Write as the user or benchmark author. Never mention a checker, hidden file, reference code, gold
label, or evaluation process in an answer. Do not volunteer unrelated requirements or solve the
optimization problem. A choice question should be answered using its option wording when an option
matches the intended contract. If the available materials genuinely do not determine an answer,
return cannot_determine; if a compound question cannot be answered unambiguously, return
ambiguous_question. Return JSON only.
"""


class IntakeAnswerer:
    """Controlled user simulator for one clarification round."""

    def __init__(
        self,
        llm: CompletionClient,
        tools: tuple[Tool, ...],
        *,
        max_turns: int = 20,
        require_source_reads: bool = True,
        max_protocol_attempts: int = 3,
    ) -> None:
        if max_protocol_attempts <= 0:
            raise ValueError("max_protocol_attempts must be positive")
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns
        self.require_source_reads = require_source_reads
        self.max_protocol_attempts = max_protocol_attempts

    async def answer(
        self,
        *,
        materials: IntakeEvaluationMaterials,
        questions: list[dict[str, Any]],
        previous_turns: list[IntakeDialogueTurn],
    ) -> list[ClarificationAnswer]:
        question_ids = [str(question.get("id") or "") for question in questions]
        if len(set(question_ids)) != len(question_ids):
            raise IntakeEvaluationError("clarification questions contain duplicate IDs")
        messages = MessageBuilder.from_prompts(INTAKE_ANSWERER_SYSTEM_PROMPT)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answers"],
            "properties": {
                "answers": {
                    "type": "array",
                    "items": ClarificationAnswer.model_json_schema(),
                }
            },
        }
        messages.add_user(
            f"Task: {materials.gold.task_id}\n"
            "Use the read-only tools to inspect the sources needed to answer this round.\n"
            f"Problem: {materials.problem_path}\n"
            f"Visible data/profile: {materials.data_path}\n"
            f"Hidden checker: {materials.hidden_checker_path}\n"
            f"Approved audit: {materials.gold_path}\n\n"
            "Previous turns:\n"
            f"{json.dumps([turn.model_dump(mode='json') for turn in previous_turns], ensure_ascii=False)}\n\n"
            f"Current questions: {json.dumps(questions, ensure_ascii=False)}\n"
            f"Required schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        _register_tools(messages, self.tools)
        last_error: IntakeEvaluationError | None = None
        for attempt in range(1, self.max_protocol_attempts + 1):
            content = await _run_reading_agent(
                self.llm,
                messages,
                self.tools,
                max_turns=self.max_turns,
                required_read_paths=(
                    _required_read_paths(materials)
                    if self.require_source_reads and attempt == 1
                    else ()
                ),
                required_read_prefixes=(
                    (f"{materials.data_path.rstrip('/')}/",)
                    if self.require_source_reads and attempt == 1
                    else ()
                ),
            )
            try:
                payload = _parse_json_object(content)
                raw_answers = payload.get("answers")
                if not isinstance(raw_answers, list):
                    raise IntakeEvaluationError("answerer report must contain an answers array")
                answers = [ClarificationAnswer.model_validate(item) for item in raw_answers]
                answer_ids = [answer.question_id for answer in answers]
                if len(set(answer_ids)) != len(answer_ids) or answer_ids != question_ids:
                    raise IntakeEvaluationError(
                        f"answerer question IDs mismatch: expected {question_ids}, got {answer_ids}"
                    )
            except (IntakeEvaluationError, ValueError) as exc:
                last_error = (
                    exc
                    if isinstance(exc, IntakeEvaluationError)
                    else IntakeEvaluationError(f"answerer report is invalid: {exc}")
                )
                if attempt == self.max_protocol_attempts:
                    break
                _append_protocol_retry(messages, content, last_error, attempt)
                continue
            return answers
        raise IntakeEvaluationError(
            f"answerer_protocol_exhausted after {self.max_protocol_attempts} attempts: {last_error}"
        ) from last_error


def _append_protocol_retry(
    messages: MessageBuilder,
    invalid_content: str,
    error: IntakeEvaluationError,
    attempt: int,
) -> None:
    messages.add_assistant(invalid_content)
    detail = str(error)
    if len(detail) > 4_000:
        detail = detail[:4_000] + "..."
    messages.add_user(
        "Your previous final response failed the required JSON protocol.\n"
        f"Protocol attempt: {attempt}\n"
        f"Validation error:\n{detail}\n\n"
        "Correct only the JSON structure and invalid fields while preserving the substantive "
        "answer. Return exactly one JSON object matching the schema. Do not include prose, "
        "Markdown fences, or additional fields. Do not call tools again unless the validation "
        "error cannot be corrected from the existing conversation."
    )


def build_intake_evaluation_tools(
    workspace: Path, settings: Settings, *, data_path: str = "data"
) -> tuple[Tool, ...]:
    """Expose only DecisionBrain's bounded, read-only workspace tools."""

    toolset = WorkspaceToolset(
        workspace,
        timeout_s=settings.solver_timeout,
        max_output_chars=settings.opt_workspace_max_output_chars,
        max_list_entries=settings.opt_workspace_max_list_entries,
        default_list_entries=settings.opt_workspace_default_list_entries,
        max_read_bytes=settings.opt_workspace_max_read_bytes,
    )
    allowed = {"list_files", "read_file", "search_file"}
    workspace_tools = tuple(tool for tool in toolset.tools(stage=None) if tool.name in allowed)
    return (*workspace_tools, *build_intake_data_tools(workspace, data_path=data_path))


def _register_tools(messages: MessageBuilder, tools: tuple[Tool, ...]) -> None:
    for tool in tools:
        messages.add_function_tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )


async def _run_reading_agent(
    llm: CompletionClient,
    messages: MessageBuilder,
    tools: tuple[Tool, ...],
    *,
    max_turns: int,
    required_read_paths: tuple[str, ...] = (),
    required_read_prefixes: tuple[str, ...] = (),
) -> str:
    tools_by_name = {tool.name: tool for tool in tools}
    read_paths: set[str] = set()
    for _ in range(max_turns):
        response = await llm.complete(messages.build(tool_choice="auto"))
        if not response.tool_calls:
            if response.content:
                missing = [path for path in required_read_paths if path not in read_paths]
                missing_prefixes = [
                    prefix
                    for prefix in required_read_prefixes
                    if not any(path.startswith(prefix) for path in read_paths)
                ]
                if missing or missing_prefixes:
                    required = missing + [f"{prefix}*" for prefix in missing_prefixes]
                    raise IntakeEvaluationError(
                        "reading agent finalized before reading required sources: "
                        + ", ".join(required)
                    )
                return response.content
            raise IntakeEvaluationError("reading agent returned no content or tool calls")
        messages.add_assistant(
            tool_calls=response.tool_calls,
            reasoning_content=response.reasoning,
        )
        for tool_call in response.tool_calls:
            called_name, called_arguments = _tool_call_name_arguments(tool_call)
            result = await _execute_read_tool(tool_call, tools_by_name)
            if (
                called_name == "read_file"
                and isinstance(called_arguments.get("path"), str)
                and not result.startswith("read_file error:")
            ):
                read_paths.add(called_arguments["path"].replace("\\", "/").lstrip("./"))
            elif (
                called_name in {"data_get", "data_keys", "data_slice"}
                and not result.startswith(f"{called_name} error:")
                and required_read_prefixes
            ):
                read_paths.add(f"{required_read_prefixes[0]}<stream-query>")
            messages.add_tool_result(str(tool_call.get("id") or ""), result)
    raise IntakeEvaluationError(f"reading agent exceeded {max_turns} turns")


async def _execute_read_tool(tool_call: dict[str, Any], tools: dict[str, Tool]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return "tool call is missing function"
    name = str(function.get("name") or "")
    tool = tools.get(name)
    if tool is None:
        return f"tool is not allowed: {name}"
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError as exc:
        return _tool_argument_contract(tool, (f"Arguments are not valid JSON: {exc.msg}",))
    issues = _tool_argument_issues(tool, arguments)
    if issues:
        return _tool_argument_contract(tool, issues)
    try:
        return await tool.handler(**arguments)
    except Exception as exc:
        return f"tool execution failed: {exc}"


def _tool_call_name_arguments(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return "", {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        return name, {}
    return name, arguments if isinstance(arguments, dict) else {}


def _required_read_paths(materials: IntakeEvaluationMaterials) -> tuple[str, ...]:
    return (
        materials.problem_path,
        materials.hidden_checker_path,
        materials.gold_path,
    )


def apply_score_policy(report: IntakeJudgeReport, gold: IntakeGoldStandard) -> IntakeJudgeReport:
    """Recompute totals and enforce routing/protocol caps independently of the judge."""

    raw_total = report.component_scores.raw_total
    expected_decision = (
        gold.expected_terminal_decision
        if report.evaluation_scope == "full_dialogue"
        else gold.expected_decision
    )
    if not report.protocol_valid or report.observed_decision == "invalid":
        cap = 0
        verdict = "protocol_failure"
    elif expected_decision == "clarify" and report.observed_decision == "proceed":
        cap = 25
        verdict = "fail"
    elif expected_decision == "proceed" and report.observed_decision == "clarify":
        cap = 40
        verdict = "fail"
    else:
        cap = 100
        final = min(raw_total, cap)
        verdict = "pass" if final >= 80 else "partial" if final >= 50 else "fail"
    total = min(raw_total, cap)
    return report.model_copy(update={"total_score": total, "applied_cap": cap, "verdict": verdict})


def normalize_report_context(
    report: IntakeJudgeReport, materials: IntakeEvaluationMaterials
) -> IntakeJudgeReport:
    """Replace judge-reported observable facts with values derived from artifacts."""

    observed = str(materials.intake_output.get("decision") or "")
    observed_decision = observed if observed in {"clarify", "proceed"} else "invalid"
    clarification_rounds = sum(
        1 for turn in materials.dialogue if turn.intake_output.get("decision") == "clarify"
    )
    scope = "full_dialogue" if clarification_rounds else "first_turn"
    expected = (
        materials.gold.expected_terminal_decision
        if scope == "full_dialogue"
        else materials.gold.expected_decision
    )
    return report.model_copy(
        update={
            "evaluation_scope": scope,
            "observed_decision": observed_decision,
            "decision_correct": observed_decision == expected,
            "clarification_rounds": clarification_rounds,
        }
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        payload = None
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            preview = text[:300].replace("\r", " ").replace("\n", " ")
            raise IntakeEvaluationError(
                f"evaluator did not return valid JSON: {exc.msg}; preview={preview!r}"
            ) from exc
    if not isinstance(payload, dict):
        raise IntakeEvaluationError("evaluator report must be a JSON object")
    return payload
