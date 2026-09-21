"""Shared Core models for inputs, state, stages, reports, and outcomes."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class AgentStage(str, Enum):
    INTAKE = "intake"
    PROBLEM_CONTRACT = "problem_contract"
    ALGORITHM_DESIGN = "algorithm_design"
    GUROBI_FORMULATOR = "gurobi_formulator"
    SOLVING = "solving"
    FEASIBILITY_REVIEW = "feasibility_review"
    EXPLANATION = "explanation"


class StreamChannel(str, Enum):
    REASONING = "reasoning"
    CODE = "code"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactDraft(_StrictModel):
    """Artifact declared by the Agent and written by Runtime."""

    relative_path: str
    content: str
    media_type: str = "application/json"
    category: str = "artifact"
    description: str = ""


class ClarificationQuestion(_StrictModel):
    """Structured question used for clarification."""

    id: str
    text: str
    answer_type: str = "text"
    options: tuple[str, ...] = ()
    context: str = ""


# Core reports progress to Runtime through AgentReport.
class StageStarted(_StrictModel):
    kind: Literal["stage_started"] = "stage_started"
    stage: AgentStage
    message: str


class StageFinished(_StrictModel):
    kind: Literal["stage_finished"] = "stage_finished"
    stage: AgentStage
    message: str = ""


class AssistantMessage(_StrictModel):
    kind: Literal["assistant_message"] = "assistant_message"
    message: str
    stage: AgentStage | None = None
    stream: StreamChannel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactProduced(_StrictModel):
    kind: Literal["artifact_produced"] = "artifact_produced"
    message: str
    artifacts: tuple[ArtifactDraft, ...] = Field(min_length=1)
    stage: AgentStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Warning(_StrictModel):
    kind: Literal["warning"] = "warning"
    message: str
    stage: AgentStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


AgentReport: TypeAlias = (
    StageStarted | StageFinished | AssistantMessage | ArtifactProduced | Warning
)


class StageCompleted(_StrictModel):
    """Generic successful stage result with stage-specific data in data."""

    stage: AgentStage
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class StageNeedsClarification(_StrictModel):
    """A stage requires user clarification."""

    stage: AgentStage
    message: str
    questions: tuple[ClarificationQuestion, ...] = ()


class StageFailed(_StrictModel):
    """A stage execution failed."""

    stage: AgentStage
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


StageResult: TypeAlias = StageCompleted | StageNeedsClarification | StageFailed


class NeedsClarification(_StrictModel):
    kind: Literal["needs_clarification"] = "needs_clarification"
    message: str
    questions: tuple[ClarificationQuestion, ...] = Field(min_length=1)


class Succeeded(_StrictModel):
    kind: Literal["succeeded"] = "succeeded"
    message: str
    summary: dict[str, Any] = Field(default_factory=dict)


class Failed(_StrictModel):
    kind: Literal["failed"] = "failed"
    message: str
    stage: AgentStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


AgentOutcome: TypeAlias = NeedsClarification | Succeeded | Failed


class ChatRequest(_StrictModel):
    """OpenAI-compatible chat request assembled by MessageBuilder."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str = "auto"


class ChatResponse:
    """Non-streaming LLM result mapped from an OpenAI chat completion."""

    __slots__ = ("content", "reasoning", "tool_calls", "raw")

    def __init__(
        self,
        content: str | None = None,
        reasoning: str | None = None,
        tool_calls: list[dict] | None = None,
        raw: dict | None = None,
    ) -> None:
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"ChatResponse(content={self.content!r}, reasoning={_truncate(self.reasoning)!r}"
            f", tool_calls={self.tool_calls!r})"
        )


def _truncate(text: str | None, max_len: int = 80) -> str | None:
    if text is None:
        return None
    return text if len(text) <= max_len else text[:max_len] + "..."


class CoreConfig(_StrictModel):
    """Immutable Core configuration constructed from Settings by Runtime."""

    solver_timeout: int = Field(ge=1)
    prompts_dir: str
    workspace_root: str
    resume_poll_interval_seconds: float = Field(gt=0)


class AgentInput(_StrictModel):
    problem_description: str = Field(min_length=1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AgentState:
    """Persistable and recoverable Agent state."""

    problem_description: str
    schema_version: str = "2"
    problem_definition: dict[str, Any] | None = None
    conversation: tuple[dict[str, Any], ...] = ()
    problem_contract: dict[str, Any] | None = None
    algorithm_design: dict[str, Any] | None = None
    gurobi_formulation: dict[str, Any] | None = None
    generated_code: str = ""
    solver_result: dict[str, Any] | None = None
    feasibility_review: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    current_stage: AgentStage | None = AgentStage.INTAKE
    pending_clarification_questions: tuple[dict[str, Any], ...] = ()
    clarification_rounds: int = 0
    feasibility_rounds: int = 0
    feasibility_directives: tuple[dict[str, Any], ...] = ()
    feasibility_audits: tuple[dict[str, Any], ...] = ()
    instance_infeasibility_reviews: tuple[dict[str, Any], ...] = ()
    pending_workspace_reset: bool = False
    pending_feasibility_handoff: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != "2":
            raise ValueError(f"Unsupported AgentState schema_version: {self.schema_version!r}")
        object.__setattr__(self, "current_stage", self._coerce_current_stage(self.current_stage))
        for name in self._tuple_field_names():
            object.__setattr__(self, name, tuple(getattr(self, name) or ()))

    @classmethod
    def _field_names(cls) -> set[str]:
        return {item.name for item in fields(cls)}

    @classmethod
    def _tuple_field_names(cls) -> set[str]:
        return {
            "conversation",
            "pending_clarification_questions",
            "feasibility_directives",
            "feasibility_audits",
            "instance_infeasibility_reviews",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentState":
        known = cls._field_names()
        unknown = set(value) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"AgentState checkpoint contains unknown fields: {names}")
        kwargs = {key: value[key] for key in known if key in value}
        return cls(**kwargs)

    @staticmethod
    def _coerce_current_stage(value: Any) -> AgentStage | None:
        if value is None:
            return None
        if isinstance(value, AgentStage):
            return value
        return AgentStage(str(value))

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _jsonable(getattr(self, item.name)) for item in fields(self)}

    def with_updates(self, **changes: Any) -> "AgentState":
        unknown = set(changes) - self._field_names()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise AttributeError(f"AgentState has no fields: {names}")
        return replace(self, **changes)

    def append_to(
        self,
        field_name: str,
        item: dict[str, Any],
        *,
        max_items: int | None = None,
    ) -> "AgentState":
        if field_name not in self._tuple_field_names():
            raise AttributeError(f"AgentState field {field_name!r} does not support append_to")
        values = (*getattr(self, field_name), item)
        if max_items is not None:
            values = values[-max_items:]
        return self.with_updates(**{field_name: values})


@dataclass(frozen=True, slots=True)
class StateUpdated:
    """Internal Core state update that Runtime does not publish to users."""

    state: AgentState
