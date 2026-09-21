"""Structured event models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.models import AgentStage
from ..schema import VersionedModel


# Runtime lifecycle event types that are not derived from Core reports or outcomes.
EVENT_RUN_CREATED = "run_created"
EVENT_RUN_STARTED = "run_started"
EVENT_USER_MESSAGE = "user_message"
EVENT_DIAGNOSTIC = "diagnostic"
EVENT_RUN_RESUMED = "run_resumed"
EVENT_RUN_FINISHED = "run_finished"
EVENT_PROGRESS_REPORTED = "progress_reported"


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value


class EventRecord(VersionedModel):
    """Persisted event record versioned by VersionedModel."""


class RunEvent(EventRecord):
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp: datetime
    type: str
    stage: AgentStage | None = None
    level: EventLevel = EventLevel.INFO
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    _timestamp_is_aware = field_validator("timestamp")(_require_aware)


class RunEventDraft(BaseModel):
    """Runtime event awaiting identity and sequence assignment by the Publisher."""

    type: str
    stage: AgentStage | None = None
    level: EventLevel = EventLevel.INFO
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid", use_enum_values=False)
