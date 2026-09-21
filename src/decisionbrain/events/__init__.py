"""Unified structured event-stream interface for DecisionBrain."""

from .models import (
    RunEvent,
    RunEventDraft,
    EVENT_DIAGNOSTIC,
    EVENT_PROGRESS_REPORTED,
    EVENT_RUN_CREATED,
    EVENT_RUN_FINISHED,
    EVENT_RUN_RESUMED,
    EVENT_RUN_STARTED,
    EVENT_USER_MESSAGE,
    EventLevel,
)
from .publisher import EventPublisher
from .sinks import (
    CompositeEventSink,
    EventSink,
    JsonlEventSink,
)

__all__ = [
    "RunEventDraft",
    "RunEvent",
    "CompositeEventSink",
    "EVENT_DIAGNOSTIC",
    "EVENT_PROGRESS_REPORTED",
    "EVENT_RUN_CREATED",
    "EVENT_RUN_FINISHED",
    "EVENT_RUN_RESUMED",
    "EVENT_RUN_STARTED",
    "EVENT_USER_MESSAGE",
    "EventLevel",
    "EventPublisher",
    "EventSink",
    "JsonlEventSink",
]
