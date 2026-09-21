"""Asynchronous event publisher."""

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from ..exceptions import EventPublishingError
from .models import RunEvent, RunEventDraft
from .sinks import EventSink


def _aware_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventPublisher:
    """Assign event identity and a strictly increasing sequence for one Run."""

    def __init__(
        self,
        sink: EventSink,
        run_id: str | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        event_id_factory: Callable[[], str] | None = None,
        initial_sequence: int = 1,
    ) -> None:
        if initial_sequence < 1:
            raise ValueError(f"initial_sequence must be >= 1; received {initial_sequence}")
        if run_id == "":
            raise ValueError("run_id cannot be empty")
        self._run_id = run_id
        self.sink = sink
        self._clock = clock or _aware_utc_now
        self._event_id_factory = event_id_factory or (lambda: uuid.uuid4().hex)
        self._next_sequence = initial_sequence
        self._failed = False
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            raise EventPublishingError("EventPublisher is not bound to a run_id")
        return self._run_id

    async def publish(self, draft: RunEventDraft) -> RunEvent:
        async with self._lock:
            run_id = self.run_id
            if self._failed:
                raise EventPublishingError(f"Publisher for Run {run_id} cannot continue after a delivery failure")
            timestamp = self._clock()
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("Publisher clock must return a timezone-aware datetime")
            event = RunEvent(
                event_id=self._event_id_factory(),
                run_id=run_id,
                sequence=self._next_sequence,
                timestamp=timestamp.astimezone(timezone.utc),
                type=draft.type,
                stage=draft.stage,
                level=draft.level,
                message=draft.message,
                payload=draft.payload,
                parent_event_id=draft.parent_event_id,
                duration_ms=draft.duration_ms,
            )
            # Validate JSON serialization before any sink receives the event.
            event.model_dump_json()
            dispatch = asyncio.create_task(self.sink.handle(event))
            try:
                await asyncio.shield(dispatch)
            except asyncio.CancelledError:
                # Cancellation may arrive after persistence but before later sinks finish.
                # Complete this delivery and advance the sequence before propagating it.
                try:
                    await dispatch
                except Exception:
                    self._failed = True
                    raise
                self._next_sequence += 1
                raise
            except Exception:
                self._failed = True
                raise
            else:
                self._next_sequence += 1
            return event
