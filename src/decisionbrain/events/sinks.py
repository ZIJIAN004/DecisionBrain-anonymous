"""Asynchronous event sink implementations."""

import asyncio
from typing import Protocol

from .models import RunEvent


class EventSink(Protocol):
    async def handle(self, event: RunEvent) -> None:
        """Consume a fully assigned event."""


class EventRepository(Protocol):
    def append_event(self, event: RunEvent) -> RunEvent:
        """Persist and return an event."""


class CompositeEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        if not sinks:
            raise ValueError("CompositeEventSink requires at least one sink")
        self.sinks = sinks

    async def handle(self, event: RunEvent) -> None:
        for sink in self.sinks:
            await sink.handle(event)


class JsonlEventSink:
    def __init__(self, repository: EventRepository) -> None:
        self.repository = repository

    async def handle(self, event: RunEvent) -> None:
        await asyncio.to_thread(self.repository.append_event, event)
