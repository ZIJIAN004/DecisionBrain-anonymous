"""Run execution and live event subscriptions for the HTTP API."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from ..config import Settings
from ..exceptions import RunCapacityError
from ..events import (
    RunEvent,
)
from ..runtime import AgentRuntime
from ..run_storage import RunRepository, RunStatus

logger = logging.getLogger(__name__)


class EventStream:
    """Detachable event subscription and sink for one Runtime execution."""

    def __init__(
        self,
        *,
        close_on_pause: bool = True,
    ) -> None:
        self._queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self._attached = True
        self._close_on_pause = close_on_pause

    async def handle(self, event: RunEvent) -> None:
        """Implement EventSink for use inside CompositeEventSink."""
        if self._attached:
            self._queue.put_nowait(event)
        if event.type == "run_finished" or (
            self._close_on_pause and event.type == "clarification_requested"
        ):
            await self.finish()

    async def finish(self) -> None:
        if self._attached:
            self._queue.put_nowait(None)
            self._attached = False

    def detach(self) -> None:
        self._attached = False
        while not self._queue.empty():
            self._queue.get_nowait()

    async def events(self) -> AsyncIterator[RunEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event


class RuntimeEventHub:
    """Attachable event broadcast sink for one active Runtime."""

    def __init__(self, *, on_run_discovered: Callable[[str], None] | None = None) -> None:
        self._streams: set[EventStream] = set()
        self._on_run_discovered = on_run_discovered
        self._run_id: str | None = None

    def attach(self, stream: EventStream) -> None:
        self._streams.add(stream)

    async def handle(self, event: RunEvent) -> None:
        if self._run_id is None:
            self._run_id = event.run_id
            if self._on_run_discovered is not None:
                self._on_run_discovered(event.run_id)
        for stream in tuple(self._streams):
            await stream.handle(event)


class RunExecutionManager:
    """Keep Run lifecycles independent of individual SSE connections."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repository: RunRepository | None = None,
        runtime_factory: Callable[[], AgentRuntime] | None = None,
    ) -> None:
        self.settings = settings or (repository.settings if repository is not None else Settings())
        self.repository = repository or RunRepository(self.settings)
        self.runtime_factory = runtime_factory or (
            lambda: AgentRuntime(self.settings, repository=self.repository)
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._runtimes: dict[str, AgentRuntime] = {}
        self._event_hubs: dict[str, RuntimeEventHub] = {}
        self._all_tasks: set[asyncio.Task[None]] = set()
        self._active_run_tasks: set[asyncio.Task[None]] = set()
        self.max_active_runs = self.settings.max_active_runs

    def start(
        self,
        *,
        workspace: Path,
        prompt: str,
        title: str = "",
        cleanup_workspace: bool = False,
    ) -> EventStream:
        self._prune_active_run_tasks()
        if len(self._active_run_tasks) >= self.max_active_runs:
            raise RunCapacityError(
                f"Active Run limit {self.max_active_runs} reached; wait for an existing Run to finish"
            )
        runtime = self.runtime_factory()
        task: asyncio.Task[None]
        hub: RuntimeEventHub

        def register(run_id: str) -> None:
            self._tasks[run_id] = task
            self._runtimes[run_id] = runtime
            self._event_hubs[run_id] = hub

        stream = EventStream()
        hub = RuntimeEventHub(on_run_discovered=register)
        hub.attach(stream)

        async def execute() -> None:
            try:
                await runtime.run(
                    workspace=workspace,
                    command=title.strip() or prompt.strip(),
                    argv=(),
                    config={"source": "api", "title": title.strip()},
                    initial_input=prompt,
                    sink=hub,
                )
            except Exception:
                logger.exception("API Run execution failed")
            finally:
                await stream.finish()
                for run_id, active in list(self._tasks.items()):
                    if active is task:
                        self._tasks.pop(run_id, None)
                        self._runtimes.pop(run_id, None)
                        self._event_hubs.pop(run_id, None)
                if cleanup_workspace:
                    await asyncio.to_thread(shutil.rmtree, workspace, True)

        task = asyncio.create_task(execute())
        self._all_tasks.add(task)
        self._active_run_tasks.add(task)
        task.add_done_callback(self._all_tasks.discard)
        task.add_done_callback(self._active_run_tasks.discard)
        return stream

    def active_run_count(self) -> int:
        self._prune_active_run_tasks()
        return len(self._active_run_tasks)

    def _prune_active_run_tasks(self) -> None:
        for task in list(self._active_run_tasks):
            if task.done():
                self._active_run_tasks.discard(task)

    def submit_user_message(self, *, run_id: str, message: str) -> EventStream:
        stream = EventStream()
        runtime = self._runtimes.get(run_id)
        hub = self._event_hubs.get(run_id)
        if runtime is None or hub is None:
            from ..exceptions import RunStateError

            raise RunStateError(f"Run {run_id} has no Runtime waiting for input")
        hub.attach(stream)

        async def submit() -> None:
            try:
                await runtime.submit_user_message(user_message=message)
            except Exception:
                logger.exception("Failed to process user input for API Run %s", run_id)
                await stream.finish()

        task = asyncio.create_task(submit())
        self._all_tasks.add(task)
        task.add_done_callback(self._all_tasks.discard)
        return stream

    async def cancel(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            raise KeyError(run_id)
        runtime = self._runtimes.get(run_id)
        if runtime is not None:
            runtime.cancel()
        task.cancel()
        await task

    async def reconcile_orphans(self) -> None:
        """Terminate non-terminal Runs left by a previous process at startup."""
        for run_record in self.repository.list_runs():
            if run_record.status not in {RunStatus.CREATED, RunStatus.RUNNING}:
                continue
            self.repository.update_record(run_record.run_id, status=RunStatus.CANCELLED)

    async def shutdown(self) -> None:
        tasks = [task for task in self._all_tasks if not task.done()]
        for task in tasks:
            for run_id, active in list(self._tasks.items()):
                if active is task and (runtime := self._runtimes.get(run_id)) is not None:
                    runtime.cancel()
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
