"""Shared Runtime orchestration helpers for CLI commands."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable
from typing import Any, Callable, Literal

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ...config import Settings
from ...events import RunEvent, CompositeEventSink
from ...run_storage import RunRepository
from ...runtime import AgentRuntime
from ...run_storage import RunStatus
from ..rich_events import RichEventSink, RichRenderer


ClarificationCommandResult = Literal["handled", "abort", "answer"]


def _stdin_is_tty() -> bool:
    if sys.stdin.isatty():
        return True
    try:
        descriptor = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    os.close(descriptor)
    return True


def resolve_cli_config(
    *,
    settings: Settings,
    debug: bool = False,
) -> Settings:
    overrides: dict[str, object] = {}
    if debug:
        overrides["debug"] = True
    return settings.model_copy(update=overrides)


def build_runtime_components(
    *, console: Console, settings: Settings, runtime: AgentRuntime, json_mode: bool = False
) -> tuple[RunRepository, CompositeEventSink, "RunEventObserver"]:
    repository = runtime.repository
    observer = RunEventObserver()
    event_console = Console(stderr=True) if json_mode else console
    renderer = RichRenderer(debug=settings.debug, show_timestamps=settings.debug)
    sink = CompositeEventSink(
        observer,
        RichEventSink(event_console, renderer),
    )
    return repository, sink, observer


class RunEventObserver:
    """Observe event state for the CLI without making business decisions."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.status: RunStatus | None = None
        self.pending_clarification_questions: list[dict[str, Any]] = []
        self.summary: str = ""
        self.error: str | None = None
        self.duration_ms: int | None = None
        self._first_timestamp = None
        self.change_count = 0

    async def handle(self, event: RunEvent) -> None:
        try:
            if self.run_id is None:
                self.run_id = event.run_id
            if self._first_timestamp is None:
                self._first_timestamp = event.timestamp

            if event.type == "clarification_requested":
                questions = event.payload.get("questions", [])
                self.pending_clarification_questions = (
                    questions if isinstance(questions, list) else []
                )
                self.status = RunStatus.NEEDS_CLARIFICATION
                self.summary = event.message
                return

            if event.type == "user_message":
                self.pending_clarification_questions = []
                self.status = RunStatus.RUNNING
                return

            if event.type == "error":
                self.error = event.message
                return

            if event.type != "run_finished":
                return

            raw_status = event.payload.get("status")
            if isinstance(raw_status, str):
                try:
                    self.status = RunStatus(raw_status)
                except ValueError:
                    self.status = RunStatus.FAILED
            self.pending_clarification_questions = []
            if self.status is RunStatus.SUCCEEDED:
                self.summary = event.message
            if self.status is RunStatus.FAILED:
                self.error = str(event.payload.get("error") or self.error or event.message)
            if self._first_timestamp is not None:
                delta = event.timestamp - self._first_timestamp
                self.duration_ms = int(delta.total_seconds() * 1000)
        finally:
            self.change_count += 1


def _question_label(question: dict[str, Any], index: int) -> str:
    identifier = question.get("id") or index
    label = str(identifier)
    if not label.lower().startswith("q"):
        label = f"Q{label}"
    return label


def _question_text(question: dict[str, Any]) -> str:
    text = question.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return str(question)


def _show_clarification_question(
    question: dict[str, Any],
    *,
    index: int,
    total: int,
    console: Console,
) -> None:
    title = _question_label(question, index)
    if total > 1:
        title = f"{title} ({index}/{total})"

    body = Text(_question_text(question))
    options = question.get("options")
    if isinstance(options, list) and options:
        body.append("\n\nOptions:")
        body.append(" / ".join(str(option) for option in options))
    console.print(Panel(body, title=title, border_style="magenta"))


def _format_clarification_answers(
    questions: list[dict[str, Any]],
    answers: list[str],
) -> str:
    if not questions:
        return answers[0]

    blocks = ["Clarification answers:"]
    for index, (question, answer) in enumerate(zip(questions, answers, strict=False), start=1):
        blocks.append(
            "\n".join(
                (
                    f"[{_question_label(question, index)}] {_question_text(question)}",
                    f"Answer: {answer}",
                )
            )
        )
    return "\n\n".join(blocks)


def collect_clarification_answers(
    questions: list[dict[str, Any]],
    *,
    console: Console,
    read_answer: Callable[[str], str],
    handle_command: Callable[[str], ClarificationCommandResult] | None = None,
) -> str | None:
    """Collect one clarification round and combine it into a Runtime user message."""
    if not questions:
        while True:
            answer = read_answer("Clarification answer> ").strip()
            if answer:
                return answer

    answers: list[str] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        while True:
            _show_clarification_question(
                question,
                index=index,
                total=total,
                console=console,
            )
            prompt = "Clarification answer> " if total == 1 else f"Clarification answer {index}/{total}> "
            answer = read_answer(prompt).strip()
            if not answer:
                continue
            if answer.startswith("/") and handle_command is not None:
                action = handle_command(answer)
                if action == "handled":
                    continue
                if action == "abort":
                    return None
            answers.append(answer)
            break
    return _format_clarification_answers(questions, answers)


async def run_runtime_with_user_feedback(
    *,
    runtime: AgentRuntime,
    run_call: Callable[[], Awaitable[None]],
    observer: RunEventObserver,
    console: Console,
    read_answer: Callable[[str], str],
    interactive: bool,
    handle_command: Callable[[str], ClarificationCommandResult] | None = None,
) -> None:
    task = asyncio.create_task(run_call())
    seen_changes = observer.change_count
    while not task.done():
        if observer.change_count == seen_changes:
            await asyncio.sleep(0.01)
            continue
        seen_changes = observer.change_count
        while observer.status is RunStatus.NEEDS_CLARIFICATION and not task.done():
            if observer.run_id is None:
                raise RuntimeError("Runtime did not publish a run_id")
            if not interactive:
                task.cancel()
                await task
                return
            try:
                answer = collect_clarification_answers(
                    observer.pending_clarification_questions,
                    console=console,
                    read_answer=read_answer,
                    handle_command=handle_command,
                )
            except EOFError:
                task.cancel()
                await task
                return
            if answer is None:
                task.cancel()
                await task
                return
            await runtime.submit_user_message(user_message=answer)
            break
    await task


def output_result(
    run_id: str,
    *,
    repository: RunRepository,
    observer: RunEventObserver,
    console: Console,
    json_mode: bool = False,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    run_record = repository.read_record(run_id)
    run_dir = repository.run_path(run_id)
    artifacts = repository.list_artifacts(run_id)
    summary = observer.summary
    if not summary and run_record.result_summary is not None:
        summary = json.dumps(run_record.result_summary, ensure_ascii=False)
    error = run_record.error_summary or observer.error
    report_path = run_dir / "agent-run.html"
    if json_mode:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": run_record.status.value,
            "summary": summary,
            "artifacts": [
                {
                    "relative_path": artifact.relative_path,
                    "category": artifact.category,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in artifacts
            ],
            "error": error,
            "report": str(report_path) if report_path.is_file() else None,
        }
        if observer.duration_ms is not None:
            payload["duration_ms"] = observer.duration_ms
        if extra_fields:
            payload.update(extra_fields)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    console.print(f"\n[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Status:[/bold] {run_record.status.value}")
    if summary:
        console.print(f"[bold]Summary:[/bold] {summary}")
    if artifacts:
        console.print(f"[bold]Artifacts ({len(artifacts)}):[/bold]")
        for artifact in artifacts:
            console.print(f"  - {artifact.relative_path}")
    if observer.duration_ms is not None:
        console.print(f"[bold]Duration:[/bold] {observer.duration_ms}ms")
    if report_path.is_file():
        console.print(f"[bold]Full Run report:[/bold] {report_path}")
    console.print(f"[dim]Run directory: {run_dir}[/dim]")
