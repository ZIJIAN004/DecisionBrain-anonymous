"""Rich event rendering and output sink for the CLI."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

from ..core.models import AgentStage
from ..events import (
    RunEvent,
    EVENT_PROGRESS_REPORTED,
    EventLevel,
)


DISPLAY_TIMEZONE = timezone(timedelta(hours=8))

STAGE_LABELS = {
    AgentStage.INTAKE: "Intake",
    AgentStage.PROBLEM_CONTRACT: "Problem Contract",
    AgentStage.ALGORITHM_DESIGN: "Algorithm Design",
    AgentStage.SOLVING: "Solving",
    AgentStage.FEASIBILITY_REVIEW: "Feasibility Review",
    AgentStage.EXPLANATION: "Explanation",
}


ROLE_COLORS: dict[str, str] = {
    "system": "bold blue",
    "developer": "bold magenta",
    "user": "bold green",
    "assistant": "bold yellow",
    "tool": "bold cyan",
}


class RichRenderer:
    def __init__(
        self,
        *,
        debug: bool = False,
        show_timestamps: bool = False,
    ) -> None:
        self.debug = debug
        self.show_timestamps = show_timestamps
        self._llm_stream_channels: dict[str, str] = {}
        self._run_started_at: dict[str, datetime] = {}
        self._llm_msg_counts: dict[str, int] = {}

    def should_render(self, event: RunEvent) -> bool:
        if self.debug:
            return True
        return event.level is not EventLevel.DEBUG

    def render(self, event: RunEvent, console: Console) -> None:
        if not self.should_render(event):
            return
        if self._is_redundant_stream_preview(event):
            return

        prefix = self._prefix(event)
        if event.type in {"stage_started", "stage_finished"}:
            self._render_stage(event, console, prefix)
        elif event.type == "assistant_message":
            console.print(
                Panel(Markdown(event.message), title=f"{prefix}Assistant", border_style="blue")
            )
        elif event.type == "user_message":
            console.print(Panel(Text(event.message), title=f"{prefix}User", border_style="cyan"))
        elif event.type == "clarification_requested":
            # Intake clarification is already shown in the AssistantMessage panel.
            if event.stage is AgentStage.INTAKE:
                return
            console.print(
                Panel(Text(event.message), title=f"{prefix}Clarification Required", border_style="magenta")
            )
        elif event.type == "artifact_created":
            path = str(event.payload.get("relative_path", event.message))
            console.print(Text.assemble(prefix, ("Artifact: ", "bold green"), path))
        elif event.type == EVENT_PROGRESS_REPORTED:
            self._render_progress(event, console, prefix)
        elif event.type in {"warning", "error"}:
            style = "yellow" if event.type == "warning" else "red"
            title = "Warning" if event.type == "warning" else "Error"
            console.print(Panel(Text(event.message), title=f"{prefix}{title}", border_style=style))
        elif event.type == "run_finished":
            self._render_finished(event, console, prefix)
        elif event.type == "diagnostic":
            console.print(Text.assemble(prefix, (event.message, "dim cyan")))
        elif event.type == "llm_stream_started":
            self._render_llm_stream_started(event, console, prefix)
        elif event.type == "llm_stream_chunk":
            self._render_llm_stream_chunk(event, console, prefix)
        elif event.type == "llm_stream_finished":
            self._render_llm_stream_finished(event, console, prefix)
        elif event.type == "tool_call":
            self._render_tool_call(event, console, prefix)
        elif event.type == "tool_result":
            self._render_tool_result(event, console, prefix)
        else:
            console.print(Text.assemble(prefix, (event.message, "white")))

    def _is_redundant_stream_preview(self, event: RunEvent) -> bool:
        if not self.debug or event.type != "assistant_message":
            return False
        return event.payload.get("stream") in {"reasoning", "code"}

    def _prefix(self, event: RunEvent) -> str:
        if not self.show_timestamps:
            return ""
        return f"[{event.timestamp.astimezone(DISPLAY_TIMEZONE).strftime('%H:%M:%S')}] "

    def _render_stage(self, event: RunEvent, console: Console, prefix: str) -> None:
        stage = STAGE_LABELS.get(event.stage, event.stage.value if event.stage else "Stage")
        if event.type == "stage_started":
            console.print(Rule(Text(f"{prefix}{stage}"), style="blue"))
        else:
            duration = f" ({event.duration_ms} ms)" if event.duration_ms is not None else ""
            console.print(Text(f"{prefix}{stage} completed{duration}", style="green"))
            if event.message:
                console.print(Text(event.message))

    def _render_finished(self, event: RunEvent, console: Console, prefix: str) -> None:
        status = str(event.payload.get("status", "finished"))
        objective = event.payload.get("objective")
        lines = [Text(event.message), Text(f"Status: {status}", style="bold")]
        if objective is not None:
            lines.append(Text(f"Objective: {objective}"))
        console.print(Panel(Group(*lines), title=f"{prefix}Run Completed", border_style="green"))

    def _render_progress(self, event: RunEvent, console: Console, prefix: str) -> None:
        summary = str(event.payload.get("summary") or event.message)
        next_step = str(event.payload.get("next_step") or "").strip()
        lines = [Text(summary)]
        if next_step:
            lines.append(Text(f"Next step: {next_step}", style="dim"))
        console.print(Panel(Group(*lines), title=f"{prefix}Progress Update", border_style="cyan"))

    def _render_llm_stream_started(self, event: RunEvent, console: Console, prefix: str) -> None:
        role = str(event.payload.get("role") or "llm")
        console.print(Rule(Text(f"{prefix}LLM {role} stream"), style="magenta"))
        self._render_llm_input(event, console, prefix)

    def _render_llm_input(self, event: RunEvent, console: Console, prefix: str) -> None:
        messages = event.payload.get("messages")
        if not isinstance(messages, list):
            return

        run_id = event.run_id
        prev_count = self._llm_msg_counts.get(run_id, 0)
        current_count = len(messages)
        delta = current_count - prev_count
        self._llm_msg_counts[run_id] = current_count

        model = str(event.payload.get("model") or "")
        reasoning_effort = str(event.payload.get("reasoning_effort") or "")
        details = []
        if model:
            details.append(f"model={model}")
        if reasoning_effort:
            details.append(f"reasoning_effort={reasoning_effort}")
        count_text = f"messages={current_count}"
        if delta > 0 and prev_count > 0:
            count_text += f" (+{delta})"
        details.append(count_text)
        console.print(Text(f"{prefix}Actual LLM input: " + "  ".join(details), style="bold cyan"))

        # Render only new messages after the first snapshot.
        start = prev_count if delta > 0 else 0
        new_messages: list[dict[str, Any]] = messages[start:]

        for msg in new_messages:
            role = str(msg.get("role") or "?")
            style = ROLE_COLORS.get(role, "white")
            content = str(msg.get("content") or "")

            # Live tool events already preserve execution order; snapshots only track counts.
            if isinstance(msg.get("tool_calls"), list) or role == "tool":
                continue

            # Ordinary assistant or user text.
            console.print(
                Panel(
                    Text(content),
                    title=f"{role}",
                    border_style=style.split()[-1] if " " in style else style,
                    title_align="left",
                )
            )

    def _render_llm_stream_chunk(self, event: RunEvent, console: Console, prefix: str) -> None:
        request_id = str(event.payload.get("request_id") or "llm")
        role = str(event.payload.get("role") or "llm")
        channel = str(event.payload.get("channel") or "content")
        previous_channel = self._llm_stream_channels.get(request_id)
        if previous_channel != channel:
            if previous_channel is not None:
                console.print()
            label = "Reasoning" if channel == "reasoning" else "Final output"
            label_style = "dim" if channel == "reasoning" else "white"
            console.print(Text(f"{prefix}LLM {role} · {label}", style=label_style))
            self._llm_stream_channels[request_id] = channel

        chunk_style = "dim" if channel == "reasoning" else "white"
        console.print(Text(event.message, style=chunk_style), end="")

    def _render_llm_stream_finished(self, event: RunEvent, console: Console, prefix: str) -> None:
        request_id = str(event.payload.get("request_id") or "llm")
        role = str(event.payload.get("role") or "llm")
        if request_id in self._llm_stream_channels:
            console.print()
            self._llm_stream_channels.pop(request_id, None)
        content_chars = event.payload.get("content_chars", 0)
        reasoning_chars = event.payload.get("reasoning_chars", 0)
        status = str(event.payload.get("status") or "ok")
        style = "green" if status == "ok" else "red"
        console.print(
            Text(
                f"{prefix}LLM {role} completed: {content_chars} output characters, "
                f"{reasoning_chars} reasoning characters",
                style=style,
            )
        )

    def _render_tool_call(self, event: RunEvent, console: Console, prefix: str) -> None:
        tool_name = str(event.payload.get("tool_name") or "?")
        arguments = event.payload.get("arguments") or {}
        tree = Tree(Text(f"{prefix}🔧 tool_call: {tool_name}", style="bold cyan"))
        tree.add(self._format_tool_args(tool_name, arguments))
        console.print(tree)

    def _render_tool_result(self, event: RunEvent, console: Console, prefix: str) -> None:
        tool_name = str(event.payload.get("tool_name") or "")
        result = event.message[:300]
        label = f"{tool_name} →" if tool_name else "→"
        console.print(Text(f"{prefix}   ✅ {label} {result}", style="green"))

    @staticmethod
    def _format_tool_args(tool_name: str, args_value: Any) -> Text:
        """Format shell arguments as code and all other tool arguments as JSON."""
        try:
            args = json.loads(args_value) if isinstance(args_value, str) else args_value
        except Exception:
            return Text(str(args_value), style="dim")

        if not isinstance(args, dict):
            return Text(json.dumps(args, ensure_ascii=False, indent=2), style="dim")

        if tool_name == "shell" and isinstance(args.get("command"), str):
            cmd = args["command"]
            workdir = args.get("workdir", ".")
            result = Text(style="dim")
            result.append(f"workdir: {workdir}\n")
            result.append("command:\n")
            result.append(cmd, style="bold white")
            return result

        return Text(json.dumps(args, ensure_ascii=False, indent=2), style="dim")

    def _display_duration_ms(self, event: RunEvent) -> int:
        if event.duration_ms is not None:
            return event.duration_ms
        started_at = self._run_started_at.setdefault(event.run_id, event.timestamp)
        elapsed = event.timestamp - started_at
        return max(0, int(elapsed.total_seconds() * 1000))

    def _payload_tree(self, payload: dict[str, Any]) -> Tree:
        tree = Tree(Text("payload", style="bold dim"), guide_style="dim")
        self._add_payload_children(tree, payload)
        return tree

    def _add_payload_children(self, tree: Tree, value: Any) -> None:
        if isinstance(value, dict):
            if not value:
                tree.add(Text("(empty)", style="dim"))
                return
            for key, item in value.items():
                self._add_payload_item(tree, str(key), item)
            return

        if isinstance(value, (list, tuple)):
            if not value:
                tree.add(Text("(empty)", style="dim"))
                return
            for index, item in enumerate(value):
                self._add_payload_item(tree, f"[{index}]", item)
            return

        tree.add(self._format_scalar(value))

    def _add_payload_item(self, tree: Tree, label: str, value: Any) -> None:
        if isinstance(value, dict):
            branch = tree.add(Text.assemble((label, "cyan"), (f" ({len(value)} fields)", "dim")))
            self._add_payload_children(branch, value)
            return

        if isinstance(value, (list, tuple)):
            branch = tree.add(Text.assemble((label, "cyan"), (f" ({len(value)} items)", "dim")))
            self._add_payload_children(branch, value)
            return

        tree.add(Text.assemble((label, "cyan"), ": ", self._format_scalar(value)))

    def _format_scalar(self, value: Any) -> Text:
        if value is None:
            return Text("null", style="italic dim")
        if isinstance(value, bool):
            return Text(str(value).lower(), style="magenta")
        if isinstance(value, (int, float)):
            return Text(str(value), style="green")

        text = str(value).replace("\r\n", "\n").replace("\n", "\\n")
        return Text(text, style="white")


class RichEventSink:
    def __init__(self, console: Console, renderer: RichRenderer | None = None) -> None:
        self.console = console
        self.renderer = renderer or RichRenderer()

    async def handle(self, event: RunEvent) -> None:
        self.renderer.render(event, self.console)
