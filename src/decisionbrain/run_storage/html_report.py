"""Self-contained HTML report for one complete Agent run."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..events.models import RunEvent
from .io import atomic_write_text
from .models import RunRecord


_HIDDEN_EVENT_TYPES = {
    "diagnostic",
    "llm_stream_started",
    "llm_stream_chunk",
    "llm_stream_finished",
    "tool_call",
    "tool_result",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _pre(value: Any, *, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<pre{class_attr}>{_escape(value)}</pre>"


def _parse_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class AgentHtmlReport:
    """Collect useful runtime events and complete LLM turns, then render one HTML file."""

    def __init__(self) -> None:
        self._events: list[RunEvent] = []
        self._turns: list[dict[str, Any]] = []

    async def handle(self, event: RunEvent) -> None:
        if event.type in _HIDDEN_EVENT_TYPES:
            return
        if event.type == "assistant_message" and event.payload.get("stream") in {
            "reasoning",
            "code",
        }:
            return
        self._events.append(event)

    async def record_turn(self, turn: Mapping[str, Any]) -> None:
        value = dict(turn)
        value["sequence"] = len(self._turns) + 1
        value["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._turns.append(value)

    def write(self, path: Path, record: RunRecord) -> None:
        atomic_write_text(path, self.render(record))

    def render(self, record: RunRecord) -> str:
        stages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for turn in self._turns:
            stages[str(turn.get("stage") or "unknown")].append(turn)

        stage_sections = "".join(
            self._render_stage(stage, turns) for stage, turns in stages.items()
        )
        if not stage_sections:
            stage_sections = '<p class="empty">This Run produced no LLM turns.</p>'

        return "".join(
            (
                '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width,initial-scale=1">',
                f"<title>DecisionBrain Run {_escape(record.run_id)}</title>",
                f"<style>{self._styles()}</style></head><body>",
                '<header class="run-header"><div><p class="eyebrow">DecisionBrain Agent Run</p>',
                f"<h1>{_escape(record.command)}</h1>",
                f'<p class="run-id">{_escape(record.run_id)}</p></div>',
                '<dl class="run-meta">',
                f"<div><dt>Status</dt><dd>{_escape(record.status.value)}</dd></div>",
                f"<div><dt>Started</dt><dd>{_escape(record.created_at.isoformat())}</dd></div>",
                f"<div><dt>Finished/updated</dt><dd>{_escape(record.updated_at.isoformat())}</dd></div>",
                f"<div><dt>Workspace</dt><dd>{_escape(record.workspace)}</dd></div>",
                "</dl></header><main>",
                '<details class="run-config"><summary>Run configuration</summary>',
                _pre(_json_text(record.config), css_class="json"),
                "</details>",
                '<section class="timeline"><h2>Run timeline</h2>',
                self._render_timeline(),
                "</section>",
                '<section class="transcript"><h2>Complete Agent transcript</h2>',
                stage_sections,
                "</section></main></body></html>",
            )
        )

    def _render_timeline(self) -> str:
        if not self._events:
            return '<p class="empty">No Run events.</p>'
        items = []
        for event in self._events:
            stage = event.stage.value if event.stage else "runtime"
            timestamp = event.timestamp.isoformat(timespec="seconds")
            payload = ""
            if event.payload:
                payload = (
                    "<details><summary>Details</summary>"
                    + _pre(_json_text(event.payload), css_class="json")
                    + "</details>"
                )
            items.append(
                '<article class="timeline-item">'
                f'<div class="timeline-meta"><time>{_escape(timestamp)}</time>'
                f"<span>{_escape(stage)}</span><span>{_escape(event.type)}</span></div>"
                f"<p>{_escape(event.message)}</p>{payload}</article>"
            )
        return "".join(items)

    def _render_stage(self, stage: str, turns: list[dict[str, Any]]) -> str:
        content = "".join(self._render_turn(turn) for turn in turns)
        return (
            '<section class="stage">'
            f'<div class="stage-heading"><h3>{_escape(stage)}</h3>'
            f"<span>{len(turns)} LLM turns</span></div>{content}</section>"
        )

    def _render_turn(self, turn: dict[str, Any]) -> str:
        request = turn.get("request") if isinstance(turn.get("request"), dict) else {}
        response = turn.get("response") if isinstance(turn.get("response"), dict) else {}
        messages = request.get("messages") if isinstance(request.get("messages"), list) else []
        tools = request.get("tools") if isinstance(request.get("tools"), list) else []
        executions = (
            turn.get("tool_executions") if isinstance(turn.get("tool_executions"), list) else []
        )

        message_html = "".join(self._render_message(message) for message in messages)
        tool_definitions = ""
        if tools:
            tool_definitions = (
                '<details class="request-tools"><summary>Available tool definitions '
                f"({len(tools)})</summary>{_pre(_json_text(tools), css_class='json')}</details>"
            )

        reasoning = response.get("reasoning")
        reasoning_html = (
            '<section class="reasoning"><h5>Reasoning</h5>' + _pre(reasoning) + "</section>"
            if reasoning
            else ""
        )
        content = response.get("content")
        content_html = (
            '<section class="response"><h5>Final output</h5>' + _pre(content) + "</section>"
            if content
            else ""
        )
        tool_html = "".join(self._render_tool_execution(item) for item in executions)
        error_html = ""
        if response.get("error"):
            error_html = (
                '<section class="error"><h5>Error</h5>' + _pre(response["error"]) + "</section>"
            )

        duration = turn.get("duration_ms")
        duration_text = f" · {duration} ms" if duration is not None else ""
        return (
            '<article class="turn">'
            f'<div class="turn-heading"><h4>Turn {turn.get("sequence", "?")}</h4>'
            f"<span>{_escape(turn.get('completed_at', ''))}{duration_text}</span></div>"
            '<details class="request"><summary>Actual LLM input '
            f"({len(messages)} messages)</summary>{message_html}{tool_definitions}</details>"
            f"{reasoning_html}{tool_html}{content_html}{error_html}</article>"
        )

    def _render_message(self, message: Any) -> str:
        if not isinstance(message, dict):
            return _pre(_json_text(message), css_class="json")
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        body = _pre(content or "")
        if message.get("reasoning_content"):
            body += "<h6>reasoning_content</h6>" + _pre(message["reasoning_content"])
        if message.get("tool_calls"):
            body += "<h6>tool_calls</h6>" + _pre(
                _json_text(message["tool_calls"]), css_class="json"
            )
        return f'<section class="message role-{_escape(role)}"><h5>{_escape(role)}</h5>{body}</section>'

    def _render_tool_execution(self, execution: Any) -> str:
        if not isinstance(execution, dict):
            return _pre(_json_text(execution), css_class="json")
        name = str(execution.get("name") or "unknown")
        arguments = _parse_arguments(execution.get("arguments") or {})
        result = execution.get("result") or ""
        arguments_html = self._render_tool_arguments(name, arguments)
        return (
            '<details class="tool" open>'
            f'<summary><span class="tool-name">{_escape(name)}</span>'
            f'<span class="tool-id">{_escape(execution.get("tool_call_id") or "")}</span></summary>'
            f'<div class="tool-grid"><section><h5>Arguments</h5>{arguments_html}</section>'
            f"<section><h5>Result</h5>{_pre(result, css_class='tool-result')}</section></div></details>"
        )

    def _render_tool_arguments(self, name: str, arguments: Any) -> str:
        if not isinstance(arguments, dict):
            return _pre(arguments)
        if name == "shell":
            parts = []
            if arguments.get("workdir") is not None:
                parts.append(f'<p class="field"><b>workdir</b> {_escape(arguments["workdir"])}</p>')
            parts.append(_pre(arguments.get("command") or "", css_class="command"))
            remaining = {
                key: value for key, value in arguments.items() if key not in {"workdir", "command"}
            }
            if remaining:
                parts.append(_pre(_json_text(remaining), css_class="json"))
            return "".join(parts)
        if name in {"read_file", "write_file", "replace_in_file"}:
            parts = []
            for key, value in arguments.items():
                parts.append(f"<h6>{_escape(key)}</h6>")
                if isinstance(value, str):
                    parts.append(_pre(value))
                else:
                    parts.append(_pre(_json_text(value), css_class="json"))
            return "".join(parts)
        return _pre(_json_text(arguments), css_class="json")

    @staticmethod
    def _styles() -> str:
        return """
:root{color-scheme:light;--ink:#17201d;--muted:#66716d;--line:#d8dedb;--paper:#fff;--soft:#f4f7f5;--accent:#087f5b;--reason:#f5f3ff;--tool:#edf8f3;--error:#fff1f0}
*{box-sizing:border-box}body{margin:0;background:#eef2f0;color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}
.run-header,main{max-width:1180px;margin:auto}.run-header{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr);gap:32px;padding:42px 28px 26px}.eyebrow{color:var(--accent);font-weight:700;margin:0 0 4px}.run-header h1{font-size:30px;margin:0;overflow-wrap:anywhere}.run-id{font-family:ui-monospace,monospace;color:var(--muted)}.run-meta{margin:0;display:grid;gap:8px}.run-meta div{display:grid;grid-template-columns:92px 1fr;border-bottom:1px solid var(--line);padding:5px 0}.run-meta dt{color:var(--muted)}.run-meta dd{margin:0;overflow-wrap:anywhere}
main{padding:0 28px 60px}h2{font-size:20px;margin:28px 0 12px}h3,h4,h5,h6{letter-spacing:0}.run-config{background:var(--paper);border:1px solid var(--line);padding:12px 16px}.timeline,.transcript{background:var(--paper);border:1px solid var(--line);padding:0 22px 22px}.timeline-item{border-top:1px solid var(--line);padding:12px 0}.timeline-item p{margin:5px 0}.timeline-meta{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:12px}.stage{border-top:3px solid var(--accent);margin-top:22px}.stage-heading,.turn-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px}.stage-heading h3{font-size:18px}.stage-heading span,.turn-heading span{color:var(--muted);font-size:12px}.turn{border:1px solid var(--line);margin:12px 0;padding:0 16px 16px;background:#fcfdfc}.turn-heading h4{font-size:15px}.request{border-top:1px solid var(--line);padding:10px 0}.message{border-left:3px solid #9aa5a0;padding:4px 10px;margin:10px 0}.message h5,.reasoning h5,.response h5,.error h5,.tool h5{margin:6px 0}.role-system{border-color:#4c6ef5}.role-user{border-color:#0b7285}.role-assistant{border-color:#e67700}.role-tool{border-color:#087f5b}.reasoning,.response,.error{margin:12px 0;padding:10px 12px}.reasoning{background:var(--reason)}.response{background:var(--soft)}.error{background:var(--error)}
details>summary{cursor:pointer;font-weight:650}.tool{background:var(--tool);border:1px solid #b9dfd0;margin:12px 0;padding:10px 12px}.tool summary{display:flex;gap:12px}.tool-name{color:var(--accent)}.tool-id{font:12px ui-monospace,monospace;color:var(--muted)}.tool-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}.field{overflow-wrap:anywhere}
pre{margin:7px 0;padding:10px 12px;background:#151b19;color:#e8eeeb;border-radius:4px;white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;tab-size:4}.json{color:#d9f99d}.command{color:#fef08a}.tool-result{color:#dbeafe}.empty{color:var(--muted)}
@media(max-width:760px){.run-header{grid-template-columns:1fr;padding:24px 16px}.run-header h1{font-size:24px}main{padding:0 12px 36px}.timeline,.transcript{padding:0 12px 14px}.stage-heading,.turn-heading{align-items:flex-start;flex-direction:column;gap:0}}
"""
