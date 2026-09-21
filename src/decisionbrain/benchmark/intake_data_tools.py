"""Bounded streaming JSON tools for Intake benchmark judges."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import ijson
from ijson.common import ObjectBuilder

from ..core.stage_agent import Tool


MAX_ITEMS = 100
MAX_OUTPUT_CHARS = 20_000
MAX_MATERIALIZED_NODES = 10_000
MAX_MATERIALIZED_SCALAR_CHARS = 20_000
VALUE_START_EVENTS = frozenset(
    {"null", "boolean", "integer", "double", "number", "string", "start_map", "start_array"}
)


class IntakeDataToolError(ValueError):
    """A bounded data query is invalid or unsafe."""


@dataclass
class _Frame:
    kind: str
    path: tuple[str, ...]
    next_index: int = 0
    current_key: str | None = None


def _parse_pointer(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise IntakeDataToolError("pointer must be an RFC 6901 JSON Pointer")
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/"))


def _path_events(source) -> Iterator[tuple[tuple[str, ...], str, Any]]:
    stack: list[_Frame] = []
    for event, value in ijson.basic_parse(source, use_float=True):
        if event == "map_key":
            if not stack or stack[-1].kind != "map":
                raise IntakeDataToolError("invalid JSON map_key event")
            stack[-1].current_key = str(value)
            yield stack[-1].path, event, value
            continue
        if event in {"end_map", "end_array"}:
            if not stack:
                raise IntakeDataToolError("invalid JSON container end event")
            frame = stack.pop()
            yield frame.path, event, value
            continue

        if not stack:
            path: tuple[str, ...] = ()
        else:
            parent = stack[-1]
            if parent.kind == "array":
                path = (*parent.path, str(parent.next_index))
                parent.next_index += 1
            else:
                if parent.current_key is None:
                    raise IntakeDataToolError("object value has no key")
                path = (*parent.path, parent.current_key)
                parent.current_key = None
        yield path, event, value
        if event == "start_map":
            stack.append(_Frame(kind="map", path=path))
        elif event == "start_array":
            stack.append(_Frame(kind="array", path=path))


def _kind_from_event(event: str) -> str:
    return {
        "null": "null",
        "boolean": "boolean",
        "integer": "integer",
        "double": "number",
        "number": "number",
        "string": "string",
        "start_array": "array",
        "start_map": "object",
    }[event]


def _build_bounded(events, first: tuple[tuple[str, ...], str, Any]) -> Any:
    builder = ObjectBuilder()
    depth = 0
    nodes = 0
    scalar_chars = 0
    current = first
    while True:
        _, event, value = current
        nodes += 1
        if event in {"string", "number", "integer", "double", "boolean", "null"}:
            scalar_chars += len(value) if isinstance(value, str) else len(str(value))
        if nodes > MAX_MATERIALIZED_NODES or scalar_chars > MAX_MATERIALIZED_SCALAR_CHARS:
            raise IntakeDataToolError(
                "selected value is too large; use data_keys or data_slice on a narrower pointer"
            )
        builder.event(event, value)
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1
        if depth == 0:
            return builder.value
        try:
            current = next(events)
        except StopIteration as exc:
            raise IntakeDataToolError("unexpected end of JSON") from exc


def _resolve_data_file(workspace: Path, data_path: str, relative_path: str) -> Path:
    data_root = (workspace / data_path).resolve()
    requested = PurePosixPath(relative_path.replace("\\", "/"))
    if requested.is_absolute() or ".." in requested.parts or str(requested) in {"", "."}:
        raise IntakeDataToolError("path must be a file relative to the data directory")
    target = (data_root / Path(*requested.parts)).resolve()
    if data_root not in target.parents or not target.is_file():
        raise IntakeDataToolError("path is not a file inside the data directory")
    if target.suffix.lower() != ".json":
        raise IntakeDataToolError("streaming data tools accept JSON files only")
    return target


def _render(tool_name: str, payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(rendered) > MAX_OUTPUT_CHARS:
        raise IntakeDataToolError(
            f"response would contain {len(rendered)} characters; narrow the query"
        )
    return rendered


def _stream_get(path: Path, pointer: str) -> Any:
    target = _parse_pointer(pointer)
    with path.open("rb") as source:
        events = iter(_path_events(source))
        for current in events:
            event_path, event, _ = current
            if event_path == target and event in VALUE_START_EVENTS:
                return _build_bounded(events, current)
    raise IntakeDataToolError(f"pointer not found: {pointer}")


def _stream_keys(path: Path, pointer: str, start: int, count: int) -> tuple[list[dict], int]:
    target = _parse_pointer(pointer)
    entries: list[dict] = []
    total = 0
    found = False
    with path.open("rb") as source:
        for event_path, event, _ in _path_events(source):
            if event_path == target and event == "start_map":
                found = True
                continue
            if found and len(event_path) == len(target) + 1 and event_path[:-1] == target:
                if event in VALUE_START_EVENTS:
                    if start <= total < start + count:
                        entries.append({"key": event_path[-1], "kind": _kind_from_event(event)})
                    total += 1
            if found and event_path == target and event == "end_map":
                return entries, total
    raise IntakeDataToolError(f"pointer does not resolve to an object: {pointer}")


def _stream_slice(path: Path, pointer: str, start: int, count: int) -> tuple[list[Any], int]:
    target = _parse_pointer(pointer)
    items: list[Any] = []
    total = 0
    found = False
    with path.open("rb") as source:
        events = iter(_path_events(source))
        for current in events:
            event_path, event, _ = current
            if event_path == target and event == "start_array":
                found = True
                continue
            if found and len(event_path) == len(target) + 1 and event_path[:-1] == target:
                if event in VALUE_START_EVENTS:
                    if start <= total < start + count:
                        items.append(_build_bounded(events, current))
                    total += 1
            if found and event_path == target and event == "end_array":
                return items, total
    raise IntakeDataToolError(f"pointer does not resolve to an array: {pointer}")


def build_intake_data_tools(workspace: Path, *, data_path: str = "data") -> tuple[Tool, ...]:
    """Build three read-only streaming tools scoped to one workspace data directory."""

    async def data_get(path: str, pointer: str = "") -> str:
        try:
            target = _resolve_data_file(workspace, data_path, path)
            value = _stream_get(target, pointer)
            return _render(
                "data_get",
                {
                    "path": path,
                    "pointer": pointer,
                    "kind": _kind_from_python(value),
                    "value": value,
                },
            )
        except (IntakeDataToolError, OSError, ijson.JSONError) as exc:
            return f"data_get error: {exc}"

    async def data_keys(path: str, pointer: str = "", start: int = 0, count: int = 50) -> str:
        try:
            _validate_page(start, count)
            target = _resolve_data_file(workspace, data_path, path)
            entries, total = _stream_keys(target, pointer, start, count)
            returned = len(entries)
            return _render(
                "data_keys",
                {
                    "path": path,
                    "pointer": pointer,
                    "start": start,
                    "returned_keys": returned,
                    "total_keys": total,
                    "has_more": start + returned < total,
                    "next_start": start + returned if start + returned < total else None,
                    "entries": entries,
                },
            )
        except (IntakeDataToolError, OSError, ijson.JSONError) as exc:
            return f"data_keys error: {exc}"

    async def data_slice(path: str, pointer: str, start: int = 0, count: int = 20) -> str:
        try:
            _validate_page(start, count)
            target = _resolve_data_file(workspace, data_path, path)
            items, total = _stream_slice(target, pointer, start, count)
            returned = len(items)
            return _render(
                "data_slice",
                {
                    "path": path,
                    "pointer": pointer,
                    "start": start,
                    "returned_items": returned,
                    "total_items": total,
                    "has_more": start + returned < total,
                    "next_start": start + returned if start + returned < total else None,
                    "items": items,
                },
            )
        except (IntakeDataToolError, OSError, ijson.JSONError) as exc:
            return f"data_slice error: {exc}"

    common_path = {
        "type": "string",
        "description": f"JSON file path relative to {data_path}/.",
    }
    pointer = {
        "type": "string",
        "description": "RFC 6901 JSON Pointer; empty string selects the document root.",
    }
    page_properties = {
        "start": {"type": "integer", "minimum": 0},
        "count": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS},
    }
    return (
        Tool(
            name="data_get",
            description="Stream one bounded scalar or small JSON subtree from visible task data.",
            parameters={
                "type": "object",
                "properties": {"path": common_path, "pointer": pointer},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=data_get,
        ),
        Tool(
            name="data_keys",
            description="Page through keys and value types of an object in visible task data.",
            parameters={
                "type": "object",
                "properties": {"path": common_path, "pointer": pointer, **page_properties},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=data_keys,
        ),
        Tool(
            name="data_slice",
            description="Page through array elements in visible task data without loading the file.",
            parameters={
                "type": "object",
                "properties": {"path": common_path, "pointer": pointer, **page_properties},
                "required": ["path", "pointer"],
                "additionalProperties": False,
            },
            handler=data_slice,
        ),
    )


def _validate_page(start: int, count: int) -> None:
    if start < 0 or not 1 <= count <= MAX_ITEMS:
        raise IntakeDataToolError(f"start must be >= 0 and count must be between 1 and {MAX_ITEMS}")


def _kind_from_python(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"
