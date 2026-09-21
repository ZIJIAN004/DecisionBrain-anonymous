"""StageAgent: per-stage tool-calling loop engine.

Each stage handler instantiates a StageAgent, supplies stage-specific tools and message
construction, and drives its tool-calling loop. StageAgent is transient and is not persisted
in AgentState.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from .contracts import AgentControl, CoreServices
from .message_builder import MessageBuilder
from .models import (
    AgentReport,
    AgentStage,
    StageCompleted,
    StageFinished,
    StageResult,
    StageStarted,
    StateUpdated,
)
from .stage_control_tools import (
    REPORT_PROGRESS_PARAMETERS,
    REPORT_PROGRESS_TOOL_NAME,
    STAGE_EVENT_PROGRESS_REPORTED,
    StageControlToolError,
    normalize_report_progress_args,
)
from .stage_outputs import STAGE_OUTPUT_FILES

logger = logging.getLogger(__name__)

StageOutput = AgentReport | StageResult | StateUpdated

DebugEventHandler = Callable[[dict[str, Any]], Awaitable[None]]
StageEventHandler = Callable[[dict[str, Any]], Awaitable[None]]
AgentTurnHandler = Callable[[dict[str, Any]], Awaitable[None]]
_MISSING_ACTUAL = object()


@dataclass(frozen=True)
class OutputValidationIssue:
    """A concrete contract violation in the final assistant output."""

    path: str
    message: str
    expected: str = ""
    actual: Any = _MISSING_ACTUAL


class StageOutputValidationError(ValueError):
    """Raised when the final assistant output is JSON but violates the stage contract."""

    def __init__(
        self,
        message: str = "阶段输出不符合契约",
        *,
        issues: Sequence[OutputValidationIssue] = (),
        file_path: str = "",
    ) -> None:
        self.issues = tuple(issues) or (OutputValidationIssue("$", message),)
        self.file_path = file_path
        super().__init__(message)
    @classmethod
    def from_json_error(
        cls,
        exc: json.JSONDecodeError,
        *,
        file_path: str = "",
    ) -> "StageOutputValidationError":
        return cls(
            "阶段输出不是合法 JSON",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message=f"JSON 语法错误：{exc.msg} (line {exc.lineno}, column {exc.colno})",
                    expected="只输出一个合法 JSON 对象",
                ),
            ),
            file_path=file_path,
        )


class StageExecutionStopped(RuntimeError):
    """Stop a stage when deterministic evidence shows further retries cannot help."""


class StageForcedCompletion(StageExecutionStopped):
    """Finish a stage from a deterministic tool outcome without another LLM turn."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def pydantic_validation_issues(
    exc: ValidationError, *, prefix: str = "$"
) -> tuple[OutputValidationIssue, ...]:
    """Convert Pydantic errors into the StageAgent feedback format."""

    issues: list[OutputValidationIssue] = []
    for error in exc.errors(include_url=False):
        path = prefix
        for part in error.get("loc", ()):
            path += f"[{part}]" if isinstance(part, int) else f".{part}"
        issues.append(
            OutputValidationIssue(
                path=path,
                message=str(error.get("msg") or "字段不符合契约"),
                expected=str(error.get("type") or "valid value"),
                actual=error.get("input", _MISSING_ACTUAL),
            )
        )
    return tuple(issues)


def unknown_tool_message(name: str, available: Any) -> str:
    """Name the tools that do exist, and the nearest match to what was called.

    A bare "unknown tool" tells the caller it was wrong without telling it what
    is right, so the same invented name comes back turn after turn.
    """

    names = sorted(available)
    suggestion = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    hint = f"；最接近的可用 tool 是 `{suggestion[0]}`" if suggestion else ""
    listed = "、".join(f"`{item}`" for item in names) or "（无）"
    return (
        f"未知 tool: {name}{hint}。当前可用的 tool 只有：{listed}。"
        "请从中选择一个重新调用，不要使用其他名称。"
    )


def parse_json_object_output(content: str) -> dict[str, Any]:
    """Strictly parse text as one JSON object.

    Stage output files require the JSON object itself, without prose or Markdown
    fences.
    """
    text = (content or "").strip()
    if not text:
        raise StageOutputValidationError(
            "阶段输出为空",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message="没有输出 JSON 对象",
                    expected="只输出一个合法 JSON 对象",
                ),
            ),
        )
    if text.startswith("```"):
        raise StageOutputValidationError(
            "阶段输出包含 Markdown 代码块",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message="阶段输出文件不能使用 markdown 代码块标记",
                    expected="直接输出 JSON 对象本身",
                ),
            ),
        )
    if not text.startswith("{"):
        raise StageOutputValidationError(
            "阶段输出不是 JSON 对象",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message="JSON 对象前存在多余文本，或输出不是对象",
                    expected="第一个非空字符必须是 `{`",
                    actual=text[:120],
                ),
            ),
        )

    try:
        parsed, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise StageOutputValidationError.from_json_error(exc) from exc
    if not isinstance(parsed, dict):
        raise StageOutputValidationError(
            "阶段输出顶层不是 JSON 对象",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message="顶层 JSON 必须是对象",
                    expected="object",
                    actual=_type_name(parsed),
                ),
            ),
        )
    trailing = text[end:].strip()
    if trailing:
        raise StageOutputValidationError(
            "阶段输出包含多余文本",
            issues=(
                OutputValidationIssue(
                    path="$",
                    message="JSON 对象后存在多余文本",
                    expected="JSON 对象结束后不能再写入解释、代码块或其他字符",
                    actual=trailing[:120],
                ),
            ),
        )
    return parsed


@dataclass(frozen=True)
class Tool:
    """Stage-specific tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[str]]  # async (**kwargs) → result_text


def _tool_argument_contract(tool: Tool, issues: Sequence[str]) -> str:
    schema = tool.parameters if isinstance(tool.parameters, dict) else {}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = {str(name) for name in schema.get("required", []) if isinstance(name, str)}

    lines = [f"tool 参数校验失败: {tool.name}", "发现的问题:"]
    lines.extend(f"- {issue}" for issue in issues)
    lines.extend(["", f"工具用途: {tool.description or '未提供'}", "允许的参数:"])
    if not properties:
        lines.append("- 无（该工具不接受参数）")
    for name, raw_definition in properties.items():
        definition = raw_definition if isinstance(raw_definition, dict) else {}
        type_name = definition.get("type", "未指定")
        if isinstance(type_name, list):
            type_name = " | ".join(str(item) for item in type_name)
        description = str(definition.get("description") or "未提供说明")
        if name in required:
            requirement = "必填"
        else:
            requirement = "可选"
        lines.append(f"- {name}: {requirement}; 类型={type_name}; 含义={description}")
    lines.append("请仅使用上述参数按契约重新调用该工具。")
    return "\n".join(lines)


def _tool_argument_issues(tool: Tool, arguments: Any) -> tuple[str, ...]:
    if not isinstance(arguments, dict):
        return (f"参数顶层必须是 object，实际为 {type(arguments).__name__}",)
    schema = tool.parameters if isinstance(tool.parameters, dict) else {}
    if not schema:
        return ()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
    issues: list[str] = []
    for error in errors:
        if error.validator == "required":
            missing = sorted(set(error.validator_value) - set(error.instance))
            issues.append(f"缺少必填参数: {', '.join(missing)}")
            continue
        if error.validator == "additionalProperties":
            properties = schema.get("properties", {})
            unknown = sorted(set(error.instance) - set(properties))
            issues.append(f"包含未允许的参数: {', '.join(unknown)}")
            continue
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        issues.append(f"参数 {path} 不符合契约: {error.message}")
    return tuple(dict.fromkeys(issues))


@dataclass
class StageAgent:
    """Per-stage tool-calling loop engine.

    A stage handler instantiates a StageAgent, supplies stage-specific tools and message
    construction, and calls run() to drive the tool-calling loop.

    Example::

        class MyAgent(StageAgent):
            def build_messages(self, state):
                mb = MessageBuilder.from_prompts(sys, dev)
                mb.add_user(f"Task: {state.problem_description}")
                return mb

            def parse_output_file(self, data, state):
                return data

        agent = MyAgent(stage=..., services=services, tools=(...))
        async for output in agent.run(state, control):
            yield output
    """

    stage: AgentStage
    services: CoreServices
    tools: tuple[Tool, ...] = ()
    tool_choice: str = "auto"

    # Internal state
    _tools_by_name: dict[str, Tool] = field(init=False, repr=False)
    _debug_event_handler: DebugEventHandler | None = field(default=None, init=False, repr=False)
    _stage_event_handler: StageEventHandler | None = field(default=None, init=False, repr=False)
    _agent_turn_handler: AgentTurnHandler | None = field(default=None, init=False, repr=False)
    _timing_handler: AgentTurnHandler | None = field(default=None, init=False, repr=False)
    _mb: MessageBuilder | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        control_tools = self._control_tools()
        control_names = {tool.name for tool in control_tools}
        collisions = sorted(tool.name for tool in self.tools if tool.name in control_names)
        if collisions:
            names = ", ".join(collisions)
            raise ValueError(f"StageAgent tool name conflicts with shared control tool(s): {names}")
        self._tools_by_name = {t.name: t for t in (*self.tools, *control_tools)}
        self._debug_event_handler = getattr(self.services, "debug_event_handler", None)
        self._stage_event_handler = getattr(self.services, "stage_event_handler", None)
        self._timing_handler = getattr(self.services, "timing_handler", None)
        self._agent_turn_handler = getattr(self.services, "agent_turn_handler", None)

    # Subclass extension points

    def build_messages(self, state: Any) -> MessageBuilder:  # AgentState
        """Build initial messages from AgentState. Subclasses must implement this."""
        raise NotImplementedError

    def output_file(self, state: Any) -> str:
        """Return the canonical workspace JSON file for this stage."""
        return STAGE_OUTPUT_FILES[self.stage]

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        """Validate and map the canonical workspace JSON object into stage data."""
        return data

    def build_validation_feedback(self, error: StageOutputValidationError) -> str:
        """Build the user message appended after an invalid final output."""
        location = f" `{error.file_path}`" if error.file_path else ""
        lines = [
            # State facts only; validation_feedback_remedy supplies the stage-specific action.
            f"阶段输出文件{location}未通过校验。",
            "具体不满足的位置与信息如下：",
        ]
        for issue in error.issues[:10]:
            line = f"- 位置 {issue.path}: {issue.message}"
            if issue.expected:
                line += f"；期望：{issue.expected}"
            if issue.actual is not _MISSING_ACTUAL:
                line += f"；实际：{_format_actual(issue.actual)}"
            lines.append(line)
        if len(error.issues) > 10:
            lines.append(f"- 另有 {len(error.issues) - 10} 个问题未列出，请先修正以上问题。")
        lines.append(self.validation_feedback_remedy(error))
        return "\n".join(lines)

    def validation_feedback_remedy(self, error: StageOutputValidationError) -> str:
        """Return the "what to do now" paragraph appended to validation feedback.

        A stage whose output file is published by a tool, or whose tool set is
        trimmed, must override this. The default names write_file/replace_in_file,
        which for such a stage is a dead end: the tool may not exist, or the file
        may be unwritable by every tool the stage has.
        """

        return (
            "请使用 write_file/replace_in_file 修正文件，并自行 read_file 或 shell json.load 验收；"
            "完成后只需发一条简短消息，不要在 message 中粘贴完整 JSON。"
        )

    def build_result(self, output: dict[str, Any], state: Any) -> StageResult:
        """Map parsed output to a stage result.

        By default successful parsing returns StageCompleted. Stages may override this hook
        to return StageNeedsClarification or StageFailed.
        """
        return StageCompleted(stage=self.stage, data=output)

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        """Hook after parse_output succeeds and before StageCompleted.
        Subclasses may yield additional events such as ArtifactProduced.
        """
        return
        yield  # pragma: no cover - make this method a generator

    # Message persistence

    def get_transcript(self, start_idx: int = 1) -> tuple[dict[str, Any], ...]:
        """Return dynamic messages after start_idx for cross-round persistence.

        The system message at index 0 is skipped by default. Subclasses may skip a larger
        static prefix. A tuple preserves AgentState immutability.
        """
        if self._mb is None:
            return ()
        messages = self._mb.messages
        return tuple(messages[start_idx:])

    def _inject_common_system_prompt(self, mb: MessageBuilder) -> None:
        common = str(getattr(self.services.prompts, "stage_common_system", "") or "").strip()
        scoped_prompts = [common]
        if self.stage in {AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}:
            scoped_prompts.append(
                str(getattr(self.services.prompts, "algorithm_library_system", "") or "").strip()
            )
        content = "\n\n".join(prompt for prompt in scoped_prompts if prompt)
        if content:
            mb.prepend_to_system(content.replace("__CURRENT_STAGE__", self.stage.value))

    def _control_tools(self) -> tuple[Tool, ...]:
        return (
            Tool(
                name=REPORT_PROGRESS_TOOL_NAME,
                description=(
                    "Publish a non-terminal user-facing progress update. "
                    "This does not finish the task."
                ),
                parameters=REPORT_PROGRESS_PARAMETERS,
                handler=self._report_progress,
            ),
        )

    async def _report_progress(self, **kwargs: Any) -> str:
        try:
            payload = normalize_report_progress_args(kwargs)
        except StageControlToolError as exc:
            return f"report_progress error: {exc}"
        await self._emit_stage_event(STAGE_EVENT_PROGRESS_REPORTED, payload)
        return "progress recorded"

    async def _emit_stage_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self._stage_event_handler is None:
            return
        await self._stage_event_handler(
            {
                "kind": kind,
                "agent_stage": self.stage.value,
                "payload": payload,
            }
        )

    # Main loop

    async def run(
        self,
        state: Any,  # AgentState
        control: AgentControl,
        *,
        suppress_stage_started: bool = False,
    ) -> AsyncIterator[StageOutput]:
        """Run the tool-calling loop and yield events plus the final StageResult.

        The signature matches StageHandler; CoreServices is injected by the constructor.
        """
        mb = self.build_messages(state)
        self._inject_common_system_prompt(mb)
        self._mb = mb
        for tool in self._tools_by_name.values():
            mb.add_function_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        if await control.is_cancelled():
            return
        if not suppress_stage_started:
            yield StageStarted(
                stage=self.stage,
                message=f"{self.stage.value} stage started",
            )

        while True:
            if await control.is_cancelled():
                return

            request = mb.build(tool_choice=self.tool_choice)
            started_at = time.perf_counter()
            try:
                response = await self.services.llm.complete(request)
            except asyncio.CancelledError:
                await self._emit_timing(
                    {
                        "kind": "llm_call",
                        "status": "interrupted",
                        **self._llm_timing_metadata(),
                    },
                    started_at,
                )
                raise
            except Exception as exc:
                await self._emit_timing(
                    {
                        "kind": "llm_call",
                        "status": "error",
                        **self._llm_timing_metadata(),
                    },
                    started_at,
                )
                await self._emit_stage_event(
                    "agent_error",
                    {
                        "flow_section": self.stage.value,
                        "error_kind": "llm_request_failure",
                        "message": str(exc),
                    },
                )
                await self._emit_agent_turn(
                    request=request,
                    response=None,
                    tool_executions=[],
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                    error=str(exc),
                )
                raise
            await self._emit_timing(
                {"kind": "llm_call", "status": "ok", **self._llm_timing_metadata()},
                started_at,
            )
            # In debug mode llm.complete() already emits every streamed delta.

            if response.tool_calls:
                mb.add_assistant(
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning,
                )
                tool_executions: list[dict[str, Any]] = []
                for tc in response.tool_calls:
                    try:
                        result = await self._execute_tool(tc)
                    except StageForcedCompletion as exc:
                        result = exc.message
                        tool_executions.append(self._tool_execution_record(tc, result))
                        mb.add_tool_result(tc["id"], result)
                        await self._emit_agent_turn(
                            request=request,
                            response=response,
                            tool_executions=tool_executions,
                            duration_ms=int((time.perf_counter() - started_at) * 1000),
                        )
                        output = await self._load_forced_completion_output(state)
                        async for extra in self._after_parse(output, state):
                            yield extra
                        yield StageFinished(
                            stage=self.stage,
                            message=f"{self.stage.value} stage completed",
                        )
                        yield self.build_result(output, state)
                        return
                    tool_executions.append(self._tool_execution_record(tc, result))
                    mb.add_tool_result(tc["id"], result)
                await self._emit_agent_turn(
                    request=request,
                    response=response,
                    tool_executions=tool_executions,
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )
                continue

            await self._emit_agent_turn(
                request=request,
                response=response,
                tool_executions=[],
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )

            # With no tool calls, validate the stage's fixed workspace output.
            mb.add_assistant(
                content=response.content or "",
                reasoning_content=response.reasoning,
            )
            validation_error: StageOutputValidationError | None = None
            output: dict[str, Any] | None = None
            try:
                output = self._load_output_file(state)
            except StageOutputValidationError as exc:
                validation_error = exc
            except json.JSONDecodeError as exc:
                validation_error = StageOutputValidationError.from_json_error(exc)

            if output is None and validation_error is None:
                validation_error = StageOutputValidationError(
                    "Stage output file could not be parsed",
                    issues=(
                        OutputValidationIssue(
                            path="$",
                            message="Could not read a workspace JSON object matching the stage contract",
                            expected=f"Write {self.output_file(state)} and satisfy the stage field contract",
                        ),
                    ),
                    file_path=self.output_file(state),
                )
            elif output is not None and not isinstance(output, dict):
                validation_error = StageOutputValidationError(
                    "Invalid stage output type",
                    issues=(
                        OutputValidationIssue(
                            path="$",
                            message="parse_output_file must return a dict",
                            expected="dict",
                            actual=_type_name(output),
                        ),
                    ),
                )

            if validation_error is not None:
                await self._emit_recoverable_error(
                    "stage_output_validation",
                    str(validation_error),
                )
                mb.add_user(self.build_validation_feedback(validation_error))
                continue

            async for extra in self._after_parse(output, state):
                yield extra
            result = self.build_result(output, state)
            yield StageFinished(
                stage=self.stage,
                message=f"{self.stage.value} stage completed",
            )
            yield result
            return

    # Internals

    def _load_output_file(self, state: Any) -> dict[str, Any]:
        relative_path = self.output_file(state)
        parsed = self._read_output_file_object(state)
        output = self.parse_output_file(parsed, state)
        if not isinstance(output, dict):
            raise StageOutputValidationError(
                "Invalid stage output type",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message="parse_output_file must return a dict",
                        expected="dict",
                        actual=_type_name(output),
                    ),
                ),
                file_path=relative_path,
            )
        return output

    async def _load_forced_completion_output(self, state: Any) -> dict[str, Any]:
        """Read the stage output after a deterministic forced completion.

        A forced completion has no next LLM turn, so validation feedback has
        nobody to act on it. The stage output was written by the Runtime itself,
        and failing the whole run over it turns a routed terminal state into a
        lost task. Record the mismatch and finish on the raw object instead.
        """

        try:
            return self._load_output_file(state)
        except (StageOutputValidationError, json.JSONDecodeError) as exc:
            await self._emit_recoverable_error("stage_output_validation", str(exc))
            return self._read_output_file_object(state)

    def _read_output_file_object(self, state: Any) -> dict[str, Any]:
        relative_path = self.output_file(state)
        root = self._stage_workspace_root()
        if root is None:
            raise StageOutputValidationError(
                "Workspace root is not configured",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message="Cannot read stage output because the workspace root is not configured",
                        expected="Runtime must configure the StageAgent workspace root",
                    ),
                ),
                file_path=relative_path,
            )
        target = self._resolve_workspace_path(root, relative_path)
        if not target.is_file():
            raise StageOutputValidationError(
                "Stage output file does not exist",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message=f"Stage output file `{relative_path}` was not found",
                        expected=f"Write {relative_path} with write_file",
                    ),
                ),
                file_path=relative_path,
            )
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise StageOutputValidationError(
                "Stage output file could not be read",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message=f"Failed to read file: {exc}",
                        expected="UTF-8 JSON text file",
                    ),
                ),
                file_path=relative_path,
            ) from exc
        try:
            parsed = parse_json_object_output(text)
        except StageOutputValidationError as exc:
            raise StageOutputValidationError(
                str(exc),
                issues=exc.issues,
                file_path=relative_path,
            ) from exc
        return parsed

    def _stage_workspace_root(self) -> Path | None:
        return self.services.workspace_root()

    def _read_workspace_json(
        self,
        relative_path: str,
    ) -> dict[str, Any] | list[Any] | None:
        """Read an internal workspace JSON value without exposing it as a tool result."""

        root = self._stage_workspace_root()
        if root is None:
            return None
        try:
            target = self._resolve_workspace_path(root, relative_path)
            if not target.is_file():
                return None
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            StageOutputValidationError,
        ):
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    @staticmethod
    def _resolve_workspace_path(root: Path, relative_path: str) -> Path:
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            raise StageOutputValidationError(
                "Stage output path cannot be absolute",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message=f"Stage output path `{relative_path}` is absolute",
                        expected="workspace-relative path",
                    ),
                ),
                file_path=relative_path,
            )
        target = (root / raw_path).resolve(strict=False)
        if target != root and root not in target.parents:
            raise StageOutputValidationError(
                "Stage output path escapes the workspace root",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message=f"Stage output path `{relative_path}` escapes the workspace root",
                        expected="path must stay inside workspace",
                    ),
                ),
                file_path=relative_path,
            )
        return target

    async def _execute_tool(self, tc: dict[str, Any]) -> str:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        tool = self._tools_by_name.get(name)
        if tool is None:
            # A name-only error made the model repeatedly guess the same missing tool; in one
            # 166-task benchmark replace_file was called 69 times across 42% of tasks.
            logger.warning("Unknown tool: %s", name)
            await self._emit_recoverable_error(
                "unknown_tool", f"Unknown tool: {name}", tool_name=name
            )
            return unknown_tool_message(name, self._tools_by_name)
        raw_arguments = fn.get("arguments", "{}")
        try:
            args: Any = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError as exc:
            result = _tool_argument_contract(
                tool,
                (f"Arguments are not valid JSON: {exc.msg}",),
            )
            await self._emit_recoverable_error(
                "tool_argument_validation",
                result,
                tool_name=name,
            )
            return result
        issues = _tool_argument_issues(tool, args)
        if issues:
            result = _tool_argument_contract(tool, issues)
            await self._emit_recoverable_error(
                "tool_argument_validation",
                result,
                tool_name=name,
            )
            return result
        await self._emit_debug_tool_call_started(tc.get("id", ""), name, args)
        started_at = time.perf_counter()
        status = "ok"
        try:
            result = await tool.handler(**args)
        except StageExecutionStopped:
            await self._emit_timing(
                self._tool_timing_event(name, args, status="stopped"),
                started_at,
            )
            raise
        except asyncio.CancelledError:
            await self._emit_timing(
                self._tool_timing_event(name, args, status="interrupted"),
                started_at,
            )
            raise
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            result = f"Tool execution failed: {exc}"
            status = "error"
        await self._emit_debug_tool_result(tc.get("id", ""), name, result)
        error_kind = _tool_result_error_kind(result)
        if error_kind is not None:
            status = "error"
        await self._emit_timing(self._tool_timing_event(name, args, status=status), started_at)
        if error_kind is not None:
            await self._emit_recoverable_error(error_kind, result, tool_name=name)
        return result

    @staticmethod
    def _tool_timing_event(
        name: str, arguments: Mapping[str, Any], *, status: str
    ) -> dict[str, Any]:
        return {
            "kind": "tool_call",
            "tool_name": name,
            "tool_arguments": dict(arguments),
            "status": status,
        }

    def _llm_timing_metadata(self) -> dict[str, int]:
        metrics = getattr(self.services.llm, "last_call_metrics", {})
        return dict(metrics) if isinstance(metrics, dict) else {}

    async def _emit_timing(self, event: dict[str, Any], started_at: float) -> None:
        if self._timing_handler is None:
            return
        payload = {
            **event,
            "stage": self.stage.value,
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
        }
        try:
            await self._timing_handler(payload)
        except Exception:
            # Timing is observability only and must not alter stage behavior.
            logger.exception("stage timing handler failed")

    async def _emit_recoverable_error(
        self,
        error_kind: str,
        message: str,
        *,
        tool_name: str = "",
    ) -> None:
        await self._emit_stage_event(
            "agent_error",
            {
                "flow_section": self.stage.value,
                "error_kind": error_kind,
                "tool_name": tool_name,
                "message": message[:1000],
            },
        )

    async def _emit_debug_tool_call_started(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        if self._debug_event_handler is None:
            return
        await self._debug_event_handler(
            {
                "kind": "tool_call_started",
                "role": self.stage.value,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )

    async def _emit_debug_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        if self._debug_event_handler is None:
            return
        await self._debug_event_handler(
            {
                "kind": "tool_result",
                "role": self.stage.value,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
            }
        )

    @staticmethod
    def _tool_execution_record(tc: dict[str, Any], result: str) -> dict[str, Any]:
        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError:
            arguments = raw_arguments
        return {
            "tool_call_id": str(tc.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": arguments,
            "result": result,
        }

    async def _emit_agent_turn(
        self,
        *,
        request: Any,
        response: Any | None,
        tool_executions: list[dict[str, Any]],
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        if self._agent_turn_handler is None:
            return
        response_payload: dict[str, Any] = {
            "content": response.content if response is not None else None,
            "reasoning": response.reasoning if response is not None else None,
            "tool_calls": response.tool_calls if response is not None else None,
        }
        if response is not None and isinstance(getattr(response, "raw", None), dict):
            usage = response.raw.get("usage")
            if isinstance(usage, dict):
                response_payload["usage"] = dict(usage)
        if error is not None:
            response_payload["error"] = error
        await self._agent_turn_handler(
            {
                "stage": self.stage.value,
                "duration_ms": duration_ms,
                "request": request.model_dump(mode="json"),
                "response": response_payload,
                "tool_executions": tool_executions,
            }
        )


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _tool_result_error_kind(result: str) -> str | None:
    first_line = result.splitlines()[0].strip() if result else ""
    if first_line.startswith("Exit code:"):
        try:
            return "tool_nonzero_exit" if int(first_line.split(":", 1)[1].strip()) != 0 else None
        except ValueError:
            return None
    if "Tool execution failed:" in result:
        return "tool_exception"
    if " error:" in first_line.lower():
        return "tool_error"
    return None


def _format_actual(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    if len(text) > 180:
        return text[:177] + "..."
    return text
