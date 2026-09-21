"""Adapt free-form Agent solutions to FrontierOR checker formats."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator, SchemaError

from ..config import Settings
from ..core.message_builder import MessageBuilder
from ..core.models import ChatRequest, ChatResponse
from ..core.stage_agent import Tool, unknown_tool_message
from ..core.workspace_tools import WorkspaceToolset
from ..run_storage.io import atomic_write_json
from .task import BenchmarkTask


ADAPTER_DIR = "benchmark_adapter"
TARGET_SCHEMA_FILE = "solution_schema.json"
RAW_CANDIDATE_FILE = "raw_candidate.json"
FORMATTER_FILE = "convert_solution.py"
ADAPTED_SOLUTION_FILE = "solution.json"


class BenchmarkAdaptationError(RuntimeError):
    """Adaptation failed or output did not match the target template."""


class CompletionClient(Protocol):
    async def complete(self, request: ChatRequest) -> ChatResponse: ...


@dataclass(frozen=True)
class AdaptationResult:
    solution_path: Path
    turns: int


class BenchmarkSolutionAdapter:
    """Restricted Agent for format mapping without reoptimization or evaluation."""

    def __init__(
        self,
        llm: CompletionClient,
        settings: Settings,
        *,
        max_turns: int = 10,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")
        self.llm = llm
        self.settings = settings
        self.max_turns = max_turns

    async def adapt(self, task: BenchmarkTask, runtime_workspace: Path) -> AdaptationResult:
        """Convert ``solution.json`` in isolation and return the checker candidate path."""

        root = self._prepare_workspace(task, runtime_workspace)
        target_template = _read_json_object(root / TARGET_SCHEMA_FILE, label="目标解模板")
        output_path = root / ADAPTED_SOLUTION_FILE
        formatter_ran = False
        validation_ran = False
        toolset = WorkspaceToolset(
            root,
            timeout_s=self.settings.solver_timeout,
            max_output_chars=self.settings.opt_workspace_max_output_chars,
            max_list_entries=self.settings.opt_workspace_max_list_entries,
            default_list_entries=self.settings.opt_workspace_default_list_entries,
            max_read_bytes=self.settings.opt_workspace_max_read_bytes,
        )
        tools = tuple(
            tool
            for tool in toolset.tools(stage=None)
            if tool.name in {"read_file", "write_file", "list_files"}
        ) + (
            Tool(
                name="run_python",
                description=(
                    f"Execute {FORMATTER_FILE} in the adapter workspace. The script must "
                    f"read {RAW_CANDIDATE_FILE} and write {ADAPTED_SOLUTION_FILE}."
                ),
                parameters={
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                    "additionalProperties": False,
                },
                handler=lambda filename: self._run_formatter(root, output_path, filename),
            ),
            Tool(
                name="validate_benchmark_solution",
                description=(
                    f"Validate {ADAPTED_SOLUTION_FILE} against the target template shape."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda: self._validate_output(output_path, target_template),
            ),
        )
        tools_by_name = {tool.name: tool for tool in tools}
        messages = self._messages(task)
        for tool in tools:
            messages.add_function_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )

        last_issues: list[str] = []
        for turn in range(1, self.max_turns + 1):
            response = await self.llm.complete(messages.build(tool_choice="auto"))
            if response.tool_calls:
                messages.add_assistant(
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning,
                )
                for tool_call in response.tool_calls:
                    function = tool_call.get("function") or {}
                    name = function.get("name") if isinstance(function, dict) else None
                    if name == "run_python":
                        formatter_ran = True
                    elif name == "validate_benchmark_solution":
                        validation_ran = True
                    result = await self._execute_tool(tool_call, tools_by_name)
                    messages.add_tool_result(str(tool_call.get("id") or ""), result)
                continue

            messages.add_assistant(
                content=response.content or "",
                reasoning_content=response.reasoning,
            )
            if not formatter_ran:
                last_issues = [f"必须先调用 run_python 执行 {FORMATTER_FILE}"]
            elif not validation_ran:
                last_issues = ["必须调用 validate_benchmark_solution 验收转换结果"]
            elif not output_path.is_file():
                last_issues = [f"缺少 {ADAPTED_SOLUTION_FILE}"]
            else:
                try:
                    candidate = _read_json_object(output_path, label="适配结果")
                except BenchmarkAdaptationError as exc:
                    last_issues = [str(exc)]
                else:
                    last_issues = validate_template_shape(target_template, candidate)
                    if not last_issues:
                        atomic_write_json(output_path, candidate)
                        return AdaptationResult(solution_path=output_path, turns=turn)
            messages.add_user(_validation_feedback(last_issues))

        detail = "; ".join(last_issues[:5]) or "Agent 未生成可验收的适配结果"
        raise BenchmarkAdaptationError(f"结果适配超过 {self.max_turns} 轮：{detail}")

    @staticmethod
    async def _run_formatter(root: Path, output_path: Path, filename: str) -> str:
        """Run the baseline-style formatter and return stderr/stdout as retry feedback."""

        if filename != FORMATTER_FILE:
            return f"FORMATTER_ERROR\n只能执行 {FORMATTER_FILE}"
        formatter = root / FORMATTER_FILE
        if not formatter.is_file():
            return f"FORMATTER_ERROR\n缺少 {FORMATTER_FILE}"
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(formatter),
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=600
            )
        except asyncio.TimeoutError:
            return f"FORMATTER_ERROR\n{FORMATTER_FILE} 执行超过 600 秒"
        except OSError as exc:
            return f"FORMATTER_ERROR\n无法启动 {FORMATTER_FILE}：{exc}"
        out = stdout.decode("utf-8", "replace")[-6000:]
        err = stderr.decode("utf-8", "replace")[-6000:]
        if proc.returncode != 0:
            return f"FORMATTER_ERROR\nexit={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        if not output_path.is_file():
            return f"FORMATTER_ERROR\n{FORMATTER_FILE} 未生成 {output_path.name}\nstdout:\n{out}"
        return f"FORMATTER_OK\nstdout:\n{out}\nstderr:\n{err}"

    @staticmethod
    async def _execute_tool(tool_call: dict[str, Any], tools: dict[str, Tool]) -> str:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return "tool call 缺少 function"
        name = str(function.get("name") or "")
        tool = tools.get(name)
        if tool is None:
            return unknown_tool_message(name, tools)
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError as exc:
            return f"tool arguments 不是合法 JSON：{exc.msg}"
        if not isinstance(arguments, dict):
            return "tool arguments 必须是 JSON object"
        try:
            return await tool.handler(**arguments)
        except Exception as exc:  # Let the adapter Agent correct tool errors without ending the batch.
            return f"tool 执行失败：{exc}"

    def _prepare_workspace(self, task: BenchmarkTask, runtime_workspace: Path) -> Path:
        runtime_workspace = runtime_workspace.expanduser().resolve()
        source_solution = runtime_workspace / "solution.json"
        if not source_solution.is_file():
            raise BenchmarkAdaptationError(f"Runtime 未产出自由格式解：{source_solution}")

        root = runtime_workspace / ADAPTER_DIR
        if root.is_symlink() or root.is_file():
            root.unlink()
        elif root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        for name in (
            "problem.md",
            "input_schema.json",
            "input.json",
            "solver_result.json",
            "feasibility_result.json",
        ):
            source = runtime_workspace / name
            if source.is_file():
                shutil.copy2(source, root / name)
        data_dir = runtime_workspace / "data"
        if data_dir.is_dir():
            shutil.copytree(data_dir, root / "data")
        shutil.copy2(source_solution, root / RAW_CANDIDATE_FILE)
        shutil.copy2(task.target_solution_schema, root / TARGET_SCHEMA_FILE)
        return root

    @staticmethod
    def _messages(task: BenchmarkTask) -> MessageBuilder:
        system = (
            "You are a result adapter, not an optimizer. You must convert the existing raw "
            f"candidate by writing a workspace-local Python script named {FORMATTER_FILE}. "
            "Use write_file to create or replace that script, then use run_python to execute it. "
            f"The script must read {RAW_CANDIDATE_FILE} and instance/schema files and write "
            f"{ADAPTED_SOLUTION_FILE} at the workspace root. Do not return the solution as chat "
            f"JSON and do not write {ADAPTED_SOLUTION_FILE} directly. Preserve candidate decisions; "
            "do not optimize, rerun a solver, or fabricate missing values. After execution, inspect "
            "the tool result and repair the script until the output validates."
        )
        developer = (
            f"The target JSON Schema is {TARGET_SCHEMA_FILE}. The raw candidate is "
            f"{RAW_CANDIDATE_FILE}; solver_result.json, problem.md, input.json, and data/ provide "
            "context. Use only the isolated workspace file tools. Finish only after running the "
            "conversion script and calling validate_benchmark_solution."
        )
        messages = MessageBuilder.from_prompts(system, developer)
        messages.add_user(
            f"任务 ID：{task.task_id}\n"
            f"问题类别：{task.problem_class}\n"
            f"目标方向：{task.objective_direction}\n"
            "Convert the saved candidate now."
        )
        return messages

    @staticmethod
    async def _validate_output(output: Path, template: dict[str, Any]) -> str:
        try:
            candidate = _read_json_object(output, label="适配结果")
        except BenchmarkAdaptationError as exc:
            return f"INVALID\n- {exc}"
        issues = validate_template_shape(template, candidate)
        try:
            Draft202012Validator.check_schema(template)
            issues.extend(
                error.message
                for error in Draft202012Validator(template).iter_errors(candidate)
            )
        except SchemaError as exc:
            issues.append(f"目标 schema 无效：{exc.message}")
        if issues:
            return "INVALID\n" + "\n".join(f"- {issue}" for issue in issues[:20])
        return "VALID"


def validate_template_shape(template: Any, candidate: Any, *, path: str = "$") -> list[str]:
    """Validate container structure and recognizable placeholders against the template."""

    issues: list[str] = []
    if isinstance(template, dict):
        if not isinstance(candidate, dict):
            return [f"{path}: 期望 object，实际 {type(candidate).__name__}"]
        for key, nested in template.items():
            if "[*]." in key:
                collection_name, item_key = key.split("[*].", 1)
                if collection_name not in candidate:
                    issues.append(f"{path}.{collection_name}: 缺少字段")
                    continue
                collection = candidate[collection_name]
                if not isinstance(collection, list):
                    issues.append(
                        f"{path}.{collection_name}: 期望 array，实际 "
                        f"{type(collection).__name__}"
                    )
                    continue
                for index, item in enumerate(collection):
                    item_path = f"{path}.{collection_name}[{index}]"
                    if not isinstance(item, dict):
                        issues.append(f"{item_path}: 期望 object，实际 {type(item).__name__}")
                    elif item_key not in item:
                        issues.append(f"{item_path}.{item_key}: 缺少字段")
                    else:
                        issues.extend(
                            validate_template_shape(
                                nested,
                                item[item_key],
                                path=f"{item_path}.{item_key}",
                            )
                        )
                continue
            if _is_dynamic_key(key):
                for actual_key, actual_value in candidate.items():
                    issues.extend(
                        validate_template_shape(
                            nested,
                            actual_value,
                            path=f"{path}.{actual_key}",
                        )
                    )
                continue
            if key not in candidate:
                if _is_optional_template_field(nested):
                    continue
                issues.append(f"{path}.{key}: 缺少字段")
                continue
            issues.extend(validate_template_shape(nested, candidate[key], path=f"{path}.{key}"))
        return issues
    if isinstance(template, list):
        if not isinstance(candidate, list):
            return [f"{path}: 期望 array，实际 {type(candidate).__name__}"]
        if template:
            for index, item in enumerate(candidate):
                issues.extend(validate_template_shape(template[0], item, path=f"{path}[{index}]"))
        return issues
    if isinstance(template, str) and template.startswith("<"):
        closing = template.find(">")
        token = template[1:closing].strip().lower() if closing > 0 else ""
        expected = _placeholder_type(token)
        if expected is not None and not _matches_type(candidate, expected):
            issues.append(f"{path}: 期望 {expected}，实际 {type(candidate).__name__}")
    return issues


def _is_dynamic_key(key: str) -> bool:
    stripped = key.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("<") and stripped.endswith(">")
    )


def _is_optional_template_field(template: Any) -> bool:
    if not isinstance(template, str):
        return False
    lowered = template.lower()
    return "variant" in lowered or "optional" in lowered


def _placeholder_type(token: str) -> str | None:
    if token == "float":
        return "number"
    if token == "int":
        return "integer"
    if token == "bool":
        return "boolean"
    if token == "str":
        return "string"
    if token.startswith("dict") or token.startswith("key:"):
        return "object"
    if token.startswith("list"):
        return "array"
    return None


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkAdaptationError(f"{label}不存在：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkAdaptationError(f"{label}不是可读 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkAdaptationError(f"{label}顶层必须是 JSON object")
    return value


def _validation_feedback(issues: list[str]) -> str:
    detail = "\n".join(f"- {issue}" for issue in issues[:20])
    return (
        f"{ADAPTED_SOLUTION_FILE} 尚未通过目标模板校验：\n{detail}\n"
        "请修正文件，再调用 validate_benchmark_solution。"
    )
