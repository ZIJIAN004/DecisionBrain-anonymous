"""Problem Contract Stage - define semantic input/solution contracts and checker."""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..contracts import CoreServices
from ..message_builder import MessageBuilder
from ..models import AgentReport, AgentStage, ArtifactDraft, ArtifactProduced
from ..stage_agent import OutputValidationIssue, StageAgent, StageOutputValidationError
from ..stage_outputs import (
    PROBLEM_CONTRACT_CHECKER_FILE,
    PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
    PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
    FIXED_SOLUTION_CONTRACT_FILE,
    STAGE_OUTPUT_FILES,
)


class ProblemContractAgent(StageAgent):
    """Problem Contract Stage: define problem-level schemas and feasibility checker."""

    def __init__(
        self,
        services: CoreServices,
        *,
        input_schema_enabled: bool = True,
        feasibility_review_enabled: bool = True,
    ):
        super().__init__(
            stage=AgentStage.PROBLEM_CONTRACT,
            services=services,
            tools=services.workspace_tools_for(AgentStage.PROBLEM_CONTRACT),
        )
        self._input_schema_enabled = input_schema_enabled
        # Ablation 3 removes validation entirely, including checker isolation.
        self._feasibility_review_enabled = feasibility_review_enabled
        self._artifacts: tuple[ArtifactDraft, ...] = ()

    def build_messages(self, state: Any) -> MessageBuilder:
        prompts = self.services.prompts
        mb = MessageBuilder.from_prompts(
            prompts.problem_contract_system,
            prompts.problem_contract_contract,
        )
        mb.add_user(
            "【当前阶段】problem_contract\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            "- `data/`: 真实数据文件；请用工具读取字段、实体、单位和约束所需数据。"
            "需要按字段名或关键词定位内容时，使用 search_file（例如 path=`data`, query=`字段名`）。\n"
            "【必须写入】\n"
            + (
                f"- `{PROBLEM_CONTRACT_INPUT_SCHEMA_FILE}`: problem-level input JSON Schema。\n"
                if self._input_schema_enabled
                else ""
            )
            + f"- `{PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE}`: problem-level semantic solution JSON Schema。\n"
            + (
                f"- `{PROBLEM_CONTRACT_CHECKER_FILE}`: 独立可行性检查器。\n"
                if self._feasibility_review_enabled
                else "- 写入 problem-level semantic solution JSON Schema，并在阶段摘要中记录业务语义契约。\n"
            )
            + f"- `{STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT]}`: 阶段摘要 JSON。\n"
            f"- 若存在 `{FIXED_SOLUTION_CONTRACT_FILE}`，必须读取并将其原样作为 "
            f"`{PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE}`；该文件是调用方固定的输出契约。\n"
            + "【阶段边界】\n"
            + (
                "- 本阶段只定义 checker 的接口并在 problem_contract.json 中记录未来文件名。"
                "`input.json`、`solution.json` 和 `feasibility_result.json` 是后续 Solving/"
                "Feasibility Review 的产物；不得在本阶段创建、覆盖、要求其存在，或以它们执行 checker CLI。\n"
                if self._feasibility_review_enabled
                else "- `input.json` 和 `solution.json` 是后续 Solving 阶段的产物；"
                "不得在本阶段创建、覆盖或要求其存在。\n"
            )
            + "- 本阶段自检只验证以下产物："
            + "、".join(
                filter(
                    None,
                    (
                        "input_schema.json" if self._input_schema_enabled else "",
                        "solution_schema.json",
                        "feasibility_checker.py" if self._feasibility_review_enabled else "",
                        "stage_outputs/problem_contract.json",
                    ),
                )
            )
            + "。\n"
            + (
                ""
                if self._input_schema_enabled
                else "- 本次运行不提供也不产出输入结构契约：不写 input_schema.json，"
                "规范化 `input.json` 的字段、类型和维度由 Solving 阶段自行读取 `data/` 确定。\n"
            )
            + "【禁止】\n"
            + (
                "- 不要读取或依赖算法设计产物；本阶段只制定业务语义契约和独立 checker。\n"
                if self._feasibility_review_enabled
                else "- 不要读取或依赖算法设计产物；本阶段只制定业务语义契约。\n"
            )
            + "- solution schema 不得暴露 MILP/CP-SAT/启发式等具体算法变量。\n"
            "写完后请自行 read_file 或 shell json.load/py_compile 验收所有文件；最终 message "
            "只需简短说明完成，不要粘贴完整 JSON。"
        )
        return mb

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        issues: list[OutputValidationIssue] = []
        if self._input_schema_enabled:
            self._load_schema(
                PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
                "$.files.input_schema",
                issues,
            )
        else:
            # Remove an unsolicited unvalidated contract so the arm differs by one variable.
            self._discard_unrequested_input_schema()
        fixed_contract = self._read_fixed_solution_contract()
        if fixed_contract is not None:
            (self._stage_workspace_root() / PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE).write_text(
                json.dumps(fixed_contract, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            self._load_schema(
                PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
                "$.files.solution_schema",
                issues,
            )
        checker = self._load_checker(issues) if self._feasibility_review_enabled else ""

        if issues:
            raise StageOutputValidationError(
                "problem_contract 产物不完整或不符合契约",
                issues=tuple(issues),
                file_path=self.output_file(state),
            )

        output = dict(data)
        files = dict(output.get("files") or {})
        if self._input_schema_enabled:
            files.setdefault("input_schema", PROBLEM_CONTRACT_INPUT_SCHEMA_FILE)
        else:
            files.pop("input_schema", None)
        files.setdefault("solution_schema", PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE)
        if self._feasibility_review_enabled:
            files.setdefault("feasibility_checker", PROBLEM_CONTRACT_CHECKER_FILE)
        else:
            files.pop("feasibility_checker", None)
        output["files"] = files
        output.setdefault("schemas_validated", True)
        if self._feasibility_review_enabled:
            output.setdefault("checker_entrypoint", "check(input_data, solution, data_dir='data')")
            output.setdefault("feasibility_result_file", "feasibility_result.json")
            output.setdefault("checker_file_chars", len(checker))
        return {"problem_contract": output}

    def _discard_unrequested_input_schema(self) -> None:
        """Delete an ``input_schema.json`` the disabled arm never asked for."""

        root = self._stage_workspace_root()
        if root is None:
            return
        path = root / PROBLEM_CONTRACT_INPUT_SCHEMA_FILE
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Cleanup failure is non-fatal because downstream prompts do not reference it.
            pass

    def _read_fixed_solution_contract(self) -> dict[str, Any] | None:
        text = self._read_workspace_text(FIXED_SOLUTION_CONTRACT_FILE)
        if text is None:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StageOutputValidationError(
                f"固定输出契约不是合法 JSON：{exc.msg}", file_path=self.output_file(None)
            ) from exc
        if not isinstance(value, dict):
            raise StageOutputValidationError("固定输出契约顶层必须是 JSON object", file_path=self.output_file(None))
        return value

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        artifacts = self._collect_artifacts(output)
        self._artifacts = tuple(artifacts)
        if artifacts:
            yield ArtifactProduced(
                stage=AgentStage.PROBLEM_CONTRACT,
                message="问题契约已生成",
                artifacts=tuple(artifacts),
            )

    def _load_schema(
        self,
        relative_path: str,
        issue_path: str,
        issues: list[OutputValidationIssue],
    ) -> dict[str, Any] | None:
        text = self._read_workspace_text(relative_path)
        if text is None:
            issues.append(
                OutputValidationIssue(
                    path=issue_path,
                    message=f"找不到 `{relative_path}`",
                    expected=f"写入合法 JSON Schema 文件 {relative_path}",
                )
            )
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            issues.append(
                OutputValidationIssue(
                    path=issue_path,
                    message=f"`{relative_path}` 不是合法 JSON：{exc.msg}",
                    expected="JSON Schema object",
                )
            )
            return None
        if not isinstance(parsed, dict):
            issues.append(
                OutputValidationIssue(
                    path=issue_path,
                    message=f"`{relative_path}` 顶层必须是 JSON object",
                    expected="object",
                    actual=type(parsed).__name__,
                )
            )
            return None
        try:
            Draft202012Validator.check_schema(parsed)
        except SchemaError as exc:
            issues.append(
                OutputValidationIssue(
                    path=issue_path,
                    message=f"`{relative_path}` 不是合法 Draft 2020-12 JSON Schema：{exc.message}",
                    expected="valid Draft 2020-12 JSON Schema",
                )
            )
        return parsed

    def _load_checker(self, issues: list[OutputValidationIssue]) -> str:
        text = self._read_workspace_text(PROBLEM_CONTRACT_CHECKER_FILE)
        if text is None:
            issues.append(
                OutputValidationIssue(
                    path="$.files.feasibility_checker",
                    message=f"找不到 `{PROBLEM_CONTRACT_CHECKER_FILE}`",
                    expected="写入包含 check(input_data, solution, data_dir='data') 的 Python 文件",
                )
            )
            return ""
        try:
            tree = ast.parse(text, filename=PROBLEM_CONTRACT_CHECKER_FILE)
        except SyntaxError as exc:
            issues.append(
                OutputValidationIssue(
                    path="$.files.feasibility_checker",
                    message=f"`{PROBLEM_CONTRACT_CHECKER_FILE}` Python 语法错误：{exc.msg}",
                    expected="valid Python source",
                )
            )
            return text
        has_check = any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "check"
            for node in tree.body
        )
        if not has_check:
            issues.append(
                OutputValidationIssue(
                    path="$.files.feasibility_checker",
                    message=f"`{PROBLEM_CONTRACT_CHECKER_FILE}` 缺少顶层 check() 函数",
                    expected="def check(input_data, solution, data_dir='data') -> dict",
                )
            )
        return text

    def _collect_artifacts(self, output: dict[str, Any]) -> list[ArtifactDraft]:
        specs = [
            (
                self.output_file(None),
                "contract/problem_contract.json",
                "application/json",
                "问题契约摘要",
            ),
            *(
                (
                    (
                        PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
                        "contract/input_schema.json",
                        "application/json",
                        "规范化输入 JSON Schema",
                    ),
                )
                if self._input_schema_enabled
                else ()
            ),
            (
                PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
                "contract/solution_schema.json",
                "application/json",
                "业务语义解 JSON Schema",
            ),
            *(
                (
                    (
                        PROBLEM_CONTRACT_CHECKER_FILE,
                        "contract/feasibility_checker.py",
                        "text/x-python",
                        "独立可行性检查器",
                    ),
                )
                if self._feasibility_review_enabled
                else ()
            ),
        ]
        artifacts: list[ArtifactDraft] = []
        for workspace_path, artifact_path, media_type, description in specs:
            content = self._read_workspace_text(workspace_path)
            if content is None:
                continue
            artifacts.append(
                ArtifactDraft(
                    relative_path=artifact_path,
                    content=content,
                    media_type=media_type,
                    category="problem_contract",
                    description=description,
                )
            )
        return artifacts

    def _read_workspace_text(self, relative_path: str) -> str | None:
        root = self._stage_workspace_root()
        if root is None:
            return None
        try:
            raw_path = Path(relative_path)
            if raw_path.is_absolute():
                return None
            target = (root / raw_path).resolve(strict=False)
            if target != root and root not in target.parents:
                return None
            if not target.is_file():
                return None
            return target.read_text(encoding="utf-8")
        except OSError:
            return None
