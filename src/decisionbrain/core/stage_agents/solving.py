"""SOLVING Stage - tool-driven solve implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import CoreServices
from ..feasibility_routing import render_for_executor as render_feasibility_repair
from ..message_builder import MessageBuilder
from ..package_policy import (
    solving_addendum,
    validate_executions,
    validate_source_imports,
)
from ..models import (
    AgentReport,
    AgentStage,
    ArtifactDraft,
    ArtifactProduced,
    ClarificationQuestion,
    StageCompleted,
    StageFailed,
    StageNeedsClarification,
    StageResult,
)
from ..stage_agent import (
    OutputValidationIssue,
    StageAgent,
    StageOutputValidationError,
    pydantic_validation_issues,
)
from ..stage_output_models import (
    RuntimeSolverTimeoutOutput,
    SolverExecutionReceipt,
    SingleSolverResultOutput,
    SolverResultOutput,
    SolvingStageOutput,
)
from ..stage_outputs import (
    PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
    PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
    FIXED_SOLUTION_CONTRACT_FILE,
    SOLVING_CODE_FILE,
    SOLVING_EXECUTION_RECEIPT_FILE,
    SOLVING_INPUT_BUILDER_FILE,
    SOLVING_INPUT_FILE,
    SOLVING_RESULT_FILE,
    SOLVING_RUNTIME_OUTCOME_FILE,
    SOLVING_SOLUTION_FILE,
    STAGE_OUTPUT_FILES,
)
from ..workspace_tools import file_sha256


DEFAULT_INPUT_FILE = SOLVING_INPUT_FILE
DEFAULT_CODE_FILE = SOLVING_CODE_FILE
DEFAULT_SOLUTION_FILE = SOLVING_SOLUTION_FILE
DEFAULT_SOLVER_RESULT_FILE = SOLVING_RESULT_FILE


class SolvingAgent(StageAgent):
    """SOLVING Stage: let the LLM solve by editing and running workspace files."""

    def __init__(
        self,
        services: CoreServices,
        *,
        feasibility_review_enabled: bool = True,
        input_schema_enabled: bool = True,
        algorithm_design_enabled: bool = True,
        problem_contract_enabled: bool = True,
    ):
        self._feasibility_review_enabled = feasibility_review_enabled
        self._input_schema_enabled = input_schema_enabled
        self._algorithm_design_enabled = algorithm_design_enabled
        self._problem_contract_enabled = problem_contract_enabled
        # No algorithm tool denotes ablation 2; use the matching prompt variant.
        self._algorithm_library_enabled = bool(
            services.algorithm_tools_for(AgentStage.SOLVING)
        )
        tools = (
            *services.workspace_tools_for(
                AgentStage.SOLVING,
                # Review-off leaves candidate validation to the solver without managed review tools.
                solving_feasibility_self_check=False,
                solving_outcome_submission=feasibility_review_enabled,
            ),
            *services.algorithm_tools_for(AgentStage.SOLVING),
        )
        super().__init__(
            stage=AgentStage.SOLVING,
            services=services,
            tools=tools,
        )
        self._artifacts: tuple[ArtifactDraft, ...] = ()

    def _input_contract_read_line(self) -> str:
        """Build the required-read entry for this arm's normalized-input schema."""

        if self._input_schema_enabled:
            return f"- `{PROBLEM_CONTRACT_INPUT_SCHEMA_FILE}`: 规范化输入 schema。\n"
        return ""

    def _solution_contract_read_line(self) -> str:
        if not self._problem_contract_enabled:
            return ""
        filename = (
            PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE
            if self._problem_contract_enabled
            else FIXED_SOLUTION_CONTRACT_FILE
        )
        return f"- `{filename}`: 业务语义解 schema。\n"

    def _solution_schema_prompt_text(self, text: str) -> str:
        """Keep generic prompt variants aligned with the visible schema file."""

        if self._problem_contract_enabled:
            return text
        return text.replace(
            "problem contract, ",
            "",
        ).replace(
            "problem contract and ",
            "",
        ).replace(
            "input/solution schema",
            "the solver's own raw solution format",
        ).replace(
            PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
            "the solver's own raw solution format (no output schema is provided)",
        ).replace(FIXED_SOLUTION_CONTRACT_FILE, "the solver's own raw solution format")

    def _stage_input_lines(self) -> str:
        lines = ""
        if self._problem_contract_enabled:
            lines += f"- `{STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT]}`: 问题契约摘要。\n"
        if self._algorithm_design_enabled:
            lines += f"- `{STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN]}`: 已确认的算法设计与求解策略。\n"
        elif self._is_gurobi_formulator_arm():
            lines += f"- `{STAGE_OUTPUT_FILES[AgentStage.GUROBI_FORMULATOR]}`: 唯一允许的完整 Gurobi formulation；只能翻译，不能重新设计。\n"
        else:
            lines += "- 根据问题描述、问题定义和可用数据确定算法、建模方案与实现策略。\n"
        return lines

    def _is_gurobi_formulator_arm(self) -> bool:
        """Whether Solving is the arm's single complete Gurobi formulator."""

        return (
            not self._algorithm_design_enabled
            and not self.services.prompts.components_enabled
            and bool(self.services.prompts.gurobi_formulator_system)
        )

    def _input_contract_source(self) -> str:
        """Describe the structure source for the required ``input.json`` output."""

        if self._input_schema_enabled:
            return "input_schema.json"
        return "你自行读取 `data/` 确定的输入结构（本次运行不提供 input schema）"

    def _algorithm_guide_lines(self, state: Any, *, branch9: bool) -> str:
        """Guide-calling rules plus this run's package-pool / single-package constraints."""

        return self._guide_call_lines(branch9=branch9) + solving_addendum(
            self.services.package_policy, self._design(state)
        )

    @staticmethod
    def _design(state: Any) -> dict[str, Any] | None:
        design = getattr(state, "algorithm_design", None)
        return design if isinstance(design, dict) else None

    def _bind_package_guard(self, state: Any) -> None:
        """Tell the toolset which packages the design selected for subprocess restriction.

        Arm B derives its available packages from the design, so the guard is rebound
        on every solving entry, including after review-triggered redesign.
        """

        toolset = getattr(self.services, "workspace_toolset", None)
        binder = getattr(toolset, "bind_solving_design", None)
        if callable(binder):
            binder(self._design(state))

    def _guide_call_lines(self, *, branch9: bool) -> str:
        """Guide-calling rules, or the neutral notice when no algorithm library is mounted."""

        if not self._algorithm_library_enabled:
            return (
                "- 本次运行不提供算法包目录，也没有 get_algorithm_guide 工具；"
                "算法设计中的组件使用 source=generated。\n"
                "- 你可以使用环境中已安装的任意 Python 包；写 solver.py 前必须用 shell "
                "验证导入、版本与接口签名，不要凭记忆假定 API 存在。\n"
            )
        if branch9:
            return (
                "- 对 selection.components 和 fallback.components 中每个 source=package 的组件，"
                "先按精确 package ID 调用 get_algorithm_guide 获取经过校验的精确接口，"
                "不要凭记忆猜测 API。\n"
                "- 写 solver.py 前用 shell 验证 manifest 版本/导入；代码必须真实调用选中算法，"
                "切换 fallback 时如实记录。\n"
            )
        if not self._algorithm_design_enabled:
            if (
                self.services.package_policy.pool == "gurobi-only"
                and not self.services.prompts.components_enabled
            ):
                return (
                    "- 在一个完整 gurobipy 精确模型中实现"
                    "全部变量、硬约束、目标和求解调用。\n"
                    "- Formulator 必须确定性完成真实数据到模型的映射，以及 Gurobi 解到"
                    " solution.json 全部业务字段的结果映射。\n"
                )
            return (
                "- 本次没有单独的 Algorithm Design 阶段；你必须在 solver.py 中直接完成完整的\n"
                "  建模、求解和业务结果映射。你可以根据问题需要选择已安装的算法包、组件、\n"
                "  分解、预处理、warm start 或其他求解策略；系统不会替你预先指定或禁止这些\n"
                "  选择。使用 package 组件前按精确 ID 获取指南并验证实际导入和接口。\n"
            )
        if not self.services.prompts.components_enabled:
            return (
                "- algorithm 中 source=package 时，先按精确 package ID 调用 "
                "get_algorithm_guide，不要凭记忆猜测 API。\n"
                "- 写 solver.py 前验证 package 版本与导入；代码必须真实调用选定 solver_id，"
                "实际调用必须与 algorithm 中的 package 和 solver_id 完全一致。\n"
            )
        return (
            "- 对 selection.components 和 fallback.components 中每个 source=package 的组件，"
            "先按精确 package ID 调用 get_algorithm_guide，不要凭记忆猜测 API。\n"
            "- 写 solver.py 前用 shell 验证 package 版本/导入；代码必须真实调用选定 "
            "solver_id，切换 fallback 时如实记录。\n"
        )

    def build_messages(self, state: Any) -> MessageBuilder:
        self._bind_package_guard(state)
        prompts = self.services.prompts
        time_budget = self._solve_time_budget()
        system_prompt = self._solution_schema_prompt_text(prompts.solving_system)
        contract_prompt = self._solution_schema_prompt_text(prompts.solving_contract)
        if not self._algorithm_design_enabled and not self._is_gurobi_formulator_arm():
            system_prompt = system_prompt.replace("algorithm_design", "your complete solving plan")
            contract_prompt = contract_prompt.replace("algorithm_design", "your complete solving plan")
        if not self._feasibility_review_enabled:
            system_prompt = self._solution_schema_prompt_text(prompts.solving_branch9_system)
            contract_prompt = self._solution_schema_prompt_text(prompts.solving_branch9_contract)
        if self._is_gurobi_formulator_arm():
            system_prompt = prompts.solving_translation_system
            contract_prompt = prompts.solving_translation_contract
        contract = contract_prompt.replace("__TIME_BUDGET__", str(time_budget))
        mb = MessageBuilder.from_prompts(system_prompt, contract)
        feasibility_repair = render_feasibility_repair(
            getattr(state, "feasibility_directives", ()),
            stage=AgentStage.SOLVING.value,
            components_enabled=self.services.prompts.components_enabled,
            algorithm_design_enabled=self._algorithm_design_enabled,
            gurobi_formulator_enabled=self._is_gurobi_formulator_arm(),
        )
        message = (
            "【当前阶段】solving\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            + self._stage_input_lines()
            + self._input_contract_read_line()
            + self._solution_contract_read_line()
            + "- `data/`: 真实数据文件；必须通过工具读取真实文件内容。\n"
            "【必须写入】\n"
            + (f"- `{DEFAULT_INPUT_FILE}`: 按 {self._input_contract_source()} 生成的规范化输入。\n"
               if self._problem_contract_enabled else "- 直接读取真实 `data/` 文件构造并求解模型。\n")
            + f"- `{DEFAULT_CODE_FILE}`: 可运行求解代码。\n"
            f"- `{DEFAULT_SOLUTION_FILE}`: 由 `run_solver` 实际执行 solver.py 生成的完整业务解；不得由 Agent 直接写入。\n"
            f"- `{DEFAULT_SOLVER_RESULT_FILE}` 和 `{STAGE_OUTPUT_FILES[AgentStage.SOLVING]}`：由 `submit_solving_outcome` 工具验收并写入，不要用 write_file 直接写入。\n"
            "【workspace】\n"
            "- shell 默认从 workspace 根目录执行；根目录文件直接使用相对路径。\n"
            "- 需要从子目录执行时，使用 shell 的相对 workdir；不要 cd 到猜测的绝对路径。\n"
            "- 你可以使用 list_files/read_file/write_file/replace_in_file/shell 完成建模与修复；必须用 `run_solver` 实际执行 solver.py，再调用 `submit_solving_outcome`。\n"
            + self._algorithm_guide_lines(state, branch9=False)
            +
            "- 详细执行信息放在 submit_solving_outcome 的 result 参数；不要直接写 solver_result.json 或 solving.json。\n"
            "- `run_solver` 会清理旧候选并生成 Runtime 执行 receipt；完成执行后调用 submit_solving_outcome，工具会按当前解 schema 验收解文件。最终 message 只需简短说明完成，"
            "不要粘贴完整 JSON 或展开完整 solution。\n"
        )
        if not self._feasibility_review_enabled:
            message = self._branch9_self_check_message(state)
        mb.add_user(message if not self._feasibility_review_enabled else message + feasibility_repair)
        return mb

    def _branch9_self_check_message(self, state: Any) -> str:
        """The legacy branch-9 Solving user prompt, used verbatim when review is off."""
        workspace_root = self._workspace_root(state)
        return (
            "【当前阶段】solving\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            + self._stage_input_lines()
            + self._input_contract_read_line()
            + self._solution_contract_read_line()
            + "- `data/`: 真实数据文件；必须通过工具读取真实文件内容。\n"
            "【必须写入】\n"
            f"- `{DEFAULT_INPUT_FILE}`: 按 {self._input_contract_source()} 生成的规范化输入。\n"
            f"- `{DEFAULT_CODE_FILE}`: 可运行求解代码。\n"
            f"- `{DEFAULT_SOLUTION_FILE}`: 由 `run_solver` 实际执行 solver.py 生成的完整业务解；不得由 Agent 直接写入。\n"
            f"- `{DEFAULT_SOLVER_RESULT_FILE}`: 精简求解摘要。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.SOLVING]}`: 阶段决策 JSON。\n"
            "【workspace】\n"
            f"- 当前工作目录: {workspace_root or '由工具环境提供'}\n"
            "- 你可以使用 list_files/read_file/write_file/replace_in_file/shell 完成建模与修复；必须使用 `run_solver` 执行 solver.py。\n"
            + self._algorithm_guide_lines(state, branch9=True)
            + "- 详细执行信息只写入 solver_result.json；solving.json 仅引用该文件。\n"
            "- 写完后请自行 read_file 或 shell json.load 验收所有 JSON 文件；最终 message 只需简短说明完成，"
            "不要粘贴完整 JSON 或展开完整 solution。"
        )

    def validation_feedback_remedy(self, error: StageOutputValidationError) -> str:
        """Point at the submission tool, not at files this stage cannot write.

        With Review enabled, both final files have an empty writer_stages set, so
        the default advice ("use write_file/replace_in_file") is refused by the ACL,
        rolled back through shell, and leaves no working path forward.
        """

        if not self._feasibility_review_enabled:
            return super().validation_feedback_remedy(error)
        return (
            f"`{STAGE_OUTPUT_FILES[AgentStage.SOLVING]}` 和 `{DEFAULT_SOLVER_RESULT_FILE}` "
            "只能由 `submit_solving_outcome` 写入，write_file、replace_in_file 和 shell 写入"
            "都会被拒绝或回滚。若问题出在求解产物或执行凭据，先用 `run_solver` 重新执行 "
            f"`{DEFAULT_CODE_FILE}`，再重新调用 `submit_solving_outcome` 提交修正后的 "
            "solver_result；完成后只需发一条简短消息，不要在 message 中粘贴完整 JSON。"
        )

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        try:
            output = SolvingStageOutput.model_validate(data)
        except ValidationError as exc:
            raise StageOutputValidationError(
                "solving 输出不符合契约",
                issues=pydantic_validation_issues(exc),
                file_path=self.output_file(state),
            ) from exc

        if output.decision == "solved":
            self._validate_solved_workspace(state)
        return output.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def _validate_solved_workspace(self, state: Any) -> None:
        issues: list[OutputValidationIssue] = []
        issues.extend(self._package_import_issues(state))

        stage_output = self._read_workspace_json(STAGE_OUTPUT_FILES[AgentStage.SOLVING])
        stage_result = stage_output.get("result") if isinstance(stage_output, dict) else None
        runtime_reference = (
            stage_result.get("runtime_solver_outcome_file")
            if isinstance(stage_result, dict)
            else None
        )
        if runtime_reference == SOLVING_RUNTIME_OUTCOME_FILE:
            runtime_data = self._read_workspace_json(SOLVING_RUNTIME_OUTCOME_FILE)
            if not isinstance(runtime_data, dict):
                issues.append(
                    OutputValidationIssue(
                        path="$.result.runtime_solver_outcome_file",
                        message="找不到 Runtime solver timeout JSON 对象",
                        expected=SOLVING_RUNTIME_OUTCOME_FILE,
                    )
                )
            else:
                try:
                    RuntimeSolverTimeoutOutput.model_validate(runtime_data)
                except ValidationError as exc:
                    issues.extend(pydantic_validation_issues(exc, prefix="$.runtime_outcome"))
            self._validate_runtime_solver_receipt(issues, state)
            for path, expected in (
                (DEFAULT_INPUT_FILE, "input JSON object"),
                (DEFAULT_CODE_FILE, "non-empty Python solver"),
            ):
                if path == DEFAULT_INPUT_FILE:
                    valid = isinstance(self._read_workspace_json(path), dict)
                else:
                    valid = bool((self._read_workspace_text(path) or "").strip())
                if not valid:
                    issues.append(
                        OutputValidationIssue(
                            path=f"$.files.{path}",
                            message=f"Runtime timeout 后找不到有效 `{path}`",
                            expected=expected,
                        )
                    )
            if issues:
                raise StageOutputValidationError(
                    "Runtime solver timeout 产物不完整或不符合契约",
                    issues=tuple(issues),
                    file_path=self.output_file(state),
                )
            return

        solver_result_data = self._read_workspace_json(DEFAULT_SOLVER_RESULT_FILE)
        solver_result: SolverResultOutput | SingleSolverResultOutput | None = None
        if not isinstance(solver_result_data, dict):
            issues.append(
                OutputValidationIssue(
                    path="$.result.solver_result_file",
                    message=f"找不到合法 JSON 对象 `{DEFAULT_SOLVER_RESULT_FILE}`",
                    expected="完整 solver_result JSON object",
                )
            )
        else:
            try:
                model = (
                    SolverResultOutput
                    if self.services.prompts.components_enabled
                    else SingleSolverResultOutput
                )
                solver_result = model.model_validate(solver_result_data)
            except ValidationError as exc:
                issues.extend(pydantic_validation_issues(exc, prefix="$.solver_result"))

        if solver_result is not None:
            self._validate_solver_execution_receipt(solver_result, issues, state)
            # Runtime-only package ablations require execution validation symmetric with design.
            issues.extend(
                OutputValidationIssue(
                    path=item.path,
                    message=item.message,
                    expected=item.expected,
                    actual=item.actual,
                )
                for item in validate_executions(
                    solver_result.model_dump(mode="json"), self.services.package_policy
                )
            )

        required_json_files = [(DEFAULT_INPUT_FILE, "input JSON object")]
        if not self._feasibility_review_enabled or solver_result is None or solver_result.solution_file:
            required_json_files.append((DEFAULT_SOLUTION_FILE, "solution JSON object"))
        for path, expected in required_json_files:
            if not isinstance(self._read_workspace_json(path), dict):
                issues.append(
                    OutputValidationIssue(
                        path=f"$.files.{path}",
                        message=f"找不到合法 JSON 对象 `{path}`",
                        expected=expected,
                    )
                )
        if not (self._read_workspace_text(DEFAULT_CODE_FILE) or "").strip():
            issues.append(
                OutputValidationIssue(
                    path=f"$.files.{DEFAULT_CODE_FILE}",
                    message=f"找不到非空 `{DEFAULT_CODE_FILE}`",
                    expected="可运行 Python solver",
                )
            )

        if solver_result is not None and solver_result.status == "failed":
            issues.append(
                OutputValidationIssue(
                    path="$.solver_result.status",
                    message="基础执行错误必须在 Solving 内修复，不能作为可审查求解终态提交",
                    expected="candidate, no-candidate, or proven infeasible status",
                    actual=solver_result.status,
                )
            )

        if issues:
            raise StageOutputValidationError(
                "Solving workspace artifacts are incomplete or violate the contract",
                issues=tuple(issues),
                file_path=self.output_file(state),
            )

    def _package_import_issues(self, state: Any) -> list[OutputValidationIssue]:
        """Validate written code statically instead of trusting receipt package IDs.

        The execution guard is authoritative but triggers only on import. This check
        catches declaration and implementation mismatches before execution.
        """

        sources = {
            name: text
            for name in (DEFAULT_CODE_FILE, SOLVING_INPUT_BUILDER_FILE)
            if (text := self._read_workspace_text(name))
        }
        if not sources:
            return []
        return [
            OutputValidationIssue(
                path=item.path,
                message=item.message,
                expected=item.expected,
                actual=item.actual,
            )
            for item in validate_source_imports(
                sources, self.services.package_policy, self._design(state)
            )
        ]

    def _validate_solver_execution_receipt(
        self,
        solver_result: SolverResultOutput | SingleSolverResultOutput,
        issues: list[OutputValidationIssue],
        state: Any,
    ) -> None:
        receipt_data = self._read_workspace_json(SOLVING_EXECUTION_RECEIPT_FILE)
        if not isinstance(receipt_data, dict):
            issues.append(
                OutputValidationIssue(
                    path="$.solver_execution_receipt",
                    message=f"找不到合法 `{SOLVING_EXECUTION_RECEIPT_FILE}`",
                    expected="Runtime solver execution receipt",
                )
            )
            return
        try:
            receipt = SolverExecutionReceipt.model_validate(receipt_data)
        except ValidationError as exc:
            issues.extend(pydantic_validation_issues(exc, prefix="$.solver_execution_receipt"))
            return
        root = Path(self._workspace_root(state))
        expected = {
            SOLVING_CODE_FILE: receipt.solver_sha256,
            SOLVING_INPUT_FILE: receipt.input_sha256,
        }
        if solver_result.solution_file == SOLVING_SOLUTION_FILE:
            if receipt.solution_file != SOLVING_SOLUTION_FILE or receipt.solution_sha256 is None:
                issues.append(
                    OutputValidationIssue(
                        path="$.solver_execution_receipt.solution_file",
                        message="候选解必须由 receipt 绑定 solution.json",
                        expected='solution_file="solution.json" and solution_sha256',
                    )
                )
            else:
                expected[SOLVING_SOLUTION_FILE] = receipt.solution_sha256
        elif receipt.solution_file is not None or receipt.solution_sha256 is not None:
            issues.append(
                OutputValidationIssue(
                    path="$.solver_execution_receipt.solution_file",
                    message="无候选终态的 receipt 不得绑定 solution.json",
                    expected="solution_file=null and solution_sha256=null",
                )
            )
        for relative_path, digest in expected.items():
            if file_sha256(root / relative_path) != digest:
                issues.append(
                    OutputValidationIssue(
                        path=f"$.solver_execution_receipt.{relative_path}",
                        message=f"receipt 与当前 {relative_path} 不一致",
                        expected="receipt hash matches current workspace file",
                    )
                )

    def _validate_runtime_solver_receipt(
        self,
        issues: list[OutputValidationIssue],
        state: Any,
    ) -> None:
        receipt_data = self._read_workspace_json(SOLVING_EXECUTION_RECEIPT_FILE)
        if not isinstance(receipt_data, dict):
            issues.append(
                OutputValidationIssue(
                    path="$.solver_execution_receipt",
                    message=f"找不到合法 `{SOLVING_EXECUTION_RECEIPT_FILE}`",
                    expected="Runtime solver execution receipt",
                )
            )
            return
        try:
            receipt = SolverExecutionReceipt.model_validate(receipt_data)
        except ValidationError as exc:
            issues.extend(pydantic_validation_issues(exc, prefix="$.solver_execution_receipt"))
            return
        root = Path(self._workspace_root(state))
        for relative_path, digest in {
            SOLVING_CODE_FILE: receipt.solver_sha256,
            SOLVING_INPUT_FILE: receipt.input_sha256,
        }.items():
            if file_sha256(root / relative_path) != digest:
                issues.append(
                    OutputValidationIssue(
                        path=f"$.solver_execution_receipt.{relative_path}",
                        message=f"receipt 与当前 {relative_path} 不一致",
                        expected="receipt hash matches current workspace file",
                    )
                )
        if receipt.solution_file is not None and file_sha256(
            root / SOLVING_SOLUTION_FILE
        ) != receipt.solution_sha256:
            issues.append(
                OutputValidationIssue(
                    path="$.solver_execution_receipt.solution.json",
                    message="receipt 与当前 solution.json 不一致",
                    expected="receipt hash matches current workspace file",
                )
            )

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        if str(output.get("decision") or "").lower() != "solved":
            return
        result = self._augment_result_from_workspace(self._result_payload(output))
        artifacts = self._collect_artifacts(result)
        self._artifacts = tuple(artifacts)
        if artifacts:
            yield ArtifactProduced(
                stage=AgentStage.SOLVING,
                message="求解文件已生成",
                artifacts=tuple(artifacts),
            )

    def build_result(self, output: dict[str, Any], state: Any) -> StageResult:
        decision = str(output.get("decision") or "").strip().lower()
        message = str(output.get("message") or "").strip()
        if decision == "solved":
            result = self._result_payload(output)
            result = self._augment_result_from_workspace(result)
            if self._artifacts:
                result.setdefault(
                    "artifacts",
                    [artifact.relative_path for artifact in self._artifacts],
                )
            return StageCompleted(
                stage=AgentStage.SOLVING,
                data={
                    "result": result,
                    "generated_code": self._read_workspace_text(
                        str(result.get("code_file") or DEFAULT_CODE_FILE)
                    ),
                },
                message=message,
            )
        if decision == "needs_clarification":
            questions = self._questions(output)
            return StageNeedsClarification(
                stage=AgentStage.SOLVING,
                message=message or "求解阶段需要补充业务信息",
                questions=questions,
            )
        return StageFailed(
            stage=AgentStage.SOLVING,
            message=message or "求解阶段失败",
            metadata={"llm_output": output},
        )

    def _result_payload(self, output: dict[str, Any]) -> dict[str, Any]:
        result = output.get("result")
        if isinstance(result, dict):
            return dict(result)
        return {
            key: value
            for key, value in output.items()
            if key not in {"decision", "message", "questions"}
        }

    def _augment_result_from_workspace(self, result: dict[str, Any]) -> dict[str, Any]:
        runtime_path = result.get("runtime_solver_outcome_file")
        if runtime_path == SOLVING_RUNTIME_OUTCOME_FILE:
            runtime_outcome = self._read_workspace_json(SOLVING_RUNTIME_OUTCOME_FILE)
            if isinstance(runtime_outcome, dict):
                merged = dict(runtime_outcome)
                merged["runtime_solver_outcome_file"] = SOLVING_RUNTIME_OUTCOME_FILE
                return merged
            return result
        solver_result_path = str(result.get("solver_result_file") or DEFAULT_SOLVER_RESULT_FILE)
        code_path = str(result.get("code_file") or DEFAULT_CODE_FILE)
        input_path = str(result.get("input_file") or DEFAULT_INPUT_FILE)
        result.setdefault("solver_result_file", solver_result_path)
        result.setdefault("code_file", code_path)
        result.setdefault("input_file", input_path)

        solver_result = self._read_workspace_json(solver_result_path)
        has_solution = (
            isinstance(solver_result, dict)
            and solver_result.get("solution_file") == DEFAULT_SOLUTION_FILE
        )
        solution_path = (
            str(result.get("solution_file") or DEFAULT_SOLUTION_FILE) if has_solution else None
        )
        if isinstance(solver_result, dict):
            merged = dict(solver_result)
            merged.update(result)
            result = merged
            result.setdefault("solver_result_file", solver_result_path)
            result.setdefault("code_file", code_path)
            result.setdefault("input_file", input_path)
        if solution_path is not None:
            result.setdefault("solution_file", solution_path)

        input_data = self._read_workspace_json(input_path)
        if isinstance(input_data, dict):
            result.setdefault("input", input_data)
        else:
            missing = list(result.get("missing_files") or [])
            if input_path not in missing:
                missing.append(input_path)
            result["missing_files"] = missing

        if solution_path is not None:
            solution = self._read_workspace_json(solution_path)
            if isinstance(solution, dict):
                result.setdefault("solution", solution)
            else:
                missing = list(result.get("missing_files") or [])
                if solution_path not in missing:
                    missing.append(solution_path)
                result["missing_files"] = missing

        if self._read_workspace_text(code_path) is None:
            missing = list(result.get("missing_files") or [])
            if code_path not in missing:
                missing.append(code_path)
            result["missing_files"] = missing

        return result

    def _collect_artifacts(self, result: dict[str, Any]) -> list[ArtifactDraft]:
        if result.get("runtime_solver_outcome_file") == SOLVING_RUNTIME_OUTCOME_FILE:
            specs = [
                (
                    SOLVING_RUNTIME_OUTCOME_FILE,
                    "result/runtime_solver_outcome.json",
                    "application/json",
                    "result",
                    "Runtime 记录的正式求解器外部超时终态",
                ),
                (
                    SOLVING_CODE_FILE,
                    "code/solver.py",
                    "text/x-python",
                    "code",
                    "生成的求解代码",
                ),
                (
                    SOLVING_INPUT_FILE,
                    "result/input.json",
                    "application/json",
                    "result",
                    "规范化求解输入",
                ),
                (
                    SOLVING_EXECUTION_RECEIPT_FILE,
                    "result/solver_execution_receipt.json",
                    "application/json",
                    "result",
                    "Runtime 记录的求解器执行凭据",
                ),
            ]
            if result.get("solution_file") == SOLVING_SOLUTION_FILE:
                specs.append(
                    (
                        SOLVING_SOLUTION_FILE,
                        "result/solution.json",
                        "application/json",
                        "result",
                        "超时前原子写出的候选解",
                    )
                )
            return self._artifacts_from_specs(specs)
        specs = [
            (
                str(result.get("code_file") or DEFAULT_CODE_FILE),
                "code/solver.py",
                "text/x-python",
                "code",
                "生成的求解代码",
            ),
            (
                str(result.get("input_file") or DEFAULT_INPUT_FILE),
                "result/input.json",
                "application/json",
                "result",
                "规范化求解输入",
            ),
            (
                str(result.get("solver_result_file") or DEFAULT_SOLVER_RESULT_FILE),
                "result/solver_result.json",
                "application/json",
                "result",
                "求解结果摘要",
            ),
            (
                SOLVING_EXECUTION_RECEIPT_FILE,
                "result/solver_execution_receipt.json",
                "application/json",
                "result",
                "Runtime 记录的求解器执行凭据",
            ),
        ]
        solution_file = result.get("solution_file")
        if isinstance(solution_file, str) and solution_file:
            specs.insert(
                2,
                (
                    solution_file,
                    "result/solution.json",
                    "application/json",
                    "result",
                    "完整业务解",
                ),
            )
        return self._artifacts_from_specs(specs)

    def _artifacts_from_specs(
        self,
        specs: list[tuple[str, str, str, str, str]],
    ) -> list[ArtifactDraft]:
        artifacts: list[ArtifactDraft] = []
        for workspace_path, artifact_path, media_type, category, description in specs:
            content = self._read_workspace_text(workspace_path)
            if content is None:
                continue
            artifacts.append(
                ArtifactDraft(
                    relative_path=artifact_path,
                    content=content,
                    media_type=media_type,
                    category=category,
                    description=description,
                )
            )
        return artifacts

    def _questions(self, output: dict[str, Any]) -> tuple[ClarificationQuestion, ...]:
        raw_questions = output.get("questions") or []
        questions: list[ClarificationQuestion] = []
        if isinstance(raw_questions, list):
            for index, question in enumerate(raw_questions, 1):
                if isinstance(question, dict):
                    text = str(question.get("text") or "").strip()
                    if not text:
                        continue
                    questions.append(
                        ClarificationQuestion(
                            id=str(question.get("id") or f"solver_clarification_{index}"),
                            text=text,
                            answer_type=str(question.get("answer_type") or "text"),
                            options=tuple(str(item) for item in question.get("options") or ()),
                            context=str(question.get("context") or ""),
                        )
                    )
                elif str(question).strip():
                    questions.append(
                        ClarificationQuestion(
                            id=f"solver_clarification_{index}",
                            text=str(question).strip(),
                        )
                    )
        if not questions:
            questions.append(
                ClarificationQuestion(
                    id="solver_clarification_1",
                    text=str(output.get("message") or "请确认求解所需的关键业务口径。"),
                )
            )
        return tuple(questions)

    def _workspace_root(self, state: Any) -> str:
        return str(getattr(self.services.config, "workspace_root", None) or "")

    def _solve_time_budget(self) -> int:
        return self.services.config.solver_timeout

    def _read_workspace_json(self, relative_path: str) -> dict[str, Any] | list[Any] | None:
        text = self._read_workspace_text(relative_path)
        if text is None:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    def _read_workspace_text(self, relative_path: str) -> str | None:
        toolset = getattr(self.services, "workspace_toolset", None)
        config_root = getattr(toolset, "root", None) or getattr(
            self.services.config, "workspace_root", None
        )
        if not config_root:
            return None
        try:
            root = Path(str(config_root)).expanduser().resolve()
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
