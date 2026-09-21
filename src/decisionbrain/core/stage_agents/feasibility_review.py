"""Independent feasibility acceptance and responsibility review stage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from ..contracts import CoreServices
from ..feasibility_routing import render_for_reviewer, visible_review_history
from ..message_builder import MessageBuilder
from ..models import AgentReport, AgentStage, ArtifactDraft, ArtifactProduced
from ..stage_agent import (
    OutputValidationIssue,
    StageAgent,
    StageExecutionStopped,
    StageOutputValidationError,
    Tool,
    pydantic_validation_issues,
)
from ..stage_output_models import (
    FeasibilityCheckReceipt,
    FeasibilityResult,
    FeasibilityReviewOutput,
)
from ..stage_outputs import (
    FEASIBILITY_REVIEW_HISTORY_FILE,
    PROBLEM_CONTRACT_CHECKER_FILE,
    PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
    PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
    SOLVING_CODE_FILE,
    SOLVING_FEASIBILITY_RECEIPT_FILE,
    SOLVING_FEASIBILITY_RESULT_FILE,
    SOLVING_INPUT_FILE,
    SOLVING_RESULT_FILE,
    SOLVING_RUNTIME_OUTCOME_FILE,
    SOLVING_SOLUTION_FILE,
    STAGE_OUTPUT_FILES,
)
from ..workspace_tools import file_sha256
class FeasibilityReviewAgent(StageAgent):
    """Independent reviewer that accepts a candidate or attributes its rejection."""

    def __init__(
        self,
        services: CoreServices,
        *,
        input_schema_enabled: bool = True,
        algorithm_design_enabled: bool = True,
    ) -> None:
        self._input_schema_enabled = input_schema_enabled
        self._algorithm_design_enabled = algorithm_design_enabled
        workspace_tools = services.workspace_tools_for(AgentStage.FEASIBILITY_REVIEW)
        read_tools = tuple(
            tool
            for tool in workspace_tools
            if tool.name in {"list_files", "read_file"}
        )
        shell_tool = next(tool for tool in workspace_tools if tool.name == "shell")
        checker_tool = next(
            tool for tool in workspace_tools if tool.name == "run_feasibility_checker"
        )
        self._run_checker = checker_tool.handler
        self._checker_executed = False
        self._has_review_history = False
        self._checker_result_before = "<missing>"
        self._last_checker_failure: tuple[str, str, str] | None = None
        self._checker_failure_count = 0
        self._checker_execution_terminal_failure = False
        self._last_validation_failure: tuple[str, str, str] | None = None
        self._validation_failure_count = 0
        reviewer_shell = Tool(
            name="shell",
            description=(
                "Run shell commands for independent feasibility review, such as focused data "
                "analysis and JSON inspection. It cannot run the checker or write its result: "
                "use run_feasibility_checker for that, and note that writes to the audited "
                "contract files, the candidate, and the checker result are rolled back here."
            ),
            parameters=shell_tool.parameters,
            handler=shell_tool.handler,
        )
        audited_checker = Tool(
            name=checker_tool.name,
            description=checker_tool.description,
            parameters=checker_tool.parameters,
            handler=self._run_audited_checker,
        )
        write_tool = next(tool for tool in workspace_tools if tool.name == "write_file")
        output_file = STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]
        verdict_writer = Tool(
            name="write_file",
            description=f"Write the feasibility verdict to `{output_file}` only.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "const": output_file},
                    "content": {"type": "string", "description": "Complete JSON object."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=write_tool.handler,
        )
        super().__init__(
            stage=AgentStage.FEASIBILITY_REVIEW,
            services=services,
            tools=(*read_tools, reviewer_shell, audited_checker, verdict_writer),
        )

    async def _run_audited_checker(self, **kwargs: Any) -> str:
        """Run the checker through Runtime and accept the result only with a receipt."""

        if self._checker_execution_terminal_failure:
            return (
                "run_feasibility_checker error: the same checker execution failed 5 times with "
                "the same checker and solution. Do not run it again; write a reject verdict with "
                "responsibility solving and a checker-independent remediation handoff."
            )
        result = await self._run_checker(**kwargs)
        result_updated = (
            self._workspace_file_fingerprint(SOLVING_FEASIBILITY_RESULT_FILE)
            != self._checker_result_before
        )
        # The receipt is Runtime-owned and binds this result to the checker,
        # input and solution it was produced from; a fresh result without a
        # matching receipt is not evidence.
        if result_updated and self._current_checker_receipt() is not None:
            self._checker_executed = True
            self._last_checker_failure = None
            self._checker_failure_count = 0
            self._checker_execution_terminal_failure = False
            return result
        signature = (
            self._workspace_file_hash(PROBLEM_CONTRACT_CHECKER_FILE),
            self._workspace_file_hash(SOLVING_SOLUTION_FILE),
            self._normalize_shell_error(result),
        )
        if signature == self._last_checker_failure:
            self._checker_failure_count += 1
        else:
            self._last_checker_failure = signature
            self._checker_failure_count = 1
        if self._checker_failure_count >= 5:
            self._checker_execution_terminal_failure = True
            return (
                result
                + "\n\nThe same checker execution failed 5 times with the same checker and "
                "solution. Do not run it again; write a reject verdict with responsibility "
                "solving and a checker-independent remediation handoff."
            )
        return result

    def _current_checker_receipt(self) -> FeasibilityCheckReceipt | None:
        """Return the receipt only when it still describes the files on disk.

        Presence of a valid ``feasibility_result.json`` proves nothing about where
        it came from.  The receipt names the checker, input, solution and result it
        was produced from, so re-hashing those four files here is what turns "a
        result exists" into "this result is the output of this check".
        """

        data = self._read_workspace_json(SOLVING_FEASIBILITY_RECEIPT_FILE)
        if not isinstance(data, dict):
            return None
        try:
            receipt = FeasibilityCheckReceipt.model_validate(data)
        except ValidationError:
            return None
        expected = {
            PROBLEM_CONTRACT_CHECKER_FILE: receipt.checker_sha256,
            SOLVING_INPUT_FILE: receipt.input_sha256,
            SOLVING_SOLUTION_FILE: receipt.solution_sha256,
            SOLVING_FEASIBILITY_RESULT_FILE: receipt.result_sha256,
        }
        root = self._stage_workspace_root()
        if root is None:
            return None
        for relative_path, digest in expected.items():
            if file_sha256(root / relative_path) != digest:
                return None
        return receipt

    def build_messages(self, state: Any) -> MessageBuilder:
        self._checker_executed = False
        self._checker_result_before = self._workspace_file_fingerprint(
            SOLVING_FEASIBILITY_RESULT_FILE
        )
        self._last_checker_failure = None
        self._checker_failure_count = 0
        self._checker_execution_terminal_failure = False
        self._last_validation_failure = None
        self._validation_failure_count = 0
        # An unfinished instance-confirmation segment may project no visible conclusion.
        self._has_review_history = bool(
            visible_review_history(
                tuple(getattr(state, "feasibility_audits", ())),
                instance_run=tuple(getattr(state, "instance_infeasibility_reviews", ())),
            )
        )
        prompts = self.services.prompts
        formulator_arm = (
            not self._algorithm_design_enabled
            and not prompts.components_enabled
            and bool(prompts.gurobi_formulator_system)
        )
        mb = MessageBuilder.from_prompts(
            prompts.feasibility_review_system,
            prompts.feasibility_review_contract,
        )
        solver_result = state.solver_result if isinstance(state.solver_result, dict) else {}
        runtime_timeout_review = (
            solver_result.get("outcome_source") == "runtime_external_timeout"
        )
        infeasible_review = solver_result.get("status") == "infeasible"
        no_candidate_review = (
            not infeasible_review and solver_result.get("solution_file") is None
        )
        execution_field = (
            "executions" if prompts.components_enabled else "execution"
        )
        design_failure_factors = (
            "设计的方法、展开方式、solver、fallback 或预算"
            if prompts.components_enabled
            else "设计的方法、展开方式、solver 或预算"
        )
        incomplete_model_examples = (
            "局部模型、预处理子问题"
            if prompts.components_enabled
            else "未覆盖全部硬约束的模型"
        )
        no_candidate_evidence = (
            "- 这是 Runtime 记录的正式 solver.py 外部超时（status=runtime_timeout）；读取 "
            f"`{SOLVING_RUNTIME_OUTCOME_FILE}`，只把其中的进程、信号、时间和候选文件状态"
            f"作为客观事实，不得要求其伪造 {execution_field}、validation、diagnosis、gap 或最优性。\n"
            f"- 结合 solver.py、算法设计和压缩账本判断责任：{design_failure_factors}"
            "不适配规模时归 algorithm_design；实现偏离设计、没有内部限时或"
            "没有在截止前原子保留 incumbent 时归 solving。\n"
            if runtime_timeout_review
            else (
                f"- 审查 solver_result.status、diagnosis、{execution_field}、solver.py、算法设计和"
                "压缩账本，判断本轮责任属于 algorithm_design 还是 solving。\n"
            )
        )
        if not self._algorithm_design_enabled:
            no_candidate_evidence = (
                f"- 审查 solver_result.status、diagnosis、{execution_field}、solver.py 和压缩账本；"
                "完整模型、数据转换、执行和业务结果映射缺陷均归 solving。\n"
            )
        candidate_instructions = (
            "【当前验收路径：求解器不可行证明】\n"
            "- solver_result.status=infeasible；本路径没有、也不得要求 solution.json。\n"
            "- 不执行 feasibility_checker.py：checker 只能检查候选解，不能验证无解证明。\n"
            f"- 必须审查 solver.py、solver_result.infeasibility_proof、实际 {execution_field}、算法设计"
            "以及全部硬约束是否由同一个完整精确模型覆盖。\n"
            "- 只有完整模型的可信 INFEASIBLE 状态或求解器证书才可归为 instance；"
            f"{incomplete_model_examples}、启发式失败、超时或代码异常必须归为 "
            "algorithm_design 或 solving。\n"
            if infeasible_review
            else (
                "【当前验收路径：无候选求解终态】\n"
                "- 本路径没有、也不得要求 solution.json，不执行 feasibility_checker.py。\n"
                + no_candidate_evidence
                +
                "- runtime_timeout、time_limit、unknown、no_solution_found、infeasible_or_unbounded 或启发式未找到解"
                "都不能归为 instance，也不能 accept。\n"
                if no_candidate_review
                else (
                    "【当前验收路径：候选解检查】\n"
                    "- 先读取 checker，再用 `run_feasibility_checker` 按它实际实现的 CLI 传参执行，"
                    "使用当前 input.json、solution.json 和 data/（如该 CLI 需要），并使其更新 "
                    "feasibility_result.json。该工具是产出 checker 结果的唯一途径；"
                    "shell 和 write_file 都写不了它。\n"
                    f"- 再读取 `{SOLVING_FEASIBILITY_RESULT_FILE}`，结合当前实现与业务约束独立裁决。\n"
                    f"- 裁决时 `{SOLVING_FEASIBILITY_RECEIPT_FILE}` 必须仍与当前 checker、input、"
                    "solution 和结果一致；任何一项在执行后发生变化都必须重新执行该工具。\n"
                    "- 若 checker 参数、路径或结果文件处理失败，先自行诊断并重新执行；只有工具"
                    "明确提示同一失败已重复 5 次时，才停止重试并 reject，责任只能是 solving。\n"
                )
            )
        )
        solving_result_requirement = (
            f"- `{SOLVING_RUNTIME_OUTCOME_FILE}`: Runtime 记录的正式求解器外部超时事实。\n"
            if runtime_timeout_review
            else f"- `{SOLVING_RESULT_FILE}`: 模型提交的求解实现及结果。\n"
        )
        mb.add_user(
            "【当前阶段】feasibility_review\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题与硬约束。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT]}`: 已通过独立审计的问题契约。\n"
            + (
                f"- `{PROBLEM_CONTRACT_INPUT_SCHEMA_FILE}` 与 "
                f"`{PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE}`: 输入和解的语义结构。\n"
                if self._input_schema_enabled
                else f"- `{PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE}`: 解的语义结构。本次运行不提供 "
                "input schema，输入结构以 `data/` 与实际 input.json 为准；"
                "不得因缺少 input_schema.json 判定责任或拒绝候选。\n"
            )
            +
            f"- `{PROBLEM_CONTRACT_CHECKER_FILE}`: 已审计的可行性规则。\n"
            + (
                f"- `{STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN]}`: 算法设计。\n"
                if self._algorithm_design_enabled
                else ""
            )
            + (
                "【本臂专用责任约束】只有全量完整 Gurobi 模型的建模问题"
                "才能归 gurobi_formulator，且回退后的修改仍必须保持全量完整建模；若完整模型正确且没有其他"
                "建模方法上的改进空间，"
                "不得归责 gurobi_formulator。翻译、实现或执行问题只能归 solving。\n"
                if formulator_arm
                else ""
            )
            +
            f"- `{STAGE_OUTPUT_FILES[AgentStage.SOLVING]}` 与 `{SOLVING_CODE_FILE}`: 阶段终态和求解实现。\n"
            + solving_result_requirement
            +
            f"- `{SOLVING_INPUT_FILE}`: 被验收的规范化输入。\n"
            + (
                f"- `{FEASIBILITY_REVIEW_HISTORY_FILE}`: 历次可行性裁决全文与 checker 结果，"
                "只有本阶段可读；作出本轮结论前必须读取它核对既往判断与修复要求。\n"
                if self._has_review_history
                else ""
            )
            + (
                ""
                if infeasible_review or no_candidate_review
                else f"- `{SOLVING_SOLUTION_FILE}`: 被验收的候选解。\n"
            )
            + candidate_instructions
            + "【工具权限】\n"
            "- shell 可用于数据分析和 JSON 验收，但不能执行 checker，也不能写入被审计的契约文件、"
            "候选解或 checker 结果：这些写入会被自动回滚。\n"
            "- 候选解路径必须用 `run_feasibility_checker` 实际执行 checker 并更新 "
            "feasibility_result.json。\n"
            "- problem contract 已完成独立审计；不得判断或输出 contract/checker 责任。\n"
            + (
                "- 拒绝时只能在 instance、algorithm_design、solving 中选择一个责任。\n"
                if self._algorithm_design_enabled
                else ("- 拒绝时 responsibility 只能是 instance、gurobi_formulator 或 solving。\n" if formulator_arm else "- 拒绝时 responsibility 只能是 instance 或 solving。\n")
            )
            +
            "- 本阶段只裁决责任，不自行执行回退或修复；Core 会按责任确定性路由。\n"
            "【必须写入】\n"
            f"- 唯一结论写入 `{STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]}`。\n"
            "写完后用 read_file 验收；最终 message 只需简短说明结论。\n"
            + render_for_reviewer(
                getattr(state, "feasibility_directives", ()),
            )
        )
        return mb

    def validation_feedback_remedy(self, error: StageOutputValidationError) -> str:
        """Name the tools this stage actually has, and the right one per problem.

        The reviewer has no replace_in_file, its write_file accepts only the verdict
        path, and its shell can neither run the checker nor write its result. A
        checker or receipt problem is fixed by running the checker again, not by
        editing a file.
        """

        return (
            "若问题出在 checker 执行或执行凭据，请重新调用 `run_feasibility_checker`；"
            "若问题出在裁决内容，请用 `write_file` 重写 "
            f"`{STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]}` 后用 `read_file` 验收。"
            "本阶段没有 replace_in_file，shell 也不能执行 checker 或写入其结果。"
            "完成后只需发一条简短消息，不要在 message 中粘贴完整 JSON。"
        )

    def build_validation_feedback(self, error: StageOutputValidationError) -> str:
        error_key = json.dumps(
            {
                "message": str(error),
                "issues": [
                    {
                        "path": issue.path,
                        "message": issue.message,
                        "expected": issue.expected,
                    }
                    for issue in error.issues
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        signature = (
            self._workspace_file_hash(PROBLEM_CONTRACT_CHECKER_FILE),
            self._workspace_file_hash(SOLVING_SOLUTION_FILE),
            error_key,
        )
        if signature == self._last_validation_failure:
            self._validation_failure_count += 1
        else:
            self._last_validation_failure = signature
            self._validation_failure_count = 1
        if self._validation_failure_count >= 5:
            raise StageExecutionStopped(
                "feasibility review stopped after the same checker, solution, and "
                "validation error repeated 5 times"
            )
        return super().build_validation_feedback(error)

    @staticmethod
    def _normalize_shell_error(result: str) -> str:
        return re.sub(r"^Wall time:.*$", "", result, flags=re.MULTILINE).strip()

    def _workspace_file_hash(self, relative_path: str) -> str:
        root = self._stage_workspace_root()
        if root is None:
            return "<workspace-unavailable>"
        target = root / relative_path
        try:
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return "<missing>"

    def _workspace_file_fingerprint(self, relative_path: str) -> str:
        root = self._stage_workspace_root()
        if root is None:
            return "<workspace-unavailable>"
        target = root / relative_path
        try:
            return f"{target.stat().st_mtime_ns}:{self._workspace_file_hash(relative_path)}"
        except OSError:
            return "<missing>"

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        try:
            output = FeasibilityReviewOutput.model_validate(data)
        except ValidationError as exc:
            raise StageOutputValidationError(
                "feasibility review 输出不符合契约",
                issues=pydantic_validation_issues(exc),
                file_path=self.output_file(state),
            ) from exc

        issues: list[OutputValidationIssue] = []
        if not self._algorithm_design_enabled and output.responsibility == "algorithm_design":
            formulator_arm = (
                not self.services.prompts.components_enabled
                and bool(self.services.prompts.gurobi_formulator_system)
            )
            expected = "gurobi_formulator" if formulator_arm else "solving"
            issues.append(
                OutputValidationIssue(
                    path="$.responsibility",
                    message=f"该工作流不存在 algorithm_design；应归 {expected}",
                    expected=f'responsibility="{expected}"',
                    actual=output.responsibility,
                )
            )
        solver_result = state.solver_result if isinstance(state.solver_result, dict) else {}
        infeasible_review = solver_result.get("status") == "infeasible"
        no_candidate_review = (
            not infeasible_review and solver_result.get("solution_file") is None
        )
        candidate_review = not infeasible_review and not no_candidate_review
        terminal_checker_failure = candidate_review and self._checker_execution_terminal_failure
        terminal_failure_rejection = (
            terminal_checker_failure
            and output.decision == "reject"
            and output.responsibility == "solving"
        )
        if candidate_review and not terminal_failure_rejection:
            if not self._checker_executed:
                issues.append(
                    OutputValidationIssue(
                        path="$.feasibility_result",
                        message=(
                            "本轮尚未通过 run_feasibility_checker 成功执行 checker 并更新 "
                            "feasibility_result.json"
                        ),
                        expected="a fresh checker execution in the current review turn",
                    )
                )
            elif self._current_checker_receipt() is None:
                # Execution alone is insufficient; the result must still belong to that execution.
                issues.append(
                    OutputValidationIssue(
                        path="$.feasibility_result",
                        message=(
                            f"{SOLVING_FEASIBILITY_RECEIPT_FILE} 缺失、非法，或其记录的 checker、"
                            "input、solution、feasibility_result 摘要与当前文件不一致；"
                            "必须重新执行 run_feasibility_checker"
                        ),
                        expected=(
                            "a receipt whose digests match the current checker, input, "
                            "solution, and result"
                        ),
                    )
                )
        feasibility_data = (
            None
            if infeasible_review or no_candidate_review
            else self._read_workspace_json(SOLVING_FEASIBILITY_RESULT_FILE)
        )
        feasibility: FeasibilityResult | None = None
        if (
            candidate_review
            and not isinstance(feasibility_data, dict)
            and not terminal_failure_rejection
        ):
            issues.append(
                OutputValidationIssue(
                    path="$.feasibility_result",
                    message=(
                        "没有本轮 checker 生成的合法 feasibility_result.json；"
                        "必须先读取 checker 后通过 run_feasibility_checker 执行它"
                    ),
                    expected="feasibility result JSON object",
                )
            )
        elif candidate_review and not terminal_failure_rejection:
            try:
                feasibility = FeasibilityResult.model_validate(feasibility_data)
            except ValidationError as exc:
                issues.extend(pydantic_validation_issues(exc, prefix="$.feasibility_result"))

        if terminal_checker_failure and not terminal_failure_rejection:
            issues.append(
                OutputValidationIssue(
                    path="$.responsibility",
                    message=(
                        "同一 checker 执行失败已重复 5 次；只能 reject 并将责任归于 solving"
                    ),
                    expected='decision="reject" and responsibility="solving"',
                )
            )

        if output.decision == "accept":
            accepted = (
                feasibility is not None
                and feasibility.schema_valid is True
                and feasibility.feasible is True
            )
            if not accepted:
                issues.append(
                    OutputValidationIssue(
                        path="$.decision",
                        message="checker 未接受当前候选，feasibility reviewer 不能放行",
                        expected="feasibility_result.schema_valid=true and feasible=true",
                        actual=(
                            feasibility.model_dump(mode="json")
                            if feasibility is not None
                            else feasibility_data
                        ),
                    )
                )
            if solver_result.get("status") in {"infeasible", "failed"}:
                issues.append(
                    OutputValidationIssue(
                        path="$.decision",
                        message="solver_result 状态不支持 accept",
                        expected='status="optimal" or "time_limit"',
                        actual=solver_result.get("status"),
                    )
                )
            if no_candidate_review:
                issues.append(
                    OutputValidationIssue(
                        path="$.decision",
                        message="没有候选解的求解终态不能 accept",
                        expected=(
                            "reject with algorithm_design or solving responsibility"
                            if self._algorithm_design_enabled
                            else "reject with solving responsibility"
                        ),
                        actual=solver_result.get("status"),
                    )
                )
        elif (
            output.responsibility == "instance"
            and solver_result.get("status") != "infeasible"
        ):
            issues.append(
                OutputValidationIssue(
                    path="$.responsibility",
                    message="没有 solver 的 infeasible 状态，不得把候选失败归因为实例不可行",
                    expected='solver_result.status="infeasible" plus a concrete proof',
                    actual=solver_result.get("status"),
                )
            )
        if issues:
            raise StageOutputValidationError(
                "feasibility review 与求解证据不一致",
                issues=tuple(issues),
                file_path=self.output_file(state),
            )
        return {
            "feasibility_review": output.model_dump(mode="json"),
            "feasibility_check": (
                feasibility.model_dump(mode="json") if feasibility is not None else None
            ),
        }

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        review = output["feasibility_review"]
        decision = str(review.get("decision") or "")
        responsibility = str(review.get("responsibility") or "")
        persisted_review = dict(review)
        if decision == "reject" and responsibility == "instance":
            prior_reviews = tuple(getattr(state, "instance_infeasibility_reviews", ()))
            persisted_review["instance_infeasibility_reviews"] = [
                *prior_reviews,
                {
                    "decision": "reject",
                    "responsibility": "instance",
                    "confidence": str(review.get("confidence") or ""),
                    "summary": str(review.get("summary") or ""),
                    "infeasibility_proof": str(review.get("infeasibility_proof") or ""),
                },
            ]
        message = "可行性验收通过"
        if decision == "reject":
            message = f"可行性验收拒绝，责任归于 {responsibility}"
        artifacts = [
            ArtifactDraft(
                relative_path="review/feasibility_review.json",
                content=json.dumps(persisted_review, ensure_ascii=False, indent=2),
                media_type="application/json",
                category="feasibility_review",
                description="独立可行性验收与责任判断",
            )
        ]
        if output["feasibility_check"] is not None:
            artifacts.insert(
                0,
                ArtifactDraft(
                    relative_path="result/feasibility_result.json",
                    content=json.dumps(
                        output["feasibility_check"],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    media_type="application/json",
                    category="result",
                    description="Feasibility Reviewer 执行的独立可行性检查结果",
                ),
            )
            receipt = self._read_workspace_json(SOLVING_FEASIBILITY_RECEIPT_FILE)
            if isinstance(receipt, dict):
                artifacts.insert(
                    1,
                    ArtifactDraft(
                        relative_path="result/feasibility_check_receipt.json",
                        content=json.dumps(receipt, ensure_ascii=False, indent=2),
                        media_type="application/json",
                        category="result",
                        description="Runtime 记录的 checker 执行凭据",
                    ),
                )
        yield ArtifactProduced(
            stage=AgentStage.FEASIBILITY_REVIEW,
            message=message,
            artifacts=tuple(artifacts),
        )
