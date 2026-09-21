"""Algorithm Design Stage - build a practical solving strategy."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from ..contracts import CoreServices
from ..feasibility_routing import render_for_executor as render_feasibility_repair
from ..message_builder import MessageBuilder
from ..models import AgentReport, AgentStage, ArtifactDraft, ArtifactProduced
from ..package_policy import design_addendum, validate_design
from ..stage_agent import (
    OutputValidationIssue,
    StageAgent,
    StageOutputValidationError,
    pydantic_validation_issues,
)
from ..stage_output_models import AlgorithmDesignOutput, SingleAlgorithmDesignOutput
from ..stage_outputs import FIXED_SOLUTION_CONTRACT_FILE, STAGE_OUTPUT_FILES


_CATALOG_COMPONENT_TERM = re.compile(r"\bcomponents?\b", flags=re.IGNORECASE)


def _single_algorithm_catalog_summary(summary: str) -> str:
    """Keep the full catalog while removing structural terminology from its prose."""

    payload = json.loads(summary)
    algorithms = payload.get("algorithms") if isinstance(payload, dict) else None
    if not isinstance(algorithms, list):
        return summary
    for algorithm in algorithms:
        if not isinstance(algorithm, dict):
            continue
        records = [algorithm]
        solvers = algorithm.get("solvers")
        if isinstance(solvers, list):
            records.extend(solver for solver in solvers if isinstance(solver, dict))
        for record in records:
            text = record.get("summary")
            if isinstance(text, str):
                record["summary"] = _CATALOG_COMPONENT_TERM.sub(
                    lambda match: (
                        "elements" if match.group(0).lower().endswith("s") else "element"
                    ),
                    text,
                )
                if _CATALOG_COMPONENT_TERM.search(record["summary"]):
                    raise ValueError(
                        "algorithm catalog summary contains unsupported structural terminology"
                    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    return rendered


class AlgorithmDesignAgent(StageAgent):
    """Algorithm Design Stage: turn the confirmed problem definition into a solve spec."""

    def __init__(self, services: CoreServices, *, problem_contract_enabled: bool = True):
        self._problem_contract_enabled = problem_contract_enabled
        super().__init__(
            stage=AgentStage.ALGORITHM_DESIGN,
            services=services,
            tools=(
                *services.workspace_tools_for(AgentStage.ALGORITHM_DESIGN),
                *services.algorithm_tools_for(AgentStage.ALGORITHM_DESIGN),
            ),
        )
        # An empty catalog denotes ablation 2. Match the prompt to the count=0 summary.
        self._algorithm_library_enabled = bool(services.algorithm_tools_for(
            AgentStage.ALGORITHM_DESIGN
        ))

    def build_messages(self, state: Any) -> MessageBuilder:
        prompts = self.services.prompts
        algorithm_catalog = self.services.algorithm_catalog_summary()
        if not prompts.components_enabled:
            algorithm_catalog = _single_algorithm_catalog_summary(algorithm_catalog)
        runtime_constraints = json.dumps(
            {
                "solver_timeout_seconds": self.services.config.solver_timeout,
                "source": "CoreConfig.solver_timeout",
            },
            ensure_ascii=False,
            indent=2,
        )
        mb = MessageBuilder.from_prompts(
            prompts.algorithm_design_system,
            prompts.algorithm_design_contract,
        )
        feasibility_repair = render_feasibility_repair(
            getattr(state, "feasibility_directives", ()),
            stage=AgentStage.ALGORITHM_DESIGN.value,
            components_enabled=self.services.prompts.components_enabled,
        )
        if self.services.prompts.components_enabled:
            package_design_terms = "selection.components 或 fallback.components"
            package_subject = "组件"
            guide_candidate_scope = "最终采用或作为 fallback 的 package"
        else:
            package_design_terms = "candidates 或 algorithm"
            package_subject = "算法"
            guide_candidate_scope = "最终采用的 package"
        mb.add_user(
            "【当前阶段】algorithm_design\n"
            "【运行时硬约束（系统注入，只读）】\n"
            f"{runtime_constraints}\n"
            "- `solver_timeout_seconds` 是你随后设计并提交的 solver 执行阶段真实总运行时间上限，不是建议值；"
            "算法设计阶段的 API/数据结构检查、探针和方案比较不受该 solver 预算限制；"
            "提交的 solver 及其全部内部调用必须遵守该预算，不得自行替换或假设其他 solver 时长。\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            "- `data/`: 真实数据文件；如需估算规模或识别字段，请通过工具读取。\n"
            + (
                f"- `{FIXED_SOLUTION_CONTRACT_FILE}`: 固定的输出解 schema；必须读取并据此设计可序列化的解。\n"
                if not self._problem_contract_enabled
                else ""
            )
            + "【不可读取】\n"
            "- problem contract 产物对本阶段不可见；不要尝试读取或依赖 "
            "`stage_outputs/problem_contract.json`、`input_schema.json`、"
            + (
                "`solution_schema.json`、"
                if self._problem_contract_enabled
                else ""
            )
            + "`feasibility_checker.py`。\n"
            + "【必须写入】\n"
            f"- 最终算法设计 JSON 必须写入 `{STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN]}`。\n"
            + (
                "【算法库目录】\n"
                f"{algorithm_catalog}\n"
                "- 上述内容是每个 package 的一句话摘要，并包含其每个 solver 的一句话功能摘要；"
                "已由系统在阶段启动时注入；"
                "不要调用工具逐包遍历完整 manifest。\n"
                "- 先根据问题结构和目录摘要形成少量 shortlist；只对需要认真比较、"
                f"{guide_candidate_scope} 先仅传 package_id 调用 get_algorithm_guide。\n"
                "- 选定 solver 后，必须同时传 package_id 和 solver_id 再次读取 solver 的完整接口与最小示例。\n"
                f"- 写入 {package_design_terms} 的 package "
                "必须已经读取完整 guide；未读取 guide 的 package ID 不得写入算法设计。\n"
                f"- 只有所有硬约束都能映射到所选 solver 的 manifest capability 时才能选择该{package_subject}。\n"
                f"- source=package 的{package_subject}必须写出 solver_id、capability_mapping 和可执行 integration。\n"
                if self._algorithm_library_enabled
                else "【算法库】\n"
                "- 本次运行不提供算法包目录，也没有 get_algorithm_guide 工具。\n"
                "- 你可以使用环境中已安装的任意 Python 包；请自行用 shell 验证其可用性、"
                "版本与接口，不要凭记忆假定 API 存在。\n"
                f"- 由于没有可引用的 manifest，所有{package_subject}都必须写成 source=generated："
                "package、solver_id、integration 为 null，capability_mapping 为空数组。\n"
            )
            + design_addendum(self.services.package_policy)
            + "写完后请自行 read_file 验收该文件；最终 message 只需简短说明完成，"
            "不要粘贴完整 JSON。\n"
            + feasibility_repair
        )
        return mb

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        try:
            model = (
                AlgorithmDesignOutput
                if self.services.prompts.components_enabled
                else SingleAlgorithmDesignOutput
            )
            output = model.model_validate(data)
        except ValidationError as exc:
            raise StageOutputValidationError(
                "algorithm_design 输出不符合契约",
                issues=pydantic_validation_issues(exc),
                file_path=self.output_file(state),
            ) from exc
        payload = output.model_dump(mode="json")
        # Package-pool and single-package ablations require runtime validation here.
        violations = validate_design(payload, self.services.package_policy)
        if violations:
            raise StageOutputValidationError(
                "algorithm_design 违反本次运行的算法包约束",
                issues=tuple(
                    OutputValidationIssue(
                        path=item.path,
                        message=item.message,
                        expected=item.expected,
                        actual=item.actual,
                    )
                    for item in violations
                ),
                file_path=self.output_file(state),
            )
        return {"algorithm_design": payload}

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        design = output.get("algorithm_design")
        if isinstance(design, dict):
            yield ArtifactProduced(
                stage=AgentStage.ALGORITHM_DESIGN,
                message="算法设计已生成",
                artifacts=(
                    ArtifactDraft(
                        relative_path="design/algorithm_design.json",
                        content=json.dumps(design, ensure_ascii=False, indent=2),
                        media_type="application/json",
                        category="algorithm_design",
                        description="求解算法设计规格",
                    ),
                ),
            )
