"""Explanation stage for user-facing result interpretation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..contracts import CoreServices
from ..feasibility_routing import has_confirmed_instance_reviews
from ..message_builder import MessageBuilder
from ..models import AgentReport, AgentStage, ArtifactDraft, ArtifactProduced
from ..stage_agent import StageAgent
from ..stage_outputs import STAGE_OUTPUT_FILES


class ExplanationAgent(StageAgent):
    """Convert solver results into a user-facing explanation."""

    def __init__(
        self,
        services: CoreServices,
        *,
        algorithm_design_enabled: bool = True,
        problem_contract_enabled: bool = True,
    ):
        self._algorithm_design_enabled = algorithm_design_enabled
        self._problem_contract_enabled = problem_contract_enabled
        super().__init__(
            stage=AgentStage.EXPLANATION,
            services=services,
            tools=services.workspace_tools_for(AgentStage.EXPLANATION),
        )

    def build_messages(self, state: Any) -> MessageBuilder:
        prompts = self.services.prompts
        mb = MessageBuilder.from_prompts(prompts.explain_system, prompts.explain_contract)
        feasibility_inputs = ""
        if isinstance(getattr(state, "feasibility_review", None), dict):
            if (state.solver_result or {}).get("status") == "infeasible":
                feasibility_inputs = (
                    f"- `{STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]}`: "
                    "独立无解证明审查结论。\n"
                    "- `solver_result.json`: 完整模型的结构化不可行证明；本路径没有 "
                    "feasibility_result.json 和 solution.json。\n"
                )
            else:
                feasibility_inputs = (
                    f"- `{STAGE_OUTPUT_FILES[AgentStage.FEASIBILITY_REVIEW]}`: "
                    "独立可行性验收结论。\n"
                    "- `feasibility_result.json`: 独立可行性检查结果。\n"
                )
        else:
            feasibility_inputs = (
                "- 本次运行未启用 Feasibility Review；不得声称独立 checker 已通过。\n"
            )
        solution_input = (
            ""
            if (state.solver_result or {}).get("status") == "infeasible"
            else (
                "- `solution.json`: 完整业务解；需要引用具体决策时读取，"
                "不要在最终解释中无意义展开。\n"
            )
        )
        upstream_inputs = ""
        if self._problem_contract_enabled:
            upstream_inputs += f"- `{STAGE_OUTPUT_FILES[AgentStage.PROBLEM_CONTRACT]}`: 问题契约摘要。\n"
        if self._algorithm_design_enabled:
            upstream_inputs += f"- `{STAGE_OUTPUT_FILES[AgentStage.ALGORITHM_DESIGN]}`: 算法设计。\n"
        message = (
            "【当前阶段】explanation\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 已确认的问题定义。\n"
            + upstream_inputs
            + f"- `{STAGE_OUTPUT_FILES[AgentStage.SOLVING]}`: 求解决策摘要。\n"
            "- `solver_result.json`: 精简求解结果摘要。\n"
        )
        message += feasibility_inputs + solution_input
        message += (
            "【必须写入】\n"
            f"- 最终业务解释 JSON 必须写入 `{STAGE_OUTPUT_FILES[AgentStage.EXPLANATION]}`。\n"
            "写完后请自行 read_file 或 shell json.load 验收该文件；最终 message 只需简短说明完成，"
            "不要粘贴完整 JSON。"
        )
        mb.add_user(message)
        instance_reviews = tuple(getattr(state, "instance_infeasibility_reviews", ()))
        if has_confirmed_instance_reviews(instance_reviews):
            mb.add_user(
                "【实例不可行已确认】\n"
                "当前没有可交付解。三次相互隔离的 Feasibility Review 均判定实例不可行。"
                "必须解释不可行的业务原因、证明范围和可行的业务调整，"
                "不得把当前结果描述成可行解、最优解或要求继续求解。\n"
                "【三次独立判定摘要】\n"
                + json.dumps(instance_reviews[-3:], ensure_ascii=False, indent=2)
            )
        return mb

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        return {"explanation": data}

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        explanation = output.get("explanation", {})
        artifacts: list[ArtifactDraft] = []
        if explanation:
            artifacts.append(
                ArtifactDraft(
                    relative_path="result/explanation.json",
                    content=json.dumps(explanation, ensure_ascii=False, indent=2),
                    media_type="application/json",
                    category="result",
                    description="业务解释",
                )
            )
        if artifacts:
            yield ArtifactProduced(
                stage=AgentStage.EXPLANATION,
                message="求解与解释已完成",
                artifacts=tuple(artifacts),
            )
