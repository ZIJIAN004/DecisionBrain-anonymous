"""Independent Gurobi Formulator stage for complete model construction."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..contracts import CoreServices
from ..feasibility_routing import render_for_executor as render_feasibility_repair
from ..message_builder import MessageBuilder
from ..models import AgentStage, ArtifactDraft, ArtifactProduced
from ..stage_agent import (
    StageAgent,
    StageOutputValidationError,
    pydantic_validation_issues,
)
from ..stage_output_models import GurobiFormulationOutput
from ..stage_outputs import STAGE_OUTPUT_FILES


class GurobiFormulatorAgent(StageAgent):
    """Build a complete Gurobi LP/MILP definition without solver code."""

    def __init__(self, services: CoreServices):
        super().__init__(
            stage=AgentStage.GUROBI_FORMULATOR,
            services=services,
            tools=tuple(
                tool for tool in services.workspace_tools_for(AgentStage.GUROBI_FORMULATOR)
                if tool.name not in {
                    "run_solver", "submit_solving_outcome", "run_feasibility_checker",
                }
            ),
        )

    def build_messages(self, state: Any) -> MessageBuilder:
        mb = MessageBuilder.from_prompts(
            self.services.prompts.gurobi_formulator_system,
            self.services.prompts.gurobi_formulator_contract,
        )
        # Review may return responsibility here; include the rejection context to avoid rebuilding
        # the same rejected formulation.
        feasibility_repair = render_feasibility_repair(
            getattr(state, "feasibility_directives", ()),
            stage=AgentStage.GUROBI_FORMULATOR.value,
            components_enabled=self.services.prompts.components_enabled,
            algorithm_design_enabled=False,
            gurobi_formulator_enabled=True,
        )
        mb.add_user(
            "【当前阶段】gurobi_formulator\n"
            "【必须读取】\n"
            "- problem.md、data/、stage_outputs/problem_contract.json、solution_schema.json。\n"
            "【必须写入】\n"
            f"- 完整 formulation JSON：`{STAGE_OUTPUT_FILES[AgentStage.GUROBI_FORMULATOR]}`。\n"
            "建立一个覆盖全部变量、硬约束、目标和结果映射的完整 Gurobi LP/MILP 模型。"
            "不得写 solver.py，不得加入启发式、分解、warm start、候选生成或 fallback。\n"
            f"solver 总预算为 {self.services.config.solver_timeout} 秒；该阶段只做规格设计。"
            + (f"\n\n{feasibility_repair}" if feasibility_repair else "")
        )
        return mb

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        try:
            output = GurobiFormulationOutput.model_validate(data)
        except ValidationError as exc:
            raise StageOutputValidationError(
                "gurobi_formulator 输出不符合契约",
                issues=pydantic_validation_issues(exc),
                file_path=self.output_file(state),
            ) from exc
        return {"gurobi_formulation": output.model_dump(mode="json")}

    async def _after_parse(self, output: dict[str, Any], state: Any):
        formulation = output.get("gurobi_formulation")
        if isinstance(formulation, dict):
            yield ArtifactProduced(
                stage=AgentStage.GUROBI_FORMULATOR,
                message="Gurobi 完整模型规格已生成",
                artifacts=(ArtifactDraft(
                    relative_path="design/gurobi_formulation.json",
                    content=json.dumps(formulation, ensure_ascii=False, indent=2),
                    media_type="application/json",
                    category="gurobi_formulation",
                    description="完整 Gurobi LP/MILP formulation",
                ),),
            )
