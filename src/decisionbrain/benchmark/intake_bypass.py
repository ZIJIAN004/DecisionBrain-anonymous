"""Prebuilt artifacts used to bypass Intake.

FrontierOR problem.md files are complete specifications reconstructed from paper models.
Running Intake would add stochastic noise without testing genuine clarification ability.

Evaluation prebuilds only Intake output and starts Runtime at Problem Contract. Core still
generates input semantics, the checker, and the contract summary. Construction is deterministic
and invokes no LLM, preserving identical starting state across models and rounds.

problem_definition is deliberately minimal: problem.md carries problem content while this file
provides only routing confirmation and objective direction.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.models import AgentStage
from ..core.stage_outputs import (
    FIXED_SOLUTION_CONTRACT_FILE,
    PROBLEM_CONTRACT_INPUT_SCHEMA_FILE,
    PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE,
    SOLVING_INPUT_FILE,
    STAGE_OUTPUT_FILES,
)
from .task import BenchmarkTask

BYPASS_MESSAGE = "benchmark 模式：问题描述已是完整规格，跳过澄清直接进入建模"


def _json_schema(value: object) -> dict:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {str(key): _json_schema(item) for key, item in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        return {"type": "array", "items": _json_schema(value[0]) if value else {}}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"type": "null"}
    return {}


def build_intake_output(task: BenchmarkTask, *, input_schema_enabled: bool = True) -> dict:
    """Build a proceed output matching the Intake contract.

    With ``input_schema_enabled=False`` (ablation 1), omit the mechanically derived
    input structure. Otherwise the arms would not differ in whether an input contract
    is provided. The model must derive the structure from ``data/`` in this arm.
    """

    instance = json.loads(task.instance_path.read_text(encoding="utf-8"))
    target_schema = json.loads(task.target_solution_schema.read_text(encoding="utf-8"))
    # Key order determines intake.json bytes and model-visible content. Preserve the pre-switch
    # ordering exactly for enabled arms.
    frontieror_contract: dict = {"input_file": SOLVING_INPUT_FILE}
    if input_schema_enabled:
        frontieror_contract["input_schema_file"] = PROBLEM_CONTRACT_INPUT_SCHEMA_FILE
    frontieror_contract["solution_schema_file"] = PROBLEM_CONTRACT_SOLUTION_SCHEMA_FILE
    if input_schema_enabled:
        frontieror_contract["input_schema"] = _json_schema(instance)
    frontieror_contract["target_solution_template"] = target_schema
    frontieror_contract["instruction"] = (
        "必须直接按 FrontierOR 输入与固定目标解契约求解；solution.json 必须匹配该目标格式。"
        if input_schema_enabled
        else "必须直接按 FrontierOR 输入与固定目标解契约求解；solution.json 必须匹配该目标格式。"
        "本次运行不提供输入结构说明，也不产出 input_schema.json；"
        "规范化输入的字段、类型和维度必须由你自行读取 `data/` 确定。"
    )

    return {
        "decision": "proceed",
        "message": BYPASS_MESSAGE,
        "questions": [],
        "problem_definition": {
            "problem_type": task.problem_class,
            "business_goal": f"求解 {task.problem_class} 类问题，完整定义见 problem.md",
            "objective": {
                "direction": task.objective_direction,
                "description": "目标函数由 problem.md 给出，此处不复述以免与原文产生歧义",
            },
            "decision_variables": [
                {
                    "name": "见 problem.md",
                    "meaning": "决策变量由 problem.md 的问题描述界定，本阶段不预先固定",
                    "type": "integer",
                }
            ],
            "parameters": [
                {
                    "name": "见 data/",
                    "meaning": "全部参数取自输入实例文件",
                    "value_or_source": "data/ 目录下的实例数据",
                }
            ],
            "constraints": [
                {
                    "name": "见 problem.md",
                    "description": "全部硬约束在 problem.md 中已完整给出，须逐条落实",
                    "type": "hard",
                    "penalty": "",
                }
            ],
            "assumptions": [
                "problem.md 为完整问题规格，不存在需要向用户确认的缺口",
                "所有数值参数以 data/ 中的实例文件为准",
            ],
            "feasibility_notes": "业务定义完整；可行性由后续阶段的独立检查器判定",
            "scale_notes": "实例规模见 data/ 中的输入文件",
            "benchmark_bypass": True,
            "frontieror_contract": frontieror_contract,
        },
    }


def materialize(
    task: BenchmarkTask,
    workspace: Path,
    *,
    input_schema_enabled: bool = True,
    problem_contract_enabled: bool = True,
) -> Path:
    """Write Intake artifacts and the fixed caller output contract."""

    target = workspace / STAGE_OUTPUT_FILES[AgentStage.INTAKE]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            build_intake_output(task, input_schema_enabled=input_schema_enabled),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if problem_contract_enabled:
        target_schema = json.loads(task.target_solution_schema.read_text(encoding="utf-8"))
        (workspace / FIXED_SOLUTION_CONTRACT_FILE).write_text(
            json.dumps(target_schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return target
