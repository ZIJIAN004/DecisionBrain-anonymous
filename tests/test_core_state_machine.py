"""Core OptimizationAgent state-machine contract."""

import asyncio
import json
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from decisionbrain.core import AgentState, OptimizationAgent
from decisionbrain.core.models import (
    AgentInput,
    AgentStage,
    ArtifactProduced,
    AssistantMessage,
    ChatResponse,
    ChatRequest,
    NeedsClarification,
    StageCompleted,
    StageStarted,
    Succeeded,
)
from decisionbrain.core.stage_flow import (
    STAGE_SEQUENCE,
    StageTransitionError,
    _prepare_pending_handoff,
    apply_stage_result,
    run_stage,
)
from decisionbrain.core.stage_output_models import FeasibilityReviewOutput
from helpers import fake_core_config, fake_core_services, fake_workspace_toolset


class Control:
    async def is_cancelled(self):
        return False


class FakeLLM:
    def __init__(self, assessments):
        self.assessments = iter(assessments)
        self.solve_calls = 0
        self.stage_attempts: dict[str, int] = {}

    async def complete(self, request: ChatRequest):
        if request.messages and request.messages[-1].get("role") == "tool":
            return ChatResponse(content="done")
        all_text = " ".join(str(m.get("content") or "") for m in request.messages)
        stage = next(
            (name for name in STAGE_SEQUENCE if self._is_stage(all_text, name.value)),
            None,
        )
        if stage is not None:
            attempts = self.stage_attempts.get(stage.value, 0) + 1
            self.stage_attempts[stage.value] = attempts
            if attempts > 1:
                recent = " | ".join(
                    str(message.get("content") or "")
                    for message in request.messages[-6:]
                    if message.get("role") == "tool"
                )
                raise AssertionError(
                    f"FakeLLM received an unexpected retry for stage {stage.value}; "
                    f"the scripted workspace output did not satisfy the current contract: {recent}"
                )
        if self._is_stage(all_text, "solving"):
            self.solve_calls += 1
            return ChatResponse(
                tool_calls=[
                    self._write_tool_call(
                        "write-input-builder",
                        "from_data_to_input.py",
                        "from pathlib import Path\nPath('input.json').write_text('{\"orders\": []}')\n",
                    ),
                    self._write_tool_call(
                        "write-code",
                        "solver.py",
                        "from pathlib import Path\nPath('solution.json').write_text('{\"routes\": []}')\n",
                    ),
                    self._tool_call(
                        "generate-input",
                        "shell",
                        {"command": "python from_data_to_input.py"},
                    ),
                    self._tool_call("run-solver", "run_solver", {}),
                    self._tool_call(
                        "submit-result",
                        "submit_solving_outcome",
                        {"message": "完成", "result": generated_solver_result(12.5)},
                    ),
                ]
            )
        if self._is_stage(all_text, "problem_contract"):
            return ChatResponse(
                tool_calls=[
                    self._write_tool_call(
                        "write-input-schema",
                        "input_schema.json",
                        {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {"orders": {"type": "array"}},
                            "required": ["orders"],
                            "additionalProperties": True,
                        },
                    ),
                    self._write_tool_call(
                        "write-solution-schema",
                        "solution_schema.json",
                        {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {"routes": {"type": "array"}},
                            "required": ["routes"],
                            "additionalProperties": True,
                        },
                    ),
                    self._write_tool_call(
                        "write-feasibility-checker",
                        "feasibility_checker.py",
                        (
                            "from pathlib import Path\n\n"
                            "def check(input_data: dict, solution: dict, "
                            "data_dir: str | Path = 'data') -> dict:\n"
                            "    return {\n"
                            "        'schema_valid': True,\n"
                            "        'feasible': True,\n"
                            "        'objective_value': 12.5,\n"
                            "        'violations': [],\n"
                            "        'warnings': [],\n"
                            "    }\n\n"
                            "if __name__ == '__main__':\n"
                            "    import argparse, json\n"
                            "    parser = argparse.ArgumentParser()\n"
                            "    parser.add_argument('--input', required=True)\n"
                            "    parser.add_argument('--solution', required=True)\n"
                            "    parser.add_argument('--data-dir', default='data')\n"
                            "    parser.add_argument('--output', required=True)\n"
                            "    args = parser.parse_args()\n"
                            "    result = check(json.loads(Path(args.input).read_text()), "
                            "json.loads(Path(args.solution).read_text()), args.data_dir)\n"
                            "    Path(args.output).write_text(json.dumps(result))\n"
                        ),
                    ),
                    self._write_tool_call(
                        "write-problem-contract-output",
                        "stage_outputs/problem_contract.json",
                        {
                            "problem_family": "routing",
                            "files": {
                                "input_schema": "input_schema.json",
                                "solution_schema": "solution_schema.json",
                                "feasibility_checker": "feasibility_checker.py",
                            },
                        },
                    ),
                ]
            )
        if self._is_stage(all_text, "intake"):
            return ChatResponse(
                tool_calls=[
                    self._write_tool_call(
                        "write-intake-output",
                        "stage_outputs/intake.json",
                        next(self.assessments),
                    )
                ]
            )
        if self._is_stage(all_text, "feasibility_review"):
            return ChatResponse(
                tool_calls=[
                    self._tool_call(
                        "run-feasibility-checker",
                        "run_feasibility_checker",
                        {
                            "args": [
                                "--input",
                                "input.json",
                                "--solution",
                                "solution.json",
                                "--output",
                                "feasibility_result.json",
                            ]
                        },
                    ),
                    self._write_tool_call(
                        "write-feasibility-review-output",
                        "stage_outputs/feasibility_review.json",
                        generated_feasibility_review(),
                    )
                ]
            )
        if self._is_stage(all_text, "explanation"):
            return ChatResponse(
                tool_calls=[
                    self._write_tool_call(
                        "write-explanation-output",
                        "stage_outputs/explanation.json",
                        {"summary": "完成"},
                    )
                ]
            )
        if self._is_stage(all_text, "algorithm_design"):
            return ChatResponse(
                tool_calls=[
                    self._write_tool_call(
                        "write-algorithm-design-output",
                        "stage_outputs/algorithm_design.json",
                        generated_algorithm_design(),
                    )
                ]
            )
        raise AssertionError("FakeLLM could not identify the requested stage")

    @staticmethod
    def _is_stage(text: str, stage: str) -> bool:
        return f"当前真实阶段名是：{stage}" in text or f"【当前阶段】{stage}" in text

    @staticmethod
    def _write_tool_call(call_id: str, path: str, content):
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {
                        "path": path,
                        "content": content,
                    },
                    ensure_ascii=False,
                ),
            },
        }

    @staticmethod
    def _tool_call(call_id: str, name: str, arguments: dict):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }


def definition():
    return {
        "problem_type": "routing",
        "business_goal": "最短距离",
        "objective": {"direction": "minimize"},
        "decision_variables": [],
        "parameters": [],
        "constraints": [],
        "assumptions": [],
    }


def generated_algorithm_design():
    return {
        "schema_version": "1.1",
        "problem_family": "routing",
        "diagnosis": {
            "summary": "small routing problem",
            "decision_type": "route",
            "key_difficulty": "capacity",
        },
        "scale": {
            "primary_items": 1,
            "resources": 1,
            "estimated_variables": 1,
            "estimated_constraints": 1,
            "risk": "low",
            "confidence": "high",
            "reason": "test fixture",
        },
        "candidates": [
            {
                "method": "greedy",
                "method_class": "constructive_heuristic",
                "source": "generated",
                "package_id": None,
                "solver_id": None,
                "fit_reason": "small instance",
                "main_risk": "solution quality",
                "recommendation": "primary",
            }
        ],
        "selection": {
            "kind": "single",
            "reason": "small fixture",
            "components": [
                {
                    "component_id": "greedy",
                    "role": "construct routes",
                    "source": "generated",
                    "package": None,
                    "solver_id": None,
                    "method": "greedy",
                    "method_class": "constructive_heuristic",
                    "optimality": "heuristic_or_incumbent",
                    "reason": "small fixture",
                    "capability_mapping": [],
                    "integration": None,
                    "formulation": None,
                }
            ],
        },
        "strategy": {
            "representation": "routes",
            "feasibility": "check routes",
            "objective": "minimize distance",
            "construction": "greedy",
            "improvement": "none",
            "stopping": "after construction",
            "output": "routes",
        },
        "fallback": {
            "trigger": "failure",
            "description": "return feasible route",
            "components": [],
        },
        "risks": [],
    }


def generated_solver_result(objective):
    return {
        "status": "time_limit",
        "objective": objective,
        "mip_gap": None,
        "optimality": "heuristic_or_incumbent",
        "input_file": "input.json",
        "solution_file": "solution.json",
        "code_file": "solver.py",
        "executions": [
            {
                "component_id": "greedy",
                "role": "construct routes",
                "source": "generated",
                "planned_package_id": None,
                "planned_solver_id": None,
                "executed_package_id": None,
                "executed_solver_id": None,
                "method": "greedy",
                "method_class": "constructive_heuristic",
                "version": None,
                "guide_consulted": False,
                "availability_check": {"ok": True, "detail": "not applicable"},
                "seed": None,
                "runtime_limit_seconds": 1,
                "fallback_used": False,
                "deviation_reason": "",
            }
        ],
        "solution_summary": {"routes": 0},
        "constraint_check": [],
        "validation": {
            "schema_check": {"ok": True, "issues": []},
            "business_semantic_check": {"ok": True, "issues": []},
            "algorithm_mapping_check": {
                "ok": True,
                "issues": [],
                "mapped_capabilities": [],
            },
            "implementation_check": {"ok": True, "issues": []},
        },
        "diagnosis": "",
    }


def generated_feasibility_result(objective=12.5, *, feasible=True):
    return {
        "schema_valid": True,
        "feasible": feasible,
        "objective_value": objective,
        "violations": [] if feasible else ["capacity"],
        "warnings": [],
    }


def generated_feasibility_review():
    return {
        "schema_version": "1.0",
        "decision": "accept",
        "responsibility": None,
        "summary": "checker 与求解结果均接受当前候选",
        "evidence": [
            {
                "source": "feasibility_result.json",
                "finding": "schema_valid=true and feasible=true",
            }
        ],
        "required_changes": [],
        "confidence": "high",
        "infeasibility_proof": "",
    }


def run(coro):
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=10))
    except TimeoutError as exc:
        raise AssertionError("core state-machine scenario did not terminate within 10 seconds") from exc


def collect_outputs(agent):
    async def scenario():
        items = []
        async for item in agent.outputs(Control()):
            items.append(item)
        return items

    return run(scenario())


def test_report_models_reject_misspelled_fields():
    with pytest.raises(ValidationError):
        StageStarted(stage=AgentStage.ALGORITHM_DESIGN, mesage="错误")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AssistantMessage(message="消息", unknown=True)  # type: ignore[call-arg]


def test_feasibility_review_responsibility_excludes_contract_and_unknown():
    base = {
        "schema_version": "1.0",
        "decision": "reject",
        "summary": "拒绝当前候选",
        "evidence": [{"source": "feasibility_result.json", "finding": "存在硬违反"}],
        "required_changes": ["修复候选"],
        "confidence": "high",
        "infeasibility_proof": "",
    }
    for responsibility in ("contract", "unknown"):
        with pytest.raises(ValidationError):
            FeasibilityReviewOutput.model_validate(
                {**base, "responsibility": responsibility}
            )


def test_feasibility_remediation_handoff_rejects_checker_details():
    payload = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "solving",
        "summary": "当前候选违反容量约束",
        "evidence": [{"source": "feasibility_result.json", "finding": "容量超限"}],
        "required_changes": ["修复容量聚合"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["vehicle_capacity"],
            "change_ids": ["capacity_aggregation"],
            "problem_requirement": "每条路线不得超过车辆容量",
            "observed_solution_behavior": "旧 solution 中存在超载路线",
            "responsibility_reason": "feasibility_checker.py:42 拒绝了 route_load",
            "required_changes": ["修复容量聚合"],
        },
    }

    with pytest.raises(ValidationError):
        FeasibilityReviewOutput.model_validate(payload)


def test_stage_sequence_is_fixed_and_explicit():
    assert list(STAGE_SEQUENCE) == [
        AgentStage.INTAKE,
        AgentStage.PROBLEM_CONTRACT,
        AgentStage.ALGORITHM_DESIGN,
        AgentStage.GUROBI_FORMULATOR,
        AgentStage.SOLVING,
        AgentStage.FEASIBILITY_REVIEW,
        AgentStage.EXPLANATION,
    ]


def test_agent_state_roundtrip_preserves_typed_internal_fields():
    state = AgentState(
        problem_description="配送问题",
        current_stage=None,
        pending_clarification_questions=[
            {"id": "q1", "text": "容量是多少？", "answer_type": "text"}
        ],
        clarification_rounds=1,
    )

    snapshot = state.to_dict()

    assert snapshot["schema_version"] == "2"
    assert snapshot["current_stage"] is None
    assert isinstance(snapshot["pending_clarification_questions"], list)
    restored = AgentState.from_dict(snapshot)
    assert restored.current_stage is None
    assert restored.pending_clarification_questions[0]["id"] == "q1"
    assert isinstance(restored.pending_clarification_questions, tuple)


def test_agent_state_is_immutable_and_not_a_mapping():
    state = AgentState(problem_description="配送问题")

    with pytest.raises(FrozenInstanceError):
        state.current_stage = AgentStage.SOLVING
    with pytest.raises(TypeError):
        state["algorithm_design"]  # type: ignore[index]

    updated = state.with_updates(current_stage=AgentStage.SOLVING)
    assert state.current_stage is AgentStage.INTAKE
    assert updated.current_stage is AgentStage.SOLVING


def test_agent_state_rejects_unknown_checkpoint_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        AgentState.from_dict({"problem_description": "配送问题", "math_modle": {}})


def test_agent_state_rejects_old_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        AgentState.from_dict({"problem_description": "配送问题", "schema_version": "1"})


def test_agent_state_rejects_legacy_checkpoint_fields():
    for field_name in (
        "agent_policy",
        "machine_state",
        "pending_step",
        "data_profile",
        "status",
        "active_stage",
        "core_config",
    ):
        with pytest.raises(ValueError, match=field_name):
            AgentState.from_dict({"problem_description": "配送问题", field_name: {}})


def test_agent_state_rejects_removed_formulation_fields():
    with pytest.raises(ValueError, match="math_model"):
        AgentState.from_dict(
            {
                "problem_description": "配送问题",
                "math_model": {"problem_family": "routing"},
            }
        )
    with pytest.raises(ValueError):
        AgentState.from_dict(
            {
                "problem_description": "配送问题",
                "current_stage": "formulation",
            }
        )


def test_full_state_machine_success_with_workspace_tools(tmp_path):
    services = fake_core_services(
        llm=FakeLLM(
            [
                {
                    "decision": "proceed",
                    "message": "信息完整",
                    "questions": [],
                    "problem_definition": definition(),
                }
            ]
        ),
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=fake_workspace_toolset(tmp_path),
    )
    agent = OptimizationAgent(services=services)
    agent.initialize(AgentInput(problem_description="配送问题"))

    items = collect_outputs(agent)
    decision = items[-1]
    assert isinstance(decision, Succeeded), decision
    paths = [
        draft.relative_path
        for report in items
        if isinstance(report, ArtifactProduced)
        for draft in report.artifacts
    ]
    assert paths == [
        "contract/problem_contract.json",
        "contract/input_schema.json",
        "contract/solution_schema.json",
        "contract/feasibility_checker.py",
        "design/algorithm_design.json",
        "code/solver.py",
        "result/input.json",
        "result/solution.json",
        "result/solver_result.json",
        "result/solver_execution_receipt.json",
        "result/feasibility_result.json",
        "result/feasibility_check_receipt.json",
        "review/feasibility_review.json",
        "result/explanation.json",
    ]
    assert decision.summary["objective"] == 12.5


def test_outputs_streams_reports_and_terminal_success(tmp_path):
    services = fake_core_services(
        llm=FakeLLM(
            [
                {
                    "decision": "proceed",
                    "message": "信息完整",
                    "questions": [],
                    "problem_definition": definition(),
                }
            ]
        ),
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=fake_workspace_toolset(tmp_path),
    )
    agent = OptimizationAgent(services=services)
    agent.initialize(AgentInput(problem_description="配送问题"))

    items = collect_outputs(agent)

    assert any(isinstance(item, StageStarted) for item in items)
    assert any(isinstance(item, AssistantMessage) for item in items)
    assert isinstance(items[-1], Succeeded)
    assert items[-1].summary["objective"] == 12.5


def test_outputs_waits_for_resume_after_clarification(tmp_path):
    services = fake_core_services(
        llm=FakeLLM(
            [
                {
                    "decision": "clarify",
                    "message": "请补充容量",
                    "questions": [{"id": "q1", "text": "容量是多少？", "answer_type": "text"}],
                    "problem_definition": definition(),
                },
                {
                    "decision": "proceed",
                    "message": "信息完整",
                    "questions": [],
                    "problem_definition": definition(),
                },
            ]
        ),
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=fake_workspace_toolset(tmp_path),
    )
    agent = OptimizationAgent(services=services)
    agent.initialize(AgentInput(problem_description="配送问题"))
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "stage_outputs" / "algorithm_design.json").write_text("{}")
    (tmp_path / "from_data_to_input.py").write_text("# stale mapping")
    (tmp_path / "input.json").write_text("{}")
    (tmp_path / "solver.py").write_text("# stale solver")
    (tmp_path / "problem.md").write_text("配送问题")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "instance.json").write_text("{}")

    async def scenario():
        iterator = agent.outputs(Control())
        while True:
            item = await anext(iterator)
            if isinstance(item, NeedsClarification):
                assert item.questions[0].text == "容量是多少？"
                break
        pending = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0.05)
        assert not pending.done()
        agent.resume("容量 200")
        resumed_item = await pending
        return resumed_item

    assert isinstance(run(scenario()), StageStarted)
    assert not (tmp_path / "stage_outputs" / "algorithm_design.json").exists()
    assert not (tmp_path / "from_data_to_input.py").exists()
    assert not (tmp_path / "input.json").exists()
    assert not (tmp_path / "solver.py").exists()
    assert (tmp_path / "problem.md").exists()
    assert (tmp_path / "data" / "instance.json").exists()


def test_clarification_resume_preserves_intake_context_and_clears_downstream_state():
    agent = OptimizationAgent(services=fake_core_services())
    problem_definition = {**definition(), "confirmed": False}
    agent.restore_state(
        AgentState(
            problem_description="配送问题",
            problem_definition=problem_definition,
            conversation=(
                {"role": "assistant", "content": "容量是多少？"},
            ),
            problem_contract={"old": "contract"},
            algorithm_design={"old": "design"},
            generated_code="# old solver",
            solver_result={"status": "failed"},
            feasibility_review={"decision": "reject"},
            explanation={"old": "explanation"},
            failure={"reason": "old failure"},
            current_stage=None,
            pending_clarification_questions=(
                {"id": "q1", "text": "容量是多少？", "answer_type": "text"},
            ),
            clarification_rounds=2,
            feasibility_rounds=3,
            feasibility_directives=({"attempt": 3},),
            feasibility_audits=({"attempt": 3},),
            instance_infeasibility_reviews=({"decision": "reject"},),
            pending_feasibility_handoff=3,
        ).to_dict()
    )

    agent.resume("容量 200")
    resumed = agent._state

    assert resumed is not None
    assert resumed.problem_description == "配送问题"
    assert resumed.problem_definition == problem_definition
    assert resumed.clarification_rounds == 2
    assert resumed.conversation == (
        {"role": "assistant", "content": "容量是多少？"},
        {"role": "user", "content": "容量 200"},
    )
    assert resumed.current_stage is AgentStage.INTAKE
    assert resumed.pending_clarification_questions == ()
    assert resumed.pending_workspace_reset is True
    assert resumed.problem_contract is None
    assert resumed.algorithm_design is None
    assert resumed.generated_code == ""
    assert resumed.solver_result is None
    assert resumed.feasibility_review is None
    assert resumed.explanation is None
    assert resumed.failure is None
    assert resumed.feasibility_rounds == 0
    assert resumed.feasibility_directives == ()
    assert resumed.feasibility_audits == ()
    assert resumed.instance_infeasibility_reviews == ()
    assert resumed.pending_feasibility_handoff == 0


def test_stage_result_mismatch_is_rejected():
    state = AgentState(problem_description="测试", current_stage=AgentStage.SOLVING)

    with pytest.raises(RuntimeError, match="mismatched result"):
        apply_stage_result(
            state,
            StageCompleted(
                stage=AgentStage.INTAKE,
                data={"reply": "", "ready": True, "problem_definition": definition()},
            ),
        )


def test_solving_result_passes_feasibility_review_before_explanation():
    state = AgentState(problem_description="测试", current_stage=AgentStage.SOLVING)
    result = StageCompleted(
        stage=AgentStage.SOLVING,
        data={"result": {"objective": 12.5, "status": "feasible"}},
    )

    awaiting_feasibility = apply_stage_result(state, result)

    assert awaiting_feasibility.current_stage is AgentStage.FEASIBILITY_REVIEW

    feasibility_result = StageCompleted(
        stage=AgentStage.FEASIBILITY_REVIEW,
        data={
            "feasibility_review": generated_feasibility_review(),
            "feasibility_check": generated_feasibility_result(),
        },
    )
    explaining = apply_stage_result(awaiting_feasibility, feasibility_result)

    assert explaining.current_stage is AgentStage.EXPLANATION


def test_solving_result_can_skip_feasibility_review_for_benchmark():
    state = AgentState(problem_description="测试", current_stage=AgentStage.SOLVING)
    result = StageCompleted(
        stage=AgentStage.SOLVING,
        data={"result": {"objective": 12.5, "status": "time_limit"}},
    )

    explaining = apply_stage_result(
        state,
        result,
        feasibility_review_enabled=False,
    )

    assert explaining.current_stage is AgentStage.EXPLANATION
    assert explaining.feasibility_review is None


def test_feasibility_rejection_routes_to_solving_with_repair_directive():
    state = AgentState(
        problem_description="测试",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        algorithm_design={"schema_version": "1.0"},
        solver_result={"status": "time_limit"},
    )
    rejected = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "solving",
        "summary": "当前候选违反容量约束",
        "evidence": [
            {
                "source": "feasibility_result.json",
                "finding": "vehicle_3 超出容量 14",
            }
        ],
        "required_changes": ["修复 solution.json 的路线载重聚合"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["vehicle_capacity"],
            "change_ids": ["capacity_aggregation"],
            "problem_requirement": "每条路线累计需求不得超过车辆容量",
            "observed_solution_behavior": "旧 solution 中存在超出车辆容量的路线",
            "responsibility_reason": "求解实现没有正确维护路线累计载重",
            "required_changes": ["修复容量聚合并在导出前完成业务约束自检"],
        },
    }

    routed = apply_stage_result(
        state,
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={
                "feasibility_review": rejected,
                "feasibility_check": None,
            },
        ),
    )

    assert routed.current_stage is AgentStage.SOLVING
    assert routed.failure is None
    assert routed.algorithm_design == {"schema_version": "1.0"}
    assert routed.solver_result is None
    assert routed.feasibility_review is None
    assert routed.feasibility_rounds == 1
    assert routed.pending_feasibility_handoff == 1
    assert routed.pending_workspace_reset is True
    directive = routed.feasibility_directives[-1]
    assert directive["responsibility"] == "solving"
    assert directive["handoff_dir"] == "feasibility_handoffs/solving/attempt1"
    assert directive["handoff"] == rejected["remediation_handoff"]
    assert routed.feasibility_audits[-1]["feasibility_review"] == rejected


def test_feasibility_rejection_routes_to_algorithm_design_and_clears_old_design():
    state = AgentState(
        problem_description="测试",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        algorithm_design={"schema_version": "1.0", "selection": "old"},
        solver_result={"status": "time_limit"},
    )
    rejected = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "algorithm_design",
        "summary": "现有设计没有覆盖联动容量约束",
        "evidence": [{"source": "algorithm_design.json", "finding": "缺少联合修复策略"}],
        "required_changes": ["重新设计覆盖联动约束的构造和修复方法"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["linked_capacity"],
            "change_ids": ["linked_capacity_repair"],
            "problem_requirement": "所有联动容量约束必须同时成立",
            "observed_solution_behavior": "旧 solution 的联合分配超过可用容量",
            "responsibility_reason": "现有设计没有定义保持联动容量的构造与修复机制",
            "required_changes": ["重新设计覆盖联动容量的构造和修复策略"],
        },
    }

    routed = apply_stage_result(
        state,
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={
                "feasibility_review": rejected,
                "feasibility_check": None,
            },
        ),
    )

    assert routed.current_stage is AgentStage.ALGORITHM_DESIGN
    assert routed.algorithm_design is None
    assert routed.solver_result is None
    assert routed.pending_feasibility_handoff == 1
    assert routed.feasibility_directives[-1]["responsibility"] == "algorithm_design"


def test_prepared_handoff_records_the_baselines_that_exist(tmp_path):
    stage_outputs = tmp_path / "stage_outputs"
    stage_outputs.mkdir()
    (stage_outputs / "algorithm_design.json").write_text('{"design": "old"}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')", encoding="utf-8")
    older = {"attempt": 1, "responsibility": "solving", "outcome": "reject"}
    pending = {
        "attempt": 2,
        "responsibility": "solving",
        "handoff_dir": "feasibility_handoffs/solving/attempt2",
    }

    directives = _prepare_pending_handoff(tmp_path, 2, (older, pending))

    # A no-candidate outcome has no old solution; list only baselines that actually exist.
    assert directives[-1]["baseline_files"] == [
        "previous_algorithm_design.json",
        "previous_solver.py",
    ]
    assert directives[0] == older
    assert "baseline_files" not in pending


def test_pending_handoff_without_workspace_fails_loudly():
    state = AgentState(
        problem_description="测试",
        current_stage=AgentStage.SOLVING,
        pending_feasibility_handoff=1,
        feasibility_directives=(
            {
                "attempt": 1,
                "responsibility": "solving",
                "handoff_dir": "feasibility_handoffs/solving/attempt1",
            },
        ),
    )
    services = fake_core_services(config=fake_core_config(workspace_root=""))

    async def drain():
        async for _ in run_stage(state, services, Control()):
            pass

    # Silent skipping would hide the old implementation and solution; fail explicitly.
    with pytest.raises(StageTransitionError):
        asyncio.run(drain())


def test_instance_feasibility_rejection_requires_three_consecutive_reviews():
    state = AgentState(
        problem_description="测试",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        solver_result={"status": "infeasible"},
    )
    rejected = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "instance",
        "summary": "完整模型证明实例不可行",
        "evidence": [{"source": "solver_result.json", "finding": "精确模型 infeasible"}],
        "required_changes": [],
        "confidence": "high",
        "infeasibility_proof": "求解器对完整模型给出不可行证明",
    }

    first = apply_stage_result(
        state,
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={
                "feasibility_review": rejected,
                "feasibility_check": None,
            },
        ),
    )

    assert first.current_stage is AgentStage.FEASIBILITY_REVIEW
    assert first.failure is None
    assert first.feasibility_review is None
    assert first.solver_result == {"status": "infeasible"}
    assert first.pending_workspace_reset is True
    assert first.instance_infeasibility_reviews == (
        {
            "decision": "reject",
            "responsibility": "instance",
            "confidence": "high",
            "summary": "完整模型证明实例不可行",
            "infeasibility_proof": "求解器对完整模型给出不可行证明",
        },
    )

    second = apply_stage_result(
        first.with_updates(pending_workspace_reset=False),
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={
                "feasibility_review": rejected,
                "feasibility_check": None,
            },
        ),
    )
    assert second.current_stage is AgentStage.FEASIBILITY_REVIEW
    assert second.feasibility_review is None

    confirmed = apply_stage_result(
        second.with_updates(pending_workspace_reset=False),
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={
                "feasibility_review": rejected,
                "feasibility_check": None,
            },
        ),
    )

    assert confirmed.current_stage is AgentStage.EXPLANATION
    assert confirmed.failure is None
    assert confirmed.feasibility_review == rejected
    assert len(confirmed.instance_infeasibility_reviews) == 3
    assert "feasibility_check" not in confirmed.solver_result
    assert "feasibility_result_file" not in confirmed.solver_result
    # Every instance decision enters cumulative audit so Review can see its previous conclusion.
    assert [item["kind"] for item in confirmed.feasibility_audits] == [
        "instance_infeasibility"
    ] * 3
    assert [item["consecutive_instance_verdict"] for item in confirmed.feasibility_audits] == [
        1,
        2,
        3,
    ]
    assert confirmed.feasibility_audits[-1]["infeasibility_proof"] == (
        "求解器对完整模型给出不可行证明"
    )


def test_low_confidence_no_longer_blocks_an_instance_confirmation():
    state = AgentState(
        problem_description="测试",
        current_stage=AgentStage.FEASIBILITY_REVIEW,
        solver_result={"status": "infeasible"},
    )
    rejected = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "instance",
        "summary": "完整模型证明实例不可行",
        "evidence": [{"source": "solver_result.json", "finding": "精确模型 infeasible"}],
        "required_changes": [],
        "confidence": "low",
        "infeasibility_proof": "求解器对完整模型给出不可行证明",
    }

    for _ in range(3):
        state = apply_stage_result(
            state.with_updates(pending_workspace_reset=False),
            StageCompleted(
                stage=AgentStage.FEASIBILITY_REVIEW,
                data={"feasibility_review": rejected, "feasibility_check": None},
            ),
        )

    # Requiring three medium/high decisions reset by one low decision caused an unbounded loop.
    assert state.current_stage is AgentStage.EXPLANATION
    assert len(state.instance_infeasibility_reviews) == 3


def test_instance_verdicts_survive_a_later_repair_rewind():
    instance_review = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "instance",
        "summary": "完整模型证明实例不可行",
        "evidence": [{"source": "solver_result.json", "finding": "精确模型 infeasible"}],
        "required_changes": [],
        "confidence": "high",
        "infeasibility_proof": "求解器对完整模型给出不可行证明",
    }
    state = apply_stage_result(
        AgentState(
            problem_description="测试",
            current_stage=AgentStage.FEASIBILITY_REVIEW,
            solver_result={"status": "infeasible"},
        ),
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={"feasibility_review": instance_review, "feasibility_check": None},
        ),
    )

    solving_review = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "solving",
        "summary": "候选未覆盖容量约束",
        "evidence": [{"source": "feasibility_result.json", "finding": "容量超限"}],
        "required_changes": ["修复容量聚合"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["vehicle_capacity"],
            "change_ids": ["capacity_aggregation"],
            "problem_requirement": "每条路线累计需求不得超过车辆容量",
            "observed_solution_behavior": "旧 solution 存在超出车辆容量的路线",
            "responsibility_reason": "当前实现没有正确维护路线累计载重",
            "required_changes": ["修复路线容量聚合并在导出前进行业务约束自检"],
        },
    }
    routed = apply_stage_result(
        state.with_updates(
            pending_workspace_reset=False,
            solver_result={"status": "time_limit", "solution_file": "solution.json"},
        ),
        StageCompleted(
            stage=AgentStage.FEASIBILITY_REVIEW,
            data={"feasibility_review": solving_review, "feasibility_check": None},
        ),
    )

    # Reset the consecutive segment while retaining the instance decision in cumulative audit.
    assert routed.instance_infeasibility_reviews == ()
    assert [item["kind"] for item in routed.feasibility_audits] == [
        "instance_infeasibility",
        "repair_attempt",
    ]
