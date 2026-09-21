import asyncio
import json
from pathlib import Path

import pytest

from decisionbrain.core.contracts import AgentControl, CoreServices
from decisionbrain.core.message_builder import MessageBuilder
from decisionbrain.core.models import (
    AgentStage,
    AgentState,
    ArtifactProduced,
    ChatRequest,
    ChatResponse,
    CoreConfig,
    StageCompleted,
    StageFinished,
    StageNeedsClarification,
    StageStarted,
)
from decisionbrain.core.prompts import PromptBundle
from decisionbrain.core.stage_agent import (
    OutputValidationIssue,
    StageAgent,
    StageExecutionStopped,
    StageForcedCompletion,
    StageOutputValidationError,
    Tool,
    unknown_tool_message,
)
from decisionbrain.core.stage_control_tools import (
    REPORT_PROGRESS_TOOL_NAME,
    STAGE_EVENT_PROGRESS_REPORTED,
)
from decisionbrain.core.stage_agents import (
    AlgorithmDesignAgent,
    ExplanationAgent,
    FeasibilityReviewAgent,
    ProblemContractAgent,
    SolvingAgent,
)
from helpers import fake_core_config, fake_workspace_toolset


CONTROL_TOOL_NAMES = [REPORT_PROGRESS_TOOL_NAME]


class FakeControl(AgentControl):
    def __init__(self, cancelled: bool = False):
        self.cancelled = cancelled

    async def is_cancelled(self) -> bool:
        return self.cancelled


class FakeLLM:
    def __init__(self, responses: list[ChatResponse]):
        self.responses = responses
        self.calls: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError(
                "FakeLLM response queue exhausted; the stage requested an unexpected LLM turn"
            )
        return self.responses.pop(0)


def make_services(
    llm=None,
    *,
    config: CoreConfig | None = None,
    workspace_toolset=None,
    stage_event_handler=None,
    debug_event_handler=None,
    agent_turn_handler=None,
    timing_handler=None,
) -> CoreServices:
    return CoreServices(
        llm=llm or FakeLLM([]),
        config=config or fake_core_config(),
        prompts=PromptBundle(
            stage_common_system="",
            algorithm_library_system="",
            intake_system="",
            intake_contract="",
            problem_contract_system="",
            problem_contract_contract="",
            algorithm_design_system="",
            algorithm_design_contract="",
            solving_system="",
            solving_contract="",
            feasibility_review_system="",
            feasibility_review_contract="",
            explain_system="",
            explain_contract="",
        ),
        workspace_toolset=workspace_toolset,
        stage_event_handler=stage_event_handler,
        debug_event_handler=debug_event_handler,
        agent_turn_handler=agent_turn_handler,
        timing_handler=timing_handler,
    )


def run(coro):
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=10))
    except TimeoutError as exc:
        raise AssertionError("stage-agent scenario did not terminate within 10 seconds") from exc


async def collect(agent: StageAgent, state=None, control=None):
    items = []
    async for output in agent.run(state or object(), control or FakeControl()):
        items.append(output)
    return items


def write_stage_output(tmp_path, stage: AgentStage, payload: dict):
    path = {
        AgentStage.INTAKE: "stage_outputs/intake.json",
        AgentStage.PROBLEM_CONTRACT: "stage_outputs/problem_contract.json",
        AgentStage.ALGORITHM_DESIGN: "stage_outputs/algorithm_design.json",
        AgentStage.SOLVING: "stage_outputs/solving.json",
        AgentStage.FEASIBILITY_REVIEW: "stage_outputs/feasibility_review.json",
        AgentStage.EXPLANATION: "stage_outputs/explanation.json",
    }[stage]
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def generated_algorithm_design() -> dict:
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
            "reason": "small deterministic fixture",
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
                    "reason": "small deterministic fixture",
                    "capability_mapping": [],
                    "integration": None,
                    "formulation": None,
                }
            ],
        },
        "strategy": {
            "representation": "ordered routes",
            "feasibility": "check every route",
            "objective": "minimize distance",
            "construction": "greedy insertion",
            "improvement": "none for fixture",
            "stopping": "after construction",
            "output": "routes",
        },
        "fallback": {
            "trigger": "primary failure",
            "description": "return simple feasible route",
            "components": [],
        },
        "risks": [],
    }


def generated_solver_result(objective: float = 0) -> dict:
    return {
        "status": "time_limit",
        "objective": objective,
        "mip_gap": None,
        "optimality": "heuristic_or_incumbent",
        "input_file": "input.json",
        "solution_file": "solution.json",
        "code_file": "solver.py",
        "feasibility_result_file": "feasibility_result.json",
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
        "solution_summary": {},
        "constraint_check": [],
        "validation": {
            "schema_check": {"ok": True, "issues": []},
            "feasibility_checker": {
                "schema_valid": True,
                "feasible": True,
                "objective_value": objective,
                "violations": [],
                "warnings": [],
            },
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


def publish_solver_receipt(toolset) -> None:
    schema = toolset.root / "solution_schema.json"
    if not schema.exists():
        schema.write_text('{"type": "object"}', encoding="utf-8")
    toolset._publish_solver_receipt(
        argv=(),
        exit_code=0,
        elapsed_seconds=0.01,
        timed_out=False,
    )


def generated_infeasible_solver_result() -> dict:
    result = generated_solver_result()
    result.update(
        {
            "status": "infeasible",
            "objective": None,
            "mip_gap": None,
            "optimality": None,
            "solution_file": None,
            "feasibility_result_file": None,
            "solution_summary": {},
            "infeasibility_proof": {
                "proof_type": "exact_solver_status",
                "solver": "CP-SAT",
                "raw_status": "INFEASIBLE",
                "model_scope": "complete",
                "hard_constraints_covered": ["assignment", "capacity", "time_windows"],
                "evidence": ["solver log ended with status INFEASIBLE"],
            },
        }
    )
    result["validation"]["feasibility_checker"] = None
    return result


def generated_no_candidate_solver_result(status: str = "time_limit") -> dict:
    result = generated_solver_result()
    result.update(
        {
            "status": status,
            "objective": None,
            "mip_gap": None,
            "optimality": None,
            "solution_file": None,
            "feasibility_result_file": None,
            "solution_summary": {},
            "diagnosis": "正常完成搜索，但没有找到候选解",
            "infeasibility_proof": None,
        }
    )
    result["validation"]["feasibility_checker"] = None
    return result


def make_json_agent(
    llm: FakeLLM,
    tmp_path,
    *,
    tools=(),
    stage_event_handler=None,
    timing_handler=None,
) -> StageAgent:
    agent = StageAgent(
        stage=AgentStage.ALGORITHM_DESIGN,
        services=make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            stage_event_handler=stage_event_handler,
            timing_handler=timing_handler,
        ),
        tools=tools,
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data
    return agent


def test_algorithm_library_prompt_is_injected_only_into_design_and_solving():
    prompts = PromptBundle(
        stage_common_system="COMMON __CURRENT_STAGE__",
        algorithm_library_system="ALGORITHM_LIBRARY_ONLY",
        intake_system="",
        intake_contract="",
        problem_contract_system="",
        problem_contract_contract="",
        algorithm_design_system="",
        algorithm_design_contract="",
        solving_system="",
        solving_contract="",
        feasibility_review_system="",
        feasibility_review_contract="",
        explain_system="",
        explain_contract="",
    )
    services = CoreServices(
        llm=FakeLLM([]),
        config=fake_core_config(),
        prompts=prompts,
    )

    for stage in AgentStage:
        agent = StageAgent(stage=stage, services=services)
        messages = MessageBuilder.from_prompts(system="STAGE_SYSTEM")

        agent._inject_common_system_prompt(messages)

        system = messages.messages[0]["content"]
        assert f"COMMON {stage.value}" in system
        if stage in {AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}:
            assert "ALGORITHM_LIBRARY_ONLY" in system
        else:
            assert "ALGORITHM_LIBRARY_ONLY" not in system


def write_problem_contract_files(tmp_path):
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"orders": {"type": "array"}},
        "required": ["orders"],
        "additionalProperties": True,
    }
    solution_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"routes": {"type": "array"}},
        "required": ["routes"],
        "additionalProperties": True,
    }
    checker = (
        "from pathlib import Path\n\n"
        "def check(input_data: dict, solution: dict, data_dir: str | Path = 'data') -> dict:\n"
        "    return {\n"
        "        'schema_valid': True,\n"
        "        'feasible': True,\n"
        "        'objective_value': 0,\n"
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
    )
    (tmp_path / "input_schema.json").write_text(
        json.dumps(input_schema),
        encoding="utf-8",
    )
    (tmp_path / "solution_schema.json").write_text(
        json.dumps(solution_schema),
        encoding="utf-8",
    )
    (tmp_path / "feasibility_checker.py").write_text(checker, encoding="utf-8")
    write_stage_output(
        tmp_path,
        AgentStage.PROBLEM_CONTRACT,
        {
            "problem_family": "routing",
            "files": {
                "input_schema": "input_schema.json",
                "solution_schema": "solution_schema.json",
                "feasibility_checker": "feasibility_checker.py",
            },
        },
    )


def test_zero_tool_stage_completes_from_workspace_output_file(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"result": "ok"})
    llm = FakeLLM([ChatResponse(content="done")])
    agent = make_json_agent(llm, tmp_path)

    items = run(collect(agent))

    assert [type(item) for item in items] == [StageStarted, StageFinished, StageCompleted]
    assert items[-1].data == {"result": "ok"}
    assert [tool["function"]["name"] for tool in llm.calls[0].tools] == CONTROL_TOOL_NAMES


def test_tool_call_loop_passes_tool_results_back_to_llm(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"done": True})
    responses = [
        ChatResponse(
            content=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                }
            ],
        ),
        ChatResponse(content="done"),
    ]
    llm = FakeLLM(responses)

    async def echo(text: str) -> str:
        return f"echo: {text}"

    agent = make_json_agent(
        llm,
        tmp_path,
        tools=(
            Tool(
                name="echo",
                description="Echo text",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                handler=echo,
            ),
        ),
    )

    items = run(collect(agent))

    assert items[-1].data == {"done": True}
    assert [tool["function"]["name"] for tool in llm.calls[0].tools] == [
        "echo",
        *CONTROL_TOOL_NAMES,
    ]
    tool_messages = [message for message in llm.calls[1].messages if message["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "tool_call_id": "call-1", "content": "echo: hello"}]


def test_stage_agent_records_llm_and_tool_timing(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"done": True})
    events = []

    async def record_timing(event):
        events.append(event)

    async def echo(text: str) -> str:
        return text

    llm = FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":"ok"}'},
                    }
                ],
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = make_json_agent(
        llm,
        tmp_path,
        tools=(
            Tool(
                name="echo",
                description="Echo text",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                handler=echo,
            ),
        ),
        timing_handler=record_timing,
    )

    run(collect(agent))

    assert [(event["kind"], event.get("tool_name")) for event in events] == [
        ("llm_call", None),
        ("tool_call", "echo"),
        ("llm_call", None),
    ]
    assert all(event["stage"] == "algorithm_design" for event in events)
    assert all(event["duration_ms"] >= 0 for event in events)


def test_stage_agent_ignores_timing_handler_failure(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"done": True})

    async def broken_timing_handler(event):
        raise RuntimeError("timing unavailable")

    agent = make_json_agent(
        FakeLLM([ChatResponse(content="done")]),
        tmp_path,
        timing_handler=broken_timing_handler,
    )

    items = run(collect(agent))

    assert items[-1].data == {"done": True}


def test_tool_argument_validation_reports_missing_fields_and_full_contract(tmp_path):
    called = False
    events = []

    async def record_event(event):
        events.append(event)

    async def replace_in_file(path: str, old_string: str, new_string: str) -> str:
        nonlocal called
        called = True
        return "replaced"

    tool = Tool(
        name="replace_in_file",
        description="Replace one exact string in a workspace file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "old_string": {"type": "string", "description": "Exact text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        handler=replace_in_file,
    )
    agent = make_json_agent(
        FakeLLM([]),
        tmp_path,
        tools=(tool,),
        stage_event_handler=record_event,
    )

    result = run(
        agent._execute_tool(
            {
                "id": "call-invalid",
                "function": {
                    "name": "replace_in_file",
                    "arguments": json.dumps({"old_string": "a", "new_string": "b"}),
                },
            }
        )
    )

    assert called is False
    assert "缺少必填参数: path" in result
    assert "工具用途: Replace one exact string" in result
    assert "path: 必填; 类型=string; 含义=" in result
    assert "old_string: 必填; 类型=string" in result
    assert "new_string: 必填; 类型=string" in result
    assert "含义=Workspace-relative file path." in result
    assert events[0]["kind"] == "agent_error"
    assert events[0]["payload"]["flow_section"] == "algorithm_design"
    assert events[0]["payload"]["error_kind"] == "tool_argument_validation"


def test_tool_argument_validation_reports_unknown_fields_and_allowed_parameters(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    read_tool = next(tool for tool in toolset.tools() if tool.name == "read_file")
    agent = make_json_agent(FakeLLM([]), tmp_path, tools=(read_tool,))

    result = run(
        agent._execute_tool(
            {
                "id": "call-invalid",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "input.json", "offset": 10, "limit": 20}),
                },
            }
        )
    )

    assert "包含未允许的参数: limit, offset" in result
    assert "path: 必填; 类型=string; 含义=" in result
    assert "max_chars: 可选; 类型=number; 含义=" in result
    assert "start_line: 可选; 类型=integer; 含义=" in result
    assert "line_count: 可选; 类型=integer; 含义=" in result
    assert "默认值=" not in result
    assert "Maximum characters to return" in result


def test_tool_argument_validation_reports_type_errors_without_calling_handler(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    read_tool = next(tool for tool in toolset.tools() if tool.name == "read_file")
    agent = make_json_agent(FakeLLM([]), tmp_path, tools=(read_tool,))

    result = run(
        agent._execute_tool(
            {
                "id": "call-invalid",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "input.json", "max_chars": "many"}),
                },
            }
        )
    )

    assert "参数 max_chars 不符合契约" in result
    assert "'many' is not of type 'number'" in result
    assert "允许的参数:" in result


def test_output_file_validation_retries_with_json_feedback(tmp_path):
    llm = FakeLLM(
        [
            ChatResponse(content="done"),
            ChatResponse(
                tool_calls=[
                    {
                        "id": "write-output",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "path": "stage_outputs/explanation.json",
                                    "content": json.dumps({"ok": True}),
                                }
                            ),
                        },
                    }
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    toolset = fake_workspace_toolset(tmp_path)
    agent = StageAgent(
        stage=AgentStage.EXPLANATION,
        services=make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        ),
        tools=toolset.tools(stage=AgentStage.EXPLANATION),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    items = run(collect(agent))

    assert len(llm.calls) == 3
    assert llm.calls[1].messages[-1]["role"] == "user"
    assert "stage_outputs/explanation.json" in llm.calls[1].messages[-1]["content"]
    assert items[-1].data == {"ok": True}


def test_build_result_hook_can_return_stage_needs_clarification(tmp_path):
    write_stage_output(tmp_path, AgentStage.SOLVING, {"question": "容量是多少？"})
    llm = FakeLLM([ChatResponse(content="done")])
    agent = StageAgent(
        stage=AgentStage.SOLVING,
        services=make_services(llm, config=fake_core_config(workspace_root=tmp_path)),
        tools=(),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data
    agent.build_result = lambda output, state: StageNeedsClarification(
        stage=AgentStage.SOLVING,
        message=output["question"],
    )

    items = run(collect(agent))

    assert isinstance(items[-1], StageNeedsClarification)
    assert items[-1].message == "容量是多少？"


def test_tool_call_loop_has_no_round_limit(tmp_path):
    write_stage_output(tmp_path, AgentStage.SOLVING, {"done": True})
    llm = FakeLLM(
        [
            ChatResponse(
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ]
            ),
            ChatResponse(
                tool_calls=[
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ]
            ),
            ChatResponse(
                tool_calls=[
                    {
                        "id": "call-3",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ]
            ),
            ChatResponse(content='{"done": true}'),
        ]
    )

    async def noop() -> str:
        return "ok"

    agent = StageAgent(
        stage=AgentStage.SOLVING,
        services=make_services(llm, config=fake_core_config(workspace_root=tmp_path)),
        tools=(Tool(name="noop", description="", parameters={}, handler=noop),),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    items = run(collect(agent))

    assert len(llm.calls) == 4
    assert isinstance(items[-1], StageCompleted)
    assert items[-1].data == {"done": True}


def test_cancellation_before_llm_call_stops_without_events():
    llm = FakeLLM([ChatResponse(content='{"unused": true}')])
    agent = make_json_agent(llm, Path("."))

    assert run(collect(agent, control=FakeControl(cancelled=True))) == []
    assert llm.calls == []


def test_cancellation_during_llm_call_records_interrupted_timing(tmp_path):
    events = []

    class CancelledLLM:
        async def complete(self, request):
            raise asyncio.CancelledError

    async def record_timing(event):
        events.append(event)

    agent = make_json_agent(CancelledLLM(), tmp_path, timing_handler=record_timing)

    with pytest.raises(asyncio.CancelledError):
        run(collect(agent))

    assert events[0]["kind"] == "llm_call"
    assert events[0]["status"] == "interrupted"


def test_workspace_tools_are_registered_on_tool_aware_stage_agents(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    expected_names = [
        "shell",
        "read_file",
        "search_file",
        "write_file",
        "list_files",
        "replace_in_file",
        "delete_file",
        *CONTROL_TOOL_NAMES,
    ]
    state = AgentState(
        problem_description="problem",
        problem_definition={"objective": "min cost"},
    )

    write_problem_contract_files(tmp_path)
    contract_llm = FakeLLM([ChatResponse(content="done")])
    contract = ProblemContractAgent(
        make_services(
            contract_llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        )
    )
    run(collect(contract, state=state))

    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, generated_algorithm_design())
    design_llm = FakeLLM([ChatResponse(content="done")])
    design = AlgorithmDesignAgent(
        make_services(
            design_llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        )
    )
    run(collect(design, state=state))

    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_solver_result()), encoding="utf-8"
    )
    publish_solver_receipt(toolset)
    write_stage_output(
        tmp_path,
        AgentStage.SOLVING,
        {
            "decision": "solved",
            "message": "done",
            "result": {"solver_result_file": "solver_result.json"},
        },
    )
    solving_llm = FakeLLM([ChatResponse(content="done")])
    solving = SolvingAgent(
        make_services(
            solving_llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        )
    )
    write_stage_output(tmp_path, AgentStage.EXPLANATION, {"summary": "ok"})
    explanation_llm = FakeLLM([ChatResponse(content="done")])
    explanation = ExplanationAgent(
        make_services(
            explanation_llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        )
    )
    run(collect(explanation, state=state.with_updates(solver_result={"status": "optimal"})))

    assert [tool["function"]["name"] for tool in contract_llm.calls[0].tools] == expected_names
    assert [tool["function"]["name"] for tool in design_llm.calls[0].tools] == expected_names
    assert [tool.name for tool in solving.tools] == [
        *expected_names[:7],
        "run_solver",
        "submit_solving_outcome",
    ]
    assert [tool["function"]["name"] for tool in explanation_llm.calls[0].tools] == expected_names


def test_problem_contract_agent_validates_required_files_and_artifacts(tmp_path):
    write_problem_contract_files(tmp_path)
    llm = FakeLLM([ChatResponse(content="done")])
    services = make_services(
        llm,
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=fake_workspace_toolset(tmp_path),
    )
    agent = ProblemContractAgent(services)
    state = AgentState(
        problem_description="problem",
        problem_definition={"objective": "min cost"},
    )

    items = run(collect(agent, state=state))

    artifact = next(item for item in items if isinstance(item, ArtifactProduced))
    assert [draft.relative_path for draft in artifact.artifacts] == [
        "contract/problem_contract.json",
        "contract/input_schema.json",
        "contract/solution_schema.json",
        "contract/feasibility_checker.py",
    ]
    completed = items[-1]
    assert isinstance(completed, StageCompleted)
    assert completed.data["problem_contract"]["files"]["input_schema"] == "input_schema.json"


def test_problem_contract_agent_marks_downstream_checker_files_as_out_of_stage_scope():
    agent = ProblemContractAgent(make_services())

    prompt = agent.build_messages(AgentState(problem_description="problem")).messages[-1]["content"]

    assert "只定义 checker 的接口并在 problem_contract.json 中记录未来文件名" in prompt
    assert "不得在本阶段创建、覆盖、要求其存在，或以它们执行 checker CLI" in prompt
    assert "本阶段自检只验证以下产物：input_schema.json、solution_schema.json、feasibility_checker.py" in prompt
    assert "使用 search_file" in prompt


def test_problem_contract_agent_retries_when_checker_missing(tmp_path):
    (tmp_path / "input_schema.json").write_text('{"type": "object"}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text('{"type": "object"}', encoding="utf-8")
    write_stage_output(tmp_path, AgentStage.PROBLEM_CONTRACT, {"problem_family": "routing"})
    llm = FakeLLM(
        [
            ChatResponse(content="done"),
            ChatResponse(
                tool_calls=[
                    {
                        "id": "write-checker",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "path": "feasibility_checker.py",
                                    "content": (
                                        "def check(input_data, solution, data_dir='data'):\n"
                                        "    return {'schema_valid': True, 'feasible': True, "
                                        "'objective_value': 0, 'violations': [], 'warnings': []}\n"
                                    ),
                                }
                            ),
                        },
                    }
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    toolset = fake_workspace_toolset(tmp_path)
    agent = ProblemContractAgent(
        make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        )
    )

    items = run(collect(agent, state=AgentState(problem_description="problem")))

    assert len(llm.calls) == 3
    assert "feasibility_checker.py" in llm.calls[1].messages[-1]["content"]
    assert isinstance(items[-1], StageCompleted)


def test_solving_agent_reads_workspace_artifacts_and_solution(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": [[0, 1, 0]]}', encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_solver_result(12.5)),
        encoding="utf-8",
    )
    write_stage_output(
        tmp_path,
        AgentStage.SOLVING,
        {
            "decision": "solved",
            "message": "完成",
            "result": {"solver_result_file": "solver_result.json"},
        },
    )
    llm = FakeLLM([ChatResponse(content="done")])
    toolset = fake_workspace_toolset(tmp_path)
    publish_solver_receipt(toolset)
    services = make_services(
        llm,
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=toolset,
    )
    agent = SolvingAgent(services)
    state = AgentState(
        problem_description="problem",
        problem_definition={"objective": "min cost"},
        problem_contract={"problem_family": "routing"},
        algorithm_design={"problem_family": "routing"},
    )

    items = run(collect(agent, state=state))

    artifact = next(item for item in items if isinstance(item, ArtifactProduced))
    assert [draft.relative_path for draft in artifact.artifacts] == [
        "code/solver.py",
        "result/input.json",
        "result/solution.json",
        "result/solver_result.json",
        "result/solver_execution_receipt.json",
    ]
    completed = items[-1]
    assert isinstance(completed, StageCompleted)
    assert completed.data["result"]["input"] == {"orders": []}
    assert completed.data["result"]["solution"] == {"routes": [[0, 1, 0]]}
    assert completed.data["generated_code"] == "print('ok')\n"


def test_solving_submits_proven_infeasible_without_solution_to_reviewer(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('INFEASIBLE')\n", encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_infeasible_solver_result()),
        encoding="utf-8",
    )
    payload = {
        "decision": "solved",
        "message": "完整模型证明不可行",
        "result": {"solver_result_file": "solver_result.json"},
    }
    toolset = fake_workspace_toolset(tmp_path)
    publish_solver_receipt(toolset)
    services = make_services(
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=toolset,
    )
    agent = SolvingAgent(services, feasibility_review_enabled=True)

    parsed = agent.parse_output_file(payload, AgentState(problem_description="problem"))
    result = agent.build_result(parsed, AgentState(problem_description="problem"))

    assert isinstance(result, StageCompleted)
    assert result.data["result"]["status"] == "infeasible"
    assert result.data["result"]["solution_file"] is None
    assert "solution.json" not in result.data["result"].get("missing_files", [])

    self_check_agent = SolvingAgent(services, feasibility_review_enabled=False)
    with pytest.raises(StageOutputValidationError) as raised:
        self_check_agent.parse_output_file(payload, AgentState(problem_description="problem"))
    assert any(issue.path == "$.files.solution.json" for issue in raised.value.issues)


def test_solving_submits_no_candidate_terminal_status_to_reviewer(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('TIME_LIMIT')\n", encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_no_candidate_solver_result()),
        encoding="utf-8",
    )
    payload = {
        "decision": "solved",
        "message": "正常结束但没有候选解",
        "result": {"solver_result_file": "solver_result.json"},
    }
    toolset = fake_workspace_toolset(tmp_path)
    publish_solver_receipt(toolset)
    services = make_services(
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=toolset,
    )

    agent = SolvingAgent(services, feasibility_review_enabled=True)
    parsed = agent.parse_output_file(payload, AgentState(problem_description="problem"))
    result = agent.build_result(parsed, AgentState(problem_description="problem"))

    assert isinstance(result, StageCompleted)
    assert result.data["result"]["status"] == "time_limit"
    assert result.data["result"]["solution_file"] is None
    assert "solution.json" not in result.data["result"].get("missing_files", [])

    self_check_agent = SolvingAgent(services, feasibility_review_enabled=False)
    with pytest.raises(StageOutputValidationError) as raised:
        self_check_agent.parse_output_file(payload, AgentState(problem_description="problem"))
    assert any(issue.path == "$.files.solution.json" for issue in raised.value.issues)


def test_solving_forced_runtime_timeout_completes_without_another_llm_turn(tmp_path):
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')\n", encoding="utf-8")
    runtime_outcome = {
        "schema_version": "1.0",
        "outcome_source": "runtime_external_timeout",
        "status": "runtime_timeout",
        "timeout_source": "runtime_watchdog",
        "timeout_seconds": 1,
        "elapsed_seconds": 1.01,
        "exit_code": -15,
        "termination_signal": "SIGTERM",
        "solver_file": "solver.py",
        "solver_execution_observed": True,
        "solution_detected": False,
        "solution_schema_valid": None,
        "solution_file": None,
        "stdout_tail": "",
    }
    (tmp_path / "runtime_solver_outcome.json").write_text(
        json.dumps(runtime_outcome), encoding="utf-8"
    )
    write_stage_output(
        tmp_path,
        AgentStage.SOLVING,
        {
            "decision": "solved",
            "message": "runtime timeout",
            "result": {"runtime_solver_outcome_file": "runtime_solver_outcome.json"},
        },
    )

    async def force_timeout() -> str:
        raise StageForcedCompletion("primary solver timed out")

    llm = FakeLLM(
        [
            ChatResponse(
                tool_calls=[
                    {
                        "id": "shell-1",
                        "type": "function",
                        "function": {"name": "force_timeout", "arguments": "{}"},
                    }
                ]
            )
        ]
    )
    services = make_services(llm, config=fake_core_config(workspace_root=tmp_path))
    agent = SolvingAgent(services)
    agent.tools = (*agent.tools, Tool("force_timeout", "force", {"type": "object"}, force_timeout))
    agent.__post_init__()

    items = run(collect(agent, state=AgentState(problem_description="problem")))

    assert len(llm.calls) == 1
    completed = next(item for item in items if isinstance(item, StageCompleted))
    assert completed.data["result"]["outcome_source"] == "runtime_external_timeout"
    assert completed.data["result"]["status"] == "runtime_timeout"


def test_solving_runtime_timeout_terminal_artifacts_validate_without_repair(tmp_path):
    # The timeout terminal route is Runtime-owned end to end: it publishes the
    # runtime outcome, the execution receipt and the stage output together. If
    # that set ever stops satisfying the Solving contract, the forced completion
    # degrades into an error the model has no turn left to answer.
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')" + chr(92) + "n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    toolset._publish_solver_receipt(
        argv=(),
        exit_code=-15,
        elapsed_seconds=30.1,
        timed_out=True,
    )
    toolset._publish_runtime_solver_timeout(
        timeout_source="runtime_watchdog",
        timeout_seconds=30,
        elapsed_seconds=30.1,
        exit_code=-15,
        termination_signal="SIGTERM",
        stdout="",
        solution_before=None,
    )
    services = make_services(
        config=fake_core_config(workspace_root=tmp_path),
        workspace_toolset=toolset,
    )
    agent = SolvingAgent(services, feasibility_review_enabled=True)
    state = AgentState(problem_description="problem")

    payload = json.loads((tmp_path / "stage_outputs/solving.json").read_text(encoding="utf-8"))
    parsed = agent.parse_output_file(payload, state)
    result = agent.build_result(parsed, state)

    assert isinstance(result, StageCompleted)
    assert result.data["result"]["status"] == "runtime_timeout"


def test_solving_without_review_does_not_require_a_checker_result(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": [[0, 1, 0]]}', encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_solver_result(12.5)), encoding="utf-8"
    )
    (tmp_path / "feasibility_result.json").write_text(
        json.dumps(
            {
                "schema_valid": True,
                "feasible": True,
                "objective_value": 12.5,
                "violations": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "decision": "solved",
        "message": "完成",
        "result": {"solver_result_file": "solver_result.json"},
    }
    toolset = fake_workspace_toolset(tmp_path)
    publish_solver_receipt(toolset)
    agent = SolvingAgent(
        make_services(config=fake_core_config(workspace_root=tmp_path), workspace_toolset=toolset),
        feasibility_review_enabled=False,
    )
    state = AgentState(problem_description="problem")
    parsed = agent.parse_output_file(payload, state)
    assert parsed["decision"] == "solved"

    result = generated_solver_result(12.5)
    result.pop("feasibility_result_file")
    (tmp_path / "solver_result.json").write_text(json.dumps(result), encoding="utf-8")
    assert agent.parse_output_file(payload, state)["decision"] == "solved"

def test_solving_without_review_does_not_receive_direct_checker_access(tmp_path):
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feasibility_checker.py").write_text(
        "import argparse\n"
        "import json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input')\n"
        "parser.add_argument('--solution')\n"
        "parser.add_argument('--data-dir')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "with open(args.output, 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'schema_valid': True, 'feasible': True, 'objective_value': 0, "
        "'violations': [], 'warnings': []}, handle)\n",
        encoding="utf-8",
    )
    toolset = fake_workspace_toolset(tmp_path)
    agent = SolvingAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=toolset,
        ),
        feasibility_review_enabled=False,
    )
    result = run(
        next(tool for tool in agent.tools if tool.name == "shell").handler(
            command=(
                "python feasibility_checker.py --input input.json --solution solution.json "
                "--data-dir data --output feasibility_result.json"
            )
        )
    )

    assert "Exit code: 0" not in result
    solving_tools = {tool.name for tool in agent.tools}
    assert "shell" in solving_tools
    assert "run_feasibility_checker" not in solving_tools
    assert "not allowed to read feasibility_checker.py" in run(
        next(tool for tool in agent.tools if tool.name == "read_file").handler(
            path="feasibility_checker.py"
        )
    )


def test_solving_prompt_uses_workspace_relative_paths(tmp_path):
    agent = SolvingAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )

    prompt = agent.build_messages(AgentState(problem_description="problem")).messages[-1]["content"]

    assert "shell 默认从 workspace 根目录执行" in prompt
    assert "使用 shell 的相对 workdir" in prompt
    assert "不要 cd 到猜测的绝对路径" in prompt
    assert str(tmp_path) not in prompt


def test_solving_agent_can_request_clarification(tmp_path):
    write_stage_output(
        tmp_path,
        AgentStage.SOLVING,
        {
            "decision": "needs_clarification",
            "message": "需要确认容量",
            "questions": [{"id": "capacity", "text": "容量是多少？"}],
        },
    )
    llm = FakeLLM([ChatResponse(content="done")])
    agent = SolvingAgent(
        make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    state = AgentState(
        problem_description="problem",
        problem_definition={"objective": "min cost"},
        problem_contract={"problem_family": "routing"},
        algorithm_design={"problem_family": "routing"},
    )

    items = run(collect(agent, state=state))

    assert isinstance(items[-1], StageNeedsClarification)
    assert items[-1].questions[0].id == "capacity"


def test_algorithm_design_rejects_formulation_for_generated_selection(tmp_path):
    payload = generated_algorithm_design()
    payload["formulation"] = {
        "paradigm": "MILP",
        "scope": "full_problem",
        "summary": "not allowed for generated selection",
        "variables": ["x"],
        "objective": "min x",
        "constraints": ["x >= 0"],
        "big_m_notes": "",
    }
    agent = AlgorithmDesignAgent(make_services(config=fake_core_config(workspace_root=tmp_path)))

    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, AgentState(problem_description="problem"))

    assert "Extra inputs are not permitted" in str(raised.value.issues[0].message)


def test_solving_rejects_detailed_result_in_stage_decision(tmp_path):
    agent = SolvingAgent(make_services(config=fake_core_config(workspace_root=tmp_path)))
    payload = {
        "decision": "solved",
        "message": "done",
        "result": {"solver_result_file": "solver_result.json", "status": "optimal"},
    }

    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, AgentState(problem_description="problem"))

    assert any(issue.path == "$.result.status" for issue in raised.value.issues)


def test_solving_rejects_incomplete_solved_workspace(tmp_path):
    agent = SolvingAgent(make_services(config=fake_core_config(workspace_root=tmp_path)))
    payload = {
        "decision": "solved",
        "message": "done",
        "result": {"solver_result_file": "solver_result.json"},
    }

    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, AgentState(problem_description="problem"))

    paths = {issue.path for issue in raised.value.issues}
    assert "$.result.solver_result_file" in paths
    assert "$.files.solver.py" in paths


def test_feasibility_review_routes_after_fifth_identical_checker_failure(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text("# checker\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )

    async def fail_checker(**kwargs) -> str:
        return "Exit code: 2\nWall time: 0.1 seconds\nOutput:\nsame failure"

    agent._run_checker = fail_checker
    agent.build_messages(AgentState(problem_description="problem"))

    for _ in range(4):
        assert "same failure" in run(agent._run_audited_checker(args=[]))
    result = run(agent._run_audited_checker(args=[]))

    assert "failed 5 times" in result
    assert agent._checker_execution_terminal_failure is True
    assert "Do not run it again" in run(agent._run_audited_checker(args=[]))


def test_feasibility_review_routes_terminal_checker_failure_to_solving(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text("# checker\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    state = AgentState(
        problem_description="problem",
        solver_result={"status": "time_limit", "solution_file": "solution.json"},
    )
    agent.build_messages(state)
    agent._checker_execution_terminal_failure = True
    payload = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "solving",
        "summary": "独立验收无法完成，候选解需要重新产出。",
        "evidence": [{"source": "checker execution", "finding": "同一执行失败重复 5 次"}],
        "required_changes": ["重新生成可验收的候选解"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["candidate_format"],
            "change_ids": ["regenerate_candidate"],
            "problem_requirement": "候选解必须满足全部业务硬约束。",
            "observed_solution_behavior": "当前候选解无法完成独立验收。",
            "responsibility_reason": "求解实现必须产出可验收的候选解。",
            "required_changes": ["重新生成候选解并完成自检。"],
        },
    }

    parsed = agent.parse_output_file(payload, state)

    assert parsed["feasibility_check"] is None
    assert parsed["feasibility_review"]["responsibility"] == "solving"

    payload["responsibility"] = "algorithm_design"
    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, state)
    assert any("只能 reject 并将责任归于 solving" in issue.message for issue in raised.value.issues)


def test_feasibility_review_shell_allows_auxiliary_commands_without_checker_execution(tmp_path):
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    agent.build_messages(AgentState(problem_description="problem"))

    shell = next(tool for tool in agent.tools if tool.name == "shell").handler
    result = run(shell(command='python -c "print(23)"'))

    assert "23" in result
    assert agent._checker_executed is False
    assert agent._checker_failure_count == 0


def test_feasibility_review_shell_cannot_modify_the_implementation_it_judges(tmp_path):
    solver = tmp_path / "solver.py"
    solver.write_text("original\n", encoding="utf-8")
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )

    shell = next(tool for tool in agent.tools if tool.name == "shell").handler
    result = run(shell(command='python -c "open(\'solver.py\', \'w\').write(\'changed\\n\')"'))

    assert "unauthorized protected file modification restored" in result
    assert solver.read_text(encoding="utf-8") == "original\n"


REVIEW_CHECKER_SOURCE = """import json
import sys

feasible = "--feasible" in sys.argv
with open("feasibility_result.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_valid": True,
            "feasible": feasible,
            "objective_value": 1470.0617,
            "violations": [] if feasible else ["objective mismatch"],
            "warnings": [],
        },
        handle,
    )
sys.exit(0 if feasible else 1)
"""


def _reviewer_with_real_checker(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text(REVIEW_CHECKER_SOURCE, encoding="utf-8")
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": []}', encoding="utf-8")
    return FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )


def test_feasibility_review_accepts_fresh_valid_result_despite_nonzero_exit(tmp_path):
    agent = _reviewer_with_real_checker(tmp_path)
    agent.build_messages(AgentState(problem_description="problem"))

    result = run(agent._run_audited_checker(args=[]))

    assert "Exit code: 1" in result
    assert agent._checker_executed is True
    assert agent._checker_failure_count == 0


def test_unknown_tool_message_lists_the_tools_and_suggests_the_nearest():
    # In a 166-task benchmark, a vague unknown-tool message triggered 69 replace_file calls.
    available = {"shell": 1, "read_file": 1, "replace_in_file": 1, "write_file": 1}

    message = unknown_tool_message("replace_file", available)

    assert "最接近的可用 tool 是 `replace_in_file`" in message
    assert "`shell`" in message and "`read_file`" in message
    assert "`write_file`" in message

    unrelated = unknown_tool_message("bash", available)
    assert "最接近" not in unrelated
    assert "`shell`" in unrelated


def test_solving_validation_feedback_points_at_the_submission_tool(tmp_path):
    # With Review enabled no actor can rewrite stage_outputs/solving.json.
    agent = SolvingAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    error = StageOutputValidationError(
        "solving 输出与工作区不一致",
        issues=(
            OutputValidationIssue(path="$.solver_execution_receipt", message="回执缺失"),
        ),
        file_path="stage_outputs/solving.json",
    )

    feedback = agent.build_validation_feedback(error)

    assert "submit_solving_outcome" in feedback
    assert "run_solver" in feedback
    assert "replace_in_file 和 shell 写入" in feedback


def test_self_check_solving_keeps_the_default_validation_remedy(tmp_path):
    agent = SolvingAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        ),
        feasibility_review_enabled=False,
    )
    error = StageOutputValidationError("solving 输出不符合契约", issues=())

    feedback = agent.build_validation_feedback(error)

    assert "请使用 write_file/replace_in_file 修正文件" in feedback
    assert "submit_solving_outcome" not in feedback


def test_feasibility_review_validation_feedback_names_its_own_tools(tmp_path):
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    error = StageOutputValidationError("feasibility review 与求解证据不一致", issues=())

    feedback = agent.build_validation_feedback(error)

    assert "run_feasibility_checker" in feedback
    assert "本阶段没有 replace_in_file" in feedback
    assert "请使用 write_file/replace_in_file 修正文件" not in feedback


def test_feasibility_review_rejects_a_result_without_a_matching_receipt(tmp_path):
    # A valid feasibility_result must belong to the current solution and checker execution.
    agent = _reviewer_with_real_checker(tmp_path)
    state = AgentState(
        problem_description="problem",
        solver_result={"status": "optimal", "solution_file": "solution.json"},
    )
    agent.build_messages(state)
    run(agent._run_audited_checker(args=["--feasible"]))
    payload = {
        "schema_version": "1.0",
        "decision": "accept",
        "responsibility": None,
        "summary": "checker 与求解结果均接受当前候选",
        "evidence": [
            {"source": "feasibility_result.json", "finding": "schema_valid=true and feasible=true"}
        ],
        "required_changes": [],
        "confidence": "high",
        "infeasibility_proof": "",
    }

    assert agent.parse_output_file(payload, state)["feasibility_review"]["decision"] == "accept"

    (tmp_path / "solution.json").write_text('{"routes": ["swapped"]}', encoding="utf-8")

    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, state)
    assert any(
        "feasibility_check_receipt.json" in issue.message for issue in raised.value.issues
    )


def test_feasibility_review_assesses_infeasibility_proof_without_checker(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('INFEASIBLE')\n", encoding="utf-8")
    (tmp_path / "solver_result.json").write_text(
        json.dumps(generated_infeasible_solver_result()),
        encoding="utf-8",
    )
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    state = AgentState(
        problem_description="problem",
        solver_result=generated_infeasible_solver_result(),
    )
    prompt = agent.build_messages(state).messages[-1]["content"]
    payload = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "instance",
        "summary": "完整精确模型证明实例不可行",
        "evidence": [{"source": "solver_result.json", "finding": "CP-SAT INFEASIBLE"}],
        "required_changes": [],
        "confidence": "high",
        "infeasibility_proof": "CP-SAT 对覆盖全部硬约束的完整模型返回 INFEASIBLE",
        "remediation_handoff": None,
    }

    parsed = agent.parse_output_file(payload, state)

    assert "不得要求 solution.json" in prompt
    assert "不执行 feasibility_checker.py" in prompt
    assert parsed["feasibility_check"] is None
    assert parsed["feasibility_review"]["responsibility"] == "instance"


def test_feasibility_review_routes_no_candidate_without_checker(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('TIME_LIMIT')\n", encoding="utf-8")
    state = AgentState(
        problem_description="problem",
        solver_result=generated_no_candidate_solver_result(),
    )
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    prompt = agent.build_messages(state).messages[-1]["content"]
    payload = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "solving",
        "summary": "实现仍有改进空间",
        "evidence": [{"source": "solver_result.json", "finding": "TIME_LIMIT 无候选"}],
        "required_changes": ["改进搜索实现"],
        "confidence": "medium",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["candidate_required"],
            "change_ids": ["improve_search"],
            "problem_requirement": "需要找到满足全部硬约束的候选解",
            "observed_solution_behavior": "本轮正常结束但没有候选解",
            "responsibility_reason": "当前实现尚未有效执行既定搜索策略",
            "required_changes": ["改进搜索实现并重新执行"],
        },
    }

    parsed = agent.parse_output_file(payload, state)

    assert "无候选求解终态" in prompt
    assert "不执行 feasibility_checker.py" in prompt
    assert parsed["feasibility_check"] is None
    assert parsed["feasibility_review"]["responsibility"] == "solving"

    payload.update(
        {
            "responsibility": "instance",
            "required_changes": [],
            "infeasibility_proof": "启发式未找到解",
            "remediation_handoff": None,
        }
    )
    with pytest.raises(StageOutputValidationError) as raised:
        agent.parse_output_file(payload, state)
    assert any(issue.path == "$.responsibility" for issue in raised.value.issues)


def test_feasibility_review_routes_runtime_timeout_without_model_fields(tmp_path):
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')\n", encoding="utf-8")
    (tmp_path / "runtime_solver_outcome.json").write_text("{}", encoding="utf-8")
    state = AgentState(
        problem_description="problem",
        solver_result={
            "schema_version": "1.0",
            "outcome_source": "runtime_external_timeout",
            "status": "runtime_timeout",
            "timeout_source": "runtime_watchdog",
            "timeout_seconds": 590,
            "elapsed_seconds": 590.1,
            "exit_code": -15,
            "termination_signal": "SIGTERM",
            "solver_file": "solver.py",
            "solver_execution_observed": True,
            "solution_detected": False,
            "solution_schema_valid": None,
            "solution_file": None,
            "stdout_tail": "",
        },
    )
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    prompt = agent.build_messages(state).messages[-1]["content"]
    payload = {
        "schema_version": "1.0",
        "decision": "reject",
        "responsibility": "algorithm_design",
        "summary": "方法不适配实例规模与预算",
        "evidence": [
            {
                "source": "runtime_solver_outcome.json",
                "finding": "正式求解器在 590 秒外部时限结束且没有候选",
            }
        ],
        "required_changes": ["选择能在预算内尽早产生可行候选的方法"],
        "confidence": "high",
        "infeasibility_proof": "",
        "remediation_handoff": {
            "requirement_ids": ["budget_compatible_method"],
            "change_ids": ["replace_oversized_formulation"],
            "problem_requirement": "在运行预算内产生完整可行候选",
            "observed_solution_behavior": "正式方法达到外部时限且没有候选",
            "responsibility_reason": "设计的方法与实例规模及预算不匹配",
            "required_changes": ["替换过重模型并优先保留可行 incumbent"],
        },
    }

    parsed = agent.parse_output_file(payload, state)

    assert "Runtime 记录" in prompt
    assert "不得要求其伪造 executions" in prompt
    assert parsed["feasibility_check"] is None
    assert parsed["feasibility_review"]["responsibility"] == "algorithm_design"


def test_feasibility_review_stops_fifth_identical_validation_failure(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text("# checker\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    agent = FeasibilityReviewAgent(
        make_services(
            config=fake_core_config(workspace_root=tmp_path),
            workspace_toolset=fake_workspace_toolset(tmp_path),
        )
    )
    agent.build_messages(AgentState(problem_description="problem"))
    error = StageOutputValidationError(
        "same validation failure",
        issues=(OutputValidationIssue("$.decision", "same evidence mismatch"),),
    )

    for _ in range(4):
        assert "same evidence mismatch" in agent.build_validation_feedback(error)
    with pytest.raises(StageExecutionStopped, match="repeated 5 times"):
        agent.build_validation_feedback(error)


def test_tool_debug_events_are_emitted(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"ok": True})
    events = []

    async def handler(event):
        events.append(event)

    async def debug_tool() -> str:
        return "result"

    llm = FakeLLM(
        [
            ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "debug_tool", "arguments": "{}"},
                    }
                ],
            ),
            ChatResponse(content="done"),
        ]
    )
    services = make_services(
        llm,
        config=fake_core_config(workspace_root=tmp_path),
        debug_event_handler=handler,
    )
    agent = StageAgent(
        stage=AgentStage.ALGORITHM_DESIGN,
        services=services,
        tools=(Tool(name="debug_tool", description="", parameters={}, handler=debug_tool),),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    run(collect(agent))

    assert [(event["kind"], event["tool_name"]) for event in events] == [
        ("tool_call_started", "debug_tool"),
        ("tool_result", "debug_tool"),
    ]


def test_complete_agent_turn_is_emitted_without_tool_content_truncation(tmp_path):
    write_stage_output(tmp_path, AgentStage.ALGORITHM_DESIGN, {"ok": True})
    turns = []

    async def record_turn(turn):
        turns.append(turn)

    async def write_file(path: str, content: str) -> str:
        return f"wrote {path}\n{content}"

    multiline = "first line\nsecond line\nthird line"
    llm = FakeLLM(
        [
            ChatResponse(
                reasoning="write the complete file",
                tool_calls=[
                    {
                        "id": "call-full",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "result.txt", "content": multiline}),
                        },
                    }
                ],
            ),
            ChatResponse(content="done", reasoning="verify complete output"),
        ]
    )
    services = make_services(
        llm,
        config=fake_core_config(workspace_root=tmp_path),
        agent_turn_handler=record_turn,
    )
    agent = StageAgent(
        stage=AgentStage.ALGORITHM_DESIGN,
        services=services,
        tools=(Tool(name="write_file", description="", parameters={}, handler=write_file),),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    run(collect(agent))

    assert len(turns) == 2
    first = turns[0]
    assert first["stage"] == "algorithm_design"
    assert first["response"]["reasoning"] == "write the complete file"
    assert first["tool_executions"][0]["arguments"]["content"] == multiline
    assert first["tool_executions"][0]["result"] == f"wrote result.txt\n{multiline}"
    assert first["request"]["messages"] == [{"role": "system", "content": "system"}]


def test_report_progress_records_stage_event_and_returns_tool_result(tmp_path):
    write_stage_output(tmp_path, AgentStage.SOLVING, {"done": True})
    events = []

    async def stage_event_handler(event):
        events.append(event)

    payload = {
        "stage": "solving",
        "summary": "已经识别出可行构造路线",
        "completed": ["读取数据", "确认容量字段"],
        "key_findings": ["订单有需求量"],
        "assumptions": ["订单不可拆分"],
        "next_step": "编写求解脚本",
    }
    llm = FakeLLM(
        [
            ChatResponse(
                tool_calls=[
                    {
                        "id": "call-progress",
                        "type": "function",
                        "function": {
                            "name": REPORT_PROGRESS_TOOL_NAME,
                            "arguments": json.dumps(payload, ensure_ascii=False),
                        },
                    }
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = StageAgent(
        stage=AgentStage.SOLVING,
        services=make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            stage_event_handler=stage_event_handler,
        ),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    items = run(collect(agent))

    assert items[-1].data == {"done": True}
    assert events == [
        {
            "kind": STAGE_EVENT_PROGRESS_REPORTED,
            "agent_stage": AgentStage.SOLVING.value,
            "payload": payload,
        }
    ]
    tool_messages = [message for message in llm.calls[1].messages if message["role"] == "tool"]
    assert tool_messages == [
        {
            "role": "tool",
            "tool_call_id": "call-progress",
            "content": "progress recorded",
        }
    ]


def test_control_tool_invalid_args_return_readable_error_without_stage_crash(tmp_path):
    write_stage_output(tmp_path, AgentStage.SOLVING, {"done": True})
    events = []

    async def stage_event_handler(event):
        events.append(event)

    llm = FakeLLM(
        [
            ChatResponse(
                tool_calls=[
                    {
                        "id": "call-bad-progress",
                        "type": "function",
                        "function": {
                            "name": REPORT_PROGRESS_TOOL_NAME,
                            "arguments": '{"stage": "solving"}',
                        },
                    }
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = StageAgent(
        stage=AgentStage.SOLVING,
        services=make_services(
            llm,
            config=fake_core_config(workspace_root=tmp_path),
            stage_event_handler=stage_event_handler,
        ),
    )
    agent.build_messages = lambda state: MessageBuilder.from_prompts(system="system")
    agent.parse_output_file = lambda data, state: data

    items = run(collect(agent))

    assert items[-1].data == {"done": True}
    assert len(events) == 1
    assert events[0]["kind"] == "agent_error"
    assert events[0]["payload"]["flow_section"] == "solving"
    assert events[0]["payload"]["error_kind"] == "tool_argument_validation"
    tool_messages = [message for message in llm.calls[1].messages if message["role"] == "tool"]
    feedback = tool_messages[0]["content"]
    assert "tool 参数校验失败: report_progress" in feedback
    assert "缺少必填参数: assumptions, completed, key_findings, next_step, summary" in feedback
    assert "summary: 必填; 类型=string; 含义=" in feedback
    assert "含义=A concise useful progress summary for the user." in feedback


def test_control_tool_name_conflict_fails_loudly():
    async def fake_report_progress(**kwargs):
        return "bad"

    try:
        StageAgent(
            stage=AgentStage.INTAKE,
            services=make_services(),
            tools=(
                Tool(
                    name=REPORT_PROGRESS_TOOL_NAME,
                    description="conflict",
                    parameters={},
                    handler=fake_report_progress,
                ),
            ),
        )
    except ValueError as exc:
        assert REPORT_PROGRESS_TOOL_NAME in str(exc)
    else:
        raise AssertionError("expected control tool name conflict")
