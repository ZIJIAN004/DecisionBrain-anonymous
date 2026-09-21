from __future__ import annotations

import asyncio
import json
from pathlib import Path

from decisionbrain.benchmark.adapter import (
    ADAPTER_DIR,
    TARGET_SCHEMA_FILE,
    BenchmarkSolutionAdapter,
    validate_template_shape,
)
from decisionbrain.benchmark.task import load_task
from decisionbrain.config import Settings
from decisionbrain.core.models import ChatResponse


class FakeLLM:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _tool_response(name: str, arguments: dict, *, call_id: str) -> ChatResponse:
    return ChatResponse(
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    )


def _task(tmp_path: Path):
    root = tmp_path / "sample"
    (root / "input" / "data").mkdir(parents=True)
    (root / "hidden").mkdir()
    (root / "input" / "problem.md").write_text("minimize routing cost", encoding="utf-8")
    (root / "input" / "data" / "instance.json").write_text(
        json.dumps({"customers": [1, 2]}), encoding="utf-8"
    )
    (root / "task.json").write_text(
        json.dumps({"direction": "min", "problem_class": "routing"}), encoding="utf-8"
    )
    (root / "hidden" / "solution_schema.json").write_text(
        json.dumps(
            {
                "objective_value": "<float>",
                "routes": [{"vehicle": "<int>", "customers": "<list[int]>"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "hidden" / "reference_solution.json").write_text(
        '{"objective_value": 10}', encoding="utf-8"
    )
    (root / "hidden" / "feasibility_check.py").write_text(
        "TOP_SECRET_CHECKER = True", encoding="utf-8"
    )
    return load_task(root)


def _runtime_workspace(tmp_path: Path, task) -> Path:
    workspace = tmp_path / "run" / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "problem.md").write_text(task.problem_md.read_text(), encoding="utf-8")
    (workspace / "data" / "instance.json").write_text(
        (task.input_dir / "data" / "instance.json").read_text(), encoding="utf-8"
    )
    (workspace / "solution.json").write_text(
        json.dumps({"cost": 12, "vehicle_paths": [[1, 2]]}), encoding="utf-8"
    )
    (workspace / "solution_schema.json").write_text(
        json.dumps({"cost": "number", "vehicle_paths": "array"}), encoding="utf-8"
    )
    (workspace / "solver_result.json").write_text(
        json.dumps({"objective": 12}), encoding="utf-8"
    )
    return workspace


def _formatter_script(candidate: dict) -> str:
    return (
        "import json\n"
        f"json.dump({candidate!r}, open('solution.json', 'w'))\n"
    )


def test_adapter_writes_valid_solution_without_copying_hidden_material(tmp_path: Path):
    task = _task(tmp_path)
    workspace = _runtime_workspace(tmp_path, task)
    candidate = {"objective_value": 12.0, "routes": [{"vehicle": 0, "customers": [1, 2]}]}
    llm = FakeLLM(
        [
            _tool_response("write_file", {"path": "convert_solution.py", "content": _formatter_script(candidate)}, call_id="write"),
            _tool_response("run_python", {"filename": "convert_solution.py"}, call_id="run"),
            _tool_response("validate_benchmark_solution", {}, call_id="validate"),
            ChatResponse(content="conversion complete"),
        ]
    )

    result = asyncio.run(BenchmarkSolutionAdapter(llm, Settings()).adapt(task, workspace))

    assert result.turns == 4
    assert json.loads(result.solution_path.read_text(encoding="utf-8")) == candidate
    adapter_files = {
        path.relative_to(workspace / ADAPTER_DIR).as_posix()
        for path in (workspace / ADAPTER_DIR).rglob("*")
        if path.is_file()
    }
    assert TARGET_SCHEMA_FILE in adapter_files
    assert "solution_schema.json" in adapter_files
    assert "feasibility_check.py" not in adapter_files
    assert "reference_solution.json" not in adapter_files
    request_text = json.dumps(
        [request.messages for request in llm.requests], ensure_ascii=False
    )
    assert "TOP_SECRET_CHECKER" not in request_text


def test_adapter_retries_after_invalid_shape(tmp_path: Path):
    task = _task(tmp_path)
    workspace = _runtime_workspace(tmp_path, task)
    valid = {"objective_value": 12, "routes": []}
    llm = FakeLLM(
        [
            _tool_response(
                "write_file",
                {"path": "convert_solution.py", "content": _formatter_script({"objective_value": 12})},
                call_id="invalid",
            ),
            _tool_response("run_python", {"filename": "convert_solution.py"}, call_id="run1"),
            _tool_response("validate_benchmark_solution", {}, call_id="validate1"),
            ChatResponse(content="retry"),
            _tool_response(
                "write_file",
                {"path": "convert_solution.py", "content": _formatter_script(valid)},
                call_id="repair",
            ),
            _tool_response("run_python", {"filename": "convert_solution.py"}, call_id="run2"),
            _tool_response("validate_benchmark_solution", {}, call_id="validate2"),
            ChatResponse(content="done"),
        ]
    )

    result = asyncio.run(BenchmarkSolutionAdapter(llm, Settings()).adapt(task, workspace))

    assert result.turns == 8
    assert json.loads(result.solution_path.read_text(encoding="utf-8")) == valid
    feedback = json.dumps(llm.requests[3].messages, ensure_ascii=False)
    assert "$.routes: 缺少字段" in feedback


def test_adapter_does_not_expose_code_execution_tool(tmp_path: Path):
    task = _task(tmp_path)
    workspace = _runtime_workspace(tmp_path, task)
    candidate = {
        "objective_value": 12,
        "routes": [{"vehicle": 0, "customers": [1, 2]}],
    }
    llm = FakeLLM(
        [
            _tool_response(
                "write_file",
                {"path": "convert_solution.py", "content": _formatter_script(candidate)},
                call_id="write-solution",
            ),
            _tool_response("run_python", {"filename": "convert_solution.py"}, call_id="run"),
            _tool_response("validate_benchmark_solution", {}, call_id="validate"),
            ChatResponse(content="conversion complete"),
        ]
    )

    result = asyncio.run(BenchmarkSolutionAdapter(llm, Settings()).adapt(task, workspace))

    assert result.turns == 4
    assert json.loads(result.solution_path.read_text(encoding="utf-8")) == candidate
    offered_tools = {tool["function"]["name"] for tool in llm.requests[0].tools}
    assert "run_python" in offered_tools
    request_text = json.dumps(llm.requests[0].messages, ensure_ascii=False)
    assert "do not optimize, rerun a solver, or fabricate missing values" in request_text


def test_template_shape_checks_nested_fields_and_known_placeholder_types():
    template = {
        "objective_value": "<float>",
        "routes": [{"vehicle": "<int>", "active": "<bool>"}],
    }

    assert validate_template_shape(
        template,
        {"objective_value": 1.5, "routes": [{"vehicle": 1, "active": True}]},
    ) == []
    issues = validate_template_shape(
        template,
        {"objective_value": "1.5", "routes": [{"vehicle": "1"}]},
    )
    assert "$.objective_value: 期望 number，实际 str" in issues
    assert "$.routes[0].vehicle: 期望 integer，实际 str" in issues
    assert "$.routes[0].active: 缺少字段" in issues


def test_template_shape_supports_dynamic_keys_dotted_list_fields_and_variants():
    template = {
        "objective_value": "<float>",
        "schedule": {"{job_id}": {"start": "<int>"}},
        "assignments": "<list[dict]>",
        "assignments[*].machine": "<int>",
        "variant_only": "<list[int]> Present only for the tardy variant.",
    }

    assert validate_template_shape(
        template,
        {
            "objective_value": 5,
            "schedule": {"job-1": {"start": 0}},
            "assignments": [{"machine": 2}],
        },
    ) == []

    missing = validate_template_shape(
        {"assignments[*].machine": "<int>"},
        {},
    )
    wrong_type = validate_template_shape(
        {"assignments[*].machine": "<int>"},
        {"assignments": {}},
    )
    assert missing == ["$.assignments: 缺少字段"]
    assert wrong_type == ["$.assignments: 期望 array，实际 dict"]
