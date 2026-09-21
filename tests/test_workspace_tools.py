"""WorkspaceToolset tests."""

import asyncio
import hashlib
import json
import ctypes
import os
import shlex
import subprocess
import sys
import time

import pytest

from decisionbrain.core.workspace_tools import (
    REVIEW_HISTORY_VIOLATION_SAMPLE,
    TRUNCATION_NOTICE,
    WorkspaceToolset,
    _parse_linux_solver_trace,
    write_feasibility_review_history,
)
from decisionbrain.core.models import AgentStage
from decisionbrain.core.stage_agent import StageForcedCompletion
from decisionbrain.core.stage_outputs import FEASIBILITY_REVIEW_HISTORY_FILE
from helpers import fake_workspace_toolset


def run(coro):
    return asyncio.run(coro)


def shell_command(*args: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    return shlex.join(args)


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid_exit(pid: int, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_exists(pid):
            return True
        time.sleep(0.02)
    return not pid_exists(pid)


def write_nested_sleep_scripts(tmp_path):
    child_pid = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    return parent, child_pid


def tool_by_name(toolset: WorkspaceToolset, name: str):
    return next(tool for tool in toolset.tools() if tool.name == name)


def stage_tool_by_name(toolset: WorkspaceToolset, stage: AgentStage, name: str):
    return next(tool for tool in toolset.tools(stage=stage) if tool.name == name)


def candidate_solver_result() -> dict:
    return {
        "status": "time_limit",
        "objective": 12.5,
        "mip_gap": None,
        "optimality": "heuristic_or_incumbent",
        "input_file": "input.json",
        "solution_file": "solution.json",
        "code_file": "solver.py",
        "feasibility_result_file": None,
        "executions": [{
            "component_id": "generated",
            "role": "construct candidate",
            "source": "generated",
            "planned_package_id": None,
            "planned_solver_id": None,
            "executed_package_id": None,
            "executed_solver_id": None,
            "method": "other",
            "method_class": "constructive_heuristic",
            "version": None,
            "guide_consulted": False,
            "availability_check": {"ok": True, "detail": "not applicable"},
            "seed": None,
            "runtime_limit_seconds": 10,
            "fallback_used": False,
            "deviation_reason": "",
        }],
        "solution_summary": {},
        "constraint_check": [],
        "validation": {
            "schema_check": {"ok": True, "issues": []},
            "feasibility_checker": None,
            "business_semantic_check": {"ok": True, "issues": []},
            "algorithm_mapping_check": {"ok": True, "issues": [], "mapped_capabilities": []},
            "implementation_check": {"ok": True, "issues": []},
        },
        "diagnosis": "candidate found before time limit",
    }


def single_algorithm_solver_result() -> dict:
    result = candidate_solver_result()
    legacy_execution = result.pop("executions")[0]
    result["execution"] = {
        key: value
        for key, value in legacy_execution.items()
        if key not in {"component_id", "role", "fallback_used", "deviation_reason"}
    }
    return result


def test_solving_submission_tool_validates_solution_and_publishes_outputs(tmp_path):
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text(
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"], "properties": {"routes": {"type": "array"}}}),
        encoding="utf-8",
    )
    toolset = fake_workspace_toolset(tmp_path)
    tools = toolset.tools(stage=AgentStage.SOLVING, solving_outcome_submission=True)
    run_solver = next(tool for tool in tools if tool.name == "run_solver")
    assert "written to solver_execution_receipt.json" in run(run_solver.handler())
    submit = next(tool for tool in tools if tool.name == "submit_solving_outcome")

    blocked = next(tool for tool in tools if tool.name == "write_file")
    assert "not allowed to write" in run(
        blocked.handler(path="solver_result.json", content="{}")
    )
    result = run(submit.handler(message="candidate ready", result=candidate_solver_result()))

    assert "passed validation" in result
    assert json.loads((tmp_path / "solver_result.json").read_text(encoding="utf-8"))["status"] == "time_limit"
    assert json.loads((tmp_path / "stage_outputs/solving.json").read_text(encoding="utf-8"))["decision"] == "solved"


def test_single_algorithm_submission_rejects_legacy_execution_list(tmp_path):
    toolset = fake_workspace_toolset(tmp_path, components_enabled=False)
    submit = next(
        tool
        for tool in toolset.tools(
            stage=AgentStage.SOLVING, solving_outcome_submission=True
        )
        if tool.name == "submit_solving_outcome"
    )

    result = run(
        submit.handler(message="candidate ready", result=candidate_solver_result())
    )

    assert "solver_result does not match its contract" in result
    assert "execution" in result
    assert not (tmp_path / "solver_result.json").exists()


def test_single_algorithm_submission_validates_and_publishes_outputs(tmp_path):
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text(
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "solution_schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["routes"],
                "properties": {"routes": {"type": "array"}},
            }
        ),
        encoding="utf-8",
    )
    toolset = fake_workspace_toolset(tmp_path, components_enabled=False)
    tools = toolset.tools(stage=AgentStage.SOLVING, solving_outcome_submission=True)
    run_solver = next(tool for tool in tools if tool.name == "run_solver")
    assert "written to solver_execution_receipt.json" in run(run_solver.handler())
    submit = next(tool for tool in tools if tool.name == "submit_solving_outcome")

    response = run(
        submit.handler(message="candidate ready", result=single_algorithm_solver_result())
    )

    assert "passed validation" in response
    stored = json.loads((tmp_path / "solver_result.json").read_text(encoding="utf-8"))
    assert "execution" in stored
    assert "executions" not in stored


def test_solving_submission_tool_rejects_solution_schema_without_publishing(tmp_path):
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('candidate')\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"]}), encoding="utf-8"
    )
    toolset = fake_workspace_toolset(tmp_path)
    submit = next(
        tool for tool in toolset.tools(stage=AgentStage.SOLVING, solving_outcome_submission=True)
        if tool.name == "submit_solving_outcome"
    )

    result = run(submit.handler(message="candidate ready", result=candidate_solver_result()))

    assert "does not match solution_schema.json" in result
    assert not (tmp_path / "solver_result.json").exists()
    assert not (tmp_path / "stage_outputs/solving.json").exists()


def _submission_tool(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    return next(
        tool
        for tool in toolset.tools(stage=AgentStage.SOLVING, solving_outcome_submission=True)
        if tool.name == "submit_solving_outcome"
    )


def test_submission_names_run_solver_when_no_receipt_exists(tmp_path):
    # The old message pointed the model to a receipt file it can never create.
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('candidate')\n", encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"]}), encoding="utf-8"
    )

    result = run(
        _submission_tool(tmp_path).handler(
            message="candidate ready", result=candidate_solver_result()
        )
    )

    assert "no solver execution receipt" in result
    assert "call run_solver" in result
    assert not (tmp_path / "solver_result.json").exists()


def test_submission_reports_every_problem_in_one_response(tmp_path):
    # Report all defects together so the model can repair them in one round.
    (tmp_path / "solution.json").write_text('{"trips": 3}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"]}), encoding="utf-8"
    )

    result = run(
        _submission_tool(tmp_path).handler(
            message="candidate ready", result=candidate_solver_result()
        )
    )

    assert "the submitted outcome does not match the workspace:" in result
    assert "missing or empty required file: solver.py" in result
    assert "path does not exist: input.json" in result
    assert "solution.json does not match solution_schema.json" in result
    assert "no solver execution receipt" in result


def test_submission_explains_a_receipt_that_bound_no_candidate(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"]}), encoding="utf-8"
    )
    (tmp_path / "solver.py").write_text("print('no candidate')\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    run(stage_tool_by_name(toolset, AgentStage.SOLVING, "run_solver").handler())
    (tmp_path / "solution.json").write_text('{"routes": []}', encoding="utf-8")

    result = run(
        _submission_tool(tmp_path).handler(
            message="candidate ready", result=candidate_solver_result()
        )
    )

    assert "bound no candidate" in result
    assert "run it again" in result


def test_submission_lists_every_file_changed_after_the_recorded_run(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        json.dumps({"type": "object", "required": ["routes"]}), encoding="utf-8"
    )
    (tmp_path / "solver.py").write_text(
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    toolset = fake_workspace_toolset(tmp_path)
    run(stage_tool_by_name(toolset, AgentStage.SOLVING, "run_solver").handler())
    (tmp_path / "solver.py").write_text("print('edited after the run')\n", encoding="utf-8")
    (tmp_path / "input.json").write_text('{"orders": [1]}', encoding="utf-8")

    result = run(
        _submission_tool(tmp_path).handler(
            message="candidate ready", result=candidate_solver_result()
        )
    )

    assert "solver.py, input.json no longer match the digests" in result
    assert "run run_solver again" in result


def test_branch9_solving_tools_do_not_expose_submission_tool(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    tools = toolset.tools(stage=AgentStage.SOLVING, solving_feasibility_self_check=True)
    assert "submit_solving_outcome" not in {tool.name for tool in tools}


def test_read_file_tool_description_explains_large_file_limit(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)

    description = tool_by_name(toolset, "read_file").description

    assert "larger than 1000000 bytes fail" in description
    assert "start_line and line_count" in description
    assert "single-line JSON" in description


def test_read_write_and_list_files_stay_inside_workspace(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    write = tool_by_name(toolset, "write_file").handler
    read = tool_by_name(toolset, "read_file").handler
    list_files = tool_by_name(toolset, "list_files").handler

    result = run(write(path="notes/todo.txt", content="hello"))
    assert result == "wrote 5 bytes to notes/todo.txt"
    assert run(read(path="notes/todo.txt")) == "hello"

    listing = run(list_files(path=".", recursive=True))
    assert "notes/" in listing
    assert "notes/todo.txt" in listing


def test_list_files_reports_sizes_and_flags_files_read_file_will_refuse(tmp_path):
    # Large-suite files can be tens of MB; expose size before an inevitably failing read.
    (tmp_path / "data").mkdir()
    (tmp_path / "data/instance.json").write_text("x" * 4096, encoding="utf-8")
    (tmp_path / "small.txt").write_text("hi", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path, max_read_bytes=1024)
    list_files = tool_by_name(toolset, "list_files").handler

    listing = run(list_files(path=".", recursive=True))

    assert "small.txt  (2 B)" in listing
    assert "data/instance.json  (4.0 KB)" in listing
    assert "[too large for read_file; use shell to extract what you need]" in listing
    # Directories have no size and only oversized files receive a marker.
    assert "data/" in listing
    assert "small.txt  (2 B) [too large" not in listing


def test_search_file_returns_literal_matches_with_locations_and_limit(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/first.txt").write_text("alpha\nneedle one\nneedle two\n", encoding="utf-8")
    (tmp_path / "data/second.txt").write_text("needle three\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    search = tool_by_name(toolset, "search_file").handler

    result = run(search(path="data", query="needle", max_results=2))

    assert result.splitlines() == ["data/first.txt:2: needle one", "data/first.txt:3: needle two"]
    assert run(search(path="data", query="NEEDLE")) == "(no matches)"
    assert "data/first.txt:2: needle one" in run(search(query="needle"))


def test_search_file_description_explains_supported_arguments_and_limits(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)

    description = tool_by_name(toolset, "search_file").description

    assert "Allowed arguments are only path" in description
    assert "defaults to ." in description
    assert "default 20, maximum 100" in description
    assert "does not support regex, glob, case_sensitive, or limit parameters" in description
    assert "scans at most 200 files" in description


def test_search_file_obeys_stage_read_policy(tmp_path):
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "stage_outputs/problem_contract.json").write_text("secret needle", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    search = stage_tool_by_name(toolset, AgentStage.ALGORITHM_DESIGN, "search_file").handler

    assert run(search(path=".", query="needle")) == "(no matches)"


def test_read_file_supports_line_ranges(tmp_path):
    (tmp_path / "lines.txt").write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    read = tool_by_name(toolset, "read_file").handler

    result = run(read(path="lines.txt", start_line=2, line_count=2))

    assert result.splitlines() == ["beta", "gamma"]


def test_read_file_line_ranges_validate_bounds(tmp_path):
    (tmp_path / "lines.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    read = tool_by_name(toolset, "read_file").handler

    assert "start_line must be a positive integer" in run(read(path="lines.txt", start_line=0))
    assert "start_line 4 exceeds file length of 2 lines" in run(
        read(path="lines.txt", start_line=4)
    )


def test_read_file_streams_requested_lines_from_file_over_read_limit(tmp_path):
    (tmp_path / "large.txt").write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")
    toolset = WorkspaceToolset(
        tmp_path,
        timeout_s=10,
        max_output_chars=12_000,
        max_list_entries=1_000,
        default_list_entries=200,
        max_read_bytes=10,
    )
    read = tool_by_name(toolset, "read_file").handler

    assert "file too large" in run(read(path="large.txt"))
    assert run(read(path="large.txt", start_line=2, line_count=2)) == "second\nthird\n"


def test_paths_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    read = tool_by_name(toolset, "read_file").handler
    write = tool_by_name(toolset, "write_file").handler

    assert "escapes workspace root" in run(read(path="../outside-secret.txt"))
    assert "absolute paths are not allowed" in run(read(path=str(outside)))
    assert "escapes workspace root" in run(write(path="../outside-secret.txt", content="x"))


def test_solving_submission_uses_fixed_contract_when_problem_contract_disabled(tmp_path):
    (tmp_path / "fixed_solution_contract.json").write_text(
        json.dumps({"type": "object", "required": ["value"]}), encoding="utf-8"
    )
    (tmp_path / "solution.json").write_text(json.dumps({"value": 1}), encoding="utf-8")
    toolset = WorkspaceToolset(
        tmp_path,
        timeout_s=10,
        max_output_chars=12_000,
        max_list_entries=1_000,
        default_list_entries=200,
        max_read_bytes=1_000_000,
        problem_contract_enabled=False,
    )
    assert toolset._solution_schema_path() == "fixed_solution_contract.json"
    assert toolset._solution_schema_problems() == []
    write = next(
        tool for tool in toolset.tools(stage=AgentStage.SOLVING) if tool.name == "write_file"
    ).handler
    assert "not allowed to write fixed_solution_contract.json" in run(
        write(path="fixed_solution_contract.json", content="{}")
    )


def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "linked-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    toolset = fake_workspace_toolset(tmp_path)
    read = tool_by_name(toolset, "read_file").handler

    assert "escapes workspace root" in run(read(path="link.txt"))


def test_write_file_creates_parent_directories(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    write = tool_by_name(toolset, "write_file").handler

    result = run(write(path="a/b/c.txt", content="created"))

    assert result == "wrote 7 bytes to a/b/c.txt"
    assert (tmp_path / "a/b/c.txt").read_text(encoding="utf-8") == "created"


def test_linux_solver_launch_uses_systemd_cpu_quota(tmp_path, monkeypatch):
    toolset = fake_workspace_toolset(tmp_path)
    toolset.solver_cpu_limit = 4
    monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
    monkeypatch.setattr(
        "decisionbrain.core.workspace_tools.shutil.which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None,
    )

    program, args = toolset._solver_launch_command(("--seed", "7"), platform_name="posix")

    assert program == "/usr/bin/systemd-run"
    assert "--property=CPUQuota=400%" in args
    assert args[-3:] == ("solver.py", "--seed", "7")
    assert not any("GRB_LICENSE_FILE" in arg for arg in args)
    assert not any("BindReadOnlyPaths" in arg for arg in args)


def test_linux_solver_launch_binds_only_explicit_gurobi_license(tmp_path, monkeypatch):
    toolset = fake_workspace_toolset(tmp_path)
    license_path = tmp_path / "gurobi.lic"
    license_path.write_text("test license", encoding="utf-8")
    monkeypatch.setenv("GRB_LICENSE_FILE", str(license_path))
    monkeypatch.setattr(
        "decisionbrain.core.workspace_tools.shutil.which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None,
    )

    program, args = toolset._solver_launch_command((), platform_name="posix")

    resolved = license_path.resolve()
    assert program == "/usr/bin/systemd-run"
    assert f"--property=BindReadOnlyPaths={resolved}:{resolved}" in args
    assert f"--setenv=GRB_LICENSE_FILE={resolved}" in args


def test_shell_uses_workdir_timeout_and_truncates_output(tmp_path):
    (tmp_path / "sub").mkdir()
    toolset = fake_workspace_toolset(tmp_path, timeout_s=1, max_output_chars=120)
    shell = tool_by_name(toolset, "shell").handler

    root_output = run(shell(command=f'{sys.executable} -c "import os; print(os.getcwd())"'))
    assert "Exit code: 0" in root_output
    assert str(tmp_path) in root_output

    output = run(
        shell(command=f'{sys.executable} -c "import os; print(os.getcwd())"', workdir="sub")
    )
    assert "Exit code: 0" in output
    assert str(tmp_path / "sub") in output

    truncated = run(shell(command=f"{sys.executable} -c \"print('x' * 1000)\"", timeout_ms=1000))
    assert "[truncated]" in truncated

    timed_out = run(
        shell(command=f'{sys.executable} -c "import time; time.sleep(2)"', timeout_ms=50)
    )
    assert "command timed out after 50 milliseconds" in timed_out

    absolute_workdir = run(shell(command="echo unreachable", workdir=str(tmp_path / "sub")))
    assert "absolute paths are not allowed" in absolute_workdir


def test_linux_trace_identifies_active_primary_solver_for_runtime_watchdog(tmp_path):
    solver = tmp_path / "solver.py"
    trace = (
        '100 execve("/bin/sh", ["/bin/sh", "-c", "python3 solver.py"], 0x0) = 0\n'
        '100 clone(child_stack=NULL, flags=SIGCHLD) = 101\n'
        '101 execve("/usr/bin/python3", ["python3", "solver.py"], 0x0) = 0\n'
    )

    parsed = _parse_linux_solver_trace(trace, cwd=tmp_path, solver=solver)

    assert parsed.solver_pids == frozenset({101})
    assert parsed.active_solver_pids == frozenset({101})
    assert not parsed.gnu_timeout_observed


def test_linux_trace_identifies_gnu_timeout_killing_primary_solver(tmp_path):
    solver = tmp_path / "solver.py"
    trace = (
        '200 execve("/usr/bin/timeout", ["timeout", "590", "python3", "solver.py"], 0x0) = 0\n'
        '200 clone(child_stack=NULL, flags=SIGCHLD) = 201\n'
        '201 execve("/usr/bin/python3", ["python3", "./solver.py"], 0x0) = 0\n'
        '201 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=200} ---\n'
        '201 +++ killed by SIGTERM +++\n'
        '200 +++ exited with 124 +++\n'
    )

    parsed = _parse_linux_solver_trace(trace, cwd=tmp_path, solver=solver)

    assert parsed.observed
    assert not parsed.active_solver_pids
    assert parsed.gnu_timeout_observed
    assert parsed.solver_killed_by == "SIGTERM"


def test_linux_trace_does_not_treat_compile_or_scratch_as_primary_solver(tmp_path):
    solver = tmp_path / "solver.py"
    trace = (
        '300 execve("/usr/bin/python3", ["python3", "-m", "py_compile", "solver.py"], 0x0) = 0\n'
        '301 execve("/usr/bin/python3", ["python3", "scratch/test_solver.py"], 0x0) = 0\n'
    )

    parsed = _parse_linux_solver_trace(trace, cwd=tmp_path, solver=solver)

    assert not parsed.observed


def test_missing_strace_is_an_optional_timeout_routing_capability(tmp_path, monkeypatch):
    toolset = fake_workspace_toolset(tmp_path)
    monkeypatch.setattr("decisionbrain.core.workspace_tools.shutil.which", lambda _name: None)

    assert run(toolset._usable_linux_strace()) is None
    assert toolset._linux_strace_probe_complete


def test_runtime_timeout_only_publishes_solution_changed_by_current_execution(tmp_path):
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "solution_schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["routes"],
                "properties": {"routes": {"type": "array"}},
            }
        ),
        encoding="utf-8",
    )
    solution = tmp_path / "solution.json"
    solution.write_text('{"routes": []}', encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    previous = toolset._file_fingerprint(solution)

    toolset._publish_runtime_solver_timeout(
        timeout_source="runtime_watchdog",
        timeout_seconds=1,
        elapsed_seconds=1.1,
        exit_code=-15,
        termination_signal="SIGTERM",
        stdout="",
        solution_before=previous,
    )
    unchanged = json.loads(
        (tmp_path / "runtime_solver_outcome.json").read_text(encoding="utf-8")
    )
    assert unchanged["status"] == "runtime_timeout"
    assert unchanged["solution_detected"] is False
    assert unchanged["solution_file"] is None

    time.sleep(0.001)
    solution.write_text('{"routes": [1]}', encoding="utf-8")
    toolset._publish_runtime_solver_timeout(
        timeout_source="runtime_watchdog",
        timeout_seconds=1,
        elapsed_seconds=1.1,
        exit_code=-15,
        termination_signal="SIGTERM",
        stdout="",
        solution_before=previous,
    )
    changed = json.loads(
        (tmp_path / "runtime_solver_outcome.json").read_text(encoding="utf-8")
    )
    assert changed["status"] == "runtime_timeout"
    assert changed["solution_detected"] is True
    assert changed["solution_schema_valid"] is True
    assert changed["solution_file"] == "solution.json"


def test_run_solver_timeout_ends_solving_instead_of_asking_for_another_turn(tmp_path):
    # A solver killed by its external time limit is a terminal fact, not one more
    # tool result. Handing it back as text leaves the model in the Solving loop,
    # where its only move is an outcome submission the timeout contract rejects.
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(SOLVER_SCHEMA, encoding="utf-8")
    (tmp_path / "solver.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path, timeout_s=0.3)
    run_solver = next(
        tool
        for tool in toolset.tools(stage=AgentStage.SOLVING, solving_outcome_submission=True)
        if tool.name == "run_solver"
    )

    with pytest.raises(StageForcedCompletion):
        run(run_solver.handler())

    outcome = json.loads((tmp_path / "runtime_solver_outcome.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "runtime_timeout"
    stage_output = json.loads((tmp_path / "stage_outputs/solving.json").read_text(encoding="utf-8"))
    assert stage_output["result"]["runtime_solver_outcome_file"] == "runtime_solver_outcome.json"
    receipt = json.loads((tmp_path / "solver_execution_receipt.json").read_text(encoding="utf-8"))
    assert receipt["timed_out"] is True
    assert (
        receipt["solver_sha256"]
        == hashlib.sha256((tmp_path / "solver.py").read_bytes()).hexdigest()
    )


def test_run_solver_timeout_stays_an_ordinary_result_without_terminal_routing(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(SOLVER_SCHEMA, encoding="utf-8")
    (tmp_path / "solver.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path, timeout_s=0.3)
    run_solver = stage_tool_by_name(toolset, AgentStage.SOLVING, "run_solver").handler

    result = run(run_solver())

    assert "Solver timed out" in result
    assert "written to solver_execution_receipt.json" in result


def test_shell_solver_timeout_publishes_the_receipt_the_terminal_contract_requires(
    tmp_path,
):
    # The shell timeout branch is Linux-and-strace only, so exercise the two
    # publications it performs. Before they were paired, a shell-launched solver
    # produced a runtime outcome with no receipt beside it, and the forced
    # completion was then rejected for an artifact the model never controlled.
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(SOLVER_SCHEMA, encoding="utf-8")
    (tmp_path / "solver.py").write_text("print('solve')\n", encoding="utf-8")
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

    receipt = json.loads((tmp_path / "solver_execution_receipt.json").read_text(encoding="utf-8"))
    assert receipt["argv"] == []
    assert receipt["timed_out"] is True
    for field, filename in (
        ("solver_sha256", "solver.py"),
        ("input_sha256", "input.json"),
    ):
        assert receipt[field] == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()


def test_shell_reports_inner_process_timing(tmp_path):
    events = []

    async def record_timing(event):
        events.append(event)

    toolset = fake_workspace_toolset(tmp_path, process_timing_handler=record_timing)
    result = run(
        toolset.shell(
            shell_command(sys.executable, "-c", "print('ok')"),
            stage=AgentStage.SOLVING,
        )
    )

    assert "Exit code: 0" in result
    assert len(events) == 1
    assert events[0]["kind"] == "shell_process"
    assert events[0]["stage"] == "solving"
    assert events[0]["duration_ms"] >= 0
    assert events[0]["exit_code"] == 0
    assert events[0]["workdir"] == "."
    assert events[0]["output_text"].strip() == "ok"
    assert events[0]["output_tail"].strip() == "ok"
    assert events[0]["output_truncated"] is False
    assert events[0]["solution_changed"] is False


def test_shell_tool_contract_requires_workspace_relative_paths(tmp_path):
    shell = tool_by_name(fake_workspace_toolset(tmp_path), "shell")

    assert "workspace root by default" in shell.description
    assert "workspace-relative paths and workdir" in shell.description
    assert "Directory-changing commands such as cd are rejected" in shell.description
    assert "do not cd to guessed absolute workspace paths" in shell.description
    assert (
        "relative to the workspace root" in shell.parameters["properties"]["workdir"]["description"]
    )
    assert (
        "absolute paths are not allowed" in shell.parameters["properties"]["workdir"]["description"]
    )


def test_shell_rejects_directory_change_commands_and_uses_relative_workdir(tmp_path):
    (tmp_path / "sub").mkdir()
    toolset = fake_workspace_toolset(tmp_path)
    shell = tool_by_name(toolset, "shell").handler

    for command in (
        "cd /workspace && pwd",
        "printf ready && cd sub",
        "echo ready; pushd sub",
        "Set-Location sub",
    ):
        result = run(shell(command=command))
        assert "directory-changing commands are not allowed" in result
        assert "Use relative workdir" in result

    allowed = run(shell(command=shell_command(sys.executable, "-c", "print('ok')"), workdir="sub"))
    assert "Exit code: 0" in allowed
    assert "ok" in allowed

    plain_text = run(shell(command=shell_command(sys.executable, "-c", "print('mention cd only')")))
    assert "Exit code: 0" in plain_text
    assert "mention cd only" in plain_text


def test_shell_timeout_terminates_nested_process_tree(tmp_path):
    parent, child_pid = write_nested_sleep_scripts(tmp_path)
    shell = tool_by_name(fake_workspace_toolset(tmp_path, timeout_s=5), "shell").handler

    result = run(
        shell(
            command=shell_command(sys.executable, str(parent)),
            timeout_ms=500,
        )
    )

    assert "command timed out after 500 milliseconds" in result
    assert child_pid.exists()
    assert wait_for_pid_exit(int(child_pid.read_text(encoding="utf-8")))


def test_shell_cancellation_terminates_nested_process_tree(tmp_path):
    parent, child_pid = write_nested_sleep_scripts(tmp_path)
    shell = tool_by_name(fake_workspace_toolset(tmp_path, timeout_s=30), "shell").handler

    async def cancel_running_shell() -> int:
        task = asyncio.create_task(shell(command=shell_command(sys.executable, str(parent))))
        for _ in range(100):
            if child_pid.exists():
                break
            await asyncio.sleep(0.02)
        assert child_pid.exists()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled shell task did not raise CancelledError")
        return int(child_pid.read_text(encoding="utf-8"))

    pid = run(cancel_running_shell())
    assert wait_for_pid_exit(pid)


def test_replace_in_file_exact_match(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    replace = tool_by_name(toolset, "replace_in_file").handler

    result = run(replace(path="notes.txt", old_string="beta", new_string="bravo"))

    assert "replace_in_file" in result
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "alpha\nbravo\ngamma\n"


def test_replace_in_file_zero_matches_shows_hint(tmp_path):
    (tmp_path / "notes.txt").write_text("  alpha\n  beta\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    replace = tool_by_name(toolset, "replace_in_file").handler

    result = run(replace(path="notes.txt", old_string="gamma", new_string="bravo"))

    assert "old_string not found" in result
    # File unchanged
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "  alpha\n  beta\n"


def test_replace_in_file_multiple_matches(tmp_path):
    (tmp_path / "notes.txt").write_text("beta\nalpha\nbeta\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    replace = tool_by_name(toolset, "replace_in_file").handler

    result = run(replace(path="notes.txt", old_string="beta", new_string="bravo"))

    assert "matches 2 times" in result
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "beta\nalpha\nbeta\n"


def test_replace_in_file_rejects_absolute_and_escape_paths(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    replace = tool_by_name(toolset, "replace_in_file").handler

    assert "escapes working directory" in run(
        replace(path="../outside.txt", old_string="secret", new_string="x")
    )
    assert "absolute paths are not allowed" in run(
        replace(path=str(outside), old_string="secret", new_string="x")
    )
    assert outside.read_text(encoding="utf-8") == "secret\n"


def test_delete_file(tmp_path):
    (tmp_path / "temp.txt").write_text("remove me\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    delete = tool_by_name(toolset, "delete_file").handler

    result = run(delete(path="temp.txt"))

    assert "deleted" in result
    assert not (tmp_path / "temp.txt").exists()


def test_delete_file_rejects_escape_and_nonexistent(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    delete = tool_by_name(toolset, "delete_file").handler

    assert "escapes working directory" in run(delete(path="../outside.txt"))
    assert "does not exist" in run(delete(path="nonexistent.txt"))


def test_stage_write_policy_blocks_non_owner_write_file(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    write = stage_tool_by_name(toolset, AgentStage.SOLVING, "write_file").handler

    result = run(
        write(path="stage_outputs/algorithm_design.json", content='{"problem_family":"routing"}')
    )

    assert "not allowed to write stage_outputs/algorithm_design.json" in result
    assert not (tmp_path / "stage_outputs/algorithm_design.json").exists()


def test_stage_write_policy_allows_owner_write_file(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    write = stage_tool_by_name(toolset, AgentStage.ALGORITHM_DESIGN, "write_file").handler

    result = run(
        write(path="stage_outputs/algorithm_design.json", content='{"problem_family":"routing"}')
    )

    assert result == "wrote 28 bytes to stage_outputs/algorithm_design.json"
    assert (tmp_path / "stage_outputs/algorithm_design.json").is_file()


def test_algorithm_design_cannot_read_problem_contract_outputs(tmp_path):
    (tmp_path / "stage_outputs").mkdir()
    (tmp_path / "stage_outputs/problem_contract.json").write_text("{}", encoding="utf-8")
    (tmp_path / "input_schema.json").write_text("{}", encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feasibility_checker.py").write_text("def check(): pass\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    tool_names = [tool.name for tool in toolset.tools(stage=AgentStage.ALGORITHM_DESIGN)]
    read = stage_tool_by_name(toolset, AgentStage.ALGORITHM_DESIGN, "read_file").handler
    list_files = stage_tool_by_name(toolset, AgentStage.ALGORITHM_DESIGN, "list_files").handler

    assert "shell" in tool_names
    assert "not allowed to read stage_outputs/problem_contract.json" in run(
        read(path="stage_outputs/problem_contract.json")
    )
    assert "not allowed to read input_schema.json" in run(read(path="input_schema.json"))
    listing = run(list_files(path=".", recursive=True))
    assert "stage_outputs/problem_contract.json" not in listing
    assert "input_schema.json" not in listing
    assert "solution_schema.json" not in listing
    assert "feasibility_checker.py" not in listing


def test_problem_contract_write_policy_blocks_other_stage_contract_files(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    solving_write = stage_tool_by_name(toolset, AgentStage.SOLVING, "write_file").handler
    contract_write = stage_tool_by_name(toolset, AgentStage.PROBLEM_CONTRACT, "write_file").handler

    blocked = run(solving_write(path="input_schema.json", content='{"type": "object"}'))
    allowed = run(contract_write(path="input_schema.json", content='{"type": "object"}'))

    assert "not allowed to write input_schema.json" in blocked
    assert allowed == "wrote 18 bytes to input_schema.json"


def test_no_stage_may_write_the_checker_result_and_solving_cannot_read_checker(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text("# audited\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    design_write = stage_tool_by_name(toolset, AgentStage.ALGORITHM_DESIGN, "write_file").handler
    solving_write = stage_tool_by_name(toolset, AgentStage.SOLVING, "write_file").handler
    solving_read = stage_tool_by_name(toolset, AgentStage.SOLVING, "read_file").handler
    solving_shell = stage_tool_by_name(toolset, AgentStage.SOLVING, "shell").handler
    reviewer_write = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "write_file"
    ).handler

    blocked = run(design_write(path="input.json", content="{}"))
    checker_read = run(solving_read(path="feasibility_checker.py"))
    checker_shell_view = run(
        solving_shell(
            command=(
                f"{sys.executable} -c \"from pathlib import Path; "
                "print(Path('feasibility_checker.py').exists())\""
            )
        )
    )
    result_write = run(solving_write(path="feasibility_result.json", content="{}"))
    reviewer_result_write = run(reviewer_write(path="feasibility_result.json", content="{}"))
    receipt_write = run(reviewer_write(path="feasibility_check_receipt.json", content="{}"))

    assert "not allowed to write input.json" in blocked
    assert "not allowed to read feasibility_checker.py" in checker_read
    assert "False" in checker_shell_view
    assert (tmp_path / "feasibility_checker.py").read_text(encoding="utf-8") == "# audited\n"
    assert "not allowed to write feasibility_result.json" in result_write
    # Only run_feasibility_checker may create review receipts; Review cannot self-author evidence.
    assert "not allowed to write feasibility_result.json" in reviewer_result_write
    assert "not allowed to write feasibility_check_receipt.json" in receipt_write


def test_read_file_declares_the_output_cap_and_silent_middle_truncation(tmp_path):
    toolset = fake_workspace_toolset(tmp_path, max_output_chars=500)
    read_file = stage_tool_by_name(toolset, AgentStage.SOLVING, "read_file")

    description = read_file.description
    max_chars = read_file.parameters["properties"]["max_chars"]["description"]

    # Middle truncation is non-fatal, so the marker is the model's only signal.
    assert "500 characters" in description
    assert TRUNCATION_NOTICE.strip() in description
    assert "the middle is dropped and no error is raised" in description
    # The limit is 80 times the whole-file threshold; document both to require chunking.
    assert str(toolset.max_read_bytes) in description
    assert "only lowers" in max_chars
    assert "12000" not in description  # 数值跟随配置，不得写死


def test_truncated_read_keeps_head_and_tail_without_erroring(tmp_path):
    body = "".join(f"LINE{index:04d}\n" for index in range(400))
    (tmp_path / "solver.py").write_text(body, encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path, max_output_chars=500)
    read_file = stage_tool_by_name(toolset, AgentStage.SOLVING, "read_file").handler

    out = run(read_file(path="solver.py"))

    assert not out.startswith("read_file error")
    assert TRUNCATION_NOTICE.strip() in out
    assert "LINE0000" in out and "LINE0399" in out
    assert "LINE0200" not in out  # 中段丢失而非尾部截断


def test_review_history_is_readable_only_by_the_reviewer(tmp_path):
    write_feasibility_review_history(
        tmp_path,
        [
            {
                "attempt": 1,
                "responsibility": "solving",
                "feasibility_review": {
                    "evidence": [{"source": "feasibility_result.json", "finding": "容量超限"}]
                },
                "feasibility_check": {"feasible": False, "violations": []},
            }
        ],
    )
    toolset = fake_workspace_toolset(tmp_path)
    original = (tmp_path / FEASIBILITY_REVIEW_HISTORY_FILE).read_text(encoding="utf-8")
    exists_probe = (
        f"{sys.executable} -c \"from pathlib import Path; "
        f"print(Path('{FEASIBILITY_REVIEW_HISTORY_FILE}').exists())\""
    )

    for stage in (AgentStage.SOLVING, AgentStage.ALGORITHM_DESIGN):
        read = stage_tool_by_name(toolset, stage, "read_file").handler
        listing = stage_tool_by_name(toolset, stage, "list_files").handler
        search = stage_tool_by_name(toolset, stage, "search_file").handler
        shell = stage_tool_by_name(toolset, stage, "shell").handler

        assert f"not allowed to read {FEASIBILITY_REVIEW_HISTORY_FILE}" in run(
            read(path=FEASIBILITY_REVIEW_HISTORY_FILE)
        )
        assert FEASIBILITY_REVIEW_HISTORY_FILE not in run(listing(path="."))
        assert "容量超限" not in run(search(query="容量超限", path="."))
        # Recursive rules do not participate in shell isolation; keep a single-file rule.
        assert "False" in run(shell(command=exists_probe))

    reviewer_read = stage_tool_by_name(toolset, AgentStage.FEASIBILITY_REVIEW, "read_file").handler
    assert "容量超限" in run(reviewer_read(path=FEASIBILITY_REVIEW_HISTORY_FILE))
    # Core writes history directly; stages cannot rewrite it.
    for stage in (AgentStage.SOLVING, AgentStage.FEASIBILITY_REVIEW):
        write = stage_tool_by_name(toolset, stage, "write_file").handler
        assert f"not allowed to write {FEASIBILITY_REVIEW_HISTORY_FILE}" in run(
            write(path=FEASIBILITY_REVIEW_HISTORY_FILE, content="tampered")
        )
    assert (tmp_path / FEASIBILITY_REVIEW_HISTORY_FILE).read_text(encoding="utf-8") == original


def test_review_history_keeps_counts_and_constraints_while_sampling_findings(tmp_path):
    violations = [
        {"constraint": "capacity" if index % 3 else "time_window", "message": f"v{index}"}
        for index in range(6888)
    ]
    write_feasibility_review_history(
        tmp_path,
        [
            {
                "attempt": 1,
                "feasibility_check": {
                    "feasible": False,
                    "objective_value": -8.5,
                    "violations": violations,
                    "warnings": [],
                },
            }
        ],
    )

    record = json.loads((tmp_path / FEASIBILITY_REVIEW_HISTORY_FILE).read_text(encoding="utf-8"))
    check = record[0]["feasibility_check"]

    assert len(check["violations"]) == REVIEW_HISTORY_VIOLATION_SAMPLE
    assert check["violations_count"] == 6888
    assert check["violations_truncated"] is True
    # Count trends and constraint profiles distinguish convergence from stagnation.
    assert check["violations_constraints"] == {"capacity": 4592, "time_window": 2296}
    assert check["objective_value"] == -8.5
    assert check["warnings_count"] == 0
    assert check["warnings_truncated"] is False


def test_only_the_reviewer_gets_the_dedicated_checker_tool(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)

    solving_names = {tool.name for tool in toolset.tools(stage=AgentStage.SOLVING)}
    reviewer_names = {
        tool.name for tool in toolset.tools(stage=AgentStage.FEASIBILITY_REVIEW)
    }

    assert "run_feasibility_checker" not in solving_names
    assert "run_feasibility_checker" in reviewer_names
    assert "shell" in reviewer_names


CHECKER_SOURCE = """import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--solution", default="solution.json")
parser.add_argument("--output", default="feasibility_result.json")
parser.add_argument("--feasible", default="true")
parser.add_argument("--also-write", default="")
args = parser.parse_args()

if args.also_write:
    with open(args.also_write, "w", encoding="utf-8") as handle:
        handle.write("tampered\\n")

feasible = args.feasible == "true"
with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_valid": True,
            "feasible": feasible,
            "objective_value": 12.5,
            "violations": [] if feasible else ["capacity"],
            "warnings": [],
        },
        handle,
    )
sys.exit(0 if feasible else 1)
"""


def _checker_workspace(tmp_path):
    (tmp_path / "feasibility_checker.py").write_text(CHECKER_SOURCE, encoding="utf-8")
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": []}', encoding="utf-8")
    return fake_workspace_toolset(tmp_path)


def test_solver_and_checker_execution_receipts_smoke(tmp_path):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(
        '{"type": "object", "required": ["routes"], '
        '"properties": {"routes": {"type": "array"}}}',
        encoding="utf-8",
    )
    (tmp_path / "solver.py").write_text(
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "feasibility_checker.py").write_text(CHECKER_SOURCE, encoding="utf-8")
    (tmp_path / "solution.json").write_text('{"routes": ["stale"]}', encoding="utf-8")
    (tmp_path / "solver_execution_receipt.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feasibility_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feasibility_check_receipt.json").write_text("{}", encoding="utf-8")

    toolset = fake_workspace_toolset(tmp_path)
    run_solver = stage_tool_by_name(toolset, AgentStage.SOLVING, "run_solver").handler
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler

    solver_result = run(run_solver())
    assert "Exit code: 0" in solver_result
    assert "written to solver_execution_receipt.json" in solver_result
    assert json.loads((tmp_path / "solution.json").read_text(encoding="utf-8")) == {
        "routes": []
    }

    checker_result = run(run_checker(args=["--solution", "solution.json"]))
    assert "Exit code: 0" in checker_result
    assert "written to feasibility_check_receipt.json" in checker_result

    for receipt_name, fields in (
        (
            "solver_execution_receipt.json",
            (
                ("input_sha256", "input.json"),
                ("solver_sha256", "solver.py"),
                ("solution_sha256", "solution.json"),
            ),
        ),
        (
            "feasibility_check_receipt.json",
            (
                ("checker_sha256", "feasibility_checker.py"),
                ("input_sha256", "input.json"),
                ("solution_sha256", "solution.json"),
                ("result_sha256", "feasibility_result.json"),
            ),
        ),
    ):
        receipt = json.loads((tmp_path / receipt_name).read_text(encoding="utf-8"))
        for field, filename in fields:
            assert receipt[field] == hashlib.sha256(
                (tmp_path / filename).read_bytes()
            ).hexdigest()


SOLVER_SCHEMA = (
    '{"type": "object", "required": ["routes"], '
    '"properties": {"routes": {"type": "array"}}}'
)


def _solver_workspace(tmp_path, solver_body):
    (tmp_path / "input.json").write_text('{"orders": []}', encoding="utf-8")
    (tmp_path / "solution_schema.json").write_text(SOLVER_SCHEMA, encoding="utf-8")
    (tmp_path / "solver.py").write_text(solver_body, encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    return stage_tool_by_name(toolset, AgentStage.SOLVING, "run_solver").handler


def test_run_solver_reports_a_bound_candidate(tmp_path):
    run_solver = _solver_workspace(
        tmp_path,
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
    )

    result = run(run_solver())

    assert "written to solver_execution_receipt.json" in result
    assert "candidate bound to solution.json" in result


def test_run_solver_reports_that_no_candidate_was_produced(tmp_path):
    run_solver = _solver_workspace(tmp_path, "print('no candidate within the budget')\n")

    result = run(run_solver())

    assert "Exit code: 0" in result
    assert "no candidate was produced" in result
    receipt = json.loads(
        (tmp_path / "solver_execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["solution_file"] is None
    assert receipt["solution_sha256"] is None


def test_run_solver_reports_schema_violations_instead_of_silently_dropping_them(tmp_path):
    # Match run_feasibility_checker by reporting invalid artifacts immediately.
    run_solver = _solver_workspace(
        tmp_path,
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"trips\": 3}', encoding='utf-8')\n",
    )

    result = run(run_solver())

    assert "Exit code: 0" in result
    assert "solution.json does not match solution_schema.json" in result
    assert "'routes' is a required property" in result
    assert "Fix solver.py" in result
    receipt = json.loads(
        (tmp_path / "solver_execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["solution_file"] is None


def test_run_solver_reports_an_unusable_solution_schema(tmp_path):
    run_solver = _solver_workspace(
        tmp_path,
        "from pathlib import Path\n"
        "Path('solution.json').write_text('{\"routes\": []}', encoding='utf-8')\n",
    )
    (tmp_path / "solution_schema.json").write_text('{"type": 17}', encoding="utf-8")

    result = run(run_solver())

    assert "solution_schema.json could not be used to validate it" in result


def test_run_feasibility_checker_writes_a_receipt_binding_the_four_inputs(tmp_path):
    toolset = _checker_workspace(tmp_path)
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler

    result = run(run_checker(args=["--solution", "solution.json"]))

    assert "Exit code: 0" in result
    assert "written to feasibility_check_receipt.json" in result
    receipt = json.loads(
        (tmp_path / "feasibility_check_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["outcome_source"] == "runtime_checker_execution"
    assert receipt["argv"] == ["--solution", "solution.json"]
    for field, name in (
        ("checker_sha256", "feasibility_checker.py"),
        ("input_sha256", "input.json"),
        ("solution_sha256", "solution.json"),
        ("result_sha256", "feasibility_result.json"),
    ):
        expected = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert receipt[field] == expected


def test_run_feasibility_checker_keeps_the_receipt_for_a_nonzero_infeasible_exit(tmp_path):
    # A checker may use a nonzero exit code for infeasibility; this is not execution failure.
    toolset = _checker_workspace(tmp_path)
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler

    result = run(run_checker(args=["--feasible", "false"]))

    assert "Exit code: 1" in result
    assert "written to feasibility_check_receipt.json" in result
    receipt = json.loads(
        (tmp_path / "feasibility_check_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["exit_code"] == 1


def test_run_feasibility_checker_discards_a_stale_receipt_when_the_result_is_invalid(tmp_path):
    toolset = _checker_workspace(tmp_path)
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler
    run(run_checker(args=[]))
    assert (tmp_path / "feasibility_check_receipt.json").is_file()

    # Invalid checker output must invalidate the old receipt.
    (tmp_path / "feasibility_checker.py").write_text(
        "import json\n"
        "with open('feasibility_result.json', 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'schema_valid': True}, handle)\n",
        encoding="utf-8",
    )
    result = run(run_checker(args=[]))

    assert "not written" in result
    assert not (tmp_path / "feasibility_check_receipt.json").exists()


def test_run_feasibility_checker_rolls_back_writes_outside_the_result_file(tmp_path):
    toolset = _checker_workspace(tmp_path)
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler

    result = run(run_checker(args=["--also-write", "solution.json"]))

    assert "modified protected files and the changes were restored" in result
    assert (tmp_path / "solution.json").read_text(encoding="utf-8") == '{"routes": []}'
    assert not (tmp_path / "feasibility_check_receipt.json").exists()


def test_run_feasibility_checker_rejects_non_string_arguments(tmp_path):
    toolset = _checker_workspace(tmp_path)
    run_checker = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "run_feasibility_checker"
    ).handler

    assert "args must be an array of strings" in run(run_checker(args=["--solution", 3]))


def test_reviewer_shell_cannot_rewrite_the_evidence_it_judges(tmp_path):
    toolset = _checker_workspace(tmp_path)
    reviewer_shell = stage_tool_by_name(
        toolset, AgentStage.FEASIBILITY_REVIEW, "shell"
    ).handler
    (tmp_path / "solver.py").write_text("original\n", encoding="utf-8")

    result = run(
        reviewer_shell(
            command=(
                f"{sys.executable} -c \"from pathlib import Path; "
                "Path('solution.json').write_text('{}'); "
                "Path('feasibility_result.json').write_text('{}')\""
            )
        )
    )

    assert "unauthorized protected file modification restored" in result
    assert (tmp_path / "solution.json").read_text(encoding="utf-8") == '{"routes": []}'
    assert not (tmp_path / "feasibility_result.json").exists()


def test_solving_self_check_view_restores_branch9_checker_permissions(tmp_path):
    checker = tmp_path / "feasibility_checker.py"
    checker.write_text("# audited\n", encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)

    default_read = stage_tool_by_name(toolset, AgentStage.SOLVING, "read_file").handler
    default_write = stage_tool_by_name(toolset, AgentStage.SOLVING, "write_file").handler
    self_check_tools = toolset.tools(
        stage=AgentStage.SOLVING,
        solving_feasibility_self_check=True,
    )
    self_check_read = next(tool for tool in self_check_tools if tool.name == "read_file").handler
    self_check_write = next(tool for tool in self_check_tools if tool.name == "write_file").handler

    assert "not allowed to read feasibility_checker.py" in run(
        default_read(path="feasibility_checker.py")
    )
    assert "not allowed to write feasibility_result.json" in run(
        default_write(path="feasibility_result.json", content="{}")
    )
    assert run(self_check_read(path="feasibility_checker.py")).strip() == "# audited"
    assert run(self_check_write(path="feasibility_result.json", content="{}")) == (
        "wrote 2 bytes to feasibility_result.json"
    )


def test_stage_write_policy_blocks_replace_in_file_to_other_stage_output(tmp_path):
    target = tmp_path / "stage_outputs/algorithm_design.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"problem_family":"routing"}\n', encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    replace = stage_tool_by_name(toolset, AgentStage.SOLVING, "replace_in_file").handler

    result = run(
        replace(
            path="stage_outputs/algorithm_design.json",
            old_string='{"problem_family":"routing"}',
            new_string='{"problem_family":"assignment"}',
        )
    )

    assert "not allowed to write stage_outputs/algorithm_design.json" in result
    assert target.read_text(encoding="utf-8") == '{"problem_family":"routing"}\n'


def test_stage_write_policy_restores_shell_changes_to_other_stage_output(tmp_path):
    target = tmp_path / "stage_outputs/algorithm_design.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"problem_family":"routing"}\n', encoding="utf-8")
    toolset = fake_workspace_toolset(tmp_path)
    shell = stage_tool_by_name(toolset, AgentStage.SOLVING, "shell").handler

    result = run(
        shell(
            command=(
                f"{sys.executable} -c "
                '"from pathlib import Path; '
                "Path('stage_outputs/algorithm_design.json').write_text('{\\\"bad\\\": true}')\""
            )
        )
    )

    assert "unauthorized protected file modification restored" in result
    assert target.read_text(encoding="utf-8") == '{"problem_family":"routing"}\n'


def test_problem_md_and_data_are_protected_but_currently_unrestricted(tmp_path):
    toolset = fake_workspace_toolset(tmp_path)
    write = stage_tool_by_name(toolset, AgentStage.EXPLANATION, "write_file").handler

    problem_result = run(write(path="problem.md", content="problem"))
    data_result = run(write(path="data/input.csv", content="id\n1\n"))

    assert problem_result == "wrote 7 bytes to problem.md"
    assert data_result == "wrote 5 bytes to data/input.csv"
