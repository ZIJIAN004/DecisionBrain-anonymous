import asyncio

from decisionbrain.runtime.timing import RunTiming, classify_shell_process


def record(timing: RunTiming, event: dict) -> None:
    asyncio.run(timing.record(event))


def test_run_timing_separates_tool_and_child_process_durations():
    timing = RunTiming()
    timing.stage_started("solving")
    record(
        timing,
        {
            "kind": "llm_call",
            "stage": "solving",
            "duration_ms": 12,
            "retry_count": 2,
            "retry_backoff_ms": 30,
        },
    )
    record(
        timing,
        {"kind": "tool_call", "stage": "solving", "tool_name": "shell", "duration_ms": 100},
    )
    record(
        timing,
        {
            "kind": "shell_process",
            "stage": "solving",
            "command": "python feasibility_checker.py --input input.json",
            "duration_ms": 34,
        },
    )
    record(
        timing,
        {
            "kind": "shell_process",
            "stage": "solving",
            "command": "python -u solver.py --seed 7",
            "duration_ms": 56,
        },
    )
    timing.stage_finished("solving")

    payload = timing.to_dict()

    assert payload["schema_version"] == "1.2"
    assert payload["stages"][0]["status"] == "completed"
    assert payload["summary"]["llm_duration_ms"] == {"solving": 12}
    assert payload["summary"]["llm_retry_count"] == 2
    assert payload["summary"]["llm_retry_backoff_ms"] == 30
    assert payload["summary"]["tool_duration_ms"] == {"shell": 100}
    assert payload["summary"]["self_checker_duration_ms"] == 34
    assert payload["summary"]["solver_execution_duration_ms"] == 56
    assert payload["process_calls"][0]["execution_kind"] == "self_checker"
    assert payload["process_calls"][1]["execution_kind"] == "solver"


def test_run_timing_closes_unfinished_stage_and_marks_composite_command_ambiguous():
    timing = RunTiming()
    timing.stage_started("feasibility_review")
    record(
        timing,
        {
            "kind": "shell_process",
            "stage": "feasibility_review",
            "command": "python solver.py && python feasibility_checker.py",
            "duration_ms": 90,
        },
    )

    payload = timing.to_dict()

    assert payload["stages"][0]["status"] == "incomplete"
    assert payload["stages"][0]["duration_ms"] >= 0
    assert payload["summary"]["incomplete_stage_count"] == 1
    assert payload["summary"]["solver_execution_count"] == 0
    assert payload["summary"]["self_checker_execution_count"] == 0
    assert payload["summary"]["ambiguous_process_count"] == 1


def test_shell_process_classifier_requires_standalone_python_script():
    assert classify_shell_process("python3 solver.py") == ("solver", False)
    assert classify_shell_process("python -u feasibility_checker.py --output out.json") == (
        "self_checker",
        False,
    )
    assert classify_shell_process("cat solver.py") == ("ambiguous", True)
    assert classify_shell_process("python -c \"print('solver.py')\"") == ("ambiguous", True)
