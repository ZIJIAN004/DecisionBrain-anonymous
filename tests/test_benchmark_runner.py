from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import decisionbrain.benchmark.execution as runner_module
from decisionbrain.algorithm_library.runtime_validation import (
    AlgorithmRuntimeValidationError,
    RuntimeValidationFailure,
)
from decisionbrain.benchmark.runner import FrontierORReport, TaskEvaluation, run_benchmark
from decisionbrain.benchmark.scoring import TaskScore
from decisionbrain.benchmark.task import (
    BenchmarkTask,
    BenchmarkTaskError,
    discover_tasks,
    load_large_max_tasks,
    normalize_large_suite_task_ids,
)
from decisionbrain.config import Settings
from decisionbrain.core.models import AgentStage, AgentState, StageCompleted
from decisionbrain.core.package_policy import PackagePolicy
from decisionbrain.core.stage_flow import apply_stage_result


def test_run_observer_latches_resource_exhaustion_from_recoverable_tool_error():
    observer = runner_module._RunObserver()
    event = SimpleNamespace(
        run_id="20260101-000000Z-abcdef01",
        type="agent_error",
        stage=SimpleNamespace(value="solving"),
        message="tool shell failed",
        payload={
            "flow_section": "solving",
            "error_kind": "tool_exception",
            "message": "OSError: [WinError 1455] 页面文件太小，无法完成操作。",
        },
    )

    asyncio.run(observer.handle(event))
    diagnostics = observer.error_diagnostics()

    assert diagnostics["resource_exhaustion_detected"] is True
    assert "WinError 1455" in diagnostics["resource_exhaustion_error"]


def test_run_observer_emits_run_and_error_events_with_run_context():
    task_events = []
    observer = runner_module._RunObserver(task_event_callback=task_events.append)
    event = SimpleNamespace(
        run_id="20260101-000000Z-abcdef01",
        type="agent_error",
        stage=SimpleNamespace(value="solving"),
        message="LLM request failed",
        payload={
            "flow_section": "solving",
            "error_kind": "llm_request_failure",
            "message": "LLM HTTP 400: context length exceeded",
        },
    )

    asyncio.run(observer.handle(event))

    assert task_events == [
        {"event": "run_created", "run_id": "20260101-000000Z-abcdef01"},
        {
            "event": "agent_error",
            "run_id": "20260101-000000Z-abcdef01",
            "stage": "solving",
            "error_kind": "llm_request_failure",
            "error": "LLM HTTP 400: context length exceeded",
        },
    ]


def test_task_progress_error_names_task_run_stage_and_error():
    message = runner_module._format_task_progress(
        {
            "event": "agent_error",
            "index": 14,
            "total": 61,
            "task_id": "desaulniers2010-large-5",
            "run_id": "20260101-000000Z-abcdef01",
            "stage": "solving",
            "error_kind": "llm_request_failure",
            "error": "LLM HTTP 400: context length exceeded",
        }
    )

    assert message == (
        "[14/61] ERROR task=desaulniers2010-large-5 "
        "run_id=20260101-000000Z-abcdef01 stage=solving "
        "kind=llm_request_failure message=LLM HTTP 400: context length exceeded"
    )


def test_score_failure_detail_explains_infeasible_checker_result():
    score = TaskScore(
        task_id="sample",
        problem_class="routing",
        executed=True,
        feasible=False,
        violated_constraints=("capacity exceeded",),
    )

    assert runner_module._score_failure_detail(score) == (
        "checker reported infeasible: capacity exceeded"
    )


def test_declared_infeasible_passes_only_for_reference_infeasible_task():
    correct = TaskScore(
        task_id="known-infeasible",
        problem_class="scheduling",
        executed=True,
        feasible=False,
        declared_infeasible=True,
        reference_feasible=False,
    )
    incorrect = TaskScore(
        task_id="known-feasible",
        problem_class="scheduling",
        executed=True,
        feasible=False,
        declared_infeasible=True,
        reference_feasible=True,
    )

    assert correct.passed is True
    assert runner_module._score_failure_detail(correct) == ""
    assert incorrect.passed is False
    assert runner_module._score_failure_detail(incorrect) == (
        "agent incorrectly declared infeasible"
    )


def test_report_marks_frontieror_direct_contract(tmp_path: Path):
    task = BenchmarkTask(
        task_id="sample",
        root=tmp_path,
        metadata={
            "problem_class": "routing",
            "formulation_type": "MIP",
            "category": "operational",
            "direction": "min",
            "instance": "large-max",
        },
    )
    score = TaskScore(
        task_id="sample",
        problem_class="routing",
        executed=True,
        feasible=True,
        objective=12,
        reference_objective=10,
        gap=0.2,
    )
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    report = FrontierORReport(
        evaluations=[
            TaskEvaluation(
                task=task,
                score=score,
                run_id="20260101-000000Z-abcdef01",
                run_status="succeeded",
                duration_seconds=1.25,
                error_diagnostics={
                    "total_errors": 2,
                    "flow_section_counts": {"solving": 2},
                    "error_kind_counts": {"tool_nonzero_exit": 2},
                    "most_repeated_flow_section": "solving",
                    "most_repeated_flow_section_count": 2,
                },
                timing_summary={
                    "wall_clock_duration_ms": 1250,
                    "stage_duration_ms": {"solving": 1000},
                    "completed_stage_count": 1,
                    "incomplete_stage_count": 0,
                    "llm_duration_ms": {"solving": 500},
                    "llm_call_count": 2,
                    "llm_retry_count": 1,
                    "llm_retry_backoff_ms": 100,
                    "tool_duration_ms": {"shell": 400},
                    "tool_call_count": 3,
                    "self_checker_duration_ms": 200,
                    "self_checker_execution_count": 1,
                    "solver_execution_duration_ms": 300,
                    "solver_execution_count": 1,
                    "ambiguous_process_duration_ms": 0,
                    "ambiguous_process_count": 0,
                },
                successful_algorithm_packages=(
                    {
                        "package_id": "ortools",
                        "solver_id": "cp_sat",
                        "version": "9.12.4544",
                    },
                ),
            )
        ],
        started_at=started,
        finished_at=started + timedelta(seconds=2),
        model="test-model",
        jobs=1,
        llm_max_attempts=8,
    ).to_dict()

    assert report["summary"]["direct_contract"] == 1
    assert report["summary"]["feasible"] == 1
    assert report["feasibility_review_enabled"] is True
    assert report["timing_enabled"] is True
    assert report["algorithm_package_stats_enabled"] is True
    assert report["suite"] == "large-max"
    assert report["summary"]["tasks_with_errors"] == 1
    assert report["summary"]["most_error_prone_flow_section"] == "solving"
    assert report["summary"]["most_error_prone_flow_section_count"] == 2
    assert report["retry_policy"] == {
        "llm_max_attempts": 8,
    }
    assert report["tasks"][0]["contract_mode"] == "frontieror_direct"
    assert report["tasks"][0]["instance_variant"] == "large-max"
    assert report["tasks"][0]["formulation_type"] == "MIP"
    assert report["tasks"][0]["category"] == "operational"
    assert report["tasks"][0]["reference_feasible"] is True
    assert report["tasks"][0]["gap"] == 0.2
    assert report["tasks"][0]["timing"]["self_checker_duration_ms"] == 200
    assert report["tasks"][0]["successful_algorithm_packages"] == [
        {"package_id": "ortools", "solver_id": "cp_sat", "version": "9.12.4544"}
    ]
    assert report["summary"]["algorithm_package_success"] == {
        "hidden_feasible_task_count": 1,
        "hidden_feasible_tasks_with_declared_packages": 1,
        "package_solver_usages": [
            {
                "package_id": "ortools",
                "solver_id": "cp_sat",
                "version": "9.12.4544",
                "successful_task_count": 1,
                "successful_tasks": ["sample"],
            }
        ],
    }
    assert report["summary"]["timing"] == {
        "timing_available_task_count": 1,
        "timing_missing_task_count": 0,
        "runtime_task_duration_sum_ms": 1250,
        "stage_duration_ms": {"solving": 1000},
        "completed_stage_count": 1,
        "incomplete_stage_count": 0,
        "llm_duration_ms": {"solving": 500},
        "llm_call_count": 2,
        "llm_retry_count": 1,
        "llm_retry_backoff_ms": 100,
        "tool_duration_ms": {"shell": 400},
        "tool_call_count": 3,
        "self_checker_duration_ms": 200,
        "self_checker_execution_count": 1,
        "solver_execution_duration_ms": 300,
        "solver_execution_count": 1,
        "ambiguous_process_duration_ms": 0,
        "ambiguous_process_count": 0,
        "benchmark_wall_clock_duration_ms": 2000,
    }


def test_report_counts_infeasibility_declarations(tmp_path: Path):
    tasks = [
        BenchmarkTask(
            task_id=task_id,
            root=tmp_path / task_id,
            metadata={
                "problem_class": "scheduling",
                "direction": "min",
                "reference_feasible": reference_feasible,
            },
        )
        for task_id, reference_feasible in (("correct", False), ("incorrect", True))
    ]
    evaluations = [
        TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=True,
                feasible=False,
                declared_infeasible=True,
                reference_feasible=task.reference_feasible,
            ),
            run_id=f"run-{task.task_id}",
            run_status="succeeded",
            duration_seconds=1,
        )
        for task in tasks
    ]
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payload = FrontierORReport(
        evaluations=evaluations,
        started_at=started,
        finished_at=started + timedelta(seconds=2),
        model="test-model",
        jobs=1,
    ).to_dict()

    assert payload["summary"]["declared_infeasible"] == 2
    assert payload["summary"]["correct_infeasibility_declarations"] == 1
    assert payload["tasks"][0]["declared_infeasible"] is True
    assert payload["tasks"][0]["timing"] is None
    assert payload["summary"]["timing"]["timing_available_task_count"] == 0
    assert payload["summary"]["timing"]["timing_missing_task_count"] == 2


def test_successful_algorithm_packages_trust_final_package_executions(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solver_result.json").write_text(
        json.dumps(
            {
                "executions": [
                    {
                        "source": "package",
                        "executed_package_id": "pyvrp.ils",
                        "executed_solver_id": "ils",
                        "version": "0.13.4",
                    },
                    {
                        "source": "package",
                        "executed_package_id": "pyvrp.ils",
                        "executed_solver_id": "ils",
                        "version": "0.13.4",
                    },
                    {
                        "source": "generated",
                        "executed_package_id": None,
                        "executed_solver_id": None,
                        "version": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    packages = runner_module._read_successful_algorithm_packages(
        workspace,
        hidden_checker_feasible=True,
    )

    assert packages == (
        {"package_id": "pyvrp.ils", "solver_id": "ils", "version": "0.13.4"},
    )
    assert (
        runner_module._read_successful_algorithm_packages(
            workspace,
            hidden_checker_feasible=False,
        )
        == ()
    )


def test_successful_solver_execution_selects_last_solution_changing_attempt():
    execution = runner_module._select_successful_solver_execution(
        {
            "attempts": [
                {"sequence": 1, "attempt_failed": True, "duration_ms": 100},
                {
                    "sequence": 2,
                    "attempt_failed": False,
                    "duration_ms": 250,
                    "status": "ok",
                    "exit_code": 0,
                    "solution_changed": True,
                },
                {
                    "sequence": 3,
                    "attempt_failed": False,
                    "duration_ms": 80,
                    "status": "ok",
                    "exit_code": 0,
                    "solution_changed": False,
                },
            ]
        },
        ({"package_id": "ortools", "solver_id": "cp_sat", "version": "9"},),
    )

    assert execution is not None
    assert execution["attempt_sequence"] == 2
    assert execution["duration_ms"] == 250
    assert execution["packages"][0]["solver_id"] == "cp_sat"


def test_report_marks_disabled_observability_without_zero_filling(tmp_path: Path):
    task = BenchmarkTask(
        task_id="sample",
        root=tmp_path,
        metadata={"problem_class": "routing", "direction": "min"},
    )
    evaluation = TaskEvaluation(
        task=task,
        score=TaskScore(
            task_id="sample",
            problem_class="routing",
            executed=True,
            feasible=True,
        ),
        run_id="run-sample",
        run_status="succeeded",
        duration_seconds=1,
        timing_summary={"wall_clock_duration_ms": 1000},
        successful_algorithm_packages=(
            {"package_id": "ortools", "solver_id": "cp_sat", "version": "9.12"},
        ),
    )
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payload = FrontierORReport(
        evaluations=[evaluation],
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        model="test-model",
        jobs=1,
        timing_enabled=False,
        algorithm_package_stats_enabled=False,
    ).to_dict()

    assert payload["timing_enabled"] is False
    assert payload["algorithm_package_stats_enabled"] is False
    assert payload["summary"]["timing"] is None
    assert payload["summary"]["algorithm_package_success"] is None
    assert payload["summary"]["package_usage"] is None


def test_compact_report_preserves_ablation_switches(tmp_path: Path):
    task = BenchmarkTask(
        task_id="sample",
        root=tmp_path,
        metadata={"problem_class": "routing", "direction": "min"},
    )
    evaluation = TaskEvaluation(
        task=task,
        score=TaskScore(task_id="sample", problem_class="routing", executed=True),
        run_id="run-sample",
        run_status="succeeded",
        duration_seconds=1,
    )
    payload = FrontierORReport(
        evaluations=[evaluation],
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        model="test-model",
        jobs=1,
        feasibility_review_enabled=False,
        input_schema_enabled=False,
        algorithm_library_enabled=False,
    ).to_dict()

    compact = runner_module._build_case_index(payload, tmp_path / "runs")
    assert compact["feasibility_review_enabled"] is False
    assert compact["input_schema_enabled"] is False
    assert compact["algorithm_library_enabled"] is False
    assert compact["components_enabled"] is True

    payload.pop("components_enabled")
    legacy_compact = runner_module._build_case_index(payload, tmp_path / "runs")
    assert legacy_compact["components_enabled"] is True


def test_package_usage_summary_and_csv_preserve_attempt_errors(tmp_path: Path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "package-usage.json").write_text(
        json.dumps(
            {
                "assessment": {
                    "plan_alignment": "deviates_from_plan",
                    "semantic_selection_judgment": "requires_independent_review",
                    "observed_package_ids": ["gurobipy"],
                    "unplanned_package_ids": ["gurobipy"],
                    "missing_guide_before_use": ["gurobipy"],
                    "solver_attempt_count": 1,
                    "failed_attempt_count": 1,
                    "errored_attempt_count": 1,
                    "error_category_counts": {"license_error": 1},
                    "solver_duration_ms": 1200,
                    "failed_attempt_duration_ms": 1200,
                    "errored_attempt_duration_ms": 1200,
                    "fallback_observed": True,
                },
                "attempts": [
                    {
                        "sequence": 1,
                        "package_ids": ["gurobipy"],
                        "duration_ms": 1200,
                        "exit_code": 1,
                        "status": "error",
                        "timed_out": False,
                        "solution_changed": False,
                        "attempt_failed": True,
                        "error_observed": True,
                        "error_category": "license_error",
                        "error_output_tail": "GurobiError: license expired",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    usage = runner_module._read_package_usage_summary(run_path)
    assert usage is not None
    task = BenchmarkTask(
        task_id="sample",
        root=tmp_path,
        metadata={"problem_class": "routing", "direction": "min"},
    )
    evaluation = TaskEvaluation(
        task=task,
        score=TaskScore(
            task_id="sample",
            problem_class="routing",
            executed=True,
            feasible=False,
        ),
        run_id="run-sample",
        run_status="succeeded",
        duration_seconds=1.2,
        package_usage_summary=usage,
    )
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload = FrontierORReport(
        evaluations=[evaluation],
        started_at=started,
        finished_at=started + timedelta(seconds=2),
        model="test-model",
        jobs=1,
    ).to_dict()

    aggregate = payload["summary"]["package_usage"]
    assert aggregate["tasks_with_package_errors"] == 1
    assert aggregate["tasks_with_failed_package_attempts"] == 1
    assert aggregate["error_category_counts"] == {"license_error": 1}
    assert aggregate["failed_attempt_duration_ms"] == 1200

    csv_path = tmp_path / "package-usage.csv"
    runner_module._write_package_usage_csv(csv_path, payload)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task_id"] == "sample"
    assert rows[0]["observed_package_ids"] == "gurobipy"
    assert json.loads(rows[0]["error_categories"]) == {"license_error": 1}
    assert rows[0]["package_usage_report"] == str(
        (run_path / "package-usage.json").resolve()
    )


def test_load_large_max_tasks_uses_fixed_largest_case(tmp_path: Path):
    task_root = tmp_path / "tasks" / "sample"
    large_root = tmp_path / "large"
    instance_dir = large_root / "sample" / "instance"
    solution_dir = large_root / "sample" / "gurobi_solution"
    instance_dir.mkdir(parents=True)
    solution_dir.mkdir(parents=True)
    for index, content in ((1, "{}"), (2, '{"largest": true}'), (3, "[]")):
        (instance_dir / f"large_instance_{index}.json").write_text(content, encoding="utf-8")
        (solution_dir / f"large_solution_{index}.json").write_text(
            '{"objective_value": 1}', encoding="utf-8"
        )
    suite_index = tmp_path / "large-max.json"
    suite_index.write_text(
        json.dumps(
            {
                "benchmark": "FrontierOR",
                "suite": "large-max",
                "case_count": 1,
                "cases": {
                    "sample": {
                        "instance_index": 2,
                        "instance_bytes": (instance_dir / "large_instance_2.json").stat().st_size,
                        "reference_feasible": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    base = BenchmarkTask(
        task_id="sample",
        root=task_root,
        metadata={"problem_class": "routing", "direction": "min"},
    )

    selected = load_large_max_tasks([base], large_root, suite_index)

    assert len(selected) == 1
    assert selected[0].task_id == "sample-large-2"
    assert selected[0].paper_id == "sample"
    assert selected[0].instance_variant == "large-max"
    assert selected[0].instance_index == 2
    assert selected[0].instance_path == instance_dir / "large_instance_2.json"
    assert selected[0].reference_feasible is False
    assert selected[0].reference_objective() is None

    (instance_dir / "large_instance_3.json").write_text("x" * 100, encoding="utf-8")
    try:
        load_large_max_tasks([base], large_root, suite_index)
    except BenchmarkTaskError as exc:
        assert "no longer the largest" in str(exc)
    else:
        raise AssertionError("changed largest instance should reject the fixed suite")


def test_frontieror65_inf_public_statements_do_not_disclose_expected_result():
    suite_root = runner_module.FRONTIEROR10_INF_ROOT
    forbidden = (
        "synthetic infeasibility",
        "no candidate solution",
        "cannot satisfy all hard constraints",
        "infeasibility_test",
    )

    statements = sorted(suite_root.glob("*/input/problem.md"))
    assert len(statements) == 10
    for statement in statements:
        text = statement.read_text(encoding="utf-8").lower()
        assert not any(phrase in text for phrase in forbidden), statement


def test_frontieror65_fea_catalog_matches_its_index():
    tasks = discover_tasks(runner_module.DEFAULT_TASKS_ROOT)
    index = json.loads(
        runner_module.DEFAULT_FRONTIEROR65_FEA_INDEX.read_text(encoding="utf-8")
    )

    assert index["suite"] == "FrontierOR65-Fea"
    assert index["case_count"] == 65
    assert len(tasks) == 65
    assert {task.paper_id for task in tasks} == set(index["cases"])
    assert all(case["reference_feasible"] for case in index["cases"].values())
    for task in tasks:
        assert not (task.input_dir / "data" / "instance.json").exists()
        assert not (task.hidden_dir / "reference_solution.json").exists()


def test_benchmark_root_contains_only_release_suites():
    benchmark_root = runner_module.PROJECT_ROOT / "benchmarks"
    assert {path.name for path in benchmark_root.iterdir() if path.is_dir()} == {
        "FrontierOR65-Fea",
        "FrontierOR10-Inf",
        "Hard32-Fea",
    }


def test_evaluate_task_scores_core_output_directly(tmp_path: Path, monkeypatch):
    task_root = tmp_path / "sample"
    (task_root / "input" / "data").mkdir(parents=True)
    (task_root / "hidden").mkdir()
    (task_root / "input" / "problem.md").write_text("完整问题", encoding="utf-8")
    (task_root / "input" / "data" / "instance.json").write_text('{"jobs": []}', encoding="utf-8")
    (task_root / "hidden" / "solution_schema.json").write_text(
        '{"objective_value": "<float>"}', encoding="utf-8"
    )
    external_instance = tmp_path / "large_instance_2.json"
    external_instance.write_text('{"jobs": ["large"]}', encoding="utf-8")
    task = BenchmarkTask(
        task_id="sample",
        root=task_root,
        metadata={
            "paper_id": "sample",
            "problem_class": "scheduling",
            "direction": "min",
            "instance": "large-max",
            "instance_index": 2,
        },
        instance_path_override=external_instance,
    )
    run_id = "20260101-000000Z-abcdef01"
    stored_run = tmp_path / "runs" / run_id
    order: list[str] = []

    class FakeRepository:
        def read_record(self, observed_run_id):
            assert observed_run_id == run_id
            return SimpleNamespace(
                status=SimpleNamespace(value="succeeded"),
                error_summary="",
            )

        def run_path(self, observed_run_id):
            assert observed_run_id == run_id
            return stored_run

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            self.repository = FakeRepository()

        async def run(self, *, sink, start_stage, workspace, **kwargs):
            order.append("runtime")
            assert start_stage.value == "problem_contract"
            assert json.loads((workspace / "data" / "instance.json").read_text()) == {
                "jobs": ["large"]
            }
            assert not (workspace / "input.json").exists()
            assert not (workspace / "input_schema.json").exists()
            assert not (workspace / "solution_schema.json").exists()
            assert json.loads(
                (workspace / "fixed_solution_contract.json").read_text(encoding="utf-8")
            ) == {
                "objective_value": "<float>"
            }
            assert not (workspace / "feasibility_checker.py").exists()
            intake = json.loads(
                (workspace / "stage_outputs" / "intake.json").read_text(encoding="utf-8")
            )
            assert intake["problem_definition"]["frontieror_contract"][
                "target_solution_template"
            ] == {"objective_value": "<float>"}
            assert "hidden" not in json.dumps(intake).lower()
            assert not (workspace / "stage_outputs" / "problem_contract.json").exists()
            workspace = stored_run / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "solution.json").write_text('{"objective_value": 12}', encoding="utf-8")
            sink.run_id = run_id

        def cancel(self):
            raise AssertionError("runtime should not be cancelled")

    def fake_score(observed_task, solution_path, *, error):
        order.append("score")
        assert observed_task is task
        assert error == ""
        assert json.loads(solution_path.read_text(encoding="utf-8")) == {"objective_value": 12}
        return TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=True,
            feasible=True,
            objective=12,
            reference_objective=10,
            gap=0.2,
        )

    monkeypatch.setattr(runner_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(runner_module, "score_task", fake_score)

    result = asyncio.run(
        runner_module.evaluate_task(
            task,
            Settings(runs_dir=tmp_path / "runs"),
            task_timeout_seconds=10,
        )
    )

    assert order == ["runtime", "score"]
    assert result.run_status == "succeeded"
    assert result.score.feasible is True


def test_evaluate_task_accepts_reviewed_instance_infeasible_without_solution(
    tmp_path: Path, monkeypatch
):
    task_root = tmp_path / "sample"
    (task_root / "input" / "data").mkdir(parents=True)
    (task_root / "hidden").mkdir()
    (task_root / "input" / "problem.md").write_text("完整问题", encoding="utf-8")
    (task_root / "input" / "data" / "instance.json").write_text("{}", encoding="utf-8")
    (task_root / "hidden" / "solution_schema.json").write_text("{}", encoding="utf-8")
    task = BenchmarkTask(
        task_id="known-infeasible",
        root=task_root,
        metadata={
            "problem_class": "scheduling",
            "direction": "min",
            "reference_feasible": False,
        },
        instance_path_override=task_root / "input" / "data" / "instance.json",
    )
    run_id = "20260101-000000Z-abcdef01"
    stored_run = tmp_path / "runs" / run_id

    class FakeRepository:
        def read_record(self, observed_run_id):
            assert observed_run_id == run_id
            return SimpleNamespace(
                status=SimpleNamespace(value="succeeded"),
                result_summary={"status": "instance_infeasible"},
                error_summary="",
            )

        def run_path(self, observed_run_id):
            assert observed_run_id == run_id
            return stored_run

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            self.repository = FakeRepository()

        async def run(self, *, sink, **kwargs):
            (stored_run / "workspace").mkdir(parents=True)
            sink.run_id = run_id

        def cancel(self):
            raise AssertionError("runtime should not be cancelled")

    def fail_if_scored(*args, **kwargs):
        raise AssertionError("hidden checker must not run for reviewed instance infeasibility")

    monkeypatch.setattr(runner_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(runner_module, "score_task", fail_if_scored)

    result = asyncio.run(
        runner_module.evaluate_task(
            task,
            Settings(runs_dir=tmp_path / "runs"),
            task_timeout_seconds=10,
        )
    )

    assert result.run_status == "succeeded"
    assert result.score.declared_infeasible is True
    assert result.score.passed is True
    assert result.error_diagnostics["total_errors"] == 0


def test_evaluate_task_does_not_score_solution_from_failed_runtime(tmp_path: Path, monkeypatch):
    task_root = tmp_path / "sample"
    (task_root / "input" / "data").mkdir(parents=True)
    (task_root / "hidden").mkdir()
    (task_root / "input" / "problem.md").write_text("完整问题", encoding="utf-8")
    (task_root / "input" / "data" / "instance.json").write_text("{}", encoding="utf-8")
    (task_root / "hidden" / "solution_schema.json").write_text(
        '{"objective_value": "<float>"}', encoding="utf-8"
    )
    task = BenchmarkTask(
        task_id="failed-after-solving",
        root=task_root,
        metadata={"problem_class": "scheduling", "direction": "min"},
        instance_path_override=task_root / "input" / "data" / "instance.json",
    )
    run_id = "20260101-000000Z-abcdef01"
    stored_run = tmp_path / "runs" / run_id
    calls: list[str] = []

    class FakeRepository:
        def read_record(self, observed_run_id):
            assert observed_run_id == run_id
            return SimpleNamespace(
                status=SimpleNamespace(value="failed"),
                error_summary="Reviewer 超时",
            )

        def run_path(self, observed_run_id):
            assert observed_run_id == run_id
            return stored_run

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            self.repository = FakeRepository()

        async def run(self, *, sink, **kwargs):
            workspace = stored_run / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "solution.json").write_text('{"cost": 12}', encoding="utf-8")
            sink.run_id = run_id

        def cancel(self):
            raise AssertionError("runtime should not be cancelled")

    def fake_score(observed_task, solution_path, *, error):
        calls.append("score")
        assert observed_task is task
        assert solution_path is None
        assert error == "Reviewer 超时"
        return TaskScore(
            task_id=task.task_id,
            problem_class=task.problem_class,
            executed=False,
            error=error,
        )

    monkeypatch.setattr(runner_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(runner_module, "score_task", fake_score)

    result = asyncio.run(
        runner_module.evaluate_task(
            task,
            Settings(runs_dir=tmp_path / "runs"),
            task_timeout_seconds=10,
        )
    )

    assert calls == ["score"]
    assert result.run_status == "failed"
    assert result.score.error == "Reviewer 超时"
    assert result.score.passed is False


def test_run_benchmark_keeps_other_tasks_when_one_runner_fails(tmp_path: Path, monkeypatch):
    tasks = [
        BenchmarkTask(
            task_id=task_id,
            root=tmp_path / task_id,
            metadata={"problem_class": "routing", "direction": "min"},
        )
        for task_id in ("broken", "healthy")
    ]

    async def fake_evaluate(task, settings, **kwargs):
        if task.task_id == "broken":
            raise RuntimeError("unexpected setup error")
        return TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=True,
                feasible=True,
            ),
            run_id="20260101-000000Z-abcdef01",
            run_status="succeeded",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)

    settings = Settings()
    report = asyncio.run(
        run_benchmark(
            tasks,
            settings,
            jobs=2,
            feasibility_review_enabled=False,
        )
    )

    assert [item.task.task_id for item in report.evaluations] == ["broken", "healthy"]
    assert report.feasibility_review_enabled is False
    assert report.evaluations[0].run_status == "runner_failed"
    assert report.evaluations[0].score.error == "Runner failed: unexpected setup error"
    assert report.evaluations[1].score.feasible is True


def test_run_benchmark_correlates_task_events_with_index_and_run_directory(
    tmp_path: Path, monkeypatch
):
    task = BenchmarkTask(
        task_id="sample-large-2",
        root=tmp_path / "sample",
        metadata={"problem_class": "routing", "direction": "min"},
    )
    run_id = "20260101-000000Z-abcdef01"

    async def fake_evaluate(task, settings, **kwargs):
        kwargs["task_event_callback"](
            {
                "event": "agent_error",
                "run_id": run_id,
                "stage": "solving",
                "error_kind": "llm_request_failure",
                "error": "LLM HTTP 400",
            }
        )
        return TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=False,
                error="LLM HTTP 400",
            ),
            run_id=run_id,
            run_status="failed",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)
    events = []
    settings = Settings(runs_dir=tmp_path / "runs")

    asyncio.run(run_benchmark([task], settings, task_event=events.append))

    error = next(event for event in events if event["event"] == "agent_error")
    assert error["task_id"] == "sample-large-2"
    assert error["index"] == 1
    assert error["total"] == 1
    assert error["run_id"] == run_id
    assert error["run_directory"] == str((tmp_path / "runs" / run_id).resolve())


def test_run_benchmark_divides_solver_cpu_budget_across_jobs(tmp_path: Path, monkeypatch):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            root=tmp_path / f"task-{index}",
            metadata={"problem_class": "routing", "direction": "min"},
        )
        for index in range(2)
    ]
    limits: list[int] = []

    async def fake_evaluate(task, settings, **kwargs):
        limits.append(kwargs["solver_cpu_limit"])
        return TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=True,
                feasible=True,
            ),
            run_id=None,
            run_status="succeeded",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)
    settings = Settings(solver_cpu_budget=24)

    asyncio.run(run_benchmark(tasks, settings, jobs=5))

    assert limits == [4, 4]


def test_run_benchmark_rejects_more_jobs_than_cpu_budget():
    with pytest.raises(ValueError, match="jobs cannot exceed SOLVER_CPU_BUDGET"):
        asyncio.run(run_benchmark([], Settings(solver_cpu_budget=24), jobs=25))


def test_run_benchmark_rejects_crossed_components_ablation_before_scheduling():
    with pytest.raises(ValueError, match="single-factor"):
        asyncio.run(
            run_benchmark(
                [],
                Settings(),
                components_enabled=False,
                feasibility_review_enabled=False,
            )
        )


def test_run_benchmark_stops_after_consecutive_network_failures(tmp_path: Path, monkeypatch):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            root=tmp_path / f"task-{index}",
            metadata={"problem_class": "routing", "direction": "min"},
        )
        for index in range(5)
    ]
    called: list[str] = []

    async def fake_evaluate(task, settings, **kwargs):
        called.append(task.task_id)
        return TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=False,
                error="Runtime 失败：All connection attempts failed",
            ),
            run_id=None,
            run_status="failed",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)

    report = asyncio.run(
        run_benchmark(
            tasks,
            Settings(),
            jobs=1,
            max_consecutive_network_failures=3,
        )
    )
    payload = report.to_dict()

    assert called == ["task-0", "task-1", "task-2"]
    assert report.termination_kind == "network_failure_threshold"
    assert payload["summary"]["completed_tasks"] == 3
    assert payload["summary"]["skipped_tasks"] == 2
    assert payload["summary"]["terminated_early"] is True


def test_run_benchmark_stops_on_memory_error_and_cancels_active_task(tmp_path: Path, monkeypatch):
    tasks = [
        BenchmarkTask(
            task_id=task_id,
            root=tmp_path / task_id,
            metadata={"problem_class": "routing", "direction": "min"},
        )
        for task_id in ("memory", "active", "never-started")
    ]
    cancelled: set[str] = set()

    async def fake_evaluate(task, settings, **kwargs):
        if task.task_id == "memory":
            kwargs["resource_exhaustion_callback"](
                "OSError: [WinError 1455] 页面文件太小，无法完成操作。"
            )
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.add(task.task_id)
            raise
        raise AssertionError("sleeping task should have been cancelled")

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)

    report = asyncio.run(
        run_benchmark(
            tasks,
            Settings(),
            jobs=2,
            stop_on_resource_exhaustion=True,
        )
    )
    payload = report.to_dict()

    assert cancelled == {"memory", "active"}
    assert report.termination_kind == "resource_exhaustion"
    assert report.evaluations == []
    assert payload["summary"]["skipped_tasks"] == 3


def test_network_failure_counter_resets_after_non_network_result(tmp_path: Path, monkeypatch):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            root=tmp_path / f"task-{index}",
            metadata={"problem_class": "routing", "direction": "min"},
        )
        for index in range(5)
    ]
    network_failures = {"task-0", "task-2", "task-3"}

    async def fake_evaluate(task, settings, **kwargs):
        network_failure = task.task_id in network_failures
        return TaskEvaluation(
            task=task,
            score=TaskScore(
                task_id=task.task_id,
                problem_class=task.problem_class,
                executed=not network_failure,
                feasible=True if not network_failure else None,
                error="All connection attempts failed" if network_failure else "",
            ),
            run_id=None,
            run_status="failed" if network_failure else "succeeded",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(runner_module, "evaluate_task", fake_evaluate)

    report = asyncio.run(
        run_benchmark(
            tasks,
            Settings(),
            jobs=1,
            max_consecutive_network_failures=3,
        )
    )

    assert report.termination_kind is None
    assert len(report.evaluations) == 5


def test_core_review_cli_switches_are_independent():
    parser = runner_module._parser()

    assert parser.parse_args([]).feasibility_review is True
    assert parser.parse_args([]).input_schema is True
    assert parser.parse_args([]).algorithm_library is True
    assert parser.parse_args([]).timing is True
    assert parser.parse_args([]).algorithm_package_stats is True
    assert parser.parse_args([]).components is True
    assert parser.parse_args([]).task_timeout_seconds == 7200
    assert parser.parse_args([]).suite == "FrontierOR65-Fea"
    assert parser.parse_args([]).stop_on_resource_exhaustion is False
    assert parser.parse_args([]).max_consecutive_network_failures == 0
    assert parser.parse_args([]).input_schema is True
    assert parser.parse_args([]).algorithm_library is True
    assert parser.parse_args([]).problem_contract is True
    assert parser.parse_args([]).cross_package is True
    assert parser.parse_args(["--no-components"]).components is False
    assert parser.parse_args(["--no-timing"]).timing is False
    assert (
        parser.parse_args(["--no-algorithm-package-stats"]).algorithm_package_stats
        is False
    )
    assert parser.parse_args(["--no-feasibility-review"]).feasibility_review is False
    for suite in ("FrontierOR65-Fea", "FrontierOR10-Inf", "Hard32-Fea"):
        assert parser.parse_args(["--suite", suite]).suite == suite
    assert parser.parse_args(["--stop-on-resource-exhaustion"]).stop_on_resource_exhaustion
    assert (
        parser.parse_args(
            ["--max-consecutive-network-failures", "3"]
        ).max_consecutive_network_failures
        == 3
    )


def test_workflow_ablation_flags_allow_only_reported_arms():
    parser = runner_module._parser()
    reported = (
        [],
        ["--package-pool", "gurobi-only"],
        [
            "--package-pool",
            "gurobi-only",
            "--workflow",
            "gurobi-formulator",
            "--no-algorithm-design",
            "--no-components",
        ],
        ["--package-pool", "gurobi-only", "--no-feasibility-review"],
    )
    for cli_args in reported:
        runner_module._validate_workflow_ablation_args(parser.parse_args(cli_args))

    unreported = (
        ["--no-algorithm-design"],
        ["--no-components"],
        ["--no-feasibility-review"],
        ["--package-pool", "gurobi-only", "--no-algorithm-design"],
    )
    for cli_args in unreported:
        with pytest.raises(ValueError, match="paper-reported"):
            runner_module._validate_workflow_ablation_args(parser.parse_args(cli_args))


def test_gurobi_workflow_endpoint_arms_are_allowed():
    formulator_review = asyncio.run(
        run_benchmark(
            [],
            Settings(),
            algorithm_design_enabled=False,
            components_enabled=False,
            package_policy=PackagePolicy(pool="gurobi-only"),
            workflow="gurobi-formulator",
        )
    )
    design_no_review = asyncio.run(
        run_benchmark(
            [],
            Settings(),
            feasibility_review_enabled=False,
            package_policy=PackagePolicy(pool="gurobi-only"),
        )
    )

    assert formulator_review.package_pool == "gurobi-only"
    assert formulator_review.algorithm_design_enabled is False
    assert formulator_review.components_enabled is False
    assert formulator_review.feasibility_review_enabled is True
    assert formulator_review.workflow == "gurobi-formulator"
    assert formulator_review.gurobi_formulator_enabled is True
    assert formulator_review.solving_mode == "translation-only"
    assert design_no_review.package_pool == "gurobi-only"
    assert design_no_review.algorithm_design_enabled is True
    assert design_no_review.components_enabled is True
    assert design_no_review.feasibility_review_enabled is False


@pytest.mark.parametrize(
    ("problem_contract", "algorithm_design", "expected"),
    [
        (True, True, AgentStage.PROBLEM_CONTRACT),
        (False, True, AgentStage.ALGORITHM_DESIGN),
        (False, False, AgentStage.SOLVING),
    ],
)
def test_intake_routes_to_first_enabled_workflow_stage(
    problem_contract, algorithm_design, expected
):
    state = AgentState(problem_description="test", current_stage=AgentStage.INTAKE)
    result = StageCompleted(
        stage=AgentStage.INTAKE,
        data={"ready": True, "problem_definition": {"objective": "minimize"}},
    )
    updated = apply_stage_result(
        state,
        result,
        problem_contract_enabled=problem_contract,
        algorithm_design_enabled=algorithm_design,
    )
    assert updated.current_stage is expected


def test_cli_exits_before_loading_tasks_when_algorithm_runtime_validation_fails(
    monkeypatch, tmp_path
):
    def fail_validation(catalog):
        raise AlgorithmRuntimeValidationError(
            [
                RuntimeValidationFailure(
                    package_id="gurobipy",
                    solver_id="milp",
                    check="minimal_call_example",
                    detail="example failed",
                )
            ]
        )

    monkeypatch.setattr(runner_module, "validate_algorithm_library_runtime", fail_validation)
    monkeypatch.setattr(
        runner_module,
        "discover_tasks",
        lambda *args, **kwargs: pytest.fail("tasks must not load after failed preflight"),
    )

    with pytest.raises(SystemExit, match="gurobipy/milp"):
        runner_module.main(["--large-root", str(tmp_path)])


def test_large_suite_task_filter_accepts_report_task_id(tmp_path: Path):
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "benchmark": "FrontierOR",
                "suite": "FrontierOR65-Fea",
                "case_count": 1,
                "cases": {"schwerdfeger2016": {"instance_index": 2}},
            }
        ),
        encoding="utf-8",
    )

    assert normalize_large_suite_task_ids(
        index,
        suite="FrontierOR65-Fea",
        requested={"schwerdfeger2016"},
    ) == {"schwerdfeger2016"}
    assert normalize_large_suite_task_ids(
        index,
        suite="FrontierOR65-Fea",
        requested={"schwerdfeger2016-large-2"},
    ) == {"schwerdfeger2016"}


def test_case_index_is_keyed_by_task_and_contains_run_directory(tmp_path: Path):
    report_payload = {
        "suite": "large-max",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "tasks": [
            {
                "task_id": "sample-large-2",
                "problem_class": "routing",
                "formulation_type": "MIP",
                "category": "operational",
                "run_id": "20260101-000000Z-abcdef01",
                "run_status": "succeeded",
                "executed": True,
                "feasible": None,
                "gap": None,
                "error": "checker timeout",
                "checker_stderr": "",
                "error_diagnostics": {"total_errors": 1},
            }
        ],
    }

    index = runner_module._build_case_index(report_payload, tmp_path / "runs")
    case = index["cases"]["sample-large-2"]

    assert case["run_id"] == "20260101-000000Z-abcdef01"
    assert case["problem_class"] == "routing"
    assert case["formulation_type"] == "MIP"
    assert case["category"] == "operational"
    assert case["run_directory"] == str((tmp_path / "runs" / "20260101-000000Z-abcdef01").resolve())
    assert case["error"] == "checker timeout"


def test_evaluate_task_full_intake_starts_without_bypass_or_gold(tmp_path, monkeypatch):
    task_root = tmp_path / "task"
    (task_root / "input" / "data").mkdir(parents=True)
    (task_root / "hidden").mkdir()
    (task_root / "input" / "problem.md").write_text("完整问题", encoding="utf-8")
    (task_root / "input" / "data" / "instance.json").write_text(
        '{"jobs": []}', encoding="utf-8"
    )
    (task_root / "hidden" / "solution_schema.json").write_text(
        '{"objective_value": "<float>"}', encoding="utf-8"
    )
    (task_root / "hidden" / "feasibility_check.py").write_text(
        "# authoritative checker", encoding="utf-8"
    )
    external_instance = tmp_path / "large_instance_1.json"
    external_instance.write_text('{"jobs": ["large"]}', encoding="utf-8")
    task = BenchmarkTask(
        task_id="sample-full",
        root=task_root,
        metadata={
            "paper_id": "sample-full",
            "problem_class": "scheduling",
            "direction": "min",
            "instance": "large-max",
            "instance_index": 1,
        },
        instance_path_override=external_instance,
    )
    run_id = "20260101-000000Z-abcdef02"
    stored_run = tmp_path / "runs" / run_id

    class FakeRepository:
        def read_record(self, observed_run_id):
            assert observed_run_id == run_id
            return SimpleNamespace(status=SimpleNamespace(value="succeeded"), error_summary="")

        def run_path(self, observed_run_id):
            assert observed_run_id == run_id
            return stored_run

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            self.repository = FakeRepository()

        async def run(self, *, sink, start_stage, workspace, config, **kwargs):
            assert start_stage.value == "intake"
            assert config["benchmark_intake_mode"] == "full"
            assert not (workspace / "stage_outputs" / "intake.json").exists()
            assert not any(workspace.rglob("gold_standard.json"))
            assert not any(workspace.rglob("feasibility_check.py"))
            assert json.loads((workspace / "fixed_solution_contract.json").read_text()) == {
                "objective_value": "<float>"
            }
            runtime_workspace = stored_run / "workspace"
            runtime_workspace.mkdir(parents=True)
            (runtime_workspace / "solution.json").write_text(
                '{"objective_value": 12}', encoding="utf-8"
            )
            sink.run_id = run_id

        def cancel(self):
            raise AssertionError("runtime should not be cancelled")

    class FakeAnswerer:
        def __init__(self, settings, workspace):
            assert workspace.name == "answerer"
            assert (workspace / "problem.md").is_file()
            assert (workspace / "data" / "instance.json").is_file()
            assert (workspace / "hidden" / "feasibility_check.py").read_text() == (
                "# authoritative checker"
            )

        async def answer(self, **kwargs):
            raise AssertionError("runtime did not request clarification")

    monkeypatch.setattr(runner_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(runner_module, "CheckerGroundedIntakeAnswerer", FakeAnswerer)
    monkeypatch.setattr(
        runner_module,
        "score_task",
        lambda observed_task, solution_path, *, error: TaskScore(
            task_id=observed_task.task_id,
            problem_class=observed_task.problem_class,
            executed=True,
            feasible=True,
        ),
    )

    result = asyncio.run(
        runner_module.evaluate_task(
            task,
            Settings(runs_dir=tmp_path / "runs"),
            task_timeout_seconds=10,
            intake_mode="full",
        )
    )

    assert result.run_status == "succeeded"
    assert result.intake_mode == "full"
    assert result.intake_dialogue == ()


def test_full_intake_attributes_answerer_failure_to_separate_stage(tmp_path, monkeypatch):
    task_root = tmp_path / "task"
    (task_root / "input" / "data").mkdir(parents=True)
    (task_root / "hidden").mkdir()
    (task_root / "input" / "problem.md").write_text("问题", encoding="utf-8")
    (task_root / "input" / "data" / "instance.json").write_text("{}", encoding="utf-8")
    (task_root / "hidden" / "solution_schema.json").write_text("{}", encoding="utf-8")
    (task_root / "hidden" / "feasibility_check.py").write_text("# checker", encoding="utf-8")
    instance = tmp_path / "large_instance_1.json"
    instance.write_text("{}", encoding="utf-8")
    task = BenchmarkTask(
        task_id="answerer-failure",
        root=task_root,
        metadata={
            "paper_id": "answerer-failure",
            "problem_class": "scheduling",
            "direction": "min",
            "instance_index": 1,
        },
        instance_path_override=instance,
    )
    run_id = "20260101-000000Z-abcdef03"

    class FakeRepository:
        def read_record(self, observed_run_id):
            return SimpleNamespace(
                status=SimpleNamespace(value="cancelled"), error_summary=""
            )

        def run_path(self, observed_run_id):
            return tmp_path / "runs" / observed_run_id

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            self.repository = FakeRepository()
            self._cancelled = asyncio.Event()

        async def run(self, *, sink, **kwargs):
            await sink.handle(
                SimpleNamespace(
                    run_id=run_id,
                    type="clarification_requested",
                    stage=SimpleNamespace(value="intake"),
                    message="请确认",
                    payload={"questions": [{"id": "q1", "text": "第一期如何编号？"}]},
                )
            )
            await self._cancelled.wait()

        def cancel(self):
            self._cancelled.set()

    class FailingAnswerer:
        def __init__(self, settings, workspace):
            pass

        async def answer(self, **kwargs):
            raise runner_module.IntakeAnswererError("protocol_exhausted", "invalid output")

    monkeypatch.setattr(runner_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(runner_module, "CheckerGroundedIntakeAnswerer", FailingAnswerer)

    def fake_score(observed_task, solution_path, *, error):
        assert solution_path is None
        assert "Intake answerer failed (protocol_exhausted)" in error
        return TaskScore(
            task_id=observed_task.task_id,
            problem_class=observed_task.problem_class,
            executed=False,
            error=error,
        )

    monkeypatch.setattr(runner_module, "score_task", fake_score)

    result = asyncio.run(
        runner_module.evaluate_task(
            task,
            Settings(runs_dir=tmp_path / "runs"),
            task_timeout_seconds=10,
            intake_mode="full",
        )
    )

    diagnostics = result.error_diagnostics
    assert diagnostics["flow_section_counts"] == {"intake_answerer": 1}
    assert diagnostics["error_kind_counts"] == {"protocol_exhausted": 1}
    assert "Intake answerer failed" in result.score.error
