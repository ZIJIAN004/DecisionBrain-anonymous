from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from decisionbrain.api.app import app
from decisionbrain.api.benchmark_execution import BenchmarkExecutionManager
from decisionbrain.benchmark.runner import FrontierORReport
from decisionbrain.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def _task(root: Path, task_id: str) -> None:
    task = root / task_id
    (task / "input").mkdir(parents=True)
    (task / "hidden").mkdir()
    (task / "input" / "problem.md").write_text("minimize cost", encoding="utf-8")
    (task / "task.json").write_text(
        json.dumps({"direction": "min", "problem_class": "routing"}),
        encoding="utf-8",
    )
    for name in ("feasibility_check.py", "solution_schema.json"):
        (task / "hidden" / name).write_text("{}", encoding="utf-8")


def test_benchmark_manager_runs_selected_task_and_records_direct_mode(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    _task(tasks_root, "sample")
    large_root = tmp_path / "large"
    (large_root / "sample" / "instance").mkdir(parents=True)
    (large_root / "sample" / "gurobi_solution").mkdir()
    instance = large_root / "sample" / "instance" / "large_instance_1.json"
    instance.write_text("{}", encoding="utf-8")
    (large_root / "sample" / "gurobi_solution" / "large_solution_1.json").write_text(
        '{"objective_value": 0}', encoding="utf-8"
    )
    suite_index = tmp_path / "index.json"
    suite_index.write_text(
        json.dumps(
            {
                "benchmark": "FrontierOR",
                "suite": "FrontierOR65-Fea",
                "case_count": 1,
                "cases": {
                    "sample": {
                        "instance_index": 1,
                        "instance_bytes": instance.stat().st_size,
                        "reference_feasible": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    async def fake_runner(
        tasks,
        settings,
        *,
        jobs,
        progress,
        feasibility_review_enabled=True,
        input_schema_enabled=True,
        algorithm_library_enabled=True,
        suite="FrontierOR65-Fea",
    ):
        captured["task_ids"] = [task.task_id for task in tasks]
        captured["feasibility_review_enabled"] = feasibility_review_enabled
        progress("[1/1] START sample")
        progress("[1/1] PASS sample")
        now = datetime.now(timezone.utc)
        return FrontierORReport(
            evaluations=[],
            started_at=now,
            finished_at=now,
            model="fake-model",
            jobs=jobs,
            feasibility_review_enabled=feasibility_review_enabled,
        )

    async def scenario():
        manager = BenchmarkExecutionManager(
            settings=Settings(),
            tasks_root=tasks_root,
            large_root=large_root,
            suite_index=suite_index,
            results_root=tmp_path / "results",
            runner=fake_runner,
            runtime_validator=lambda: None,
        )
        job = manager.start(
            feasibility_review_enabled=True,
            task_ids=["sample"],
            jobs=1,
        )
        await manager._tasks[job.benchmark_id]
        return job

    job = asyncio.run(scenario())

    assert captured == {
        "task_ids": ["sample-large-1"],
        "feasibility_review_enabled": True,
    }
    assert job.status == "completed"
    assert job.completed == 1
    assert job.report["feasibility_review_enabled"] is True
    assert Path(job.report_path).is_file()


def test_benchmark_api_and_frontend_controls_are_registered():
    paths = set(app.openapi()["paths"])
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "/api/benchmarks/frontieror" in paths
    assert "/api/benchmarks/frontieror/tasks" in paths
    assert "/api/benchmarks/frontieror/{benchmark_id}" in paths
    assert "/api/benchmarks/frontieror/{benchmark_id}/cancel" in paths
    for element_id in (
        "open-benchmark",
        "benchmark-dialog",
        "benchmark-task",
        "benchmark-feasibility-review-enabled",
        "start-benchmark",
        "cancel-benchmark",
    ):
        assert f'id="{element_id}"' in html
    assert "function startBenchmark()" in html
    assert "function renderBenchmarkJob(job)" in html
    assert "benchmark-run-link" in html
