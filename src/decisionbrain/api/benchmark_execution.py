"""Background management for web-triggered FrontierOR benchmarks."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..algorithm_library import LocalAlgorithmCatalog, validate_algorithm_library_runtime
from ..benchmark.runner import (
    DEFAULT_FRONTIEROR65_FEA_INDEX,
    DEFAULT_SUITE,
    DEFAULT_TASKS_ROOT,
    FrontierORReport,
    run_benchmark,
)
from ..benchmark.task import (
    BenchmarkTask,
    discover_tasks,
    load_large_max_tasks,
    normalize_large_suite_task_ids,
)
from ..config import Settings
from ..run_storage.io import atomic_write_json


BenchmarkRunner = Callable[..., Awaitable[FrontierORReport]]
RuntimeValidator = Callable[[], None]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BenchmarkJob:
    benchmark_id: str
    feasibility_review_enabled: bool
    input_schema_enabled: bool
    algorithm_library_enabled: bool
    task_ids: tuple[str, ...]
    jobs: int
    status: str = "queued"
    completed: int = 0
    progress: list[str] = field(default_factory=list)
    report_path: str = ""
    report: dict[str, Any] | None = None
    error: str = ""
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark": "FrontierOR",
            "feasibility_review_enabled": self.feasibility_review_enabled,
            "input_schema_enabled": self.input_schema_enabled,
            "algorithm_library_enabled": self.algorithm_library_enabled,
            "task_ids": list(self.task_ids),
            "jobs": self.jobs,
            "status": self.status,
            "total": len(self.task_ids),
            "completed": self.completed,
            "progress": self.progress[-50:],
            "report_path": self.report_path,
            "report": self.report,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class BenchmarkExecutionManager:
    """Start, inspect, and cancel web benchmarks; run one batch per process."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        tasks_root: Path = DEFAULT_TASKS_ROOT,
        large_root: Path | None = None,
        suite_index: Path = DEFAULT_FRONTIEROR65_FEA_INDEX,
        results_root: Path | None = None,
        runner: BenchmarkRunner = run_benchmark,
        runtime_validator: RuntimeValidator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.tasks_root = tasks_root.expanduser().resolve()
        configured_large_root = large_root
        if configured_large_root is None:
            configured = os.environ.get("FRONTIEROR_LARGE_ROOT", "").strip()
            configured_large_root = Path(configured) if configured else None
        self.large_root = (
            configured_large_root.expanduser().resolve() if configured_large_root else None
        )
        self.suite_index = suite_index.expanduser().resolve()
        self.results_root = (
            results_root.expanduser().resolve()
            if results_root is not None
            else Path(__file__).resolve().parents[3] / "benchmark-results"
        )
        self.runner = runner
        self.runtime_validator = runtime_validator or (
            lambda: validate_algorithm_library_runtime(
                LocalAlgorithmCatalog(self.settings.opt_algorithm_manifests_dir)
            )
        )
        self._jobs: dict[str, BenchmarkJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def list_tasks(self) -> list[dict[str, str]]:
        tasks = self._load_large_tasks()
        return [
            {
                "task_id": task.task_id,
                "problem_class": task.problem_class,
                "direction": task.objective_direction,
            }
            for task in tasks
        ]

    def _load_large_tasks(self, task_ids: Sequence[str] = ()) -> list[BenchmarkTask]:
        if self.large_root is None:
            raise ValueError(
                "FrontierOR large data directory is not configured; set FRONTIEROR_LARGE_ROOT"
            )
        requested = set(task_ids) if task_ids else None
        paper_ids = normalize_large_suite_task_ids(
            self.suite_index,
            suite=DEFAULT_SUITE,
            requested=requested,
        )
        base = discover_tasks(self.tasks_root, only=paper_ids)
        return load_large_max_tasks(
            base,
            self.large_root,
            self.suite_index,
            suite=DEFAULT_SUITE,
        )

    def start(
        self,
        *,
        feasibility_review_enabled: bool = True,
        input_schema_enabled: bool = True,
        algorithm_library_enabled: bool = True,
        task_ids: Sequence[str] = (),
        limit: int | None = 1,
        jobs: int = 1,
    ) -> BenchmarkJob:
        if jobs <= 0:
            raise ValueError("jobs must be greater than zero")
        if any(task.status in {"queued", "running"} for task in self._jobs.values()):
            raise RuntimeError("A FrontierOR benchmark is already running")
        if algorithm_library_enabled:
            # Ablation 2 omits the algorithm library, so unused manifests must not block startup.
            self.runtime_validator()
        selected = self._load_large_tasks(task_ids)
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            raise ValueError("No runnable FrontierOR tasks were found")

        benchmark_id = f"frontieror-{_now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        job = BenchmarkJob(
            benchmark_id=benchmark_id,
            feasibility_review_enabled=feasibility_review_enabled,
            input_schema_enabled=input_schema_enabled,
            algorithm_library_enabled=algorithm_library_enabled,
            task_ids=tuple(task.task_id for task in selected),
            jobs=jobs,
        )
        self._jobs[benchmark_id] = job
        task = asyncio.create_task(self._execute(job, selected))
        self._tasks[benchmark_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(benchmark_id, None))
        return job

    def get(self, benchmark_id: str) -> BenchmarkJob:
        try:
            return self._jobs[benchmark_id]
        except KeyError as exc:
            raise KeyError(f"Benchmark does not exist: {benchmark_id}") from exc

    async def cancel(self, benchmark_id: str) -> BenchmarkJob:
        job = self.get(benchmark_id)
        task = self._tasks.get(benchmark_id)
        if task is None or task.done():
            raise RuntimeError(f"Benchmark {benchmark_id} cannot currently be cancelled")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return job

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, job: BenchmarkJob, tasks: Sequence[BenchmarkTask]) -> None:
        job.status = "running"
        job.started_at = _now()

        def progress(message: str) -> None:
            job.progress.append(message)
            if " START " not in message:
                job.completed = min(job.completed + 1, len(job.task_ids))

        try:
            report = await self.runner(
                tasks,
                self.settings,
                jobs=job.jobs,
                progress=progress,
                feasibility_review_enabled=job.feasibility_review_enabled,
                input_schema_enabled=job.input_schema_enabled,
                algorithm_library_enabled=job.algorithm_library_enabled,
                suite=DEFAULT_SUITE,
            )
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "Cancelled by user"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        else:
            payload = report.to_dict()
            output = self.results_root / f"{job.benchmark_id}.json"
            try:
                atomic_write_json(output, payload)
            except Exception as exc:
                job.status = "failed"
                job.error = f"Failed to write report: {exc}"
            else:
                job.report = payload
                job.report_path = str(output)
                job.completed = len(job.task_ids)
                job.status = "completed"
        finally:
            job.finished_at = _now()
