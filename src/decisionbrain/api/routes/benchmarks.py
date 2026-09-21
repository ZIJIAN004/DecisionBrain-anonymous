"""Web API for FrontierOR benchmarks."""

from fastapi import APIRouter, HTTPException, Request, status

from ..benchmark_execution import BenchmarkExecutionManager
from ..schemas import StartBenchmarkRequest

router = APIRouter(prefix="/api/benchmarks/frontieror", tags=["benchmarks"])


def _manager(request: Request) -> BenchmarkExecutionManager:
    return request.app.state.benchmark_manager


@router.get("/tasks")
async def list_tasks(request: Request):
    return {"tasks": _manager(request).list_tasks()}


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_benchmark(payload: StartBenchmarkRequest, request: Request):
    try:
        job = _manager(request).start(
            feasibility_review_enabled=payload.feasibility_review_enabled,
            input_schema_enabled=payload.input_schema_enabled,
            algorithm_library_enabled=payload.algorithm_library_enabled,
            task_ids=payload.task_ids,
            limit=payload.limit,
            jobs=payload.jobs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_dict()


@router.get("/{benchmark_id}")
async def get_benchmark(benchmark_id: str, request: Request):
    try:
        return _manager(request).get(benchmark_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{benchmark_id}/cancel")
async def cancel_benchmark(benchmark_id: str, request: Request):
    try:
        return (await _manager(request).cancel(benchmark_id)).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
