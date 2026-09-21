"""DecisionBrain API service entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import Settings
from ..exceptions import (
    ArtifactNotFoundError,
    DecisionBrainError,
    InvalidRunPathError,
    RunNotFoundError,
    RunStateError,
    RunStorageError,
)
from ..paths import INDEX_HTML
from .benchmark_execution import BenchmarkExecutionManager
from .execution import RunExecutionManager
from .routes.benchmarks import router as benchmarks_router
from .routes.runs import router as runs_router


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    run_manager = RunExecutionManager(settings=settings)
    benchmark_manager = BenchmarkExecutionManager(settings=settings)
    app.state.run_manager = run_manager
    app.state.benchmark_manager = benchmark_manager
    await run_manager.reconcile_orphans()
    try:
        yield
    finally:
        await benchmark_manager.shutdown()
        await run_manager.shutdown()


app = FastAPI(title="DecisionBrain Agent Runtime API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(runs_router)
app.include_router(benchmarks_router)


@app.exception_handler(DecisionBrainError)
async def decisionbrain_error_handler(_request: Request, exc: DecisionBrainError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(RunNotFoundError)
@app.exception_handler(ArtifactNotFoundError)
async def not_found_handler(_request: Request, exc: RunStorageError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidRunPathError)
async def invalid_path_handler(_request: Request, exc: InvalidRunPathError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RunStateError)
async def run_state_handler(_request: Request, exc: RunStateError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(RunStorageError)
async def run_storage_handler(_request: Request, exc: RunStorageError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def root():
    # Inject only non-secret presentation configuration into the static UI.
    settings = Settings()
    with INDEX_HTML.open(encoding="utf-8") as f:
        html = f.read().replace("__DBN_UI_LANGUAGE__", settings.ui_language)
        return HTMLResponse(html)


@app.get("/info")
async def info():
    return {
        "service": "decisionbrain-agent-runtime",
        "endpoints": [
            "POST /api/runs",
            "POST /api/runs/{run_id}/continue",
            "POST /api/runs/{run_id}/cancel",
            "GET  /api/runs",
            "GET  /api/runs/{run_id}",
            "GET  /api/runs/{run_id}/events",
            "GET  /api/runs/{run_id}/report",
            "GET  /api/runs/{run_id}/artifacts",
            "GET  /api/runs/{run_id}/artifacts/{artifact_id}",
            "GET  /api/benchmarks/frontieror/tasks",
            "POST /api/benchmarks/frontieror",
            "GET  /api/benchmarks/frontieror/{benchmark_id}",
            "POST /api/benchmarks/frontieror/{benchmark_id}/cancel",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def run():
    import uvicorn

    settings = Settings()
    host = settings.host
    port = settings.port
    logger.info("DecisionBrain service started at http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
