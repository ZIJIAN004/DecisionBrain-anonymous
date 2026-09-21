"""Run-centric HTTP API and RunEvent SSE transport."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...exceptions import BadRequestError, RunStateError
from ...run_storage import InputWorkspace
from ..execution import EventStream, RunExecutionManager
from ..schemas import ContinueRunRequest

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _manager(request: Request) -> RunExecutionManager:
    return request.app.state.run_manager


def _encode_sse(event) -> str:
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event.type}\ndata: {data}\n\n"


_DISPLAY_HIDDEN_EVENT_TYPES = {
    "diagnostic",
    "llm_stream_started",
    "llm_stream_chunk",
    "llm_stream_finished",
    "tool_call",
    "tool_result",
}
_DISPLAY_HIDDEN_STREAMS = {"reasoning", "code"}


def _is_display_event(event) -> bool:
    if getattr(event.level, "value", event.level) == "debug":
        return False
    if event.type in _DISPLAY_HIDDEN_EVENT_TYPES:
        return False
    if event.type == "assistant_message" and event.payload.get("stream") in _DISPLAY_HIDDEN_STREAMS:
        return False
    return True


def _filter_events_for_view(events, view: str):
    if view == "display":
        return [event for event in events if _is_display_event(event)]
    return events


def _stream_response(stream: EventStream, *, view: str = "all") -> StreamingResponse:
    async def generate():
        try:
            async for event in stream.events():
                if view == "display" and not _is_display_event(event):
                    continue
                yield _encode_sse(event)
        finally:
            stream.detach()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("")
async def create_run_record(
    request: Request,
    prompt: str = Form(..., min_length=1),
    title: str = Form(""),
    view: str = Form("all"),
    files: list[UploadFile] | None = File(default=None),
):
    prompt = prompt.strip()
    if not prompt:
        raise BadRequestError("prompt cannot be empty")
    workspace = InputWorkspace.create(prefix="decisionbrain-api-")
    try:
        workspace.write_problem_description(prompt)
        for upload in files or []:
            workspace.add_bytes(upload.filename or "uploaded", await upload.read())
    except Exception:
        workspace.cleanup()
        raise
    try:
        stream = _manager(request).start(
            workspace=workspace.root,
            prompt=prompt,
            title=title,
            cleanup_workspace=True,
        )
    except Exception:
        workspace.cleanup()
        raise
    return _stream_response(stream, view=view)


@router.post("/{run_id}/continue")
async def continue_run(
    run_id: str,
    payload: ContinueRunRequest,
    request: Request,
    view: str = Query("all"),
):
    manager = _manager(request)
    message = payload.message.strip()
    if not message:
        raise BadRequestError("message cannot be empty")
    stream = manager.submit_user_message(run_id=run_id, message=message)
    return _stream_response(stream, view=view)


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    manager = _manager(request)
    manager.repository.read_record(run_id)
    try:
        await manager.cancel(run_id)
    except KeyError as exc:
        raise RunStateError(f"Run {run_id} is not active in this process") from exc
    return manager.repository.read_record(run_id)


@router.get("")
async def list_runs(request: Request):
    return {"runs": _manager(request).repository.list_runs()}


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request):
    return _manager(request).repository.read_record(run_id)


@router.get("/{run_id}/events")
async def get_events(run_id: str, request: Request, view: str = Query("all")):
    events = _manager(request).repository.read_events(run_id)
    return {"events": _filter_events_for_view(events, view)}


@router.get("/{run_id}/report")
async def get_report(run_id: str, request: Request):
    repository = _manager(request).repository
    repository.read_record(run_id)
    path = repository.run_path(run_id) / "agent-run.html"
    if not path.is_file():
        raise BadRequestError(f"Run {run_id} has not produced an HTML report")
    return FileResponse(path, media_type="text/html", filename=f"{run_id}.html")


@router.get("/{run_id}/artifacts")
async def list_artifacts(run_id: str, request: Request):
    artifacts = _manager(request).repository.list_artifacts(run_id)
    return {"artifacts": artifacts}


@router.get("/{run_id}/artifacts/{artifact_id}")
async def download_artifact(run_id: str, artifact_id: str, request: Request):
    repository = _manager(request).repository
    metadata = repository.get_artifact_metadata(run_id, artifact_id)
    path = repository.get_artifact_path(run_id, artifact_id)
    return FileResponse(path, media_type=metadata.media_type, filename=Path(path).name)
