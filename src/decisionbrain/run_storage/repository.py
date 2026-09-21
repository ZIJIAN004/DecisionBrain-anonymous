"""Filesystem-backed Run repository."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..events.models import RunEvent
from ..exceptions import (
    CorruptEventLogError,
    InvalidRunPathError,
    RunNotFoundError,
    RunStateError,
    RunStorageError,
)
from .io import atomic_write_bytes, atomic_write_json
from .models import (
    ArtifactIndex,
    ArtifactMetadata,
    InputSnapshotReport,
    RunRecord,
    RunStatus,
    SnapshotRecord,
)

if TYPE_CHECKING:
    from .artifacts import ArtifactStore


RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}Z-[0-9a-f]{8}$")
TERMINAL_STATUSES = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
ALLOWED_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.NEEDS_CLARIFICATION,
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.NEEDS_CLARIFICATION: {
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
}
DEFAULT_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".cache",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "runs",
        "venv",
    }
)
_UNSET = object()
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Configuration timestamps must include an explicit timezone")
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Configuration value {type(value).__name__} cannot be serialized safely")


def _aware_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRepository:
    """Create and maintain append-only Run records."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.runs_dir = self.settings.runs_dir.expanduser().resolve()
        self._clock = clock or _aware_utc_now
        self._lock = threading.RLock()

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RunStorageError("Repository clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _validate_run_id(self, run_id: str) -> None:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise InvalidRunPathError(f"Invalid Run ID: {run_id!r}")

    def run_path(self, run_id: str, *, require_exists: bool = True) -> Path:
        self._validate_run_id(run_id)
        path = self.runs_dir / run_id
        resolved = path.resolve(strict=False)
        if resolved.parent != self.runs_dir:
            raise InvalidRunPathError(f"Run path escapes the runs directory: {run_id!r}")
        if require_exists and not path.is_dir():
            raise RunNotFoundError(f"Run does not exist: {run_id}")
        return path

    def create_run_record(
        self,
        *,
        run_id: str,
        workspace: Path,
        command: str,
        argv: Sequence[str] = (),
        config: Mapping[str, Any] | BaseModel | None = None,
        parent_run_id: str | None = None,
        input_summary: Mapping[str, Any] | None = None,
        max_snapshot_file_size_bytes: int | None = None,
    ) -> RunRecord:
        self._validate_run_id(run_id)
        workspace_path = workspace.expanduser().resolve()
        if not workspace_path.is_dir():
            raise RunStorageError(f"Workspace does not exist or is not a directory: {workspace_path}")
        if not command.strip():
            raise RunStorageError("command cannot be empty")
        if parent_run_id is not None:
            self._validate_run_id(parent_run_id)

        config_value = _jsonable(config or {})
        summary_value = _jsonable(input_summary or {})
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        run_dir = self.run_path(run_id, require_exists=False)
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise RunStorageError(f"Run {run_id} already exists")

        now = self._now()
        run_record = RunRecord(
            run_id=run_id,
            created_at=now,
            updated_at=now,
            status=RunStatus.CREATED,
            workspace=str(workspace_path),
            command=command,
            argv=list(argv),
            config=config_value,
            parent_run_id=parent_run_id,
            input_summary=summary_value,
        )
        try:
            for name in ("workspace", "artifacts"):
                (run_dir / name).mkdir()
            atomic_write_json(run_dir / "run.json", run_record.model_dump(mode="json"))
            atomic_write_json(
                run_dir / "artifacts" / ".index.json",
                ArtifactIndex().model_dump(mode="json"),
            )
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        self.snapshot_workspace(
            run_id,
            max_file_size_bytes=max_snapshot_file_size_bytes,
        )
        run_record = self.read_record(run_id)

        return run_record

    def read_record(self, run_id: str) -> RunRecord:
        path = self.run_path(run_id) / "run.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return RunRecord.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise RunStorageError(f"Cannot read Run record {path}: {exc}") from exc

    def list_runs(self) -> list[RunRecord]:
        """Return readable Runs in reverse creation order."""
        if not self.runs_dir.exists():
            return []
        runs: list[RunRecord] = []
        for path in self.runs_dir.iterdir():
            if not path.is_dir() or not RUN_ID_PATTERN.fullmatch(path.name):
                continue
            try:
                runs.append(self.read_record(path.name))
            except RunStorageError as exc:
                logger.warning("Skipping unreadable Run %s: %s", path.name, exc)
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def update_record(
        self,
        run_id: str,
        *,
        status: RunStatus | str | None = None,
        input_summary: Mapping[str, Any] | object = _UNSET,
        result_summary: Mapping[str, Any] | None | object = _UNSET,
        error_summary: str | None | object = _UNSET,
    ) -> RunRecord:
        with self._lock:
            run_record = self.read_record(run_id)
            has_changes = (
                any(value is not _UNSET for value in (input_summary, result_summary, error_summary))
                or status is not None
            )
            if run_record.status in TERMINAL_STATUSES and has_changes:
                raise RunStateError(
                    f"Run {run_id} is terminal ({run_record.status.value}) and cannot be modified"
                )

            new_status = RunStatus(status) if status is not None else run_record.status
            if new_status != run_record.status:
                allowed = ALLOWED_TRANSITIONS.get(run_record.status, set())
                if new_status not in allowed:
                    raise RunStateError(
                        f"Transition from {run_record.status.value} to {new_status.value} is not allowed"
                    )
            update: dict[str, Any] = {"status": new_status, "updated_at": self._now()}
            if input_summary is not _UNSET:
                update["input_summary"] = _jsonable(input_summary)
            if result_summary is not _UNSET:
                update["result_summary"] = (
                    None if result_summary is None else _jsonable(result_summary)
                )
            if error_summary is not _UNSET:
                update["error_summary"] = error_summary
            updated = run_record.model_copy(update=update)
            atomic_write_json(
                self.run_path(run_id) / "run.json",
                updated.model_dump(mode="json"),
            )
            return updated

    def append_event(self, event: RunEvent) -> RunEvent:
        """Legacy event persistence for existing integrations.

        Normal Runtime execution no longer writes events.jsonl.
        """

        with self._lock:
            existing = self.read_events(event.run_id)
            expected = len(existing) + 1
            if event.sequence != expected:
                raise CorruptEventLogError(
                    f"New event for Run {event.run_id} has sequence={event.sequence}; expected {expected}"
                )
            encoded = (event.model_dump_json() + "\n").encode("utf-8")
            path = self.run_path(event.run_id) / "events.jsonl"
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    written = os.write(descriptor, encoded)
                    if written != len(encoded):
                        raise RunStorageError(
                            f"Event log wrote only {written}/{len(encoded)} bytes; final line may be incomplete"
                        )
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise RunStorageError(f"Cannot append event log {path}: {exc}") from exc
            return event

    def read_events(
        self,
        run_id: str,
        *,
        tolerate_truncated_final_line: bool = False,
    ) -> list[RunEvent]:
        path = self.run_path(run_id) / "events.jsonl"
        if not path.exists():
            return []
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RunStorageError(f"Cannot read event log {path}: {exc}") from exc
        lines = content.splitlines(keepends=True)
        events: list[RunEvent] = []
        for index, raw_line in enumerate(lines, start=1):
            final_line = index == len(lines)
            if not raw_line.strip():
                raise CorruptEventLogError(f"Event log line {index} is empty")
            try:
                event = RunEvent.model_validate_json(raw_line)
            except (ValueError, ValidationError) as exc:
                incomplete_final_line = final_line and not raw_line.endswith((b"\n", b"\r"))
                if tolerate_truncated_final_line and incomplete_final_line:
                    break
                raise CorruptEventLogError(f"Event log line {index} is corrupt: {exc}") from exc
            expected = len(events) + 1
            if event.sequence != expected:
                raise CorruptEventLogError(
                    f"Event log line {index} has sequence={event.sequence}; expected {expected}"
                )
            if event.run_id != run_id:
                raise CorruptEventLogError(
                    f"Event log line {index} belongs to Run {event.run_id}; expected {run_id}"
                )
            events.append(event)
        return events

    def snapshot_workspace(
        self,
        run_id: str,
        *,
        workspace: Path | None = None,
        max_file_size_bytes: int | None = None,
        ignored_names: set[str] | frozenset[str] | None = None,
    ) -> InputSnapshotReport:
        run_dir = self.run_path(run_id)
        run_record = self.read_record(run_id)
        if run_record.status in TERMINAL_STATUSES:
            raise RunStateError(
                f"Run {run_id} is terminal ({run_record.status.value}); cannot create an input snapshot"
            )
        source_root = (workspace or Path(run_record.workspace)).expanduser().resolve()
        if not source_root.is_dir():
            raise RunStorageError(f"Workspace does not exist or is not a directory: {source_root}")
        limit = (
            self.settings.snapshot_max_file_size_bytes
            if max_file_size_bytes is None
            else max_file_size_bytes
        )
        if limit <= 0:
            raise RunStorageError("max_file_size_bytes must be greater than zero")
        ignored = DEFAULT_IGNORED_NAMES | frozenset(ignored_names or ())
        target_root = run_dir / "workspace"
        if (run_dir / "input.snapshot.json").exists() or any(target_root.iterdir()):
            raise RunStorageError(f"Run {run_id} already has an input snapshot; overwrite is disabled")
        report = InputSnapshotReport(
            run_id=run_id,
            workspace=str(source_root),
            created_at=self._now(),
            max_file_size_bytes=limit,
        )

        for current, directories, files in os.walk(source_root, topdown=True, followlinks=False):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directories):
                source = current_path / name
                relative = source.relative_to(source_root)
                try:
                    resolved = source.resolve()
                except OSError as exc:
                    report.errors.append(
                        SnapshotRecord(
                            relative_path=relative.as_posix(), reason=f"resolve_error: {exc}"
                        )
                    )
                    continue
                reason: str | None = None
                if source.is_symlink():
                    reason = "symlink"
                elif name in ignored:
                    reason = "ignored_directory"
                elif resolved == self.runs_dir or self.runs_dir in resolved.parents:
                    reason = "runs_directory"
                if reason:
                    report.ignored.append(
                        SnapshotRecord(relative_path=relative.as_posix(), reason=reason)
                    )
                else:
                    kept_directories.append(name)
            directories[:] = kept_directories

            for name in sorted(files):
                source = current_path / name
                relative = source.relative_to(source_root)
                relative_text = relative.as_posix()
                if source.is_symlink():
                    report.ignored.append(
                        SnapshotRecord(relative_path=relative_text, reason="symlink")
                    )
                    continue
                if name in ignored:
                    report.ignored.append(
                        SnapshotRecord(relative_path=relative_text, reason="ignored_file")
                    )
                    continue
                try:
                    size = source.stat().st_size
                    if size > limit:
                        report.ignored.append(
                            SnapshotRecord(
                                relative_path=relative_text,
                                reason="file_too_large",
                                size_bytes=size,
                            )
                        )
                        continue
                    target = target_root / relative
                    if target.exists():
                        raise RunStorageError(f"Input snapshot target already exists: {target}")
                    content = source.read_bytes()
                    if len(content) > limit:
                        report.ignored.append(
                            SnapshotRecord(
                                relative_path=relative_text,
                                reason="file_too_large",
                                size_bytes=len(content),
                            )
                        )
                        continue
                    atomic_write_bytes(target, content)
                except RunStorageError:
                    raise
                except OSError as exc:
                    report.errors.append(
                        SnapshotRecord(
                            relative_path=relative_text,
                            reason=f"copy_error: {exc}",
                            size_bytes=None,
                        )
                    )
                    continue
                report.copied_files += 1
                report.copied_bytes += len(content)

        atomic_write_json(
            run_dir / "input.snapshot.json",
            report.model_dump(mode="json"),
        )
        self.update_record(
            run_id,
            input_summary={
                "copied_files": report.copied_files,
                "copied_bytes": report.copied_bytes,
                "ignored_files": len(report.ignored),
                "errors": len(report.errors),
            },
        )
        return report

    def _artifact_store(self, run_id: str) -> "ArtifactStore":
        from .artifacts import ArtifactStore

        self.run_path(run_id)
        return ArtifactStore(self, run_id)

    def write_artifact_text(
        self,
        run_id: str,
        relative_path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
        media_type: str | None = None,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self._artifact_store(run_id).write_text(
            relative_path,
            content,
            encoding=encoding,
            media_type=media_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def write_artifact_json(
        self,
        run_id: str,
        relative_path: str | Path,
        value: Any,
        *,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self._artifact_store(run_id).write_json(
            relative_path,
            value,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def write_artifact_bytes(
        self,
        run_id: str,
        relative_path: str | Path,
        content: bytes,
        *,
        media_type: str | None = None,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self._artifact_store(run_id).write_bytes(
            relative_path,
            content,
            media_type=media_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def copy_artifact_file(
        self,
        run_id: str,
        source: Path,
        relative_path: str | Path,
        *,
        media_type: str | None = None,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self._artifact_store(run_id).copy_file(
            source,
            relative_path,
            media_type=media_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def list_artifacts(self, run_id: str) -> list[ArtifactMetadata]:
        return self._artifact_store(run_id).list_artifacts()

    def get_artifact_metadata(self, run_id: str, artifact_id: str) -> ArtifactMetadata:
        return self._artifact_store(run_id).get_metadata(artifact_id)

    def get_artifact_path(self, run_id: str, artifact_id: str) -> Path:
        return self._artifact_store(run_id).get_path(artifact_id)
