"""Domain models for DecisionBrain Run records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schema import VersionedModel


class RunStatus(str, Enum):
    """Persisted lifecycle state for one Run."""

    CREATED = "created"
    RUNNING = "running"
    NEEDS_CLARIFICATION = "needs_clarification"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PersistedModel(VersionedModel):
    """Base for top-level persisted models versioned by VersionedModel."""


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value


class RunRecord(PersistedModel):
    run_id: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    status: RunStatus = RunStatus.CREATED
    workspace: str = Field(min_length=1)
    command: str = Field(min_length=1)
    argv: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] | None = None
    error_summary: str | None = None

    _created_at_is_aware = field_validator("created_at")(_require_aware)
    _updated_at_is_aware = field_validator("updated_at")(_require_aware)


class ArtifactMetadata(PersistedModel):
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    category: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    producer: str = Field(min_length=1)
    description: str = ""

    _created_at_is_aware = field_validator("created_at")(_require_aware)

    @field_validator("relative_path")
    @classmethod
    def relative_path_stays_in_artifacts(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "artifacts"
            or ".." in path.parts
        ):
            raise ValueError("Artifact metadata path must be under artifacts/")
        return path.as_posix()


class ArtifactIndex(PersistedModel):
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)


class SnapshotRecord(BaseModel):
    relative_path: str
    reason: str
    size_bytes: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class InputSnapshotReport(PersistedModel):
    run_id: str
    workspace: str
    created_at: datetime
    max_file_size_bytes: int = Field(gt=0)
    copied_files: int = Field(default=0, ge=0)
    copied_bytes: int = Field(default=0, ge=0)
    ignored: list[SnapshotRecord] = Field(default_factory=list)
    errors: list[SnapshotRecord] = Field(default_factory=list)

    _created_at_is_aware = field_validator("created_at")(_require_aware)
