"""Stable domain interface for Run records."""

from ..schema import CURRENT_SCHEMA_VERSION
from .input_workspace import FolderWorkspace, InputWorkspace
from .html_report import AgentHtmlReport
from .models import (
    ArtifactIndex,
    ArtifactMetadata,
    InputSnapshotReport,
    RunRecord,
    RunStatus,
    SnapshotRecord,
)
from .repository import RunRepository

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "AgentHtmlReport",
    "ArtifactIndex",
    "ArtifactMetadata",
    "FolderWorkspace",
    "InputSnapshotReport",
    "InputWorkspace",
    "RunRecord",
    "RunRepository",
    "RunStatus",
    "SnapshotRecord",
]
