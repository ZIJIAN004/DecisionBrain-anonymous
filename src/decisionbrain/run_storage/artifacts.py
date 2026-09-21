"""Artifact storage confined to a Run directory."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..exceptions import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    InvalidRunPathError,
    RunStorageError,
)
from .io import atomic_write_bytes, atomic_write_json
from .models import ArtifactIndex, ArtifactMetadata

if TYPE_CHECKING:
    from .repository import RunRepository


class ArtifactStore:
    INDEX_NAME = ".index.json"

    def __init__(self, repository: RunRepository, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id
        self.run_dir = repository.run_path(run_id)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.index_path = self.artifacts_dir / self.INDEX_NAME
        self._lock = repository._lock

    def _target(self, relative_path: str | Path) -> tuple[Path, str]:
        raw = str(relative_path)
        pure = PurePosixPath(raw.replace("\\", "/"))
        if (
            not raw
            or pure.is_absolute()
            or (len(raw) >= 3 and raw[1] == ":" and raw[2] in {"/", "\\"})
            or pure == PurePosixPath(".")
            or ".." in pure.parts
            or pure.parts == (self.INDEX_NAME,)
        ):
            raise InvalidRunPathError(f"Invalid artifact relative path: {raw!r}")
        target = self.artifacts_dir.joinpath(*pure.parts)
        resolved = target.resolve(strict=False)
        artifacts_root = self.artifacts_dir.resolve()
        if resolved != artifacts_root and artifacts_root not in resolved.parents:
            raise InvalidRunPathError(f"Artifact path escapes the current Run: {raw!r}")
        current = self.artifacts_dir
        for part in pure.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise InvalidRunPathError(f"Artifact path contains a symbolic link: {raw!r}")
        return target, pure.as_posix()

    def _read_index(self) -> ArtifactIndex:
        try:
            return ArtifactIndex.model_validate_json(self.index_path.read_bytes())
        except (OSError, ValueError, ValidationError) as exc:
            raise RunStorageError(f"Cannot read artifact index {self.index_path}: {exc}") from exc

    def _write(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        media_type: str | None,
        category: str,
        producer: str,
        description: str,
        overwrite: bool,
    ) -> ArtifactMetadata:
        target, normalized = self._target(relative_path)
        with self._lock:
            index = self._read_index()
            existing = next(
                (
                    item
                    for item in index.artifacts
                    if item.relative_path == f"artifacts/{normalized}"
                ),
                None,
            )
            if (target.exists() or existing is not None) and not overwrite:
                raise ArtifactConflictError(f"Artifact already exists: {normalized}")
            if target.exists() and target.is_dir():
                raise ArtifactConflictError(f"Artifact target is a directory: {normalized}")
            if target.is_symlink():
                raise InvalidRunPathError(f"Artifact target is a symbolic link: {normalized}")

            guessed_type, _encoding = mimetypes.guess_type(normalized)
            metadata = ArtifactMetadata(
                artifact_id=existing.artifact_id if existing else uuid.uuid4().hex,
                run_id=self.run_id,
                relative_path=f"artifacts/{normalized}",
                media_type=media_type or guessed_type or "application/octet-stream",
                category=category,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                created_at=self.repository._now(),
                producer=producer,
                description=description,
            )
            artifacts = [
                item for item in index.artifacts if item.artifact_id != metadata.artifact_id
            ]
            artifacts.append(metadata)
            artifacts.sort(key=lambda item: item.relative_path)
            previous_content = target.read_bytes() if target.exists() else None
            atomic_write_bytes(target, content)
            try:
                atomic_write_json(
                    self.index_path,
                    ArtifactIndex(artifacts=artifacts).model_dump(mode="json"),
                )
            except Exception:
                if previous_content is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(target, previous_content)
                raise
            return metadata

    def write_text(
        self,
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
        effective_type = media_type
        if effective_type is None and str(relative_path).lower().endswith((".txt", ".md", ".csv")):
            guessed, _encoding = mimetypes.guess_type(str(relative_path))
            effective_type = guessed or "text/plain"
        return self._write(
            relative_path,
            content.encode(encoding),
            media_type=effective_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def write_json(
        self,
        relative_path: str | Path,
        value: Any,
        *,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        try:
            content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RunStorageError(f"Artifact JSON is not serializable: {exc}") from exc
        return self._write(
            relative_path,
            content,
            media_type="application/json",
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def write_bytes(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        media_type: str | None = None,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self._write(
            relative_path,
            bytes(content),
            media_type=media_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def copy_file(
        self,
        source: Path,
        relative_path: str | Path,
        *,
        media_type: str | None = None,
        category: str = "artifact",
        producer: str = "decisionbrain",
        description: str = "",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        source_path = source.expanduser()
        if not source_path.is_file():
            raise RunStorageError(f"Artifact source does not exist or is not a regular file: {source_path}")
        try:
            content = source_path.read_bytes()
        except OSError as exc:
            raise RunStorageError(f"Cannot read artifact source {source_path}: {exc}") from exc
        return self._write(
            relative_path,
            content,
            media_type=media_type,
            category=category,
            producer=producer,
            description=description,
            overwrite=overwrite,
        )

    def list_artifacts(self) -> list[ArtifactMetadata]:
        with self._lock:
            return list(self._read_index().artifacts)

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata:
        with self._lock:
            for metadata in self._read_index().artifacts:
                if metadata.artifact_id == artifact_id:
                    return metadata
        raise ArtifactNotFoundError(f"Artifact does not exist: {artifact_id}")

    def get_path(self, artifact_id: str) -> Path:
        """Return the artifact path after validating it against metadata."""
        metadata = self.get_metadata(artifact_id)
        prefix = "artifacts/"
        if not metadata.relative_path.startswith(prefix):
            raise RunStorageError(f"Invalid artifact path: {metadata.relative_path}")
        path, _normalized = self._target(metadata.relative_path[len(prefix) :])
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact content does not exist: {artifact_id}")
        return path
