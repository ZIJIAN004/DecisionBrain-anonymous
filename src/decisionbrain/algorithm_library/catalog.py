"""Local YAML-backed algorithm package manifest catalog."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import ValidationError

from .models import AlgorithmManifest


class AlgorithmCatalogError(ValueError):
    """The manifest catalog is missing, invalid, or internally inconsistent."""


class AlgorithmCatalog(Protocol):
    def list_available(self) -> tuple[AlgorithmManifest, ...]: ...

    def get(self, package_id: str) -> AlgorithmManifest: ...


def load_algorithm_manifest(path: Path) -> AlgorithmManifest:
    """Load one manifest with the same strict validation used by the catalog."""

    return LocalAlgorithmCatalog._load_manifest(path.expanduser().resolve())


class LocalAlgorithmCatalog:
    """Eagerly load and validate package manifests from one local directory."""

    def __init__(
        self,
        manifests_dir: Path,
        *,
        allowed_package_ids: Collection[str] | None = None,
    ) -> None:
        self.manifests_dir = manifests_dir.expanduser().resolve()
        self._manifests = self._load_manifests()
        # Package-pool ablation (arm A): filter at this layer so list_available and get
        # expose the same pool. A guide cannot remain readable after leaving the summary.
        if allowed_package_ids is not None:
            allowed = frozenset(allowed_package_ids)
            unknown = sorted(allowed - set(self._manifests))
            if unknown:
                raise AlgorithmCatalogError(
                    "package pool references unknown package_id: " + ", ".join(unknown)
                )
            self._manifests = {
                package_id: manifest
                for package_id, manifest in self._manifests.items()
                if package_id in allowed
            }
            if not self._manifests:
                raise AlgorithmCatalogError("package pool selected no available manifest")

    def list_available(self) -> tuple[AlgorithmManifest, ...]:
        manifests = [
            manifest for manifest in self._manifests.values() if manifest.status != "disabled"
        ]
        status_order = {"available": 0, "experimental": 1, "deprecated": 2}
        manifests.sort(key=lambda item: (status_order[item.status], item.id))
        return tuple(manifests)

    def get(self, package_id: str) -> AlgorithmManifest:
        normalized = str(package_id or "").strip()
        if not normalized:
            raise AlgorithmCatalogError("package_id must be a non-empty string")
        try:
            return self._manifests[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._manifests)) or "none"
            raise AlgorithmCatalogError(
                f"unknown package_id {normalized!r}; available: {available}"
            ) from exc

    def _load_manifests(self) -> dict[str, AlgorithmManifest]:
        paths = self._yaml_paths(self.manifests_dir, label="algorithm manifests")
        manifests: dict[str, AlgorithmManifest] = {}
        for path in paths:
            manifest = self._load_manifest(path)
            if manifest.id in manifests:
                raise AlgorithmCatalogError(
                    f"duplicate algorithm manifest id {manifest.id!r} in {path}"
                )
            manifests[manifest.id] = manifest
        return manifests

    @staticmethod
    def _yaml_paths(directory: Path, *, label: str) -> list[Path]:
        if not directory.is_dir():
            raise AlgorithmCatalogError(f"{label} directory does not exist: {directory}")
        paths = sorted((*directory.glob("*.yaml"), *directory.glob("*.yml")))
        if not paths:
            raise AlgorithmCatalogError(f"{label} directory is empty: {directory}")
        return paths

    @staticmethod
    def _load_manifest(path: Path) -> AlgorithmManifest:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise AlgorithmCatalogError(
                f"failed to read algorithm manifest {path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise AlgorithmCatalogError(f"algorithm manifest must be an object: {path}")
        try:
            return AlgorithmManifest.model_validate(raw)
        except ValidationError as exc:
            raise AlgorithmCatalogError(f"invalid algorithm manifest {path}: {exc}") from exc
