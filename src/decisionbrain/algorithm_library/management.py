"""Safe installation of validated algorithm package manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import AlgorithmCatalogError, LocalAlgorithmCatalog, load_algorithm_manifest
from .models import AlgorithmManifest


@dataclass(frozen=True)
class ManifestInstallResult:
    manifest: AlgorithmManifest
    path: Path
    replaced: bool


def install_algorithm_manifest(
    source: Path,
    manifests_dir: Path,
    *,
    replace: bool = False,
) -> ManifestInstallResult:
    """Validate and atomically install one manifest into a live catalog."""

    source = source.expanduser().resolve()
    manifests_dir = manifests_dir.expanduser().resolve()
    if not source.is_file():
        raise AlgorithmCatalogError(f"algorithm manifest does not exist: {source}")

    manifest = load_algorithm_manifest(source)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    destination = manifests_dir / f"{manifest.id}.yaml"
    existed = destination.exists()
    if existed and not replace:
        raise AlgorithmCatalogError(
            f"algorithm manifest {manifest.id!r} already exists: {destination}; use --replace"
        )

    previous = destination.read_bytes() if existed else None
    temporary = destination.with_suffix(".yaml.tmp")
    try:
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        LocalAlgorithmCatalog(manifests_dir)
    except Exception:
        temporary.unlink(missing_ok=True)
        if previous is None:
            destination.unlink(missing_ok=True)
        else:
            destination.write_bytes(previous)
        raise

    return ManifestInstallResult(manifest=manifest, path=destination, replaced=existed)
