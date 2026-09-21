"""Commands for managing the Agent-visible algorithm package catalog."""

from pathlib import Path

import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ...algorithm_library import (
    LocalAlgorithmCatalog,
    load_algorithm_manifest,
    validate_algorithm_library_runtime,
)
from ...algorithm_library.management import install_algorithm_manifest
from ...paths import ALGORITHM_MANIFESTS_DIR
from ..context import get_cli_context


algorithms_app = typer.Typer(
    name="algorithms",
    help="Add and inspect algorithm packages available to the Agent.",
    no_args_is_help=True,
)


class _AlgorithmCatalogSettings(BaseSettings):
    """Configuration required by catalog commands, independent of Agent runtime."""

    manifests_dir: Path = Field(
        default=ALGORITHM_MANIFESTS_DIR,
        validation_alias="OPT_ALGORITHM_MANIFESTS_DIR",
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _configured_manifests_dir() -> Path:
    return _AlgorithmCatalogSettings().manifests_dir


class _SingleManifestCatalog:
    def __init__(self, manifest):
        self.manifest = manifest

    def list_available(self):
        return (self.manifest,)

    def get(self, package_id: str):
        if package_id != self.manifest.id:
            raise KeyError(package_id)
        return self.manifest


@algorithms_app.command("add")
def add_algorithm_command(
    ctx: typer.Context,
    manifest: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="A complete YAML package manifest using a supported strict interface profile.",
    ),
    manifests_dir: Path | None = typer.Option(
        None,
        "--manifests-dir",
        file_okay=False,
        resolve_path=True,
        help="Target catalog directory; defaults to OPT_ALGORITHM_MANIFESTS_DIR.",
    ),
    replace: bool = typer.Option(False, "--replace", help="Replace the same package ID."),
    check_runtime: bool = typer.Option(
        False,
        "--check-runtime",
        help="Also run import and minimal-call checks for the resulting catalog.",
    ),
) -> None:
    """Validate and connect an algorithm package manifest to the Agent catalog."""

    cli = get_cli_context(ctx)
    target = manifests_dir or _configured_manifests_dir()
    if check_runtime:
        validate_algorithm_library_runtime(_SingleManifestCatalog(load_algorithm_manifest(manifest)))
    result = install_algorithm_manifest(manifest, target, replace=replace)
    catalog = LocalAlgorithmCatalog(target)
    action = "Replaced" if result.replaced else "Added"
    cli.console.print(
        f"[green]{action}[/green] {result.manifest.id} at {result.path} "
        f"({len(catalog.list_available())} available packages)"
    )


@algorithms_app.command("list")
def list_algorithms_command(ctx: typer.Context) -> None:
    """List packages currently visible to the Agent."""

    cli = get_cli_context(ctx)
    catalog = LocalAlgorithmCatalog(_configured_manifests_dir())
    for manifest in catalog.list_available():
        cli.console.print(
            f"{manifest.id}\t{manifest.distribution.verified_version}\t{manifest.status}"
        )
