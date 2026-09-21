"""CLI dependencies with lazily loaded runtime configuration."""

import typer
from rich.console import Console

from ..config import Settings


class CLIContext:
    """Keep metadata-only commands independent from Agent runtime settings."""

    __slots__ = ("_settings", "console")

    def __init__(self, *, console: Console, settings: Settings | None = None) -> None:
        self.console = console
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings


def get_cli_context(ctx: typer.Context) -> CLIContext:
    """Return initialized CLI dependencies from the Typer context."""

    if not isinstance(ctx.obj, CLIContext):
        ctx.obj = CLIContext(console=Console())
    return ctx.obj
