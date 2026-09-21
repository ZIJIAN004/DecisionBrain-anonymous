"""Initialize a DecisionBrain workspace with ``dbn init``."""

from pathlib import Path

import typer

from ..context import get_cli_context

# ---------------------------------------------------------------------------
# Template content
# ---------------------------------------------------------------------------

_PROBLEM_MD_TEMPLATE = """\
# Optimization Problem

## Objective

<!-- Describe the objective, such as minimizing cost or maximizing profit. -->

## Constraints

<!-- List all constraints, such as capacity, budget, or time-window limits. -->

## Data

<!-- Place relevant files under data/ and describe them here. -->

## Notes

<!-- Add any other relevant information. -->
"""

_GITIGNORE_TEMPLATE = """\
/runs/
*.log
"""

# Template files
_TEMPLATES: dict[str, str] = {
    "problem.md": _PROBLEM_MD_TEMPLATE,
    ".gitignore": _GITIGNORE_TEMPLATE,
}

# Template directories
_TEMPLATE_DIRS = ["data"]


def _resolve_path(path_arg: str | None) -> Path:
    """Resolve a user path, defaulting to the current directory."""
    raw = path_arg or "."
    return Path(raw).expanduser().resolve()


def init_command(
    ctx: typer.Context,
    path: str = typer.Argument(
        ".",
        help="Directory to initialize; defaults to the current directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite template files without deleting the directory or unrelated files.",
    ),
) -> None:
    """Initialize a DecisionBrain workspace in the current or specified directory.

    Create a problem.md template, data/ directory, and .gitignore.
    """
    cli = get_cli_context(ctx)
    target = _resolve_path(path)

    # Validate the target path.
    if target.exists() and not target.is_dir():
        cli.console.print(f"[red]Error: [/red]Path exists and is not a directory: {target}")
        raise typer.Exit(code=1)

    # Create a missing target directory.
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    # Confirm the target remains a directory.
    if not target.is_dir():
        cli.console.print(f"[red]Error: [/red]Target path is not a directory: {target}")
        raise typer.Exit(code=1)

    # Check template conflicts unless --force was supplied.
    conflicts = [name for name in _TEMPLATES if (target / name).exists()] + [
        "data/"
        for _ in [1]  # Check once.
        if (target / "data").exists()
    ]

    if conflicts and not force:
        cli.console.print(
            "[red]The following files already exist; use --force to overwrite them:[/red]\n"
            + "\n".join(f"  - {c}" for c in conflicts)
        )
        raise typer.Exit(code=1)

    # Create template directories.
    for dirname in _TEMPLATE_DIRS:
        dir_path = target / dirname
        dir_path.mkdir(parents=True, exist_ok=True)

    # Write template files.
    created: list[str] = []
    overwritten: list[str] = []

    for filename, content in _TEMPLATES.items():
        file_path = target / filename
        existed = file_path.exists()
        file_path.write_text(content, encoding="utf-8")
        if existed:
            overwritten.append(filename)
        else:
            created.append(filename)

    # Render the result.
    cli.console.print(f"\n[bold green]Workspace initialized: [/bold green]{target}")
    if created:
        cli.console.print(f"  Created: [dim]{', '.join(created)}[/dim]")
    if overwritten:
        cli.console.print(f"  Overwritten: [yellow]{', '.join(overwritten)}[/yellow]")
    if "data/" in conflicts:
        cli.console.print("  [dim]data/ already exists; skipped[/dim]")

    cli.console.print("\n[bold]Next steps:[/bold]")
    cli.console.print(f"  [cyan]dbn run {target}[/cyan]     # Run the optimization")
    cli.console.print("  [cyan]dbn chat[/cyan]             # Start an interactive conversation")
    cli.console.print()
