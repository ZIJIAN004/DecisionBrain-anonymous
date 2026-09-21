"""Adapter for the ``dbn doctor`` command."""

import json

import typer
from rich.table import Table

from ..context import get_cli_context
from ..doctor import CheckStatus, DoctorReport, run_doctor


def _render_table(report: DoctorReport, console) -> None:
    table = Table(title="DecisionBrain Environment Diagnostics")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Level")
    table.add_column("Status")
    table.add_column("Details")
    status_styles = {
        CheckStatus.PASS: "[green]PASS[/green]",
        CheckStatus.FAIL: "[red]FAIL[/red]",
        CheckStatus.WARN: "[yellow]WARN[/yellow]",
    }
    for check in report.checks:
        table.add_row(check.name, check.level.value, status_styles[check.status], check.message)
    console.print(table)
    if report.ok:
        console.print("[green]All required checks passed.[/green]")
    else:
        console.print("[red]One or more required checks failed.[/red]")


def doctor_command(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Write check results as machine-readable JSON.",
    ),
) -> None:
    """Check the CLI environment and optional external tools."""

    cli = get_cli_context(ctx)
    report = run_doctor(cli.settings)
    if json_output:
        typer.echo(json.dumps(report.as_json_dict(), ensure_ascii=False, indent=2))
    else:
        _render_table(report, cli.console)
    if not report.ok:
        raise typer.Exit(code=1)
