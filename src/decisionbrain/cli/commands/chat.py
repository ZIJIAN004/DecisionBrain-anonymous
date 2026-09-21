"""Interactive Run client that does not require an existing workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...exceptions import ConfigurationError, UserInputError
from ...run_storage import InputWorkspace
from ...runtime import AgentRuntime
from ..context import get_cli_context
from ..exit_codes import STATUS_TO_EXIT_CODE, ExitCode
from ._runtime import (
    _stdin_is_tty,
    build_runtime_components,
    output_result,
    resolve_cli_config,
    run_runtime_with_user_feedback,
)


def _show_help(console) -> None:
    table = Table(title="Chat Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    rows = (
        ("/add PATH", "Add one data file before the first message"),
        ("/files", "List staged data files"),
        ("/status", "Show the current Run status"),
        ("/artifacts", "List artifacts from the current Run"),
        ("/run-id", "Show the current Run ID"),
        ("/help", "Show this help"),
        ("/quit", "Exit"),
    )
    for command, description in rows:
        table.add_row(command, description)
    console.print(table)


def _show_files(workspace: InputWorkspace, console) -> None:
    files = workspace.list_files()
    if not files:
        console.print("[dim]No data files have been added.[/dim]")
        return
    console.print("[bold]Staged data:[/bold]")
    for path in files:
        console.print(f"  - {path.name}")


def _show_run_command(command: str, run_id: str, repository, console) -> None:
    if command == "/status":
        run_record = repository.read_record(run_id)
        console.print(f"[bold]Current status:[/bold] {run_record.status.value}")
    elif command == "/run-id":
        console.print(f"[bold]Run ID:[/bold] {run_id}")
    elif command == "/artifacts":
        artifacts = repository.list_artifacts(run_id)
        if not artifacts:
            console.print("[dim]No artifacts are available.[/dim]")
        for artifact in artifacts:
            console.print(f"  - {artifact.relative_path}")


def chat_command(
    ctx: typer.Context,
    files: list[str] | None = typer.Option(
        None,
        "--file",
        help="Add a data file; repeatable. Relative paths resolve from the startup directory.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show trace events, timestamps, configuration, and snapshot diagnostics.",
    ),
) -> None:
    """Create and follow a Run from a question and local data."""
    import asyncio

    cli = get_cli_context(ctx)
    if not _stdin_is_tty():
        cli.console.print("[red]Input error: [/red]dbn chat requires an interactive terminal.")
        raise typer.Exit(code=ExitCode.USER_INPUT_ERROR)

    invocation_dir = Path.cwd().resolve()
    workspace = InputWorkspace.create(prefix="decisionbrain-cli-")
    try:
        for path in files or []:
            added = workspace.add_local_file(path, base_dir=invocation_dir)
            cli.console.print(f"[green]Added: [/green]{added.name}")

        cli.console.print(
            Panel(
                Text("Use /add PATH to attach data before the first ordinary message starts the Run."),
                title="DecisionBrain Chat",
                border_style="blue",
            )
        )

        prompt = ""
        while not prompt:
            line = input("Question> ").strip()
            if not line:
                continue
            if line == "/quit":
                raise typer.Exit(code=ExitCode.SUCCESS)
            if line == "/help":
                _show_help(cli.console)
                continue
            if line == "/files":
                _show_files(workspace, cli.console)
                continue
            if line.startswith("/add"):
                path = line[4:].strip()
                if not path:
                    cli.console.print("[yellow]Usage: /add PATH[/yellow]")
                    continue
                added = workspace.add_local_file(path, base_dir=invocation_dir)
                cli.console.print(f"[green]Added: [/green]{added.name}")
                continue
            if line.startswith("/"):
                cli.console.print("[yellow]The Run has not started; enter /help for commands.[/yellow]")
                continue
            prompt = line

        workspace.write_problem_description(prompt)

        settings = resolve_cli_config(
            settings=cli.settings,
            debug=debug,
        )
        runtime = AgentRuntime(settings)
        repository, sink, observer = build_runtime_components(
            console=cli.console,
            settings=settings,
            runtime=runtime,
        )

        def handle_clarification_command(line: str):
            command = line.lower()
            if command == "/quit":
                return "abort"
            if command == "/help":
                _show_help(cli.console)
                return "handled"
            if command in {"/status", "/artifacts", "/run-id"}:
                if observer.run_id is None:
                    cli.console.print("[yellow]The Run has not started.[/yellow]")
                    return "handled"
                _show_run_command(command, observer.run_id, repository, cli.console)
                return "handled"
            if command == "/files":
                if observer.run_id is None:
                    cli.console.print("[yellow]The Run has not started.[/yellow]")
                    return "handled"
                data_dir = repository.run_path(observer.run_id) / "workspace" / "data"
                for path in sorted(data_dir.iterdir() if data_dir.is_dir() else []):
                    if path.is_file():
                        cli.console.print(f"  - {path.name}")
                return "handled"
            if command.startswith("/add"):
                cli.console.print("[yellow]Files cannot be added after a Run starts; create a new Run.[/yellow]")
                return "handled"
            if command.startswith("/"):
                cli.console.print("[yellow]Unknown command; enter /help for available commands.[/yellow]")
                return "handled"
            return "answer"

        asyncio.run(
            run_runtime_with_user_feedback(
                runtime=runtime,
                run_call=lambda: runtime.run(
                    workspace=workspace.root,
                    command=prompt,
                    argv=sys.argv[1:],
                    config=settings.to_runtime_config() | {"source": "cli-chat"},
                    initial_input=prompt,
                    sink=sink,
                    max_snapshot_file_size_bytes=settings.snapshot_max_file_size_bytes,
                ),
                observer=observer,
                console=cli.console,
                read_answer=input,
                interactive=True,
                handle_command=handle_clarification_command,
            )
        )
        workspace.cleanup()
        if observer.run_id is None:
            raise RuntimeError("Runtime did not publish a run_id")
        run_id = observer.run_id

        output_result(run_id, repository=repository, observer=observer, console=cli.console)
        status = repository.read_record(run_id).status
        raise typer.Exit(code=STATUS_TO_EXIT_CODE.get(status, ExitCode.INTERNAL_FAILURE))
    except EOFError:
        raise typer.Exit(code=ExitCode.NEEDS_CLARIFICATION)
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=ExitCode.CANCELLED)
    except (UserInputError, ConfigurationError) as exc:
        cli.console.print(f"[red]Input error: [/red]{exc}")
        raise typer.Exit(code=ExitCode.USER_INPUT_ERROR)
    except typer.Exit:
        raise
    except Exception as exc:
        if debug:
            raise
        cli.console.print(f"[red]Run failed: [/red]{exc}")
        raise typer.Exit(code=ExitCode.INTERNAL_FAILURE)
    finally:
        workspace.cleanup()
