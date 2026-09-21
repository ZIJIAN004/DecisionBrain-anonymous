"""Run an optimization from a directory containing problem.md and optional data."""

from __future__ import annotations

import sys

import typer

from ...exceptions import ConfigurationError, UserInputError
from ...run_storage import FolderWorkspace
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


def run_command(
    ctx: typer.Context,
    folder: str = typer.Argument(
        ".",
        help="Input directory containing problem.md and optional data/.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show trace events, timestamps, configuration, and snapshot diagnostics.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Write machine-readable JSON to stdout and human-readable logs to stderr.",
    ),
) -> None:
    """Run a workspace and handle clarification in an interactive terminal."""
    import asyncio

    cli = get_cli_context(ctx)
    try:
        workspace = FolderWorkspace.from_path(folder)
        if not workspace.root.is_dir():
            raise UserInputError(f"Input path does not exist or is not a directory: {workspace.root}")
        if not workspace.has_problem:
            raise UserInputError(f"Input directory has no non-empty problem.md: {workspace.root}")

        issues = workspace.validate()
        if issues:
            raise UserInputError("；".join(issues))

        settings = resolve_cli_config(
            settings=cli.settings,
            debug=debug,
        )
        runtime = AgentRuntime(settings)
        repository, sink, observer = build_runtime_components(
            console=cli.console,
            settings=settings,
            runtime=runtime,
            json_mode=json_output,
        )
        asyncio.run(
            run_runtime_with_user_feedback(
                runtime=runtime,
                run_call=lambda: runtime.run(
                    workspace=workspace.root,
                    command=f"dbn run {folder}",
                    argv=sys.argv[1:],
                    config=settings.to_runtime_config() | {"source": "cli-folder"},
                    initial_input=workspace.problem_description,
                    sink=sink,
                    max_snapshot_file_size_bytes=settings.snapshot_max_file_size_bytes,
                ),
                observer=observer,
                console=cli.console,
                read_answer=input,
                interactive=_stdin_is_tty(),
            )
        )
        if observer.run_id is None:
            raise RuntimeError("Runtime did not publish a run_id")
        run_id = observer.run_id

        output_result(
            run_id,
            repository=repository,
            observer=observer,
            console=cli.console,
            json_mode=json_output,
        )
        status = repository.read_record(run_id).status
        raise typer.Exit(code=STATUS_TO_EXIT_CODE.get(status, ExitCode.INTERNAL_FAILURE))
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
