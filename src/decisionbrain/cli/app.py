"""Decision Brain CLI application entry point."""

import os

import typer
from rich.console import Console

from .. import __version__
from ..exceptions import ORAgentError
from .commands.chat import chat_command
from .commands.algorithms import algorithms_app
from .commands.dev import dev_app
from .commands.doctor import doctor_command
from .commands.init import init_command
from .commands.resume import resume_command
from .commands.run import run_command
from .context import CLIContext


CONFIG_HELP = """
[bold]Global configuration (environment variables)[/bold]

[cyan]DBN_RUNS_DIR[/cyan]: Run record directory.

[cyan]DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES[/cyan]: per-file input snapshot size limit.

[cyan]DBN_DEBUG[/cyan]: retain exception tracebacks when true.

Configuration is loaded from the repository-root [code].env[/code] and process environment;
process variables take precedence. [code]--debug[/code] temporarily enables full event rendering
and diagnostic output.
"""


app = typer.Typer(
    name="dbn",
    help=(
        "Decision Brain OR Agent: initialize a workspace, run an optimization, "
        "or start an interactive conversation."
    ),
    no_args_is_help=True,
    invoke_without_command=True,
    rich_markup_mode="rich",
    epilog=CONFIG_HELP,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def cli_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    if version:
        typer.echo(f"Decision Brain {__version__}")
        raise typer.Exit()
    ctx.obj = CLIContext(console=Console())


app.command("doctor")(doctor_command)
app.add_typer(algorithms_app, name="algorithms")
app.add_typer(dev_app, name="dev")
app.command("init")(init_command)
app.command("run")(run_command)
app.command("resume")(resume_command)
app.command("chat")(chat_command)


def _debug_enabled() -> bool:
    return os.environ.get("DBN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    """Run the CLI and render expected project exceptions consistently."""

    try:
        app()
    except ORAgentError as exc:
        if _debug_enabled():
            raise
        Console(stderr=True).print(f"[red]Error: [/red]{exc}")
        raise SystemExit(exc.exit_code) from None
    except Exception as exc:
        if _debug_enabled():
            raise
        Console(stderr=True).print(f"[red]Unexpected error: [/red]{exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
