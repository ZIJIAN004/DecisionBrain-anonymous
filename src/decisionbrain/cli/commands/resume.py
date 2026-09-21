"""Restart a stage from an existing Run with ``dbn resume``."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from ...core.models import AgentStage
from ...core.package_policy import PackagePolicy
from ...core.stage_flow import STAGE_SEQUENCE
from ...exceptions import ConfigurationError, UserInputError
from ...run_storage import RunRepository
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

# Derive this set from the stage sequence so new stages cannot be omitted.
_VALID_STAGES = frozenset(stage.value for stage in STAGE_SEQUENCE)


def _validate_stage_for_parent(stage: str, config: dict[str, object]) -> None:
    if stage == AgentStage.GUROBI_FORMULATOR.value and not (
        config.get("gurobi_formulator_enabled") is True
        or config.get("workflow") == "gurobi-formulator"
    ):
        raise UserInputError(
            "The parent Run did not enable the gurobi-formulator workflow; cannot resume here"
        )


def _resolve_run_id(run_id: str, settings_runs_dir: Path) -> Path:
    """Resolve a Run ID or relative/absolute path to a Run directory."""
    # The supplied path is itself a Run directory.
    cwd = Path.cwd()
    if cwd.name == run_id and (cwd / "run.json").is_file():
        return cwd

    # The current directory contains RUN_ID/.
    candidate = cwd / run_id
    if candidate.is_dir() and (candidate / "run.json").is_file():
        return candidate

    # The current directory contains runs/RUN_ID/.
    candidate = cwd / "runs" / run_id
    if candidate.is_dir() and (candidate / "run.json").is_file():
        return candidate

    # Fall back to DBN_RUNS_DIR.
    candidate = settings_runs_dir / run_id
    if candidate.is_dir() and (candidate / "run.json").is_file():
        return candidate

    raise UserInputError(f"Run not found: {run_id}")


def resume_command(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run ID to resume from."),
    stage: str = typer.Argument(
        ...,
        help=f"Stage at which execution restarts ({' / '.join(sorted(_VALID_STAGES))}).",
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
    """Create a new Run from existing stage outputs and restart at a selected stage."""
    import asyncio

    cli = get_cli_context(ctx)
    try:
        if stage not in _VALID_STAGES:
            raise UserInputError(
                f"Invalid stage {stage!r}; expected one of: {', '.join(sorted(_VALID_STAGES))}"
            )

        settings = resolve_cli_config(settings=cli.settings, debug=debug)
        run_dir = _resolve_run_id(run_id, settings.runs_dir)

        # Read the authoritative Run ID from run.json rather than trusting the directory name.
        repo = RunRepository(settings)
        record = repo.read_record(run_dir.name)
        actual_run_id = record.run_id
        _validate_stage_for_parent(stage, record.config)

        parent_config = record.config
        runtime = AgentRuntime(
            settings,
            input_schema_enabled=parent_config.get("input_schema_enabled", True),
            algorithm_library_enabled=parent_config.get("algorithm_library_enabled", True),
            feasibility_review_enabled=parent_config.get("feasibility_review_enabled", True),
            algorithm_design_enabled=parent_config.get("algorithm_design_enabled", True),
            problem_contract_enabled=parent_config.get("problem_contract_enabled", True),
            components_enabled=parent_config.get("components_enabled", True),
            package_policy=PackagePolicy(
                pool=parent_config.get("package_pool", "full"),
                cross_package_enabled=parent_config.get("cross_package_enabled", True),
            ),
        )
        repository, sink, observer = build_runtime_components(
            console=cli.console,
            settings=settings,
            runtime=runtime,
            json_mode=json_output,
        )

        resolved_stage = AgentStage(str(stage))

        asyncio.run(
            run_runtime_with_user_feedback(
                runtime=runtime,
                run_call=lambda: runtime.resume_run(
                    sink=sink,
                    parent_run_id=actual_run_id,
                    resume_stage=resolved_stage,
                    command=f"dbn resume {actual_run_id} {stage}",
                    argv=sys.argv[1:],
                    config=settings.to_runtime_config() | {"source": "cli-resume"},
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
        new_run_id = observer.run_id

        output_result(
            new_run_id,
            repository=repository,
            observer=observer,
            console=cli.console,
            json_mode=json_output,
            extra_fields={
                "parent_run_id": actual_run_id,
                "resume_stage": stage,
            },
        )
        new_status = repository.read_record(new_run_id).status
        raise typer.Exit(code=STATUS_TO_EXIT_CODE.get(new_status, ExitCode.INTERNAL_FAILURE))
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
