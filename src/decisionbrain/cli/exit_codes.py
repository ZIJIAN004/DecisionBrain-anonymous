"""Central CLI exit-code definitions.

Commands reference ``ExitCode.XXX`` rather than embedding numeric codes.
"""

from ..run_storage import RunStatus


class ExitCode:
    """Shared CLI exit codes."""

    SUCCESS = 0
    NEEDS_CLARIFICATION = 2
    USER_INPUT_ERROR = 3  # Workspace or user-input error
    INTERNAL_FAILURE = 4  # Agent, tool, or unknown internal error
    CANCELLED = 130  # User cancellation (Ctrl+C / SIGINT)


# RunStatus to exit-code mapping
STATUS_TO_EXIT_CODE: dict[RunStatus, int] = {
    RunStatus.CREATED: ExitCode.INTERNAL_FAILURE,  # Not a terminal status
    RunStatus.RUNNING: ExitCode.INTERNAL_FAILURE,  # Not a terminal status
    RunStatus.SUCCEEDED: ExitCode.SUCCESS,
    RunStatus.NEEDS_CLARIFICATION: ExitCode.NEEDS_CLARIFICATION,
    RunStatus.FAILED: ExitCode.INTERNAL_FAILURE,
    RunStatus.CANCELLED: ExitCode.CANCELLED,
}
