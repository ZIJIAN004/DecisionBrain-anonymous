"""Tests for centralized ExitCode definitions."""

from decisionbrain.cli.exit_codes import STATUS_TO_EXIT_CODE, ExitCode
from decisionbrain.run_storage import RunStatus


def test_exit_codes_are_unique():
    """Exit-code values are unique."""
    values = [
        ExitCode.SUCCESS,
        ExitCode.NEEDS_CLARIFICATION,
        ExitCode.USER_INPUT_ERROR,
        ExitCode.INTERNAL_FAILURE,
        ExitCode.CANCELLED,
    ]
    assert len(values) == len(set(values))


def test_status_to_exit_code_coverage():
    """Every status except CREATED has an exit-code mapping."""
    for status in RunStatus:
        if status == RunStatus.CREATED:
            continue
        assert status in STATUS_TO_EXIT_CODE, f"{status} has no exit-code mapping"
