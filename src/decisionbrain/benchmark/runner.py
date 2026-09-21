"""Stable benchmark entry point.

The implementation is split across dedicated benchmark modules while this
module preserves ``python -m decisionbrain.benchmark.runner`` and historical
imports used by experiment launchers and external callers.
"""

from __future__ import annotations

from .execution import (
    DEFAULT_FRONTIEROR65_FEA_INDEX,
    DEFAULT_SUITE,
    DEFAULT_TASKS_ROOT,
    FRONTIEROR10_INF_INDEX,
    FRONTIEROR10_INF_ROOT,
    HARD32_FEA_ROOT,
    HARD32_FEA_SELECTION,
    FrontierORReport,
    TaskEvaluation,
    _parser as _parser,
    _validate_workflow_ablation_args as _validate_workflow_ablation_args,
    evaluate_task,
    main,
    run_benchmark,
)

__all__ = [
    "DEFAULT_FRONTIEROR65_FEA_INDEX",
    "DEFAULT_SUITE",
    "DEFAULT_TASKS_ROOT",
    "FRONTIEROR10_INF_INDEX",
    "FRONTIEROR10_INF_ROOT",
    "HARD32_FEA_ROOT",
    "HARD32_FEA_SELECTION",
    "FrontierORReport",
    "TaskEvaluation",
    "evaluate_task",
    "main",
    "run_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
