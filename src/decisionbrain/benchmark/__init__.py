"""Benchmark support for task discovery, Intake bypass, execution, and scoring.

This package is evaluation-only. Product runtime code must not depend on it.
"""

from .adapter import BenchmarkAdaptationError, BenchmarkSolutionAdapter
from .intake_evaluation import (
    ClarificationAnswer,
    IntakeAnswerer,
    IntakeDialogueTurn,
    IntakeEvaluationMaterials,
    IntakeEvaluator,
    IntakeGoldStandard,
    IntakeJudgeReport,
    build_intake_evaluation_tools,
)
from .intake_data_tools import IntakeDataToolError, build_intake_data_tools
from .task import BenchmarkTask, discover_tasks, load_large_max_tasks, load_task

__all__ = [
    "BenchmarkAdaptationError",
    "BenchmarkSolutionAdapter",
    "BenchmarkTask",
    "ClarificationAnswer",
    "IntakeAnswerer",
    "IntakeDialogueTurn",
    "IntakeEvaluationMaterials",
    "IntakeDataToolError",
    "IntakeEvaluator",
    "IntakeGoldStandard",
    "IntakeJudgeReport",
    "build_intake_evaluation_tools",
    "build_intake_data_tools",
    "discover_tasks",
    "load_large_max_tasks",
    "load_task",
]
