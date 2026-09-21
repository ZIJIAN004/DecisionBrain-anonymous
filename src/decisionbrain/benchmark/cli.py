"""Command-line contract for benchmark execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.package_policy import PACKAGE_POOLS


def build_parser(*, default_tasks_root: Path, default_suite: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FrontierOR tasks through the Core Agent and hidden checker."
    )
    parser.add_argument("--root", type=Path, default=default_tasks_root, help="Task collection directory")
    parser.add_argument("--task", action="append", default=[], help="Run only this task ID; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N sorted tasks")
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent tasks (default: 1)")
    parser.add_argument(
        "--intake-mode",
        choices=("bypass", "full"),
        default="bypass",
        help="Use the historical bypass flow or full Intake without gold data (default: bypass)",
    )
    parser.add_argument(
        "--max-clarification-rounds",
        type=int,
        default=5,
        help="Maximum automatic feedback rounds in full Intake mode (default: 5)",
    )
    parser.add_argument(
        "--feasibility-review",
        dest="feasibility_review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable the complete validation component (default: enabled). Disabling "
            "implements the paper's gurobipy-only No review arm: no generated checker, stage "
            "self-check, independent review, responsibility routing, or retry loop."
        ),
    )
    parser.add_argument(
        "--algorithm-design",
        dest="algorithm_design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Algorithm Design; disabling is only for the paper's No design arm",
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Runtime and benchmark timing statistics (default: enabled)",
    )
    parser.add_argument(
        "--algorithm-package-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track source=package usage for hidden-checker passing solutions",
    )
    parser.add_argument(
        "--components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structured components; disabling is valid only for the No design arm",
    )
    parser.add_argument(
        "--package-pool",
        choices=tuple(sorted(PACKAGE_POOLS)),
        default="full",
        help=(
            "Algorithm package pool (default: full). The paper's library ablation uses "
            "gurobi-only, hiding other manifests and blocking their imports, including NumPy "
            "and SciPy; all components must use source=package."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=("standard", "gurobi-formulator"),
        default="standard",
        help="Explicit workflow; gurobi-formulator is only for the paper's No design arm",
    )
    parser.set_defaults(
        input_schema=True,
        algorithm_library=True,
        problem_contract=True,
        cross_package=True,
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=7200,
        help="Timeout for each Core Runtime in seconds (default: 7200)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Evaluation report JSON path")
    parser.add_argument(
        "--suite",
        choices=("FrontierOR65-Fea", "FrontierOR10-Inf", "Hard32-Fea"),
        default=default_suite,
        help="Evaluation suite; all instances are loaded from --large-root",
    )
    parser.add_argument(
        "--large-root", type=Path, default=None, help="External FrontierOR large-data root"
    )
    parser.add_argument(
        "--stop-on-resource-exhaustion",
        action="store_true",
        help="Stop the batch when memory or virtual-memory exhaustion is detected",
    )
    parser.add_argument(
        "--max-consecutive-network-failures",
        type=int,
        default=0,
        help="Stop after this many consecutive task-level network failures; 0 disables",
    )
    return parser


def validate_reported_ablation(args: argparse.Namespace) -> None:
    """Accept only the full system and the three ablation arms reported in the paper."""

    configuration = (
        args.package_pool,
        args.workflow,
        args.algorithm_design,
        args.components,
        args.feasibility_review,
    )
    reported_arms = {
        ("full", "standard", True, True, True),
        ("gurobi-only", "standard", True, True, True),
        ("gurobi-only", "gurobi-formulator", False, False, True),
        ("gurobi-only", "standard", True, True, False),
    }
    if configuration not in reported_arms:
        raise ValueError(
            "Only paper-reported configurations are supported: Full, gurobipy-only, "
            "gurobipy-only/no-design, or gurobipy-only/no-review"
        )
