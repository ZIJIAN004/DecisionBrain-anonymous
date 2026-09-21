#!/usr/bin/env python3
"""Feasibility checker for the Set Partitioning Problem (Zhang2025).

Hard constraints (numbered top-to-bottom from the formulation):
  Constraint 1: A x = 1_M  (each element covered exactly once)
  Constraint 2: x_j in {0, 1}  (binary decision variables)
  Constraint 3: objective_value matches c^T x (Tier C anti-gaming check)
"""

import argparse
import json


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    M = instance["parameters"]["num_trips_M"]
    N = instance["parameters"]["num_duties_N"]
    duties = instance["duties"]
    costs = instance.get("costs")
    selected = solution["selected_duties"]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Build x vector (guard against out-of-range indices to avoid IndexError;
    # they're caught by Constraint 2 below).
    x = [0] * N
    for j in selected:
        if 0 <= j < N:
            x[j] = 1

    # ------------------------------------------------------------------
    # Constraint 1: A x = 1_M  (each element i must be covered exactly once)
    # ------------------------------------------------------------------
    coverage = [0] * M
    for j in range(N):
        if x[j] == 1:
            for trip_id in duties[j]:
                coverage[trip_id] += 1

    for i in range(M):
        lhs = float(coverage[i])
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_set.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if lhs == 0:
                violations.append(f"Element (trip) {i} is not covered by any selected duty")
            else:
                violations.append(f"Element (trip) {i} is covered by {int(lhs)} duties instead of exactly 1")
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 2: x_j in {0, 1}  (binary)
    # ------------------------------------------------------------------
    for j in selected:
        if j < 0 or j >= N:
            violated_set.add(2)
            violations.append(f"Selected duty index {j} is out of range [0, {N-1}]")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(j),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": max(abs(0.0), eps),
                "ratio": 1.0 / eps,
            })

    from collections import Counter
    counts = Counter(selected)
    for j, cnt in counts.items():
        if cnt > 1:
            violated_set.add(2)
            lhs = float(cnt)
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(f"Duty {j} selected {cnt} times (must be 0 or 1)")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 3 (NEW): objective_value must equal c^T x.
    # Full recompute: every variable that determines the obj (the chosen
    # duty indices) is in the solution; costs come from the instance.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if costs is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for j in selected:
                if 0 <= j < N:
                    true_obj += float(costs[j])
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_set.add(3)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_j(costs[j]*x_j)="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for Set Partitioning Problem")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {result['feasible']} | Violated constraints: {result['violated_constraints']}")


if __name__ == "__main__":
    main()
