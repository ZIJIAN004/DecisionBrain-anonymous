#!/usr/bin/env python3
"""
Feasibility checker for RCPSP/max-pi solutions (with Tier C obj recompute).

Checks all hard constraints from Watermeyer & Zimmermann (2023):
  C1: Temporal constraints   S_j >= S_i + delta_ij  for all (i,j) in E
  C2: Resource constraints   sum_i r^c_{ik}(S_i) <= R_k  for all k in R
  C3: Project start          S_0 = 0
  C4: Deadline               S_{n+1} <= d_bar
  C5: Non-negativity/integrality  S_i in Z_{>=0}  for all i in V
  C6: Objective consistency  objective_value == S_{n+1}  (Tier C defense)
"""

import argparse
import json
import math


TOL = 1e-5
EPS = 1e-5


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_start_times(instance, solution):
    """Extract start times list from either efficient or gurobi solution format."""
    num_nodes = instance["num_nodes"]
    if "schedule" in solution and solution["schedule"] is not None:
        # efficient format: {"0": 0, "1": 17, ...}
        sched = solution["schedule"]
        S = [sched[str(i)] for i in range(num_nodes)]
    elif "start_times" in solution and solution["start_times"] is not None:
        # gurobi format: [0, 17, 13, ...]
        S = list(solution["start_times"])
    else:
        raise ValueError("Solution contains no schedule or start_times.")
    return S


def resource_usage(start_time, processing_time, time_periods):
    """
    r^u_{ik}(S_i) = |{p in Pi_k : S_i < p <= S_i + p_i}|
    Counts periods in Pi_k in the half-open interval (S_i, S_i + p_i].
    """
    count = 0
    for p in time_periods:
        if start_time < p <= start_time + processing_time:
            count += 1
    return count


def make_violation(constraint_idx, lhs, rhs, raw_excess):
    normalizer = max(abs(rhs), EPS)
    return {
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(raw_excess / normalizer),
    }


def check_feasibility(instance, solution):
    """Run full feasibility check including Tier C objective recompute.

    Accepts the solution dict (not a raw start-time list) so the reported
    objective_value can be compared against the recomputed value.
    """
    num_nodes = instance["num_nodes"]
    deadline = instance["deadline"]
    processing_times = instance["processing_times"]
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]

    S = extract_start_times(instance, solution)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ---- Constraint 1: Temporal constraints ----
    # S_j >= S_i + delta_ij  for all (i,j) in E
    for tc in temporal_constraints:
        i = tc["from"]
        j = tc["to"]
        delta = tc["delta"]
        lhs = S[j]
        rhs = S[i] + delta
        # This is a >= constraint: lhs >= rhs, violation = rhs - lhs
        violation_amount = rhs - lhs
        if violation_amount > TOL:
            violated_constraints.add(1)
            violations.append(
                f"Temporal constraint violated: S[{j}]={S[j]} < S[{i}]+delta={S[i]}+{delta}={rhs}"
            )
            violation_magnitudes.append(make_violation(1, lhs, rhs, violation_amount))

    # ---- Constraint 2: Resource constraints ----
    # sum_i r^c_{ik}(S_i) <= R_k  for all k in R
    for res in resources:
        k = res["resource_id"]
        capacity = res["capacity"]
        time_periods = res["time_periods"]
        demands = res["demands"]

        total_consumption = 0
        for i in range(num_nodes):
            r_d = demands[i]
            if r_d == 0:
                continue
            p_i = processing_times[i]
            r_u = resource_usage(S[i], p_i, time_periods)
            total_consumption += r_u * r_d

        lhs = total_consumption
        rhs = capacity
        # <= constraint: violation = lhs - rhs
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(2)
            violations.append(
                f"Resource {k} capacity exceeded: consumption={total_consumption} > capacity={capacity}"
            )
            violation_magnitudes.append(make_violation(2, lhs, rhs, violation_amount))

    # ---- Constraint 3: Project start S_0 = 0 ----
    lhs = S[0]
    rhs = 0
    violation_amount = abs(lhs - rhs)
    if violation_amount > TOL:
        violated_constraints.add(3)
        violations.append(f"Project start violated: S[0]={S[0]} != 0")
        violation_magnitudes.append(make_violation(3, lhs, rhs, violation_amount))

    # ---- Constraint 4: Deadline S_{n+1} <= d_bar ----
    sink = num_nodes - 1
    lhs = S[sink]
    rhs = deadline
    violation_amount = lhs - rhs
    if violation_amount > TOL:
        violated_constraints.add(4)
        violations.append(
            f"Deadline violated: S[{sink}]={S[sink]} > deadline={deadline}"
        )
        violation_magnitudes.append(make_violation(4, lhs, rhs, violation_amount))

    # ---- Constraint 5: Non-negativity and integrality ----
    for i in range(num_nodes):
        # Non-negativity: S_i >= 0
        if S[i] < -TOL:
            violated_constraints.add(5)
            violations.append(f"Non-negativity violated: S[{i}]={S[i]} < 0")
            lhs_val = S[i]
            rhs_val = 0
            va = abs(lhs_val)
            violation_magnitudes.append(make_violation(5, lhs_val, rhs_val, va))

        # Integrality
        if abs(S[i] - round(S[i])) > TOL:
            violated_constraints.add(5)
            violations.append(
                f"Integrality violated: S[{i}]={S[i]} is not integer"
            )
            lhs_val = S[i]
            rhs_val = round(S[i])
            va = abs(lhs_val - rhs_val)
            violation_magnitudes.append(make_violation(5, lhs_val, rhs_val, va))

    # ---- Constraint 6: Objective consistency (Tier C) ----
    # The objective is the start time of the project-end activity (node
    # num_nodes-1). All variables that determine the obj are in the
    # solution, so a full recompute is exact: true_obj == S[sink].
    # Reject when the reported objective_value disagrees.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(S[sink])
            # Objective is an integer count of time periods; use a tight
            # tolerance (0.5 absolute so any integer mismatch >= 1 fires,
            # 0.1% relative for very large values). NaN/inf reported
            # values are treated as violating regardless.
            tol = max(0.5, 1e-3 * abs(true_obj))
            bad = (not math.isfinite(reported)) or (abs(reported - true_obj) > tol)
            if bad:
                abs_diff = (abs(reported - true_obj)
                            if math.isfinite(reported) else float("inf"))
                violated_constraints.add(6)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from S[{sink}]={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                lhs_val = float(reported) if math.isfinite(reported) else float("inf")
                rhs_val = float(true_obj)
                raw_excess = float(abs_diff) if math.isfinite(abs_diff) else float("inf")
                violation_magnitudes.append(make_violation(6, lhs_val, rhs_val, raw_excess))

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for RCPSP/max-pi solutions"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
