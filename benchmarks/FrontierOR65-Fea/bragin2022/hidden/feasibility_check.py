#!/usr/bin/env python3
"""
Feasibility checker for the Generalized Assignment Problem (GAP)
from Bragin & Tucker (2022).

Constraints (numbered top-to-bottom from the formulation):
  Constraint 1: sum_i x[i][j] = 1 for all j  (each job assigned to exactly one machine)
  Constraint 2: sum_j a[i][j] * x[i][j] <= b[i] for all i  (machine capacity)
  Constraint 3: x[i][j] in {0, 1}  (binary integrality)
  Constraint 4: reported objective_value matches sum_{i,j} c[i][j] * x[i][j]
                (Tier C obj-consistency check — full recompute, since the
                solution carries every variable the objective depends on)
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def check_feasibility(instance, solution):
    num_machines = instance["num_machines"]
    num_jobs = instance["num_jobs"]
    cost = instance["cost_matrix"]            # c[i][j]
    resource = instance["resource_matrix"]    # a[i][j]
    capacity = instance["capacities"]         # b[i]

    assignments = solution.get("assignments", {})

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Build assignment matrix from solution
    # assignments maps str(job_index) -> machine_index
    x = [[0] * num_jobs for _ in range(num_machines)]
    assigned_jobs = set()
    for job_str, machine in assignments.items():
        j = int(job_str)
        i = int(machine)
        if 0 <= i < num_machines and 0 <= j < num_jobs:
            x[i][j] = 1
            assigned_jobs.add(j)

    # -------------------------------------------------------------------------
    # Constraint 1: sum_i x[i][j] = 1 for all j (assignment equality)
    # -------------------------------------------------------------------------
    unassigned_jobs = []
    multi_assigned_jobs = []
    for j in range(num_jobs):
        lhs = sum(x[i][j] for i in range(num_machines))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
            if lhs == 0:
                unassigned_jobs.append(j)
            else:
                multi_assigned_jobs.append(j)

    if unassigned_jobs:
        violations.append(
            f"Constraint 1 violated: jobs {unassigned_jobs} are not assigned to any machine"
        )
    if multi_assigned_jobs:
        violations.append(
            f"Constraint 1 violated: jobs {multi_assigned_jobs} are assigned to multiple machines"
        )

    # -------------------------------------------------------------------------
    # Constraint 2: sum_j a[i][j] * x[i][j] <= b[i] for all i (capacity)
    # -------------------------------------------------------------------------
    capacity_violated_machines = []
    for i in range(num_machines):
        lhs = sum(resource[i][j] * x[i][j] for j in range(num_jobs))
        rhs = float(capacity[i])
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
            capacity_violated_machines.append(
                f"machine {i} (usage {lhs} > capacity {rhs})"
            )

    if capacity_violated_machines:
        violations.append(
            f"Constraint 2 violated: capacity exceeded on {', '.join(capacity_violated_machines)}"
        )

    # -------------------------------------------------------------------------
    # Constraint 3: x[i][j] in {0, 1} (binary integrality)
    # -------------------------------------------------------------------------
    non_binary_vars = []
    for job_str, machine in assignments.items():
        j = int(job_str)
        i = int(machine)
        # Check that machine index is valid
        if i < 0 or i >= num_machines:
            violated_constraints.add(3)
            rhs_val = float(num_machines - 1)
            normalizer = max(abs(rhs_val), EPS)
            violation_amount = abs(i - max(0, min(i, num_machines - 1)))
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(i),
                "rhs": rhs_val,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer
            })
            non_binary_vars.append(f"x[{i}][{j}] has invalid machine index")
        if j < 0 or j >= num_jobs:
            violated_constraints.add(3)
            rhs_val = float(num_jobs - 1)
            normalizer = max(abs(rhs_val), EPS)
            violation_amount = abs(j - max(0, min(j, num_jobs - 1)))
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(j),
                "rhs": rhs_val,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer
            })
            non_binary_vars.append(f"x[{i}][{j}] has invalid job index")

    if non_binary_vars:
        violations.append(
            f"Constraint 3 violated: {', '.join(non_binary_vars)}"
        )

    # -------------------------------------------------------------------------
    # Constraint 4: objective consistency (Tier C defense).
    # Recompute true_obj = sum_{i,j} c[i][j] * x[i][j] from the solution
    # variables and compare to the reported objective_value. The GAP solution
    # carries every obj-determining variable (the full assignment dict), so a
    # full recompute applies. Tolerance: max(1e-3 absolute, 1e-3 relative)
    # plus a 0.5 integer-floor since cost coefficients are integers in this
    # benchmark and any genuine LP/MIP-precision deviation stays well below
    # half a unit, while LLM exploits typically lie by orders of magnitude.
    # -------------------------------------------------------------------------
    reported_raw = solution.get("objective_value")
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(
                cost[i][j] * x[i][j]
                for i in range(num_machines) for j in range(num_jobs)
            ))
            abs_diff = abs(reported - true_obj)
            tol = max(0.5, 1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff) / float(normalizer),
                })
                violations.append(
                    f"Constraint 4 violated: reported objective_value={reported} "
                    f"differs from recomputed sum_(i,j) c[i][j]*x[i][j]={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )

    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Check feasibility of a GAP solution")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    feasible = result["feasible"]
    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
