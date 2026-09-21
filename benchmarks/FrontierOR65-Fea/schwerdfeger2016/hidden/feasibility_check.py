"""
Feasibility checker for P || NSSWD (Workload Balancing on Identical Parallel Machines).

Checks the hard constraints from the mathematical formulation:
  Constraint 1 (Eq. 3): sum_j p_j * x_{ij} = C_i,  for each machine i
  Constraint 2 (Eq. 4): sum_i x_{ij} = 1,           for each job j
  Constraint 3 (Eq. 5): x_{ij} in {0, 1},           for all i, j
  Constraint 4 (Eq. 2): objective_value consistency with assignment
                        (Tier C: full recompute of NSSWD from x_{ij})
"""

import argparse
import json
import math


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    n = instance["num_jobs"]
    m = instance["num_machines"]
    p = instance["processing_times"]  # list of length n

    assignment = solution.get("assignment", {})
    machine_completion_times = solution.get("machine_completion_times", {})

    # Build assignment matrix: for each job, which machines is it assigned to?
    job_assignment_count = [0] * n
    job_assignments = [[] for _ in range(n)]  # which machines each job is on

    for machine_key, job_list in assignment.items():
        for j in job_list:
            if 0 <= j < n:
                job_assignment_count[j] += 1
                job_assignments[j].append(machine_key)

    # ------------------------------------------------------------------
    # Constraint 1 (Eq. 3): sum_j p_j * x_{ij} = C_i for each machine i
    # ------------------------------------------------------------------
    for machine_key, job_list in assignment.items():
        computed_completion = sum(p[j] for j in job_list if 0 <= j < n)
        reported_completion = machine_completion_times.get(machine_key, 0)
        if isinstance(reported_completion, str):
            reported_completion = float(reported_completion)

        lhs = computed_completion
        rhs = reported_completion
        violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 1 violated on machine {machine_key}: "
                f"sum(p_j) = {lhs}, reported C_i = {rhs}, diff = {violation_amount:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 2 (Eq. 4): sum_i x_{ij} = 1 for each job j
    # ------------------------------------------------------------------
    for j in range(n):
        count = job_assignment_count[j]
        lhs = count
        rhs = 1
        violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if count == 0:
                msg = f"Constraint 2 violated: job {j} is not assigned to any machine"
            else:
                machines_str = ", ".join(str(mk) for mk in job_assignments[j])
                msg = (
                    f"Constraint 2 violated: job {j} is assigned to "
                    f"{count} machines ({machines_str}) instead of exactly 1"
                )
            violations.append(msg)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 3 (Eq. 5): x_{ij} in {0, 1} — domain/integrality
    # Check that all job indices in assignments are valid (0..n-1)
    # and that the solution uses exactly m machines.
    # ------------------------------------------------------------------
    for machine_key, job_list in assignment.items():
        for j in job_list:
            if not isinstance(j, int) or j < 0 or j >= n:
                violated_constraints.add(3)
                lhs = float(j) if isinstance(j, (int, float)) else 0.0
                rhs = 0.0  # expected: valid index in [0, n-1]
                violation_amount = 1.0  # binary domain violation
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 3 violated: invalid job index {j} "
                    f"on machine {machine_key} (valid range: 0..{n-1})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(lhs) if isinstance(j, (int, float)) else 0.0,
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # Check number of machines matches
    num_machines_in_sol = len(assignment)
    if num_machines_in_sol != m:
        violated_constraints.add(3)
        lhs = float(num_machines_in_sol)
        rhs = float(m)
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violations.append(
            f"Constraint 3 violated: solution has {num_machines_in_sol} machines "
            f"but instance requires {m}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # ------------------------------------------------------------------
    # Constraint 4 (Eq. 2): objective_value consistency (Tier C)
    # Full recompute of NSSWD = (1/mu) * sqrt(sum_i (C_i - mu)^2) from the
    # assignment. The solution carries every variable the objective
    # depends on (the per-machine job lists), so an exact recomputation
    # is possible. Reject when the reported value disagrees beyond a
    # tight tolerance: this catches obj=0 / obj=MAX_FLOAT exploits.
    # ------------------------------------------------------------------
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None and m > 0 and sum(p) > 0:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj):
            mu = sum(p) / m
            # Use the same C_i the assignment implies (ignore reported C_i:
            # constraint 1 already validates that link).
            C = [sum(p[j] for j in job_list if 0 <= j < n)
                 for job_list in assignment.values()]
            sq_dev = sum((c - mu) ** 2 for c in C)
            true_obj = math.sqrt(sq_dev) / mu if mu > 0 else 0.0
            abs_diff = abs(reported_obj - true_obj)
            # Tight tolerance: Gurobi-reported obj matches recompute to
            # exact float on the reference solutions, so 1e-8 absolute
            # floor with 1e-6 relative is safe against FP noise while
            # still catching obj=0 exploits on small-obj instances
            # (true_obj ~5e-4 here, so diff ~5e-4 >> 1e-8).
            obj_tol = max(1e-8, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint 4 violated: reported objective_value="
                    f"{reported_obj} differs from recomputed NSSWD="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported_obj),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })
        elif reported_obj is not None:
            # Non-finite (inf / nan) reported objective: also a violation.
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 violated: reported objective_value is "
                f"non-finite ({reported_obj_raw})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float("inf") if reported_obj == float("inf") else 0.0,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    # Build result
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for P || NSSWD"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result"
    )
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
