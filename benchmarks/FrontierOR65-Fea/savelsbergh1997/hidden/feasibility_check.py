"""
Feasibility checker for the Generalized Assignment Problem (GAP).

Reference: Savelsbergh (1997), "A Branch-and-Price Algorithm for the
Generalized Assignment Problem", Operations Research 45(6):831-841.

Checks the Standard Integer Programming Formulation (Formulation 1):

  max  sum_{i,j} p_{ij} x_{ij}

  subject to
    (1) sum_{i=1..m} x_{ij} = 1,          for j = 1,...,n
    (2) sum_{j=1..n} w_{ij} x_{ij} <= c_i, for i = 1,...,m
    (3) x_{ij} in {0, 1},                  for all i, j
    (4) reported objective_value must equal recomputed
        sum_{i,j} p_{ij} x_{ij} within tolerance (anti-gaming check).
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the GAP standard IP formulation.

    Returns a dict with feasibility results.
    """
    m = instance["num_agents"]
    n = instance["num_jobs"]
    profits = instance["profits"]        # profits[i][j]
    weights = instance["weights"]      # weights[i][j]
    capacities = instance["capacities"]  # capacities[i]

    assignment = solution["assignment"]  # dict: str(j) -> agent i

    violations = []
    violation_magnitudes = []

    # ---------------------------------------------------------------
    # Build the x_{ij} matrix from the assignment dict
    # ---------------------------------------------------------------
    x = [[0] * n for _ in range(m)]
    for j_str, i_val in assignment.items():
        j = int(j_str)
        i = int(i_val)
        if 0 <= i < m and 0 <= j < n:
            x[i][j] += 1  # use += to detect duplicates

    # ---------------------------------------------------------------
    # Constraint (1): sum_{i=1..m} x_{ij} = 1, for each j = 1,...,n
    #   Equality constraint: violation_amount = |LHS - RHS|
    # ---------------------------------------------------------------
    constraint_1_violated = False
    for j in range(n):
        lhs = sum(x[i][j] for i in range(m))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            constraint_1_violated = True
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            if lhs == 0:
                violations.append(
                    f"Constraint (1): Job {j} is not assigned to any agent "
                    f"(sum_i x_{{i,{j}}} = {lhs}, expected 1)"
                )
            elif lhs > 1:
                violations.append(
                    f"Constraint (1): Job {j} is assigned to {lhs} agents "
                    f"(sum_i x_{{i,{j}}} = {lhs}, expected 1)"
                )
            else:
                violations.append(
                    f"Constraint (1): Job {j} assignment sum is {lhs} "
                    f"(expected 1)"
                )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ---------------------------------------------------------------
    # Constraint (2): sum_{j=1..n} w_{ij} x_{ij} <= c_i, for each i
    #   <= constraint: violation_amount = max(LHS - RHS, 0)
    # ---------------------------------------------------------------
    constraint_2_violated = False
    for i in range(m):
        lhs = sum(weights[i][j] * x[i][j] for j in range(n))
        rhs = float(capacities[i])
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            constraint_2_violated = True
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint (2): Agent {i} capacity exceeded "
                f"(load = {lhs}, capacity = {rhs}, excess = {violation_amount})"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ---------------------------------------------------------------
    # Constraint (3): x_{ij} in {0, 1} for all i, j
    #   Each x_{ij} must be exactly 0 or 1.
    #   Treat as equality to nearest integer: violation = |x_{ij} - round(x_{ij})|
    # ---------------------------------------------------------------
    constraint_3_violated = False
    for i in range(m):
        for j in range(n):
            val = x[i][j]
            if val not in (0, 1):
                nearest_bin = round(val)
                violation_amount = abs(val - nearest_bin)
                if violation_amount > TOL:
                    constraint_3_violated = True
                    rhs = float(nearest_bin)
                    normalizer = max(abs(rhs), EPS)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint (3): x_{{{i},{j}}} = {val} is not binary "
                        f"(expected 0 or 1)"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": float(val),
                        "rhs": rhs,
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(ratio),
                    })
                elif val > 1:
                    # x_{ij} > 1 means job assigned multiple times to same agent
                    constraint_3_violated = True
                    rhs = 1.0
                    violation_amount = abs(val - rhs)
                    normalizer = max(abs(rhs), EPS)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint (3): x_{{{i},{j}}} = {val} is not binary "
                        f"(expected 0 or 1)"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": float(val),
                        "rhs": rhs,
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(ratio),
                    })

    # Also check that assignment keys reference valid jobs in range [0, n)
    for j_str, i_val in assignment.items():
        j = int(j_str)
        i = int(i_val)
        if j < 0 or j >= n:
            violations.append(
                f"Constraint (1): Job index {j} is out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })
        if i < 0 or i >= m:
            violations.append(
                f"Constraint (3): Agent index {i} is out of range [0, {m-1}] "
                f"for job {j}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(i),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": EPS,
                "ratio": 1.0 / EPS,
            })

    # ---------------------------------------------------------------
    # Constraint (4): objective consistency (anti-gaming).
    #   Reported objective_value must equal recomputed
    #     sum_{i,j} p_{ij} x_{ij}
    #   within tolerance. Catches LLM-evolved exploits that fabricate
    #   objective_value while keeping assignment feasible.
    # ---------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for i in range(m):
                for j in range(n):
                    if x[i][j]:
                        true_obj += float(profits[i][j]) * x[i][j]
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor. Profits are
            # integers so any honest reporter will match exactly; the slack
            # only forgives float-roundtrip noise.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint (4): Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_{{i,j}} p_{{ij}} x_{{ij}}={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # ---------------------------------------------------------------
    # Build result
    # ---------------------------------------------------------------
    violated_constraints = sorted(set(
        vm["constraint"] for vm in violation_magnitudes
    ))
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the GAP (Savelsbergh 1997)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON.")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    with open(args.solution_path, 'r') as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — {len(result['violations'])} violation(s) found:")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
