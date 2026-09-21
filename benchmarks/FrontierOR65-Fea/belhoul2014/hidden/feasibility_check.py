#!/usr/bin/env python3
"""
Feasibility checker for the Multi-Objective Assignment Problem (MOAP) /
Linearized Compromise Assignment Problem (LCAP) from Zheng et al. (2014).

Checks all hard constraints listed in math_model.txt, numbered strictly
from top to bottom:

  Constraint  1  (MOAP-1):  sum_j x_ij = 1,  i=1,...,n
  Constraint  2  (MOAP-2):  sum_i x_ij = 1,  j=1,...,n
  Constraint  3  (MOAP-3):  x_ij in {0,1}
  Constraint  4  (CAP-1):   sum_j x_ij = 1,  i=1,...,n
  Constraint  5  (CAP-2):   sum_i x_ij = 1,  j=1,...,n
  Constraint  6  (CAP-3):   x_ij in {0,1}
  Constraint  7  (LCAP-1):  mu >= lambda_k * (sum_ij c^k_ij x_ij - z_bar_k), k=1,...,p
  Constraint  8  (LCAP-2):  sum_j x_ij = 1,  i=1,...,n
  Constraint  9  (LCAP-3):  sum_i x_ij = 1,  j=1,...,n
  Constraint 10  (LCAP-4):  x_ij in {0,1}
  Constraint 11  (LCAP-5):  mu unrestricted
  Constraint 12  (RLCAP-1): sum_j x_ij = 1,  i=1,...,n
  Constraint 13  (RLCAP-2): sum_i x_ij = 1,  j=1,...,n
  Constraint 14  (RLCAP-3): mu >= lambda_k * (sum_ij c^k_ij x_ij - z_bar_k), k=1,...,p
  Constraint 15  (RLCAP-4): x_ij >= 0
  Constraint 16  (RLCAP-5): mu unrestricted
  Constraint 17  (2a):      u_i + v_j <= sum_k lambda_k c^k_ij omega_k  (dual; skipped)
  Constraint 18  (2b):      sum_k omega_k = 1                           (dual; skipped)
  Constraint 19  (2c):      omega_k > 0                                 (dual; skipped)
  Constraint 20  (obj):     objective_value == max_k lambda_k * (z_k - z_bar_k)
                            (Tier C: catches reported-objective fabrication; the
                            LCAP-1 inequality alone only rules out under-reporting,
                            so a candidate that inflates mu to sys.float_info.max
                            would otherwise slip through.)
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_assignment_matrix(assignment, n):
    """Build n x n binary matrix from assignment list."""
    x = [[0] * n for _ in range(n)]
    for i, j in enumerate(assignment):
        x[i][j] = 1
    return x


def compute_objective_per_criterion(assignment, cost_matrices, n, p):
    """z_k(x) = sum_i sum_j c^k_ij * x_ij  for each k."""
    z = [0.0] * p
    for k in range(p):
        for i in range(n):
            j = assignment[i]
            z[k] += cost_matrices[k][i][j]
    return z


def add_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs):
    """Record a constraint violation with normalized magnitude."""
    raw_excess = abs(lhs - rhs)
    # For = constraints, raw_excess is |LHS - RHS|
    # For >= constraints (mu >= rhs_val), raw_excess is max(0, rhs - lhs)
    # Caller computes the appropriate raw_excess and passes lhs, rhs accordingly.
    normalizer = max(abs(rhs), EPS)
    ratio = raw_excess / normalizer
    violations_list.append(msg)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": raw_excess,
        "normalizer": normalizer,
        "ratio": ratio,
    })


def check_row_assignment(x_mat, n, constraint_idx, label):
    """Check sum_j x_ij = 1 for all i (= constraint)."""
    violated_msgs = []
    violated_mags = []
    for i in range(n):
        row_sum = sum(x_mat[i][j] for j in range(n))
        diff = abs(row_sum - 1.0)
        if diff > TOL:
            add_violation(
                violated_msgs, violated_mags, constraint_idx,
                f"{label}: Row {i} sum = {row_sum}, expected 1",
                row_sum, 1.0,
            )
    return violated_msgs, violated_mags


def check_col_assignment(x_mat, n, constraint_idx, label):
    """Check sum_i x_ij = 1 for all j (= constraint)."""
    violated_msgs = []
    violated_mags = []
    for j in range(n):
        col_sum = sum(x_mat[i][j] for i in range(n))
        diff = abs(col_sum - 1.0)
        if diff > TOL:
            add_violation(
                violated_msgs, violated_mags, constraint_idx,
                f"{label}: Column {j} sum = {col_sum}, expected 1",
                col_sum, 1.0,
            )
    return violated_msgs, violated_mags


def check_binary(x_mat, n, constraint_idx, label):
    """Check x_ij in {0, 1} for all i, j."""
    violated_msgs = []
    violated_mags = []
    for i in range(n):
        for j in range(n):
            val = x_mat[i][j]
            dist_to_nearest = min(abs(val - 0.0), abs(val - 1.0))
            if dist_to_nearest > TOL:
                # LHS = distance from binary, RHS = 0 (should be zero distance)
                add_violation(
                    violated_msgs, violated_mags, constraint_idx,
                    f"{label}: x[{i}][{j}] = {val}, not binary",
                    dist_to_nearest, 0.0,
                )
    return violated_msgs, violated_mags


def check_lcap_linearization(mu, z_vals, lam, z_bar, p, constraint_idx, label):
    """Check mu >= lambda_k * (z_k - z_bar_k) for all k (>= constraint)."""
    violated_msgs = []
    violated_mags = []
    for k in range(p):
        rhs_val = lam[k] * (z_vals[k] - z_bar[k])
        violation_amount = rhs_val - mu  # positive if mu < rhs_val
        if violation_amount > TOL:
            add_violation(
                violated_msgs, violated_mags, constraint_idx,
                f"{label}: mu ({mu:.6f}) < lambda_{k}*(z_{k} - z_bar_{k}) = {rhs_val:.6f}",
                mu, rhs_val,
            )
    return violated_msgs, violated_mags


def check_nonnegativity(x_mat, n, constraint_idx, label):
    """Check x_ij >= 0 for all i, j (>= constraint)."""
    violated_msgs = []
    violated_mags = []
    for i in range(n):
        for j in range(n):
            val = x_mat[i][j]
            if val < -TOL:
                violation_amount = -val
                add_violation(
                    violated_msgs, violated_mags, constraint_idx,
                    f"{label}: x[{i}][{j}] = {val} < 0",
                    val, 0.0,
                )
    return violated_msgs, violated_mags


def run_feasibility_check(instance, solution):
    n = instance["n"]
    p = instance["p"]
    cost_matrices = instance["cost_matrices"]
    z_bar = instance["reference_point"]
    lam = instance["search_direction_lambda"]

    assignment = solution["assignment"]

    # Build assignment matrix
    x_mat = build_assignment_matrix(assignment, n)

    # Use assignment_matrix from solution if available (for consistency check)
    if "assignment_matrix" in solution:
        x_mat_sol = solution["assignment_matrix"]
        # Check consistency between assignment list and matrix
        for i in range(n):
            for j in range(n):
                if x_mat[i][j] != x_mat_sol[i][j]:
                    # Use the assignment list as ground truth
                    pass

    # Compute objective values per criterion from the assignment
    z_vals = compute_objective_per_criterion(assignment, cost_matrices, n, p)

    # Determine mu (objective_value)
    mu = solution.get("objective_value")
    if mu is None:
        # Compute from the achievement function
        mu = max(lam[k] * (z_vals[k] - z_bar[k]) for k in range(p))

    violated_constraints = set()
    all_msgs = []
    all_mags = []

    # Helper to accumulate results
    def accumulate(constraint_idx, msgs, mags):
        if msgs:
            violated_constraints.add(constraint_idx)
            all_msgs.extend(msgs)
            all_mags.extend(mags)

    # --- Constraint 1 (MOAP-1): sum_j x_ij = 1 for all i ---
    msgs, mags = check_row_assignment(x_mat, n, 1, "MOAP-1")
    accumulate(1, msgs, mags)

    # --- Constraint 2 (MOAP-2): sum_i x_ij = 1 for all j ---
    msgs, mags = check_col_assignment(x_mat, n, 2, "MOAP-2")
    accumulate(2, msgs, mags)

    # --- Constraint 3 (MOAP-3): x_ij in {0,1} ---
    msgs, mags = check_binary(x_mat, n, 3, "MOAP-3")
    accumulate(3, msgs, mags)

    # --- Constraint 4 (CAP-1): sum_j x_ij = 1 for all i ---
    msgs, mags = check_row_assignment(x_mat, n, 4, "CAP-1")
    accumulate(4, msgs, mags)

    # --- Constraint 5 (CAP-2): sum_i x_ij = 1 for all j ---
    msgs, mags = check_col_assignment(x_mat, n, 5, "CAP-2")
    accumulate(5, msgs, mags)

    # --- Constraint 6 (CAP-3): x_ij in {0,1} ---
    msgs, mags = check_binary(x_mat, n, 6, "CAP-3")
    accumulate(6, msgs, mags)

    # --- Constraint 7 (LCAP-1): mu >= lambda_k*(z_k - z_bar_k) for all k ---
    msgs, mags = check_lcap_linearization(mu, z_vals, lam, z_bar, p, 7, "LCAP-1")
    accumulate(7, msgs, mags)

    # --- Constraint 8 (LCAP-2): sum_j x_ij = 1 for all i ---
    msgs, mags = check_row_assignment(x_mat, n, 8, "LCAP-2")
    accumulate(8, msgs, mags)

    # --- Constraint 9 (LCAP-3): sum_i x_ij = 1 for all j ---
    msgs, mags = check_col_assignment(x_mat, n, 9, "LCAP-3")
    accumulate(9, msgs, mags)

    # --- Constraint 10 (LCAP-4): x_ij in {0,1} ---
    msgs, mags = check_binary(x_mat, n, 10, "LCAP-4")
    accumulate(10, msgs, mags)

    # --- Constraint 11 (LCAP-5): mu unrestricted ---
    # Trivially satisfied for any real mu; no check needed.

    # --- Constraints 12-16 (RLCAP-1..RLCAP-5) SKIPPED ---
    # These are the LP-relaxation (RLCAP) constraints. For any binary x
    # that satisfies the LCAP constraints (assignment 6, 7; binary 10),
    # the relaxation is automatically satisfied (row/col assignment is
    # identical; x_ij >= 0 is weaker than x_ij in {0,1}). Checking them
    # only duplicates the LCAP checks above.
    #
    # --- Constraints 17-19 (Dual: 2a, 2b, 2c) SKIPPED ---
    # Dual variables (u, v, omega) are not part of the primal solution
    # structure and not produced by a generated algorithm.

    # --- Constraint 20 (obj consistency): mu == max_k lambda_k*(z_k - z_bar_k) ---
    # The LCAP linearization in constraint 7 only enforces mu >= achievement,
    # so a candidate could pass by reporting an arbitrarily large mu. The
    # actual achievement-function value at the chosen assignment is
    # max_k lambda_k*(z_k - z_bar_k); the reported objective_value must
    # equal it within tolerance. All variables that determine the achievement
    # function (the assignment) are present in the solution, so we perform
    # a full recompute rather than a lower-bound check.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = max(lam[k] * (z_vals[k] - z_bar[k]) for k in range(p))
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                add_violation(
                    all_msgs, all_mags, 20,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed "
                    f"max_k lambda_k*(z_k - z_bar_k)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})",
                    reported, true_obj,
                )
                violated_constraints.add(20)

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": all_msgs,
        "violation_magnitudes": all_mags,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MOAP/LCAP (Zheng et al. 2014)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file",
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON solution file",
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON feasibility result",
    )
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = run_feasibility_check(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
