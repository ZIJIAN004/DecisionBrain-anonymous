"""
Feasibility Checker for CDLP (Choice-Based Deterministic Linear Programming)
=============================================================================
Paper: Bront, Mendez-Diaz, Vulcano (2009)
"A Column Generation Algorithm for Choice-Based Network Revenue Management"
Operations Research 57(3):769-784

Checks a candidate CDLP solution against the three hard constraints plus an
objective-consistency check (Tier C defence against self-reported-objective
exploits):
  Constraint 1 (Capacity):     sum_S lambda * Q_i(S) * t(S) <= c_i   for each leg i
  Constraint 2 (Time):         sum_S t(S) <= T
  Constraint 3 (Non-negativity): t(S) >= 0                           for all S
  Constraint 4 (Obj consistency):
        reported objective_value must equal sum_S lambda * R(S) * t(S)
        within a small tolerance.  Because the solution lists every active
        column (offer_set, time_allocated) the objective can be fully
        recomputed from the solution + instance data, so a tight equality
        check is appropriate.
"""

import argparse
import json
import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_problem_data(data):
    """Extract problem parameters from instance JSON."""
    n = len(data["products"])
    m = len(data["network"]["legs"])
    L = len(data["segments"])
    T = data["booking_horizon"]["T"]
    lam = data["lambda"]

    r = np.array([p["fare"] for p in data["products"]], dtype=float)

    A = np.zeros((m, n), dtype=float)
    for j, prod in enumerate(data["products"]):
        for leg_id in prod["legs_used"]:
            A[leg_id - 1, j] = 1.0

    c = np.array([leg["capacity"] for leg in data["network"]["legs"]], dtype=float)

    segments = []
    for seg in data["segments"]:
        seg_info = {
            "lambda_l": seg["lambda_l"],
            "consideration_set": [pid - 1 for pid in seg["consideration_set"]],
            "v": {},
            "v0": seg["no_purchase_preference"]
        }
        for idx, pid in enumerate(seg["consideration_set"]):
            seg_info["v"][pid - 1] = seg["preference_vector"][idx]
        segments.append(seg_info)

    p_l = np.array([seg["lambda_l"] / lam for seg in segments])

    return {
        "n": n, "m": m, "L": L, "T": T, "lam": lam,
        "r": r, "A": A, "c": c,
        "segments": segments, "p_l": p_l
    }


def compute_choice_probs(S_set, prob_data):
    """Compute P_j(S) for all products j using MNL with overlapping segments."""
    segments = prob_data["segments"]
    p_l = prob_data["p_l"]
    n = prob_data["n"]
    P = np.zeros(n)
    for j in S_set:
        for l_idx, seg in enumerate(segments):
            if j in seg["v"]:
                denom = seg["v0"]
                for h in S_set:
                    if h in seg["v"]:
                        denom += seg["v"][h]
                P[j] += p_l[l_idx] * seg["v"][j] / denom
    return P


def compute_R_and_Q(S_set, prob_data):
    """Compute R(S) and Q(S) for an offer set S."""
    r = prob_data["r"]
    A = prob_data["A"]
    P = compute_choice_probs(S_set, prob_data)
    R_S = sum(r[j] * P[j] for j in S_set)
    Q_S = A @ P
    return R_S, Q_S


def extract_columns_and_times(solution):
    """
    Extract offer sets and their time allocations from a candidate solution.
    Returns list of (S_set_0indexed, t_value) tuples, or None if no primal
    variables are present.
    """
    if "active_columns" not in solution:
        return None

    columns = []
    for col in solution["active_columns"]:
        # offer_set is 1-indexed in the solution JSON
        S_set = set(pid - 1 for pid in col["offer_set"])
        t_val = col["time_allocated"]
        columns.append((S_set, t_val))
    return columns


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the CDLP formulation plus objective
    consistency.

    Constraints:
      1: Capacity  -- sum_S lambda * Q_i(S) * t(S) <= c_i  for each leg i
      2: Time      -- sum_S t(S) <= T
      3: Non-negativity -- t(S) >= 0 for all S
      4: Objective consistency -- reported objective_value == sum_S lambda * R(S) * t(S)
    """
    tol = 1e-5
    eps = 1e-5

    prob_data = build_problem_data(instance)
    m = prob_data["m"]
    T = prob_data["T"]
    lam = prob_data["lam"]
    c = prob_data["c"]

    columns = extract_columns_and_times(solution)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    if columns is None:
        # Solution has no primal t(S) variables (e.g., simulation-based DCOMP).
        # Constraints 1-3 cannot be evaluated, but a reported objective_value
        # with no supporting columns is still an obj-consistency violation
        # (recomputed obj is 0 in this case).
        reported_obj = solution.get("objective_value")
        try:
            reported = float(reported_obj) if reported_obj is not None else None
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} but no active_columns present so recomputed "
                    f"sum_S lambda*R(S)*t(S) = 0.0 "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer)
                })
        if not violated_constraints:
            return {
                "feasible": True,
                "violated_constraints": [],
                "violations": [
                    "No primal t(S) variables in solution; CDLP constraints not evaluated"
                ],
                "violation_magnitudes": []
            }
        return {
            "feasible": False,
            "violated_constraints": sorted(violated_constraints),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes
        }

    # ------------------------------------------------------------------
    # Constraint 1: Capacity constraint (one per leg)
    #   sum_S lambda * Q_i(S) * t(S)  <=  c_i   for i = 1, ..., m
    # ------------------------------------------------------------------
    # Compute R(S) and Q(S) for every active column (R(S) reused by constraint 4).
    R_per_col = []
    capacity_usage = np.zeros(m)
    for S_set, t_val in columns:
        R_S, Q_S = compute_R_and_Q(S_set, prob_data)
        R_per_col.append(R_S)
        capacity_usage += lam * Q_S * t_val

    for i in range(m):
        lhs = capacity_usage[i]
        rhs = c[i]
        violation_amount = lhs - rhs  # positive means violated (LHS > RHS)
        if violation_amount > tol:
            violated_constraints.add(1)
            leg_info = instance["network"]["legs"][i]
            violations.append(
                f"Capacity constraint violated on leg {leg_info['leg_id']} "
                f"({leg_info['origin']}->{leg_info['destination']}): "
                f"usage {lhs:.6f} > capacity {rhs:.6f}"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # ------------------------------------------------------------------
    # Constraint 2: Time constraint
    #   sum_S t(S)  <=  T
    # ------------------------------------------------------------------
    total_time = sum(t_val for _, t_val in columns)
    lhs = total_time
    rhs = float(T)
    violation_amount = lhs - rhs
    if violation_amount > tol:
        violated_constraints.add(2)
        violations.append(
            f"Time constraint violated: total time allocated {lhs:.6f} > T = {rhs:.6f}"
        )
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer)
        })

    # ------------------------------------------------------------------
    # Constraint 3: Non-negativity
    #   t(S) >= 0  for all S
    # ------------------------------------------------------------------
    for idx, (S_set, t_val) in enumerate(columns):
        lhs = 0.0  # RHS of t(S) >= 0 rewritten: 0 <= t(S), so check 0 - t(S)
        rhs_val = 0.0
        # For a >= constraint: violation_amount = RHS - LHS = 0 - t_val
        violation_amount = rhs_val - t_val  # positive means t_val < 0
        if violation_amount > tol:
            violated_constraints.add(3)
            offer_set_1idx = sorted(j + 1 for j in S_set)
            violations.append(
                f"Non-negativity violated for offer set {offer_set_1idx}: "
                f"t(S) = {t_val:.6f} < 0"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(t_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # ------------------------------------------------------------------
    # Constraint 4: Objective consistency (Tier C defence)
    #   reported objective_value  ==  sum_S lambda * R(S) * t(S)
    # Full recompute is exact: every variable that determines the obj
    # (the active columns and their time allocations) is present in the
    # solution.  Tolerance: 0.1% relative, with a 1e-3 absolute floor.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    try:
        reported = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        true_obj = float(sum(lam * R_per_col[i] * columns[i][1] for i in range(len(columns))))
        abs_diff = abs(reported - true_obj)
        obj_tol = max(1e-3, 1e-3 * abs(true_obj))
        if abs_diff > obj_tol:
            violated_constraints.add(4)
            violations.append(
                f"Objective consistency violated: reported objective_value="
                f"{reported} differs from recomputed sum_S lambda*R(S)*t(S)="
                f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            normalizer = max(abs(true_obj), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer)
            })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CDLP (Bront et al. 2009)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Feasibility: {status}")
    if result["violated_constraints"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
