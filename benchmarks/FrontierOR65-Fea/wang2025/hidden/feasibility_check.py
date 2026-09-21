#!/usr/bin/env python3
"""
Feasibility checker for Approximately Submodular Function Maximization (ASFM).

Checks candidate solutions against the hard constraints from the mathematical
formulation in math_model.txt.

Constraints numbered top-to-bottom from the formulation sections:
  Constraint 1 (Eq. 1):  |S| <= k, S subset of N
  Constraint 2 (Eq. 1):  z <= f(S*)  (objective consistency with the
                          original problem; the Eq. 10 reformulation
                          cuts for S' != S* are BIP tractability
                          artifacts and are NOT part of Eq. 1's
                          feasible region)
  Constraint 3 (Eq. 10): sum_{i in N} y_i <= k
  Constraint 4 (Eq. 10): y_i in {0, 1}, for all i in N
  Constraints 5-7 (Eq. 11): implied by 2-4 (Q subset of F)
  Constraints 8-12 (Eq. 17): branch-and-cut node specific, not applicable to final solution
  Constraint 13 (Tier C):  Objective consistency |z - f(S*)| <= tol.
                          Constraint 2 only catches over-reporting
                          (z > f(S*)); this bilateral check additionally
                          catches under-reporting exploits (e.g. obj=0)
                          that misrepresent the solution's true value.
"""

import argparse
import json
import numpy as np


# ============================================================================
# Set function oracle (same as in efficient_algorithm.py / gurobi_code.py)
# ============================================================================

def build_set_function(instance):
    """Return a callable f(frozenset) based on instance type."""
    inst_type = instance["instance_type"]
    data = np.array(instance["data_matrix"])
    m, n = data.shape

    if inst_type == "LOC":
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            return float(np.sum(np.max(data[:, cols], axis=1)))
        return f, n, m

    elif inst_type == "COV":
        weights = np.array(instance["node_weights"])
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            covered = np.max(data[:, cols], axis=1)
            return float(np.dot(weights, covered))
        return f, n, m

    elif inst_type == "INF":
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            survival = np.prod(1.0 - data[:, cols], axis=1)
            return float(np.sum(1.0 - survival))
        return f, n, m

    else:
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            return float(np.sum(np.max(data[:, cols], axis=1)))
        return f, n, m


# ============================================================================
# Feasibility checking
# ============================================================================

def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    k = instance["k"]
    gamma_bar = instance["gamma"]

    f, n_actual, m = build_set_function(instance)

    selected = solution["selected_elements"]
    z = solution["objective_value"]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Build y vector from selected elements
    y = [0] * n
    valid_selected = []
    for elem in selected:
        if isinstance(elem, int) and 0 <= elem < n:
            y[elem] = 1
            valid_selected.append(elem)

    # ------------------------------------------------------------------
    # Constraint 1 (Eq. 1): |S| <= k, S ⊆ N
    # ------------------------------------------------------------------

    # Check S ⊆ N: all elements must be in {0, ..., n-1}
    for elem in selected:
        if not isinstance(elem, int) or elem < 0 or elem >= n:
            violated_set.add(1)
            lhs_val = float(elem) if isinstance(elem, (int, float)) else 0.0
            rhs_val = float(n - 1)
            raw = max(lhs_val - rhs_val, 0.0) if isinstance(elem, (int, float)) and elem >= n else 1.0
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Element {elem} is not in N = {{0, ..., {n-1}}}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw,
                "normalizer": normalizer,
                "ratio": raw / normalizer,
            })

    # Check |S| <= k
    card = len(selected)
    if card - k > tol:
        violated_set.add(1)
        raw = float(card - k)
        rhs_val = float(k)
        normalizer = max(abs(rhs_val), eps)
        violations.append(f"Cardinality |S| = {card} exceeds k = {k}")
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": float(card),
            "rhs": rhs_val,
            "raw_excess": raw,
            "normalizer": normalizer,
            "ratio": raw / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 2 (Eq. 1): z <= f(S*)  (objective consistency).
    #
    # Eq. 10 is a BIP reformulation of Eq. 1; its exponential family
    # of cuts indexed by S' is a tractability artifact of that
    # reformulation, not part of Eq. 1's feasible region.  Enumerating
    # the cuts for S' != S* over-enforces the reformulation and can
    # reject valid solutions.  Only the tight cut at S' = S* is
    # semantically meaningful here; it reduces to z <= f(S*).
    # ------------------------------------------------------------------

    S_star = frozenset(valid_selected)
    f_S_star = f(S_star)

    # Check z <= f(S*) (tightest upper-bound constraint)
    if z - f_S_star > tol:
        violated_set.add(2)
        raw = z - f_S_star
        normalizer = max(abs(f_S_star), eps)
        violations.append(
            f"Reported objective z = {z} exceeds f(S*) = {f_S_star} "
            f"(violates upper-bound constraint for S' = S*)"
        )
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(z),
            "rhs": float(f_S_star),
            "raw_excess": float(raw),
            "normalizer": float(normalizer),
            "ratio": float(raw / normalizer),
        })

    # ------------------------------------------------------------------
    # Constraint 3 (Eq. 10): sum_{i in N} y_i <= k
    # ------------------------------------------------------------------

    sum_y = sum(y)
    if sum_y - k > tol:
        violated_set.add(3)
        raw = float(sum_y - k)
        rhs_val = float(k)
        normalizer = max(abs(rhs_val), eps)
        violations.append(f"Sum of y_i = {sum_y} exceeds k = {k}")
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": float(sum_y),
            "rhs": rhs_val,
            "raw_excess": raw,
            "normalizer": normalizer,
            "ratio": raw / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 4 (Eq. 10): y_i in {0, 1} for all i in N
    # ------------------------------------------------------------------

    # Check for duplicate elements (would imply y_i > 1)
    if len(set(selected)) != len(selected):
        violated_set.add(4)
        num_dups = len(selected) - len(set(selected))
        violations.append(
            f"Duplicate elements in selected_elements ({num_dups} duplicate(s))"
        )
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": float(len(selected)),
            "rhs": float(len(set(selected))),
            "raw_excess": float(num_dups),
            "normalizer": max(float(len(set(selected))), eps),
            "ratio": float(num_dups) / max(float(len(set(selected))), eps),
        })

    # Check all elements are non-negative integers
    for elem in selected:
        if not isinstance(elem, int):
            violated_set.add(4)
            violations.append(
                f"Element {elem} is not an integer (type: {type(elem).__name__})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(elem) if isinstance(elem, (int, float)) else 0.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": eps,
                "ratio": 1.0 / eps,
            })

    # ------------------------------------------------------------------
    # Constraint 13 (Tier C obj-recompute): |z - f(S*)| <= tol.
    #
    # Constraint 2 only catches over-reporting (z > f(S*)); this
    # bilateral check additionally catches under-reporting exploits
    # (e.g. obj=0 with a non-empty S*), where the candidate
    # misrepresents the solution's true value via a fabricated
    # objective_value. Only fires when reported z and f(S*) are both
    # finite numbers; non-numeric or NaN z is ignored here to avoid
    # masking other failure modes.
    # ------------------------------------------------------------------

    try:
        reported = float(z)
        reported_ok = np.isfinite(reported)
    except (TypeError, ValueError):
        reported = None
        reported_ok = False

    if reported_ok:
        abs_diff = abs(reported - f_S_star)
        # 0.1% relative tolerance with 1e-3 absolute floor
        obj_tol = max(1e-3, 1e-3 * abs(f_S_star))
        if abs_diff > obj_tol:
            violated_set.add(13)
            normalizer = max(abs(f_S_star), eps)
            violations.append(
                f"Objective consistency violated: reported objective_value = "
                f"{reported} differs from recomputed f(S*) = {f_S_star} "
                f"(|diff| = {abs_diff:.6g}, tol = {obj_tol:.6g})"
            )
            violation_magnitudes.append({
                "constraint": 13,
                "lhs": float(reported),
                "rhs": float(f_S_star),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer),
            })

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------

    result = {
        "feasible": len(violated_set) == 0,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for ASFM candidate solutions"
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

    with open(args.instance_path) as fp:
        instance = json.load(fp)
    with open(args.solution_path) as fp:
        solution = json.load(fp)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as fp:
        json.dump(result, fp, indent=2)

    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
