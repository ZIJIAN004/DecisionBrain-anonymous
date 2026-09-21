"""
Feasibility checker for the Quadratic Multiknapsack Problem (QMKP-QP).
Source: Bergman (2019), INFORMS Journal on Computing.

Constraints (numbered top-to-bottom from the QMKP-QP formulation):
  1. Capacity:   sum_i w_i * x_{i,k} <= C_k,  for all k in [m]
  2. Assignment: sum_k x_{i,k} <= 1,           for all i in [n]
  3. Binary:     x_{i,k} in {0,1},             for all i in [n], k in [m]
  6. Objective consistency: reported objective_value must equal the
     recomputed QMKP-QP objective
     sum_{i,k} p_i * x_{i,k}
       + sum_{i<j,k} p_{i,j} * x_{i,k} * x_{j,k}
     from the assignment in the solution (Tier C defense against
     LLM score-gaming exploits that fabricate objective_value).
"""

import argparse
import json


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    m = instance["m"]
    weights = instance["weights"]
    capacities = instance["capacities"]

    assignment = solution.get("assignment", [])

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # ── Constraint 3 (index validity): validate BEFORE downstream lookups.
    # Fix_7 reviewer item 2: weights[i] was indexed before the bounds
    # check ran, so i >= n would raise IndexError and i < 0 would silently
    # wrap via Python's negative indexing.
    # Fix_7 reviewer item 1: assignment is now a list of [item, knapsack]
    # pairs (schema reshape) so multi-assignment is expressible and
    # constraint 2 below is no longer a tautology.
    valid_pairs = []
    for entry in assignment:
        item = int(entry[0])
        kk = int(entry[1])
        item_ok = 0 <= item < n
        k_ok = 0 <= kk < m
        if not item_ok:
            violated_set.add(3)
            violations.append(
                f"Item index {item} out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(item),
                "rhs": float(n - 1),
                "raw_excess": float(abs(item - (n - 1)) if item >= n else abs(item)),
                "normalizer": max(float(n - 1), eps),
                "ratio": float(abs(item - (n - 1)) if item >= n else abs(item))
                       / max(float(n - 1), eps),
            })
        if not k_ok:
            violated_set.add(3)
            violations.append(
                f"Knapsack index {kk} for item {item} out of range [0, {m-1}]"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(kk),
                "rhs": float(m - 1),
                "raw_excess": float(abs(kk - (m - 1)) if kk >= m else abs(kk)),
                "normalizer": max(float(m - 1), eps),
                "ratio": float(abs(kk - (m - 1)) if kk >= m else abs(kk))
                       / max(float(m - 1), eps),
            })
        if item_ok and k_ok:
            valid_pairs.append((item, kk))

    # ── Constraint 1: Capacity ──
    # sum_i w_i * x_{i,k} <= C_k, for each k in [m]
    for k in range(m):
        lhs = sum(weights[i] for i, kk in valid_pairs if kk == k)
        rhs = capacities[k]
        violation_amount = lhs - rhs  # for <= constraint
        if violation_amount > tol:
            violated_set.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Capacity exceeded on knapsack {k}: "
                f"total weight {lhs} > capacity {rhs}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ── Constraint 2: Assignment ──
    # sum_k x_{i,k} <= 1, for each i in [n].
    # The list-of-pairs schema lets the same item index appear more than
    # once; the prior dict-keyed schema structurally collapsed duplicates,
    # making this branch unreachable.
    counts = {}
    for i, kk in valid_pairs:
        counts[i] = counts.get(i, 0) + 1
    for i, count in counts.items():
        violation_amount = count - 1
        if violation_amount > tol:
            violated_set.add(2)
            normalizer = max(abs(1), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Item {i} assigned to {count} knapsacks (max 1 allowed)"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(count),
                "rhs": float(1),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    _domain_check_vars_binary = []
    _domain_check_vars_integer = [("assignment", assignment)]

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 4: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_constraints.add(4)
                        violations.append(
                            f"Constraint 4 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 4,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 5: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > tol:
                    violated_constraints.add(5)
                    violations.append(
                        f"Constraint 5 (integer domain): {var_name}[{key}] = {v} is not integer")
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), eps),
                        "ratio": float(frac / max(abs(round(v)), eps)),
                    })

    # ── Constraint 6: Objective consistency (Tier C) ──
    # Recompute the QMKP-QP objective from the solution's assignment and
    # compare against the reported objective_value. The solution contains
    # every variable the objective depends on (x_{i,k}), so a full
    # recompute applies. Profits and pairwise_profits are integers per
    # math_model.txt, so the true objective is integral; a tolerance of
    # max(0.5, 1e-6 * |true_obj|) flags any integer-level mismatch while
    # absorbing float-formatting noise (e.g. trailing ".0" round-trips).
    profits = instance.get("profits")
    pairwise_profits = instance.get("pairwise_profits")
    reported_obj = solution.get("objective_value")
    if profits is not None and pairwise_profits is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Dedup per knapsack: x_{i,k} is binary, so an item appearing
            # twice in the same knapsack contributes only once to the obj
            # (the duplicate is already flagged separately by constraint 2).
            items_per_knap = {}
            for i, kk in valid_pairs:
                items_per_knap.setdefault(kk, set()).add(i)
            linear_part = 0.0
            for kk, items in items_per_knap.items():
                for i in items:
                    linear_part += float(profits[i])
            quad_part = 0.0
            for kk, items in items_per_knap.items():
                items_sorted = sorted(items)
                for a in range(len(items_sorted)):
                    i = items_sorted[a]
                    row_i = pairwise_profits[i]
                    for b in range(a + 1, len(items_sorted)):
                        j = items_sorted[b]
                        quad_part += float(row_i[j])
            true_obj = linear_part + quad_part
            abs_diff = abs(reported - true_obj)
            tol_obj = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_set.add(6)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed QMKP-QP objective="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    feasible = len(violated_set) == 0
    violated_constraints = sorted(violated_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for QMKP-QP (Bergman 2019)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {result['feasible']}")
    if result['violated_constraints']:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
