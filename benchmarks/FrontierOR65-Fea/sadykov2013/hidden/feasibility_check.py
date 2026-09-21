#!/usr/bin/env python3
"""
Feasibility checker for Bin Packing Problem with Conflicts (BPPC).

Checks all hard constraints from the compact formulation (1) in Sadykov & Vanderbeck (2013):
  Constraint 1 (1b): Each item assigned to at least one bin.
  Constraint 2 (1c): Bin capacity constraint.
  Constraint 3 (1d): Conflict constraint -- two conflicting items cannot share a bin.
  Constraint 4 (1e): y_k in {0, 1}.
  Constraint 5 (1f): x_{ik} in {0, 1}.
  Constraint 6:     Objective consistency -- reported objective_value must equal
                    the recomputed number of used bins sum_k y_k.
"""

import argparse
import json


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_bins(solution):
    """Parse bins from either efficient or gurobi solution format.

    Returns list of (bin_id, list_of_item_ids).
    """
    bins_data = solution.get("bins", {})

    # Efficient format: list of {"bin_id": ..., "items": [...]}
    if isinstance(bins_data, list):
        return [(b["bin_id"], b["items"]) for b in bins_data]

    # Gurobi format: dict of {"k": {"items": [...], ...}}
    if isinstance(bins_data, dict):
        result = []
        for k, v in bins_data.items():
            result.append((int(k), v["items"]))
        return result

    return []


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["num_items"]
    W = instance["bin_capacity"]
    items_data = instance["items"]
    conflict_edges = instance["conflict_edges"]

    weights = {it["id"]: it["weight"] for it in items_data}
    item_ids = set(it["id"] for it in items_data)

    bins = parse_bins(solution)
    # Number of bins K = total bins in solution
    K = len(bins)

    # Reconstruct assignments: which items are in which bin
    # bin_items[k] = set of items in bin k
    bin_items = {}
    for bin_id, items in bins:
        bin_items[bin_id] = set(items)

    # Build item-to-bins mapping
    item_to_bins = {}
    for bin_id, items_in_bin in bin_items.items():
        for i in items_in_bin:
            item_to_bins.setdefault(i, set()).add(bin_id)

    # y_k = 1 if bin k is used (has items), 0 otherwise
    # Since all bins in the solution are listed as used, y_k = 1 for all listed bins
    y = {bin_id: 1 for bin_id in bin_items}

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # --- Constraint 1 (1b): sum_k x_{ik} >= 1 for all i ---
    for i in item_ids:
        count = len(item_to_bins.get(i, set()))
        lhs = float(count)
        rhs = 1.0
        violation_amount = rhs - lhs  # >= constraint: how much RHS exceeds LHS
        if violation_amount > tol:
            violated_constraints.add(1)
            violations.append(f"Item {i} is not assigned to any bin (assigned {count} times)")
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # --- Constraint 2 (1c): sum_i w_i * x_{ik} <= W * y_k for all k ---
    for bin_id, items_in_bin in bin_items.items():
        total_weight = sum(weights.get(i, 0) for i in items_in_bin)
        y_k = y.get(bin_id, 0)
        lhs = float(total_weight)
        rhs = float(W * y_k)
        violation_amount = lhs - rhs  # <= constraint: how much LHS exceeds RHS
        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"Capacity exceeded on bin {bin_id}: total weight {total_weight} > {W}"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # --- Constraint 3 (1d): x_{ik} + x_{jk} <= y_k for all (i,j) in E, all k ---
    conflict_set = set()
    for edge in conflict_edges:
        i, j = edge[0], edge[1]
        conflict_set.add((min(i, j), max(i, j)))

    for bin_id, items_in_bin in bin_items.items():
        items_list = sorted(items_in_bin)
        items_set = set(items_list)
        y_k = y.get(bin_id, 0)
        for idx_a in range(len(items_list)):
            for idx_b in range(idx_a + 1, len(items_list)):
                a, b = items_list[idx_a], items_list[idx_b]
                pair = (min(a, b), max(a, b))
                if pair in conflict_set:
                    lhs = 2.0  # x_{ik} + x_{jk} = 1 + 1 = 2
                    rhs = float(y_k)  # y_k = 1
                    violation_amount = lhs - rhs
                    if violation_amount > tol:
                        violated_constraints.add(3)
                        violations.append(
                            f"Conflict violated: items {a} and {b} both in bin {bin_id}"
                        )
                        normalizer = max(abs(rhs), eps)
                        violation_magnitudes.append({
                            "constraint": 3,
                            "lhs": lhs,
                            "rhs": rhs,
                            "raw_excess": violation_amount,
                            "normalizer": normalizer,
                            "ratio": violation_amount / normalizer,
                        })

    # --- Constraint 4 (1e): y_k in {0, 1} ---
    for bin_id, y_val in y.items():
        val = float(y_val)
        if abs(val - round(val)) > tol or round(val) not in (0, 1):
            violated_constraints.add(4)
            lhs = val
            rhs_nearest = round(val)
            violation_amount = abs(val - rhs_nearest)
            rhs = rhs_nearest
            normalizer = max(abs(rhs), eps)
            violations.append(f"y_{bin_id} = {val} is not binary")
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # --- Constraint 5 (1f): x_{ik} in {0, 1} ---
    # In the solution representation, items are either in a bin or not,
    # so x values are implicitly 0 or 1. Check for duplicates (item in same bin twice)
    # and non-integer assignments.
    for bin_id, items_in_bin in bin_items.items():
        for i in items_in_bin:
            x_val = 1.0  # item is assigned
            if abs(x_val - round(x_val)) > tol or round(x_val) not in (0, 1):
                violated_constraints.add(5)
                violation_amount = abs(x_val - round(x_val))
                rhs = round(x_val)
                normalizer = max(abs(rhs), eps)
                violations.append(f"x_{i},{bin_id} = {x_val} is not binary")
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": x_val,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # --- Constraint 6: Objective consistency (Tier C defense against obj fabrication) ---
    # Objective (1a) is min sum_k y_k = number of used bins. Every bin listed in the
    # solution has y_k = 1 (consistent with the original checker's y dict). Full
    # recompute: true_obj = len(bins). Integer-valued, so a 0.5 tolerance catches
    # any off-by-one mismatch.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(y.values()))  # = len(bin_items) = number of used bins
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5  # integer-valued objective
            if abs_diff > obj_tol:
                violated_constraints.add(6)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_k y_k="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for BPPC (Sadykov & Vanderbeck 2013)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
