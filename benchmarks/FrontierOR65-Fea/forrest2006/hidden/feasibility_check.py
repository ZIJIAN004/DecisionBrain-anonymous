"""
Feasibility checker for the Multiple Knapsack Problem with Color Constraints (MKCP).

Based on: Forrest, Kalagnanam, and Ladanyi (2006)
"A Column-Generation Approach to the Multiple Knapsack Problem with Color Constraints"
INFORMS Journal on Computing 18(1), pp. 129-134.

Checks hard constraints from the natural (original) formulation (Section A):
  Constraint (2): Knapsack capacity
  Constraint (3): Each order assigned to at most one slab
  Constraint (4): At most K=2 distinct colors per slab
  Constraint (5a): Linking x_ij <= y_{c_i, j}
  Constraint (5b): x_ij in {0, 1}
  Constraint (5c): y_cj in {0, 1}
  Constraint (5d): z_j in {0, 1}
  Constraint (6): Objective consistency -- the reported objective_value must
                  agree with the objective recomputed from the solution
                  variables (Tier C anti-gaming defense; see below).

This is the obj-recompute variant of feasibility_check.py. It is identical to
the original except for the new Constraint (6) block inserted just before the
final feasibility verdict. Constraint (6) recomputes the linear-form objective
(Eq. 1' of math_model.txt):

    obj = sum_{i in N} sum_{j in M_i} 2*w_i*x_ij  -  sum_{j in M} W_j*z_j

from the assignments (x) and the derived used-slab set (z). Every variable the
objective depends on is present in the solution, so this is a FULL recompute
(not a lower bound). A candidate that fabricates objective_value while keeping
a feasible route is rejected with violated constraint 6.
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    violations = []
    violated_constraints = set()
    violation_magnitudes = []

    n = instance["num_orders"]
    m = instance["num_slabs"]
    w = instance["order_weights"]
    W = instance["slab_weights"]
    colors = instance["order_colors"]
    eligible = instance["eligible_slabs_per_order"]
    K = instance["max_colors_per_knapsack"]

    # Build M_i (eligible slabs for order i) and N_j (eligible orders for slab j)
    M_i = [set(eligible[i]) for i in range(n)]
    N_j = [set() for _ in range(m)]
    for i in range(n):
        for j in M_i[i]:
            N_j[j].add(i)

    # Reconstruct decision variables from solution
    assignments = solution.get("assignments", [])

    # x_ij: 1 if order i assigned to slab j
    x = {}
    for a in assignments:
        i = a["order"]
        j = a["slab"]
        x[(i, j)] = 1

    # Derive z_j: 1 if any order is assigned to slab j
    slabs_used = set()
    for a in assignments:
        slabs_used.add(a["slab"])
    z = {j: (1 if j in slabs_used else 0) for j in range(m)}

    # Derive y_cj: 1 if any order of color c is assigned to slab j
    y = {}
    for a in assignments:
        i = a["order"]
        j = a["slab"]
        c = colors[i]
        y[(c, j)] = 1

    # Also check that assigned (i,j) pairs respect eligibility (implicit domain)
    for a in assignments:
        i = a["order"]
        j = a["slab"]
        if j not in M_i[i]:
            violated_constraints.add(5)
            violations.append(
                f"Order {i} assigned to slab {j} but slab {j} is not in eligible set M_{i} = {sorted(M_i[i])}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": EPS,
                "ratio": 1.0 / EPS,
            })

    # ------------------------------------------------------------------
    # Constraint (2): Knapsack capacity
    #   sum_{i in N_j} w_i * x_ij <= W_j * z_j,  for all j in M
    # ------------------------------------------------------------------
    for j in range(m):
        lhs = sum(w[i] * x.get((i, j), 0) for i in N_j[j])
        rhs = W[j] * z[j]
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint (2) violated on slab {j}: "
                f"total assigned weight {lhs:.4f} > capacity {rhs:.4f} (W_j={W[j]}, z_j={z[j]})"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint (3): Each order assigned to at most one slab
    #   sum_{j in M_i} x_ij <= 1,  for all i in N
    # ------------------------------------------------------------------
    for i in range(n):
        lhs = sum(x.get((i, j), 0) for j in M_i[i])
        rhs = 1.0
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(3)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            assigned_slabs = [j for j in M_i[i] if x.get((i, j), 0) == 1]
            violations.append(
                f"Constraint (3) violated for order {i}: "
                f"assigned to {int(lhs)} slabs {assigned_slabs} (max 1)"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint (4): At most K=2 distinct colors per slab
    #   sum_{c in C_j} y_cj <= K,  for all j in M
    # ------------------------------------------------------------------
    for j in range(m):
        # Count distinct colors of orders actually assigned to slab j
        colors_on_slab = set()
        for i in N_j[j]:
            if x.get((i, j), 0) == 1:
                colors_on_slab.add(colors[i])
        lhs = float(len(colors_on_slab))
        rhs = float(K)
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(4)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint (4) violated on slab {j}: "
                f"{int(lhs)} distinct colors {sorted(colors_on_slab)} assigned (max {K})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint (5a): Linking  x_ij <= y_{c_i, j}
    #   for all i in N, j in M_i
    #   Since y is derived from x (y_{c,j}=1 iff any order of color c is on j),
    #   this is always satisfied by construction. But we still check explicitly.
    # ------------------------------------------------------------------
    for (i, j), xval in x.items():
        if xval == 1:
            c = colors[i]
            yval = y.get((c, j), 0)
            lhs = float(xval)
            rhs = float(yval)
            violation_amount = lhs - rhs
            if violation_amount > TOL:
                violated_constraints.add(5)
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint (5a) violated: x_{{{i},{j}}}={xval} > y_{{{c},{j}}}={yval}"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Constraint (5b): x_ij in {0, 1}
    #   By construction from the solution format (assignments list), x values
    #   are always 0 or 1. Check anyway.
    # ------------------------------------------------------------------
    for (i, j), xval in x.items():
        if xval not in (0, 1):
            violated_constraints.add(5)
            violations.append(
                f"Constraint (5b) violated: x_{{{i},{j}}}={xval} not in {{0, 1}}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(xval),
                "rhs": 1.0,
                "raw_excess": abs(xval - round(xval)),
                "normalizer": max(1.0, EPS),
                "ratio": abs(xval - round(xval)) / max(1.0, EPS),
            })

    # ------------------------------------------------------------------
    # Constraint (5c): y_cj in {0, 1}
    #   By construction y values are always 0 or 1. Check anyway.
    # ------------------------------------------------------------------
    for (c, j), yval in y.items():
        if yval not in (0, 1):
            violated_constraints.add(5)
            violations.append(
                f"Constraint (5c) violated: y_{{{c},{j}}}={yval} not in {{0, 1}}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(yval),
                "rhs": 1.0,
                "raw_excess": abs(yval - round(yval)),
                "normalizer": max(1.0, EPS),
                "ratio": abs(yval - round(yval)) / max(1.0, EPS),
            })

    # ------------------------------------------------------------------
    # Constraint (5d): z_j in {0, 1}
    #   By construction z values are always 0 or 1. Check anyway.
    # ------------------------------------------------------------------
    for j_idx, zval in z.items():
        if zval not in (0, 1):
            violated_constraints.add(5)
            violations.append(
                f"Constraint (5d) violated: z_{{{j_idx}}}={zval} not in {{0, 1}}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(zval),
                "rhs": 1.0,
                "raw_excess": abs(zval - round(zval)),
                "normalizer": max(1.0, EPS),
                "ratio": abs(zval - round(zval)) / max(1.0, EPS),
            })

    # ------------------------------------------------------------------
    # Constraint (6): Objective consistency (Tier C anti-gaming defense)
    #   The eval pipeline otherwise trusts solution["objective_value"]; an
    #   LLM-evolved candidate can keep a feasible route but report a
    #   fabricated objective. Here we RECOMPUTE the linear-form objective
    #   (Eq. 1' of math_model.txt) directly from the solution variables:
    #
    #       obj = sum_{i in N} sum_{j in M_i} 2*w_i*x_ij  -  sum_{j in M} W_j*z_j
    #
    #   All obj-determining variables are present -- x_ij from `assignments`
    #   and z_j derived from the used-slab set (same z used by Constraint 2)
    #   -- so this is a FULL recompute, not a lower bound. We reject when the
    #   reported value disagrees beyond a 0.1% relative tolerance with a
    #   1e-3 absolute floor (the floor covers the Gurobi-optimal obj==0
    #   instances, whose reported values carry sub-1e-3 floating-point noise).
    #
    #   This check is append-only: it can add constraint 6 but never alters
    #   the verdict of constraints 2-5.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            obj_x = sum(2.0 * w[i] * xval for (i, j), xval in x.items())
            obj_z = sum(W[j] * zval for j, zval in z.items())
            true_obj = obj_x - obj_z
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(6)
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint (6) violated: reported objective_value={reported} "
                    f"differs from recomputed 2*sum(w_i*x_ij) - sum(W_j*z_j)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MKCP (Forrest et al. 2006)."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance.",
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution.",
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result.",
    )
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Feasibility: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
