"""
Feasibility checker for the Multicommodity Uncapacitated Fixed-charge
Network Design (MUFND) problem.

Checks all hard constraints from the formulation in Zetina, Contreras,
Cordeau (2019):
  Constraint 1 (tag 2): Flow conservation
  Constraint 2 (tag 3): Linking  x^k_{ij} <= y_{ij}
  Constraint 3 (tag 4): Non-negativity  x^k_{ij} >= 0
  Constraint 4 (tag 5): Integrality  y_{ij} in {0,1}
  Constraint 5 (Tier C obj recompute): reported objective_value must equal
      sum_{(i,j)} f_{ij}*y_{ij} + sum_k sum_{(i,j)} W^k * c^k_{ij} * x^k_{ij}
      within 0.1% relative / 1e-3 absolute tolerance.
"""

import argparse
import json


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    num_nodes = instance["num_nodes"]
    num_commodities = instance["num_commodities"]
    arcs = [tuple(a) for a in instance["arcs"]]
    commodities = instance["commodities"]

    # Build arc set for quick lookup
    arc_set = set(arcs)

    # Reconstruct y_{ij} from solution
    open_arcs = solution.get("open_arcs", {})
    y = {}
    for i, j in arcs:
        key = f"{i}_{j}"
        y[i, j] = open_arcs.get(key, 0)

    # Reconstruct x^k_{ij} from solution
    routings = solution.get("routings", {})
    x = {}
    for k in range(num_commodities):
        for i, j in arcs:
            x[k, i, j] = 0.0
    for k_str, flows in routings.items():
        k = int(k_str)
        for arc_key, val in flows.items():
            parts = arc_key.split("_")
            i, j = int(parts[0]), int(parts[1])
            if (i, j) in arc_set:
                x[k, i, j] = val

    # Build adjacency
    arcs_out = {n: [] for n in range(num_nodes)}
    arcs_in = {n: [] for n in range(num_nodes)}
    for i, j in arcs:
        arcs_out[i].append((i, j))
        arcs_in[j].append((i, j))

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1 (tag 2): Flow conservation
    # sum_j x^k_{ji} - sum_j x^k_{ij} = b^k_i
    # where b^k_i = -1 if i=o_k, 1 if i=d_k, 0 otherwise
    # This is an equality constraint.
    # ------------------------------------------------------------------
    for k in range(num_commodities):
        ok = commodities[k]["origin"]
        dk = commodities[k]["destination"]
        for n in range(num_nodes):
            if n == ok:
                rhs = -1.0
            elif n == dk:
                rhs = 1.0
            else:
                rhs = 0.0

            inflow = sum(x[k, i, j] for i, j in arcs_in[n])
            outflow = sum(x[k, i, j] for i, j in arcs_out[n])
            lhs = inflow - outflow

            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(1)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Flow conservation violated for commodity {k} at node {n}: "
                    f"LHS={lhs:.6f}, RHS={rhs:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Constraint 2 (tag 3): Linking  x^k_{ij} <= y_{ij}
    # ------------------------------------------------------------------
    for k in range(num_commodities):
        for i, j in arcs:
            lhs = x[k, i, j]
            rhs = y[i, j]
            violation_amount = lhs - rhs  # how much LHS exceeds RHS
            if violation_amount > tol:
                violated_constraints.add(2)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Linking violated for commodity {k} on arc ({i},{j}): "
                    f"x={lhs:.6f} > y={rhs}"
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
    # Constraint 3 (tag 4): Non-negativity  x^k_{ij} >= 0
    # ------------------------------------------------------------------
    for k in range(num_commodities):
        for i, j in arcs:
            lhs = x[k, i, j]
            rhs = 0.0
            violation_amount = rhs - lhs  # how much RHS exceeds LHS
            if violation_amount > tol:
                violated_constraints.add(3)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Non-negativity violated for commodity {k} on arc ({i},{j}): "
                    f"x={lhs:.6f} < 0"
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
    # Constraint 4 (tag 5): Integrality  y_{ij} in {0, 1}
    # Treated as equality to nearest integer: violation = |y - round(y)|
    # ------------------------------------------------------------------
    for i, j in arcs:
        y_val = y[i, j]
        nearest_int = round(y_val)
        violation_amount = abs(y_val - nearest_int)
        if violation_amount > tol:
            violated_constraints.add(4)
            rhs = nearest_int
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Integrality violated for arc ({i},{j}): y={y_val}"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": y_val,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 5 (Tier C): Objective consistency.
    # Full recompute -- every variable the objective depends on (y, x)
    # is present in the solution.
    #   obj = sum_{(i,j)} f_{ij}*y_{ij}
    #       + sum_k sum_{(i,j)} W^k * c^k_{ij} * x^k_{ij}
    # Reject when reported objective_value disagrees beyond tolerance.
    # ------------------------------------------------------------------
    fixed_costs = instance.get("fixed_costs", {})
    variable_costs = instance.get("variable_costs", {})
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            fixed_total = 0.0
            for i, j in arcs:
                key = f"{i}_{j}"
                f_ij = float(fixed_costs.get(key, 0))
                fixed_total += f_ij * float(y[i, j])
            variable_total = 0.0
            for k in range(num_commodities):
                Wk = float(commodities[k]["demand"])
                for i, j in arcs:
                    key = f"{i}_{j}"
                    c_list = variable_costs.get(key)
                    if not c_list or k >= len(c_list):
                        continue
                    c_k_ij = float(c_list[k])
                    xv = x[k, i, j]
                    if xv == 0.0:
                        continue
                    variable_total += Wk * c_k_ij * float(xv)
            true_obj = fixed_total + variable_total
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(5)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total cost="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
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
        description="Feasibility checker for MUFND solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    feasible = result["feasible"]
    violated_constraints = result["violated_constraints"]
    violations = result["violations"]

    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations[:10]:
            print(f"  {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    main()
