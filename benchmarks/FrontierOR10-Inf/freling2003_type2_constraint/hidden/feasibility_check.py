"""
Feasibility Checker for the Multiperiod Single-Sourcing Problem (MPSSP)
from Freling, Romeijn, Romero Morales, Wagelmans (2003),
Operations Research 51(6):922-939.

Checks each hard constraint of formulation (P_0) one by one:
  Constraint 1: sigma_t * sum_j d_j x_{ij} + I_{it} <= b_{it} + I_{i,t-1}  for all i,t
  Constraint 2: sum_i x_{ij} = 1  for all j
  Constraint 3: x_{ij} in {0,1}  for all i,j
  Constraint 4: I_{i0} = 0  for all i
  Constraint 5: I_{it} >= 0  for all i,t
  Constraint 6: objective consistency -- the reported objective_value must
                equal the objective recomputed from the solution variables:
                  obj = sum_{i,j} (sum_t c_{ijt}) x_{ij} + sum_{i,t} h_{it} I_{it}
                This is a Tier-C defense against fabricated objective values
                (e.g. reporting obj=0 or obj=sys.float_info.max). All variables
                that determine the objective (x from `assignment`, I from
                `inventory`) are present in the solution, so a full recompute
                is exact.
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def check_feasibility(instance, solution):
    m = instance["parameters"]["num_facilities"]
    n = instance["parameters"]["num_customers"]
    T = instance["parameters"]["num_periods"]
    sigma = instance["seasonal_factors"]
    demands = instance["demands"]
    capacities = instance["capacities"]  # capacities[i][t]

    # Parse assignment: x_{ij} — solution stores {str(j): i}
    assignment = solution["assignment"]  # dict str(j) -> i
    # Build x[i][j]
    x = [[0] * n for _ in range(m)]
    for j_str, i_val in assignment.items():
        j = int(j_str)
        i = int(i_val)
        if 0 <= i < m and 0 <= j < n:
            x[i][j] = 1

    # Parse inventory: I[i][t] — solution stores {str(i): [I_{i,1}, ..., I_{i,T}]}
    inventory_raw = solution.get("inventory", {})
    # I_it indexed as I[i][t] for t=1..T (0-indexed: t=0..T-1)
    # I_{i,0} = 0 by constraint 4
    I = [[0.0] * T for _ in range(m)]
    for i_str, inv_list in inventory_raw.items():
        i = int(i_str)
        if 0 <= i < m:
            for t in range(min(T, len(inv_list))):
                I[i][t] = float(inv_list[t])

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1: sigma_t * sum_j(d_j * x_{ij}) + I_{it} <= b_{it} + I_{i,t-1}
    #   for i=1..m, t=1..T
    #   Rearranged: LHS = sigma_t * sum_j(d_j * x_{ij}) + I_{it}
    #               RHS = b_{it} + I_{i,t-1}
    # ------------------------------------------------------------------
    for i in range(m):
        for t in range(T):
            demand_sum = sum(demands[j] * x[i][j] for j in range(n))
            lhs = sigma[t] * demand_sum + I[i][t]
            I_prev = I[i][t - 1] if t > 0 else 0.0
            rhs = capacities[i][t] + I_prev
            violation_amount = lhs - rhs  # <= constraint
            if violation_amount > TOL:
                violated_constraints.add(1)
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 1 violated: facility {i}, period {t+1}: "
                    f"LHS={lhs:.6f} > RHS={rhs:.6f} (excess={violation_amount:.6f})"
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
    # Constraint 2: sum_i x_{ij} = 1  for all j=1..n
    # ------------------------------------------------------------------
    for j in range(n):
        lhs = sum(x[i][j] for i in range(m))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            if lhs == 0:
                msg = f"Constraint 2 violated: customer {j} not assigned to any facility"
            elif lhs > 1:
                facs = [i for i in range(m) if x[i][j] == 1]
                msg = (
                    f"Constraint 2 violated: customer {j} assigned to multiple "
                    f"facilities {facs}"
                )
            else:
                msg = (
                    f"Constraint 2 violated: customer {j} assignment sum={lhs:.6f} != 1"
                )
            violations.append(msg)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 3: x_{ij} in {0, 1}  for all i,j
    # ------------------------------------------------------------------
    for i in range(m):
        for j in range(n):
            val = x[i][j]
            # Check if val is binary (0 or 1)
            violation_amount_0 = abs(val - 0.0)
            violation_amount_1 = abs(val - 1.0)
            violation_amount = min(violation_amount_0, violation_amount_1)
            if violation_amount > TOL:
                violated_constraints.add(3)
                lhs = float(val)
                # For binary constraint, closest bound
                rhs = 0.0 if violation_amount_0 < violation_amount_1 else 1.0
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 3 violated: x[{i}][{j}]={val} is not binary"
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
    # Constraint 4: I_{i,0} = 0  for all i
    # ------------------------------------------------------------------
    # The solution stores inventory for t=1..T. I_{i,0} is implicitly 0.
    # However, we check if the solution explicitly provides I_{i,0}.
    # In the solution format, inventory[str(i)] is a list of T values
    # corresponding to periods 1..T. I_{i,0} = 0 is implicit.
    # We still verify: if the solution had an I_{i,0} field, it must be 0.
    # Since the solution format gives [I_{i,1}, ..., I_{i,T}], I_{i,0}=0
    # is always satisfied by construction. But we check anyway for safety.
    # The initial inventory is not stored in the solution — it is always 0.
    # This constraint is always satisfied by the solution format.
    # We still record the check for completeness.
    # (No violation possible here since I_{i,0} is hardcoded as 0.)

    # ------------------------------------------------------------------
    # Constraint 5: I_{it} >= 0  for all i=1..m, t=1..T
    # ------------------------------------------------------------------
    for i in range(m):
        for t in range(T):
            lhs = I[i][t]
            rhs = 0.0
            violation_amount = rhs - lhs  # >= constraint: how much RHS exceeds LHS
            if violation_amount > TOL:
                violated_constraints.add(5)
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 5 violated: I[{i}][{t+1}]={lhs:.6f} < 0"
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
    # Constraint 6: objective consistency (Tier-C anti-gaming check).
    #   The reported objective_value must equal the objective recomputed
    #   from the solution variables:
    #     obj = sum_{i,j} (sum_t c_{ijt}) x_{ij} + sum_{i,t} h_{it} I_{it}
    #   All obj-determining variables (x from `assignment`, I from
    #   `inventory`) are present in the solution, so this is an exact
    #   full recompute. Tolerance: max(1e-3 absolute, 1e-3 relative).
    # ------------------------------------------------------------------
    transport_costs = instance["transportation_costs"]  # c_{ijt}, shape m x n x T
    holding_costs = instance["holding_costs"]           # h_{it}, shape m x T
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            transport_total = 0.0
            for i in range(m):
                for j in range(n):
                    if x[i][j]:
                        transport_total += sum(
                            transport_costs[i][j][t] for t in range(T)
                        )
            holding_total = 0.0
            for i in range(m):
                for t in range(T):
                    holding_total += holding_costs[i][t] * I[i][t]
            true_obj = float(transport_total + holding_total)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(6)
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint 6 violated: objective consistency: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_ij(sum_t c_ijt)x_ij + sum_it h_it I_it={true_obj} "
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
        description="Feasibility checker for MPSSP (Freling et al. 2003)."
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

    save_json(args.result_path, result)
    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
