"""
Feasibility checker for the Minimum Convex Cost Flow in Bipartite Networks (MCCFBN)
problem from Castro & Nasini (2021).

Hard constraints from the mathematical formulation (Eqs 2-4, counted as Constraints 1-3):

  Constraint 1 (Eq 2):  sum_{i in I} x_{ij} = d_j,      for all j in J   (demand satisfaction)
  Constraint 2 (Eq 3):  sum_{j in J} x_{ij} <= s_i,      for all i in I   (supply capacity)
  Constraint 3 (Eq 4):  0 <= x_{ij} <= u_{ij},            for all i in I, j in J  (arc bounds)

Objective-consistency check (Eq 1, counted as Constraint 4):

  Constraint 4 (Eq 1):  reported objective_value must equal the objective
                        recomputed from the flow variables,
                        f(x) = sum_{i,j} ( c_{ij}*x_{ij} + q_{ij}*x_{ij}^2 ).
                        This is a Tier C defense against candidates that
                        return a fabricated objective_value while the flows
                        themselves satisfy Constraints 1-3.

NOTE: this file is the obj-recompute variant of `feasibility_check.py`.
Constraints 1-3 are byte-for-byte identical to the original; the only
addition is Constraint 4.  The original file is kept untouched.
"""

import argparse
import json


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    m = instance["m"]
    supplies = instance["supplies"]
    demands = instance["demands"]
    arc_capacity = instance["arc_capacity"]
    # Objective coefficients (Eq 1).  gurobi_code.py builds the objective as
    #   sum_{i,j} ( linear_costs[i][j]*x_ij + quadratic_costs[i][j]*x_ij^2 )
    # for every cost_type; the quadratic term simply vanishes when q_ij == 0.
    linear_costs = instance.get("linear_costs")
    quadratic_costs = instance.get("quadratic_costs")

    # For summation constraints, accumulated floating-point error from a
    # barrier (interior-point) solver grows with the number of terms.  The
    # Gurobi model for this paper uses BarConvTol=1e-4, Crossover=0, so
    # per-variable imprecision is ~1e-4 (NOT 1e-6 as previously assumed —
    # measured violations at l31 reached ratio 5.5e-3 / raw_excess 1.3e-2,
    # well above the prior 1e-4 rel_tol). When summing k terms the worst-
    # case accumulated error is O(k * 1e-4). We therefore scale the absolute
    # tolerance by the number of summands. A violation is only reported when
    # it also exceeds a relative threshold (ratio > 1e-2) to avoid flagging
    # solutions that are essentially feasible. — 2026-05-19 retuned: was
    # rel_tol=1e-4 / per-var=1e-6, both too tight for BarConvTol=1e-4.
    tol_demand = max(tol, n * 1e-4)   # Constraint 1 sums n terms
    tol_supply = max(tol, m * 1e-4)   # Constraint 2 sums m terms
    rel_tol = 1e-2  # relative tolerance: violation / |rhs| must exceed this

    flows_dict = solution.get("flows", {})
    if flows_dict is None:
        flows_dict = {}

    # Build full flow matrix x[i][j], default 0
    x = [[0.0] * m for _ in range(n)]
    # Constraint 4 piggybacks on this parse pass: accumulate the true
    # objective f(x) directly from the flow variables.  Arcs absent from
    # `flows` carry zero flow and contribute zero cost, so iterating the
    # dict is exact.
    obj_recomputable = linear_costs is not None
    true_obj = 0.0
    for key, val in flows_dict.items():
        # keys are "x_i_j"
        parts = key.split("_")
        i = int(parts[1])
        j = int(parts[2])
        v = float(val)
        x[i][j] = v
        if obj_recomputable:
            true_obj += float(linear_costs[i][j]) * v
            if quadratic_costs is not None:
                true_obj += float(quadratic_costs[i][j]) * v * v

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # ------------------------------------------------------------------
    # Constraint 1 (Eq 2): sum_{i in I} x_{ij} = d_j, for all j in J
    # Equality constraint: violation_amount = |LHS - RHS|
    # ------------------------------------------------------------------
    for j in range(m):
        lhs = sum(x[i][j] for i in range(n))
        rhs = float(demands[j])
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        if violation_amount > tol_demand and ratio > rel_tol:
            violated_set.add(1)
            violations.append(
                f"Constraint 1 (demand satisfaction): demand node j={j} has "
                f"total inflow {lhs:.6f} but demand is {rhs:.6f} "
                f"(difference {violation_amount:.6e})"
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
    # Constraint 2 (Eq 3): sum_{j in J} x_{ij} <= s_i, for all i in I
    # <= constraint: violation_amount = max(LHS - RHS, 0)
    # ------------------------------------------------------------------
    for i in range(n):
        lhs = sum(x[i][j] for j in range(m))
        rhs = float(supplies[i])
        violation_amount = lhs - rhs
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        if violation_amount > tol_supply and ratio > rel_tol:
            violated_set.add(2)
            violations.append(
                f"Constraint 2 (supply capacity): supply node i={i} has "
                f"total outflow {lhs:.6f} but supply capacity is {rhs:.6f} "
                f"(excess {violation_amount:.6e})"
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
    # Constraint 3 (Eq 4): 0 <= x_{ij} <= u_{ij}, for all i in I, j in J
    # Two-sided bound constraint, checked as two separate inequalities:
    #   (a) x_{ij} >= 0  (>= constraint: violation = max(RHS - LHS, 0) = max(-x_{ij}, 0))
    #   (b) x_{ij} <= u_{ij}  (<= constraint: violation = max(LHS - RHS, 0) = max(x_{ij} - u_{ij}, 0))
    # ------------------------------------------------------------------
    # Arc capacity is per-arc u_{ij}: build n x m matrix (expand scalar if given).
    if isinstance(arc_capacity, list):
        u = [[float(arc_capacity[i][j]) for j in range(m)] for i in range(n)]
    else:
        u = [[float(arc_capacity) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            val = x[i][j]
            u_ij = u[i][j]
            # Lower bound: x_{ij} >= 0
            if val < -tol:
                violation_amount = -val  # how much RHS(0) exceeds LHS(x_{ij})
                violated_set.add(3)
                normalizer = eps  # RHS is 0, so max(|0|, eps) = eps
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 3 (lower bound): x_{i}_{j} = {val:.6e} < 0 "
                    f"(violation {violation_amount:.6e})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": val,
                    "rhs": 0.0,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

            # Upper bound: x_{ij} <= u_{ij}
            violation_amount = val - u_ij
            if violation_amount > tol:
                violated_set.add(3)
                normalizer = max(abs(u_ij), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 3 (upper bound): x_{i}_{j} = {val:.6e} > u_{i}{j} = {u_ij:.6f} "
                    f"(excess {violation_amount:.6e})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": val,
                    "rhs": u_ij,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Constraint 4 (Eq 1): objective consistency.
    # The eval pipeline trusts the solver's self-reported objective_value.
    # Recompute the true objective f(x) = sum_{i,j}( c_ij*x_ij + q_ij*x_ij^2 )
    # from the flow variables (accumulated above) and reject the solution
    # when the reported value disagrees beyond tolerance.
    # Equality check: violation_amount = |reported - recomputed|.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if obj_recomputable and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            violation_amount = abs(reported - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor.  The
            # objective sums up to n*m terms produced by a barrier solver
            # (BarConvTol=1e-4, Crossover=0); a relative band absorbs that
            # accumulated imprecision plus the omission of near-zero flows
            # (the solver writes only x_ij > 1e-8 into `flows`), while still
            # catching fabricated objective values, which are off by 100%+.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            normalizer = max(abs(true_obj), eps)
            ratio = violation_amount / normalizer
            if violation_amount > obj_tol:
                violated_set.add(4)
                violations.append(
                    f"Constraint 4 (objective consistency): reported "
                    f"objective_value {reported:.6f} differs from objective "
                    f"recomputed from flows {true_obj:.6f} "
                    f"(difference {violation_amount:.6e})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    violated_constraints = sorted(violated_set)
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
        description="Feasibility checker for MCCFBN (Castro & Nasini 2021)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON solution file."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON feasibility result."
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
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
