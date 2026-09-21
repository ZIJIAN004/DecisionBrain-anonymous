#!/usr/bin/env python3
"""
Feasibility checker for the Stochastic Capacitated Facility Location (SFL) problem.
Rahmaniani et al. (2020), Appendix D.2, Equations (D.4)-(D.7).

Constraint numbering (top to bottom in formulation section):
  Constraint 1: (D.5) sum_i x_{ij}^s >= d_j^s       for all j in M, s in S  [demand satisfaction, >=]
  Constraint 2: (D.6) sum_j x_{ij}^s <= u_i * y_i    for all i in N, s in S  [capacity, <=]
  Constraint 3: (D.7) sum_i u_i * y_i >= max_s sum_j d_j^s                   [complete recourse, >=]
  Constraint 4: (D.4) Objective consistency: reported objective_value must equal
                f^T y + sum_s p_s * min_x { sum_{i,j} c_{ij} x_{ij}^s : D.5, D.6, x>=0 }
                (Tier C defense against fabricated objective values.)

Since solutions only contain y (facility opening decisions) and not x (flow variables),
constraints 1 and 2 are checked by solving an LP per scenario to determine if feasible
x exists given y. If demand cannot be fully met, constraint 1 (D.5) is reported violated.
Constraint 2 (D.6) is verified explicitly on the LP-optimal x.
Constraint 3 (D.7) is checked directly from y.

Constraint 4 recomputes the true objective by solving a cost-minimization transportation
LP per scenario (the recourse problem) and comparing the expected total cost against the
reported objective_value.
"""

import json
import argparse
import numpy as np
from scipy.optimize import linprog


def check_feasibility(data, sol):
    tol = 1e-5
    eps = 1e-5

    N = data["num_facilities"]
    M = data["num_customers"]
    S = data["num_scenarios"]
    cap = data["facilities"]["capacities"]
    demands = data["scenarios"]["demands"]
    y = sol["y"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 3 (D.7): sum_i u_i * y_i >= max_s sum_j d_j^s
    # Type: >= constraint. Violation = RHS - LHS when RHS > LHS.
    # ------------------------------------------------------------------
    D_max = max(sum(demands[s][j] for j in range(M)) for s in range(S))
    lhs_3 = sum(cap[i] * y[i] for i in range(N))
    rhs_3 = float(D_max)
    viol_3 = rhs_3 - lhs_3
    if viol_3 > tol:
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3 (D.7) violated: sum(u_i*y_i)={lhs_3:.6f} < "
            f"max_s(sum_j d_j^s)={rhs_3:.6f}"
        )
        normalizer = max(abs(rhs_3), eps)
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_3,
            "rhs": rhs_3,
            "raw_excess": viol_3,
            "normalizer": normalizer,
            "ratio": viol_3 / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraints 1 (D.5) and 2 (D.6): checked via LP per scenario.
    #
    # For each scenario s we solve:
    #   min  sum_j slack_j
    #   s.t. sum_i x_{ij} + slack_j >= d_j^s   for all j   (D.5 relaxed)
    #        sum_j x_{ij}          <= u_i*y_i   for all i   (D.6 enforced)
    #        x >= 0, slack >= 0
    #
    # If total slack > 0, constraint 1 (D.5) is violated for that scenario.
    # Constraint 2 (D.6) is then verified explicitly on the found x.
    # ------------------------------------------------------------------
    feasibility_ok_per_scenario = [True] * S
    for s in range(S):
        d_s = demands[s]
        n_x = N * M
        n_slack = M
        n_vars = n_x + n_slack

        # Objective: minimize total demand shortfall
        c_obj = np.zeros(n_vars)
        c_obj[n_x:] = 1.0

        A_ub_rows = []
        b_ub_vals = []

        # D.5 (>=): sum_i x_{ij} + slack_j >= d_j^s
        #   => -sum_i x_{ij} - slack_j <= -d_j^s
        for j in range(M):
            row = np.zeros(n_vars)
            for i in range(N):
                row[i * M + j] = -1.0
            row[n_x + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(-float(d_s[j]))

        # D.6 (<=): sum_j x_{ij} <= u_i * y_i
        for i in range(N):
            row = np.zeros(n_vars)
            for j in range(M):
                row[i * M + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(float(cap[i] * y[i]))

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, None)] * n_vars

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            # Should not happen with slack variables, but handle defensively
            violated_constraints.add(1)
            violations.append(
                f"Constraint 1 (D.5) violated: LP solver failed for scenario {s}"
            )
            feasibility_ok_per_scenario[s] = False
            continue

        # Check D.5 violations via slack values
        if res.fun > tol:
            violated_constraints.add(1)
            feasibility_ok_per_scenario[s] = False
            slack = res.x[n_x:]
            shortfall_details = []
            for j in range(M):
                if slack[j] > tol:
                    # LHS = sum_i x_{ij}^s = d_j^s - slack_j (actual flow)
                    lhs_val = float(d_s[j]) - slack[j]
                    rhs_val = float(d_s[j])
                    viol_amt = slack[j]
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": viol_amt,
                        "normalizer": normalizer,
                        "ratio": viol_amt / normalizer,
                    })
                    shortfall_details.append(f"customer {j} short by {slack[j]:.6f}")
            violations.append(
                f"Constraint 1 (D.5) violated in scenario {s}: "
                f"total shortfall={res.fun:.6f} ({'; '.join(shortfall_details)})"
            )

        # Explicitly verify D.6 on the LP-optimal x
        x_vals = res.x[:n_x].reshape(N, M)
        for i in range(N):
            flow_i = float(np.sum(x_vals[i, :]))
            cap_rhs = float(cap[i] * y[i])
            viol_6 = flow_i - cap_rhs  # <= constraint: violation when LHS > RHS
            if viol_6 > tol:
                violated_constraints.add(2)
                normalizer = max(abs(cap_rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": flow_i,
                    "rhs": cap_rhs,
                    "raw_excess": viol_6,
                    "normalizer": normalizer,
                    "ratio": viol_6 / normalizer,
                })
                violations.append(
                    f"Constraint 2 (D.6) violated: facility {i}, scenario {s}, "
                    f"flow={flow_i:.6f} > capacity={cap_rhs:.6f}"
                )

    # ------------------------------------------------------------------
    # Constraint 4: Objective consistency (Tier C anti-exploit defense).
    #
    # Recompute the optimal objective for the given y by solving the recourse
    # problem (a transportation LP) per scenario, then compare to the reported
    # objective_value. The full-recompute formula:
    #
    #   true_obj = sum_i f_i * y_i
    #            + sum_s p_s * min_{x>=0} { sum_{i,j} c_{ij} x_{ij}^s
    #                                       : sum_i x_{ij}^s >= d_j^s (D.5),
    #                                         sum_j x_{ij}^s <= u_i*y_i (D.6) }
    #
    # Skip the check if any structural constraint (1, 2, or 3) already
    # marks the solution infeasible -- in that case the obj is undefined,
    # and the existing violations already cause feasible=False.
    # ------------------------------------------------------------------
    structural_violation = bool(violated_constraints & {1, 2, 3})
    reported_obj_raw = sol.get("objective_value")
    try:
        reported_obj = float(reported_obj_raw) if reported_obj_raw is not None else None
    except (TypeError, ValueError):
        reported_obj = None

    if reported_obj is not None and not structural_violation:
        fixed_costs = data["facilities"]["fixed_costs"]
        routing_costs = data["routing_costs"]
        probs = data["scenarios"]["probabilities"]

        setup_cost = float(sum(fixed_costs[i] * y[i] for i in range(N)))

        # Build the cost vector for the routing LP (same A_ub structure as above
        # but without slack variables, since structural feasibility is already
        # confirmed for every scenario).
        n_x = N * M
        c_route = np.empty(n_x)
        for i in range(N):
            for j in range(M):
                c_route[i * M + j] = float(routing_costs[i][j])

        # D.6 capacity rows are scenario-independent (depend only on y).
        cap_rows = np.zeros((N, n_x))
        cap_rhs_arr = np.empty(N)
        for i in range(N):
            for j in range(M):
                cap_rows[i, i * M + j] = 1.0
            cap_rhs_arr[i] = float(cap[i] * y[i])

        # D.5 demand rows depend on scenario; build template once.
        demand_template = np.zeros((M, n_x))
        for j in range(M):
            for i in range(N):
                demand_template[j, i * M + j] = -1.0

        recompute_failed_for_scenario = False
        expected_recourse = 0.0
        for s in range(S):
            d_s = demands[s]
            demand_rhs = np.array([-float(d_s[j]) for j in range(M)])
            A_ub_s = np.vstack([demand_template, cap_rows])
            b_ub_s = np.concatenate([demand_rhs, cap_rhs_arr])
            res = linprog(
                c_route,
                A_ub=A_ub_s,
                b_ub=b_ub_s,
                bounds=[(0, None)] * n_x,
                method="highs",
            )
            if not res.success or res.fun is None:
                recompute_failed_for_scenario = True
                break
            expected_recourse += float(probs[s]) * float(res.fun)

        if not recompute_failed_for_scenario:
            true_obj = setup_cost + expected_recourse
            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative tolerance with 1.0 absolute floor (obj is on the
            # order of 1e3 to 1e5 cost units; floating-point recompute noise
            # is well under 1).
            tol_obj = max(1.0, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported_obj),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": float(abs_diff) / normalizer,
                })
                violations.append(
                    f"Constraint 4 (D.4) violated: reported objective_value="
                    f"{reported_obj} differs from recomputed setup+E[recourse]="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    vc_sorted = sorted(violated_constraints)
    return {
        "feasible": len(vc_sorted) == 0,
        "violated_constraints": vc_sorted,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if violation_magnitudes else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SFL (Rahmaniani et al. 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)

    result = check_feasibility(data, sol)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
