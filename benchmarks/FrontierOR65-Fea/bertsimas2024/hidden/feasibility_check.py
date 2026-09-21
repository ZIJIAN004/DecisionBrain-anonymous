#!/usr/bin/env python3
"""
Feasibility checker for Stochastic Multi-commodity Capacitated
Fixed-charge Network Design (MCFND) from Bertsimas et al. (2024).

Constraints (in paper order, Problem (1), page 4):
  1. Flow conservation: A x^{k,r} = d^{k,r}  for all k, r
  2. Capacity: sum_k x_{i,j}^{k,r} <= u_{i,j}  for all (i,j), r
  3. Linking + nonnegativity: x >= 0, x_{i,j}^{k,r} = 0 if z_{i,j} = 0
  4. Cardinality + binary: sum z_{i,j} <= c_0, z_{i,j} in {0,1}
  5. Objective consistency (lower-bound check, Tier C anti-exploit):
        reported objective_value must be >= construction(z) +
        (1/R) * sum_r min_x_feasible(r,z) sum_e f_e * sum_k x_{e,k,r}
     The quadratic congestion penalty in the true objective is >= 0, so the
     linear-only LP minimum gives a sound lower bound on per-scenario cost.
     Solution carries only first-stage z; second-stage flows are implied.

The solution provides only z (binary design variables). Flow variables x
are determined by solving the second-stage LP for each scenario given z.
"""

import argparse
import json
import math
import numpy as np
from scipy.optimize import linprog


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of MCFND solution."
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

    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violation_messages = {}   # constraint_idx -> list of message strings
    violation_magnitudes = []

    N = data["num_nodes"]
    E = data["num_edges"]
    K = data["num_commodities"]
    R = data["num_scenarios"]
    c_0 = data["c_0"]
    edges = data["edges"]
    caps = data["capacities"]
    demands = data["demands"]  # [r][k][n]
    flow_costs = data["flow_costs"]
    construction_costs = data["construction_costs"]

    z_dict = sol.get("z", {})

    # ----------------------------------------------------------------
    # Handle empty/infeasible solution (no z provided or solver infeasible)
    # ----------------------------------------------------------------
    solver_status = sol.get("status")
    if not z_dict:
        if solver_status in (3, 4, 5):
            # Gurobi status 3=infeasible, 4=inf_or_unbd, 5=unbounded:
            # no solution exists, so feasibility is indeterminate (null).
            result = {
                "feasible": None,
                "violated_constraints": [],
                "violations": [
                    f"Solver returned status {solver_status} "
                    f"(infeasible/unbounded); no solution to check."
                ],
                "violation_magnitudes": [],
            }
        else:
            # Solution file exists but z is empty for an unknown reason
            result = {
                "feasible": None,
                "violated_constraints": [],
                "violations": [
                    "No design variables (z) provided in the solution; "
                    "no solution to check."
                ],
                "violation_magnitudes": [],
            }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        return

    z = np.array([float(z_dict.get(str(e), 0.0)) for e in range(E)])

    # ================================================================
    # Constraint 4: Cardinality + Binary domain
    #   sum_{(i,j)} z_{i,j} <= c_0,  z_{i,j} in {0,1}
    # ================================================================

    # 4a: Cardinality constraint  (<=)
    z_sum = float(np.sum(z))
    card_excess = max(0.0, z_sum - c_0)
    if card_excess > tol:
        violated_constraints.add(4)
        violation_messages.setdefault(4, []).append(
            f"Cardinality violated: sum(z) = {z_sum:.4f} > c_0 = {c_0}"
        )
        normalizer = max(abs(float(c_0)), eps)
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": z_sum,
            "rhs": float(c_0),
            "raw_excess": card_excess,
            "normalizer": normalizer,
            "ratio": card_excess / normalizer,
        })

    # 4b: Binary domain  (z in {0,1})
    binary_viols = []
    for e in range(E):
        dist = min(abs(z[e]), abs(z[e] - 1.0))
        if dist > tol:
            binary_viols.append(e)
            nearest = round(z[e])
            normalizer = max(abs(nearest), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(z[e]),
                "rhs": float(nearest),
                "raw_excess": float(dist),
                "normalizer": normalizer,
                "ratio": dist / normalizer,
            })
    if binary_viols:
        violated_constraints.add(4)
        msg = (f"Binary domain violated for {len(binary_viols)} edge(s): "
               f"e.g. z[{binary_viols[0]}] = {z[binary_viols[0]]:.6f}")
        violation_messages.setdefault(4, []).append(msg)

    # ================================================================
    # Constraints 1, 2, 3: checked via second-stage LP per scenario
    #
    # For each scenario r, we solve:
    #   find x >= 0 on active edges (z_e >= 0.5) such that
    #     A x^{k,r} = d^{k,r}              (constraint 1)
    #     sum_k x_{e}^{k,r} <= u_e          (constraint 2)
    #     x_{e}^{k,r} = 0 for inactive e    (constraint 3, linking)
    #
    # We MINIMIZE the linear transportation cost f_e * sum_k x_{e,k,r}.
    # The LP's feasibility is independent of the objective, so this still
    # serves as the feasibility test. When feasible, res.fun also gives the
    # tight lower bound on per-scenario transport cost used by C5 below.
    #
    # If infeasible, we solve an elastic LP to quantify violations.
    # ================================================================

    active_edges = [e for e in range(E) if z[e] >= 0.5]
    n_act = len(active_edges)

    # Build node-to-active-edge incidence
    out_act = [[] for _ in range(N)]
    in_act = [[] for _ in range(N)]
    for ai, e in enumerate(active_edges):
        out_act[edges[e][0]].append(ai)
        in_act[edges[e][1]].append(ai)

    flow_viol_count = 0
    cap_viol_count = 0
    link_viol_scenarios = []

    # Per-scenario obj bounds (for C5 envelope check):
    #   transport_lower_sum: sum over r of min linear transport cost (LP min).
    #   transport_upper_sum: sum over r of full (linear + quadratic) cost
    #     evaluated at the LP-min flow. Since the LP-min flow is feasible,
    #     its full cost upper-bounds the true second-stage QP optimum.
    transport_lower_sum = 0.0
    transport_upper_sum = 0.0
    transport_bound_valid = True
    gamma_val = float(data["gamma"])

    for r in range(R):
        n_vars = K * n_act

        # --- No active edges: any nonzero demand violates constraints ---
        if n_vars == 0:
            any_demand = False
            for k in range(K):
                for n in range(N):
                    d = demands[r][k][n]
                    if abs(d) > tol:
                        any_demand = True
                        # Flow conservation (C1) cannot be satisfied because
                        # linking (C3) forces all x = 0
                        violated_constraints.add(1)
                        violated_constraints.add(3)
                        flow_viol_count += 1
                        normalizer = max(abs(d), eps)
                        violation_magnitudes.append({
                            "constraint": 1,
                            "lhs": 0.0,
                            "rhs": d,
                            "raw_excess": abs(d),
                            "normalizer": normalizer,
                            "ratio": abs(d) / normalizer,
                        })
            if any_demand:
                transport_bound_valid = False
                if 3 not in link_viol_scenarios:
                    link_viol_scenarios.append(r)
            # No active edges but no demand: transport_lower_sum += 0
            continue

        # --- Build flow conservation (equality) constraints ---
        n_eq = K * N
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)
        for k in range(K):
            for n in range(N):
                row = k * N + n
                b_eq[row] = demands[r][k][n]
                for ai in out_act[n]:
                    A_eq[row, k * n_act + ai] = 1.0
                for ai in in_act[n]:
                    A_eq[row, k * n_act + ai] = -1.0

        # --- Build capacity (inequality) constraints ---
        A_ub = np.zeros((n_act, n_vars))
        b_ub = np.array([caps[active_edges[ai]] for ai in range(n_act)])
        for ai in range(n_act):
            for k in range(K):
                A_ub[ai, k * n_act + ai] = 1.0

        bounds = [(0, None)] * n_vars

        # --- LP: minimize linear transport cost; doubles as feasibility test ---
        # Cost vector: c[k*n_act + ai] = flow_costs[active_edges[ai]] for all k.
        # (Per-edge flow cost is shared across commodities per instance schema.)
        c_lp = np.empty(n_vars)
        for ai in range(n_act):
            fe = flow_costs[active_edges[ai]]
            for k in range(K):
                c_lp[k * n_act + ai] = fe

        res = linprog(
            c_lp,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
            options={"presolve": True},
        )

        if res.success:
            # Feasible: constraints 1, 2, 3 satisfied for this scenario.
            # res.fun = min linear transport cost = LB on scenario cost.
            transport_lower_sum += float(res.fun)
            # Compute upper bound: use LP-min flow as a feasible witness and
            # add its quadratic penalty term to the linear cost.
            xv = np.asarray(res.x)
            quad_r = 0.0
            for ai in range(n_act):
                # F_{e,r} = sum_k x_{e,k,r}
                Fer = 0.0
                for k in range(K):
                    Fer += float(xv[k * n_act + ai])
                quad_r += Fer * Fer
            quad_r = quad_r / (2.0 * gamma_val)
            transport_upper_sum += float(res.fun) + quad_r
            continue

        # Could not get an LP-min; cannot tighten the obj LB for this scenario.
        transport_bound_valid = False

        # --- Infeasible: solve elastic LP to quantify violations ---
        #
        # Extended variables:
        #   x (n_vars) | s_plus (n_eq) | s_minus (n_eq) | t (n_act)
        #
        # Modified constraints:
        #   A_eq x + s_plus - s_minus = b_eq   (relaxed flow conservation)
        #   A_ub x - t <= b_ub                  (relaxed capacity)
        #   all >= 0
        #
        # Objective: min sum(s_plus) + sum(s_minus) + sum(t)

        n_total = n_vars + 2 * n_eq + n_act
        c_ext = np.zeros(n_total)
        c_ext[n_vars:] = 1.0

        A_eq_ext = np.zeros((n_eq, n_total))
        A_eq_ext[:, :n_vars] = A_eq
        for i in range(n_eq):
            A_eq_ext[i, n_vars + i] = 1.0              # s_plus
            A_eq_ext[i, n_vars + n_eq + i] = -1.0      # s_minus

        A_ub_ext = np.zeros((n_act, n_total))
        A_ub_ext[:, :n_vars] = A_ub
        for i in range(n_act):
            A_ub_ext[i, n_vars + 2 * n_eq + i] = -1.0  # -t

        bounds_ext = [(0, None)] * n_total

        res_ext = linprog(
            c_ext,
            A_ub=A_ub_ext, b_ub=b_ub,
            A_eq=A_eq_ext, b_eq=b_eq,
            bounds=bounds_ext, method="highs",
            options={"presolve": True},
        )

        if not res_ext.success:
            violated_constraints.add(1)
            flow_viol_count += 1
            violation_messages.setdefault(1, []).append(
                f"Scenario {r}: elastic LP also failed"
            )
            continue

        xv = res_ext.x

        # --- Flow conservation violations (Constraint 1) ---
        scenario_has_flow_viol = False
        for i in range(n_eq):
            sp = xv[n_vars + i]
            sm = xv[n_vars + n_eq + i]
            viol_amount = sp + sm
            if viol_amount > tol:
                k_idx = i // N
                n_idx = i % N
                violated_constraints.add(1)
                scenario_has_flow_viol = True
                flow_viol_count += 1
                rhs = b_eq[i]
                lhs = rhs + sp - sm  # actual net outflow
                raw = abs(lhs - rhs)
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(raw),
                    "normalizer": normalizer,
                    "ratio": raw / normalizer,
                })

        # --- Capacity violations (Constraint 2) ---
        for i in range(n_act):
            tv = xv[n_vars + 2 * n_eq + i]
            if tv > tol:
                e_idx = active_edges[i]
                violated_constraints.add(2)
                cap_viol_count += 1
                lhs = b_ub[i] + tv
                rhs = b_ub[i]
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(tv),
                    "normalizer": normalizer,
                    "ratio": tv / normalizer,
                })

        # If flow conservation is violated, it may be because the linking
        # constraint (C3) prevents routing through inactive edges.
        if scenario_has_flow_viol:
            link_viol_scenarios.append(r)

    # ================================================================
    # Constraint 3: Linking + Nonnegativity
    #
    # x >= 0 is enforced by LP bounds (always satisfied).
    # x = 0 on inactive edges is enforced by construction (always satisfied).
    #
    # However, if flow conservation (C1) is violated because needed edges
    # are inactive, the root cause is the linking constraint (C3).
    # We check whether removing the linking restriction would make the
    # problem feasible.
    # ================================================================
    if link_viol_scenarios and 1 in violated_constraints:
        # Check if the problem would be feasible with all edges active
        # (i.e., without the linking constraint)
        out_all = [[] for _ in range(N)]
        in_all = [[] for _ in range(N)]
        for e_idx in range(E):
            out_all[edges[e_idx][0]].append(e_idx)
            in_all[edges[e_idx][1]].append(e_idx)

        for r in link_viol_scenarios[:3]:  # check a few scenarios
            n_vars_all = K * E
            n_eq_all = K * N

            A_eq_all = np.zeros((n_eq_all, n_vars_all))
            b_eq_all = np.zeros(n_eq_all)
            for k in range(K):
                for n in range(N):
                    row = k * N + n
                    b_eq_all[row] = demands[r][k][n]
                    for ei in out_all[n]:
                        A_eq_all[row, k * E + ei] = 1.0
                    for ei in in_all[n]:
                        A_eq_all[row, k * E + ei] = -1.0

            A_ub_all = np.zeros((E, n_vars_all))
            b_ub_all = np.array(caps)
            for ei in range(E):
                for k in range(K):
                    A_ub_all[ei, k * E + ei] = 1.0

            res_all = linprog(
                np.zeros(n_vars_all),
                A_ub=A_ub_all, b_ub=b_ub_all,
                A_eq=A_eq_all, b_eq=b_eq_all,
                bounds=[(0, None)] * n_vars_all, method="highs",
                options={"presolve": True},
            )

            if res_all.success:
                # Feasible without linking => linking is the cause
                violated_constraints.add(3)
                violation_messages.setdefault(3, []).append(
                    "Linking constraint prevents feasible flow routing: "
                    "deactivated edges block required flow paths"
                )
                # Compute violation magnitude for constraint 3:
                # For each inactive edge, the linking says x = 0 if z = 0.
                # The violation is that flow "needs" to go through inactive edges.
                # We approximate by the total flow violation magnitude.
                # (Individual linking violations are hard to isolate.)
                break

    # Also check for fractional z as a constraint 3 issue
    # (linking is x=0 if z=0; fractional z makes this ambiguous)
    for e in range(E):
        if tol < z[e] < 0.5:
            # z is fractional and close to 0; we treated edge as inactive
            # but z != 0 strictly, so "x=0 if z=0" may not apply
            violated_constraints.add(3)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(z[e]),
                "rhs": 0.0,
                "raw_excess": float(z[e]),
                "normalizer": eps,
                "ratio": float(z[e] / eps),
            })
            violation_messages.setdefault(3, []).append(
                f"Fractional z[{e}] = {z[e]:.6f}: linking constraint "
                f"ambiguous (z not binary)"
            )
            break  # one message is enough

    # ================================================================
    # Constraint 5: Objective consistency (envelope check)
    #
    # True objective:
    #   obj = sum_e c_e * z_e
    #       + (1/R) * sum_r sum_e [ f_e * sum_k x_{e,k,r}
    #                              + (1/(2*gamma)) * (sum_k x_{e,k,r})^2 ]
    #
    # The solution only carries first-stage z; the second-stage flows x
    # are implied as the QP optimum given z. We bracket the true obj with:
    #
    #   obj_lower = sum_e c_e * z_e
    #             + (1/R) * sum_r min_{x feasible} sum_e f_e * sum_k x_{e,k,r}
    #     (drops the >=0 quadratic penalty; the inner min is the per-scenario
    #      LP res.fun computed above)
    #
    #   obj_upper = sum_e c_e * z_e
    #             + (1/R) * sum_r [ linear(x_lin_r) + quadratic(x_lin_r) ]
    #     (x_lin_r is the LP-min feasible flow we already have; its full cost
    #      upper-bounds the second-stage QP optimum since the QP minimum is
    #      attained over the same feasible set)
    #
    # Reject the reported objective_value when it falls outside [obj_lower,
    # obj_upper] (with a small tolerance). The lower side catches obj=0 /
    # under-reporting exploits; the upper side catches obj=sys.float_info.max
    # / over-reporting exploits. Only applied when constraints 1-4 are
    # satisfied (so the per-scenario LPs were all feasible).
    #
    # Tolerance: max(1e-3, 1e-3 * |bound|) (0.1% relative, 1e-3 floor).
    # ================================================================
    reported_obj = sol.get("objective_value")
    obj_check_applicable = (
        len(violated_constraints) == 0
        and transport_bound_valid
        and reported_obj is not None
    )
    if obj_check_applicable:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is None or not math.isfinite(reported):
            # Non-finite / unparseable reported obj is itself an obj-
            # consistency violation: a real solver would return a finite
            # cost or an explicit infeasible status.
            violated_constraints.add(5)
            violation_messages.setdefault(5, []).append(
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj!r} is not a finite number"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float("nan") if reported is None else float(reported),
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })
        else:
            construction = float(sum(construction_costs[e] * z[e]
                                     for e in range(E)))
            mean_transport_lower = transport_lower_sum / float(R)
            mean_transport_upper = transport_upper_sum / float(R)
            obj_lower = construction + mean_transport_lower
            obj_upper = construction + mean_transport_upper
            shortfall = obj_lower - reported  # > 0 if under-reported
            excess = reported - obj_upper      # > 0 if over-reported
            tol_lo = max(1e-3, 1e-3 * abs(obj_lower))
            tol_hi = max(1e-3, 1e-3 * abs(obj_upper))
            if shortfall > tol_lo:
                violated_constraints.add(5)
                violation_messages.setdefault(5, []).append(
                    f"Objective consistency violated: reported objective_value"
                    f"={reported} < lower bound={obj_lower:.6f} "
                    f"(construction={construction:.6f}, "
                    f"mean_min_transport={mean_transport_lower:.6f}; "
                    f"quadratic penalty omitted as it is >= 0). "
                    f"shortfall={shortfall:.4g}, tol={tol_lo:.4g}"
                )
                normalizer = max(abs(obj_lower), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(obj_lower),
                    "raw_excess": float(shortfall),
                    "normalizer": normalizer,
                    "ratio": float(shortfall / normalizer),
                })
            elif excess > tol_hi:
                violated_constraints.add(5)
                violation_messages.setdefault(5, []).append(
                    f"Objective consistency violated: reported objective_value"
                    f"={reported} > upper bound={obj_upper:.6f} "
                    f"(construction={construction:.6f}, "
                    f"mean_full_cost_at_LP_min_flow="
                    f"{mean_transport_upper:.6f}; this feasible flow's full "
                    f"linear+quadratic cost upper-bounds the second-stage "
                    f"QP optimum). excess={excess:.4g}, tol={tol_hi:.4g}"
                )
                normalizer = max(abs(obj_upper), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(obj_upper),
                    "raw_excess": float(excess),
                    "normalizer": normalizer,
                    "ratio": float(excess / normalizer),
                })

    # ================================================================
    # Build aggregated violation messages
    # ================================================================
    if 1 in violated_constraints and 1 not in violation_messages:
        violation_messages[1] = [
            f"Flow conservation violated in {flow_viol_count} instance(s) "
            f"across scenarios"
        ]
    elif 1 in violated_constraints:
        violation_messages.setdefault(1, []).append(
            f"Flow conservation violated in {flow_viol_count} instance(s) total"
        )

    if 2 in violated_constraints:
        violation_messages.setdefault(2, []).append(
            f"Edge capacity exceeded in {cap_viol_count} instance(s) "
            f"across scenarios"
        )

    # Flatten messages in constraint order
    violations_list = []
    for c in sorted(violated_constraints):
        for msg in violation_messages.get(c, []):
            violations_list.append(msg)

    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations_list,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if result['violated_constraints']:
        print(f"Violated constraints: {result['violated_constraints']}")


if __name__ == "__main__":
    main()
