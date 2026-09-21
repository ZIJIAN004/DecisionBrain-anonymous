"""
Feasibility checker for the Resource Loading Problem (RLP).

Checks candidate solutions against the Execution-Interval formulation
(Section 4.1) from Song, Kis & Leus (2020).

Hard constraints checked (numbered top-to-bottom in the EI formulation):
  1. Constraint (15): Intensity bounds linking y and a variables
  2. Constraint (16): Exactly one execution interval per order
  3. Constraint (17): Binary domain for a variables (interval feasibility)
  4. Constraint (6):  Total intensity per order equals 1
  5. Constraint (7):  Nonregular capacity z_t >= sum_j p_j*y_jt - C_t
  6. Constraint (8):  Non-negativity of intensity y_jt >= 0
  7. Constraint (9):  Non-negativity of nonregular capacity z_t >= 0
  8. Objective consistency (Eq.14): reported objective_value must equal
     sum_j w_j * max(0, l_chosen_j - d_j) + sigma * sum_t z_t
"""

import json
import argparse
import math


def enumerate_execution_intervals(order, H):
    """
    Enumerate all feasible execution intervals for an order.
    (k, l) in E_j iff:
      r_j <= k <= l <= H,
      (l - k + 1) * LB_j <= 1,
      (l - k + 1) * UB_j >= 1.
    """
    r_j = order['release_date_r']
    LB_j = order['intensity_lower_bound_LB']
    UB_j = order['intensity_upper_bound_UB']

    intervals = []
    for k in range(r_j, H + 1):
        for l in range(k, H + 1):
            length = l - k + 1
            if length * LB_j <= 1.0 + 1e-9 and length * UB_j >= 1.0 - 1e-9:
                intervals.append((k, l))
    return intervals


def check_feasibility(instance, solution):
    """Check all hard constraints of the EI formulation."""

    tol = 1e-5
    eps = 1e-5

    H = instance['planning_horizon_H']
    capacities = instance['capacities']
    sigma = instance['unit_cost_nonregular_capacity_sigma']
    orders_data = {o['order_id']: o for o in instance['orders']}
    orders_sol = solution.get('orders_solution', [])
    nonregular_by_period = solution.get('nonregular_capacity_by_period', {})

    violations = []
    violation_magnitudes = []

    # Build solution lookup
    sol_by_id = {}
    for os in orders_sol:
        sol_by_id[os['order_id']] = os

    # Precompute feasible intervals for each order
    feasible_intervals = {}
    for order in instance['orders']:
        j = order['order_id']
        feasible_intervals[j] = enumerate_execution_intervals(order, H)

    # =========================================================================
    # Constraint 1 — (15): Intensity bounds linking y and a
    #   LB_j * sum_{(k,l): k<=t<=l} a^j_{kl} <= y_{jt}
    #                                             <= UB_j * sum_{(k,l): k<=t<=l} a^j_{kl}
    #
    # In the integer solution, exactly one a^j_{kl}=1 for the chosen interval.
    # So for t in [k_chosen, l_chosen], the bounds become LB_j <= y_jt <= UB_j.
    # For t outside the chosen interval, y_jt must be 0.
    # =========================================================================
    for order in instance['orders']:
        j = order['order_id']
        LB_j = order['intensity_lower_bound_LB']
        UB_j = order['intensity_upper_bound_UB']
        r_j = order['release_date_r']

        if j not in sol_by_id:
            continue

        osol = sol_by_id[j]
        interval = osol.get('chosen_interval')
        if interval is None:
            continue

        k_chosen, l_chosen = interval

        intensities = osol.get('intensities', {})

        for t in range(r_j, H + 1):
            y_jt = float(intensities.get(str(t), 0.0))
            in_interval = (k_chosen <= t <= l_chosen)

            if in_interval:
                # sum of covering a's = 1, so bounds are LB_j and UB_j
                # Check lower bound: LB_j <= y_jt
                lb_violation = LB_j - y_jt
                if lb_violation > tol:
                    rhs_val = LB_j
                    lhs_val = y_jt
                    raw_excess = lb_violation
                    normalizer = max(abs(rhs_val), eps)
                    ratio = raw_excess / normalizer
                    violations.append(
                        f"Constraint 1 (Eq.15 lower): Order {j}, period {t}: "
                        f"y_jt={y_jt:.8f} < LB_j={LB_j:.8f}"
                    )
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": raw_excess,
                        "normalizer": normalizer,
                        "ratio": ratio
                    })

                # Check upper bound: y_jt <= UB_j
                ub_violation = y_jt - UB_j
                if ub_violation > tol:
                    lhs_val = y_jt
                    rhs_val = UB_j
                    raw_excess = ub_violation
                    normalizer = max(abs(rhs_val), eps)
                    ratio = raw_excess / normalizer
                    violations.append(
                        f"Constraint 1 (Eq.15 upper): Order {j}, period {t}: "
                        f"y_jt={y_jt:.8f} > UB_j={UB_j:.8f}"
                    )
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": raw_excess,
                        "normalizer": normalizer,
                        "ratio": ratio
                    })
            else:
                # Outside interval: y_jt must be 0 (sum of covering a's = 0)
                if y_jt > tol:
                    lhs_val = y_jt
                    rhs_val = 0.0
                    raw_excess = y_jt
                    normalizer = max(abs(rhs_val), eps)
                    ratio = raw_excess / normalizer
                    violations.append(
                        f"Constraint 1 (Eq.15): Order {j}, period {t}: "
                        f"y_jt={y_jt:.8f} > 0 but period is outside chosen interval "
                        f"[{k_chosen},{l_chosen}]"
                    )
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": raw_excess,
                        "normalizer": normalizer,
                        "ratio": ratio
                    })

    # =========================================================================
    # Constraint 2 — (16): Exactly one execution interval per order
    #   sum_{(k,l) in E_j} a^j_{kl} = 1
    #
    # In the solution, each order should have exactly one chosen_interval.
    # =========================================================================
    for order in instance['orders']:
        j = order['order_id']
        if j not in sol_by_id:
            lhs_val = 0.0
            rhs_val = 1.0
            raw_excess = 1.0
            normalizer = max(abs(rhs_val), eps)
            ratio = raw_excess / normalizer
            violations.append(
                f"Constraint 2 (Eq.16): Order {j} has no chosen interval"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio
            })
        else:
            osol = sol_by_id[j]
            if osol.get('chosen_interval') is None:
                lhs_val = 0.0
                rhs_val = 1.0
                raw_excess = 1.0
                normalizer = max(abs(rhs_val), eps)
                ratio = raw_excess / normalizer
                violations.append(
                    f"Constraint 2 (Eq.16): Order {j} has null chosen interval"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": ratio
                })

    # =========================================================================
    # Constraint 3 — (17): Binary domain / interval feasibility
    #   a^j_{kl} in {0,1}, and (k,l) must be in E_j
    #
    # Check that the chosen interval is a valid member of E_j.
    # =========================================================================
    for order in instance['orders']:
        j = order['order_id']
        if j not in sol_by_id:
            continue
        osol = sol_by_id[j]
        interval = osol.get('chosen_interval')
        if interval is None:
            continue

        k_chosen, l_chosen = interval
        valid_intervals = feasible_intervals[j]

        if (k_chosen, l_chosen) not in valid_intervals:
            # Compute how far off the interval is from feasibility
            r_j = order['release_date_r']
            LB_j = order['intensity_lower_bound_LB']
            UB_j = order['intensity_upper_bound_UB']
            length = l_chosen - k_chosen + 1

            detail_parts = []
            if k_chosen < r_j:
                detail_parts.append(f"k={k_chosen} < r_j={r_j}")
            if l_chosen > H:
                detail_parts.append(f"l={l_chosen} > H={H}")
            if length * LB_j > 1.0 + 1e-9:
                detail_parts.append(
                    f"length*LB={length * LB_j:.6f} > 1"
                )
            if length * UB_j < 1.0 - 1e-9:
                detail_parts.append(
                    f"length*UB={length * UB_j:.6f} < 1"
                )

            detail = "; ".join(detail_parts) if detail_parts else "unknown reason"

            # Violation amount: measure how much interval length conditions are off
            viol_lb = max(0.0, length * LB_j - 1.0)
            viol_ub = max(0.0, 1.0 - length * UB_j)
            viol_release = max(0.0, r_j - k_chosen)
            viol_horizon = max(0.0, l_chosen - H)
            raw_excess = max(viol_lb, viol_ub, float(viol_release), float(viol_horizon))
            if raw_excess < tol:
                raw_excess = 1.0  # flag as violated anyway
            normalizer = max(1.0, eps)
            ratio = raw_excess / normalizer

            violations.append(
                f"Constraint 3 (Eq.17): Order {j}, interval [{k_chosen},{l_chosen}] "
                f"is not a feasible execution interval ({detail})"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": 0.0,
                "rhs": 0.0,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # =========================================================================
    # Constraint 4 — (6): Total intensity equals 1
    #   sum_{t=r_j}^{H} y_{jt} = 1 for all j
    # =========================================================================
    for order in instance['orders']:
        j = order['order_id']
        r_j = order['release_date_r']
        if j not in sol_by_id:
            continue
        osol = sol_by_id[j]
        intensities = osol.get('intensities', {})

        total_intensity = sum(float(v) for v in intensities.values())

        raw_excess = abs(total_intensity - 1.0)
        if raw_excess > tol:
            lhs_val = total_intensity
            rhs_val = 1.0
            normalizer = max(abs(rhs_val), eps)
            ratio = raw_excess / normalizer
            violations.append(
                f"Constraint 4 (Eq.6): Order {j}: sum of intensities = "
                f"{total_intensity:.8f} != 1.0"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # =========================================================================
    # Constraint 5 — (7): Nonregular capacity
    #   z_t >= sum_{j} p_j * y_{jt} - C_t for t = 1,...,H
    #
    # The solution reports z_t via nonregular_capacity_by_period.
    # z_t must be >= max(0, sum_j p_j*y_jt - C_t).
    # =========================================================================
    for t in range(1, H + 1):
        C_t = capacities[t - 1]

        # Compute resource usage in period t
        resource_usage = 0.0
        for order in instance['orders']:
            j = order['order_id']
            p_j = order['work_content_p']
            if j not in sol_by_id:
                continue
            osol = sol_by_id[j]
            intensities = osol.get('intensities', {})
            y_jt = float(intensities.get(str(t), 0.0))
            resource_usage += p_j * y_jt

        required_z = resource_usage - C_t  # z_t >= this value

        z_t = float(nonregular_by_period.get(str(t), 0.0))

        # z_t >= resource_usage - C_t
        lhs_val = z_t
        rhs_val = required_z
        violation_amount = required_z - z_t  # how much RHS exceeds LHS
        if violation_amount > tol:
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 5 (Eq.7): Period {t}: z_t={z_t:.8f} < "
                f"resource_usage - C_t = {resource_usage:.8f} - {C_t} = "
                f"{required_z:.8f}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # =========================================================================
    # Constraint 6 — (8): Non-negativity of intensity y_jt >= 0
    # =========================================================================
    for order in instance['orders']:
        j = order['order_id']
        if j not in sol_by_id:
            continue
        osol = sol_by_id[j]
        intensities = osol.get('intensities', {})

        for t_str, y_val in intensities.items():
            y_jt = float(y_val)
            if -y_jt > tol:  # y_jt < -tol
                lhs_val = y_jt
                rhs_val = 0.0
                raw_excess = -y_jt
                normalizer = max(abs(rhs_val), eps)
                ratio = raw_excess / normalizer
                violations.append(
                    f"Constraint 6 (Eq.8): Order {j}, period {t_str}: "
                    f"y_jt={y_jt:.8f} < 0"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": ratio
                })

    # =========================================================================
    # Constraint 7 — (9): Non-negativity of nonregular capacity z_t >= 0
    # =========================================================================
    for t in range(1, H + 1):
        z_t = float(nonregular_by_period.get(str(t), 0.0))
        if -z_t > tol:  # z_t < -tol
            lhs_val = z_t
            rhs_val = 0.0
            raw_excess = -z_t
            normalizer = max(abs(rhs_val), eps)
            ratio = raw_excess / normalizer
            violations.append(
                f"Constraint 7 (Eq.9): Period {t}: z_t={z_t:.8f} < 0"
            )
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # =========================================================================
    # Constraint 8 — (14): Objective consistency (Tier C anti-exploit check)
    #   reported objective_value must equal
    #     sum_j w_j * max(0, l_chosen_j - d_j) + sigma * sum_t z_t
    #
    # All variables that determine the EI-formulation objective (the chosen
    # interval per order, which fixes T^j_{kl}=max(0,l-d_j), and z_t) are
    # present in the solution, so a full recompute is exact up to FP noise.
    # =========================================================================
    reported_obj_raw = solution.get('objective_value')
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj):
            tardiness_cost = 0.0
            for order in instance['orders']:
                j = order['order_id']
                d_j = order['due_date_d']
                w_j = order['tardiness_cost_w']
                if j not in sol_by_id:
                    continue
                interval = sol_by_id[j].get('chosen_interval')
                if interval is None:
                    continue
                _, l_chosen = interval
                T_j = max(0, int(l_chosen) - int(d_j))
                tardiness_cost += float(w_j) * float(T_j)

            z_sum = 0.0
            for v in nonregular_by_period.values():
                try:
                    z_sum += float(v)
                except (TypeError, ValueError):
                    pass

            true_obj = tardiness_cost + float(sigma) * z_sum

            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                lhs_val = reported_obj
                rhs_val = true_obj
                raw_excess = abs_diff
                normalizer = max(abs(rhs_val), eps)
                ratio = raw_excess / normalizer
                violations.append(
                    f"Constraint 8 (Eq.14): Objective consistency violated: "
                    f"reported objective_value={reported_obj} differs from "
                    f"recomputed sum_j w_j*max(0,l-d_j) + sigma*sum_t z_t="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": ratio
                })
        else:
            # Non-finite or unparseable reported objective is itself an
            # exploit signal (e.g. NaN, inf, sys.float_info.max sometimes
            # parses as a finite huge number but inf does not).
            lhs_val = float('nan') if reported_obj is None else reported_obj
            rhs_val = 0.0
            raw_excess = 1.0
            normalizer = max(abs(rhs_val), eps)
            ratio = raw_excess / normalizer
            violations.append(
                f"Constraint 8 (Eq.14): Objective consistency violated: "
                f"reported objective_value={reported_obj_raw} is not a finite number"
            )
            violation_magnitudes.append({
                "constraint": 8,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # Build result
    violated_constraints = sorted(set(vm['constraint'] for vm in violation_magnitudes))
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Resource Loading Problem (RLP) "
                    "using the Execution-Interval formulation."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    with open(args.solution_path, 'r') as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['feasible']:
        print(f"FEASIBLE: No constraint violations detected.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
