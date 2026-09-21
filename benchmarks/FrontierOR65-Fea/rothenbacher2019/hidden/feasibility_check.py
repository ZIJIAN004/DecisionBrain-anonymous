#!/usr/bin/env python3
"""
Feasibility checker for the Periodic Vehicle Routing Problem with Time Windows
(PVRPTW) with Flexible Schedule Structures.

Based on: Rothenbaecher (2019), Transportation Science.
Mathematical formulation: Route-based Extended Set-Partitioning (Eq. 1a-1f).

Constraints are numbered top-to-bottom from the formulation section in
math_model.txt:
  Constraint 1  (1b): Schedule selection — exactly one schedule per customer
  Constraint 2  (1c): Linking — route schedule parts match selected schedules
  Constraint 3  (1d): Fleet — at most m routes per day
  Constraint 4  (1e): Binary integrality of schedule variables z_n^s
  Constraint 5  (1f): Binary integrality of route variables lambda_r^p

Route feasibility conditions (implicit in set R^p):
  Constraint 6:  Vehicle capacity
  Constraint 7:  Max route duration
  Constraint 8:  Customer time windows
  Constraint 9:  Depot time window
  Constraint 10: Schedule compatibility (visit day/delivery fits offered schedule)
  Constraint 11: Elementary routes (each customer at most once per route)
  Constraint 12: Objective consistency (reported objective_value must equal
                 the sum over all routes of the Euclidean depot-to-customer-...-
                 to-depot tour length; recomputed independently from the
                 reported per-route 'cost' field to defend against fabricated
                 objective values).
"""

import argparse
import json
import math


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # -------------------------------------------------------------------------
    # An empty solution (solver returned infeasible / no incumbent) does NOT
    # satisfy constraint 1 (each customer must receive a schedule). Earlier
    # versions returned feasible=True for an empty solution under a
    # "vacuously true" reading; that is wrong — feasibility_check should
    # strictly verify the constraints against the original solution
    # structure (per the workflow guideline).
    # -------------------------------------------------------------------------
    if solution.get("status") in ("infeasible", "no_solution"):
        return {
            "feasible": False,
            "violated_constraints": [1],
            "violations": [
                "Solver reported infeasible/no_solution — no candidate solution "
                "to verify; constraint 1 (every customer must have a schedule) "
                "cannot be satisfied."
            ],
            "violation_magnitudes": [{
                "constraint": 1, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            }],
        }
    if solution.get("objective_value") is None and not solution.get("routes"):
        return {
            "feasible": False,
            "violated_constraints": [1],
            "violations": [
                "Empty solution (no objective_value and no routes) — constraint "
                "1 (every customer must have a schedule) cannot be satisfied."
            ],
            "violation_magnitudes": [{
                "constraint": 1, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            }],
        }

    # -------------------------------------------------------------------------
    # Preprocessing: build data structures
    # -------------------------------------------------------------------------
    depot = instance["depot"]
    customers = instance["customers"]
    num_days = instance["num_days"]
    num_vehicles = instance["num_vehicles"]
    capacity = instance["vehicle_capacity"]
    max_duration = instance["max_route_duration"]

    cust_by_id = {c["id"]: c for c in customers}

    # Build location list and distance / travel-time functions
    all_locs = [depot] + customers
    loc_by_id = {loc["id"]: loc for loc in all_locs}

    def dist(id1, id2):
        l1, l2 = loc_by_id[id1], loc_by_id[id2]
        return euclidean_distance(l1["x"], l1["y"], l2["x"], l2["y"])

    def travel_time(id1, id2):
        """t_{o1,o2} = distance + service_time at o1."""
        return dist(id1, id2) + loc_by_id[id1]["service_time"]

    depot_id = depot["id"]
    depot_tw = depot["time_window"]

    sol_routes = solution.get("routes", {})
    sol_schedules = solution.get("schedules", {})

    def record(constraint_idx, msg, lhs_val, rhs_val, violation_amount):
        violated_set.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs_val), eps)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # =====================================================================
    # Constraint 1 (1b): Schedule selection — exactly one schedule per customer
    # Σ_{s ∈ S_n} z_n^s = 1  ∀ n ∈ N
    # =====================================================================
    for cust in customers:
        cid = cust["id"]
        cid_str = str(cid)
        if cid_str in sol_schedules:
            count = 1  # one schedule entry per customer in solution
        else:
            count = 0
        rhs = 1.0
        violation_amount = abs(count - rhs)
        if violation_amount > tol:
            record(1,
                   f"Customer {cid}: {count} schedule(s) selected, expected exactly 1",
                   count, rhs, violation_amount)

    # If schedules present, validate schedule_index is within range
    for cid_str, sched_info in sol_schedules.items():
        cid = int(cid_str)
        if cid not in cust_by_id:
            record(1,
                   f"Schedule assigned to unknown customer {cid}",
                   1, 0, 1.0)
            continue
        cust = cust_by_id[cid]
        s_idx = sched_info.get("schedule_index", -1)
        if s_idx < 0 or s_idx >= len(cust["schedules"]):
            record(1,
                   f"Customer {cid}: schedule_index {s_idx} out of range [0, {len(cust['schedules'])-1}]",
                   s_idx, len(cust["schedules"]) - 1, 1.0)

    # =====================================================================
    # Constraint 4 (1e): Binary integrality of z_n^s
    # z_n^s ∈ {0, 1}
    # =====================================================================
    # In the JSON solution, schedules are represented by a single chosen index
    # per customer, so they are inherently binary. No check needed beyond
    # Constraint 1.

    # =====================================================================
    # Constraint 5 (1f): Binary integrality of λ_r^p
    # λ_r^p ∈ {0, 1}
    # In the JSON solution, each route is either listed or not, so inherently
    # binary. No additional check needed.
    # =====================================================================

    # =====================================================================
    # Constraint 3 (1d): Fleet — at most m routes per day
    # Σ_{r ∈ R^p} λ_r^p ≤ m  ∀ p ∈ P
    # =====================================================================
    for day in range(num_days):
        day_str = str(day)
        day_routes = sol_routes.get(day_str, [])
        num_routes = len(day_routes)
        rhs = float(num_vehicles)
        violation_amount = max(0.0, num_routes - rhs)
        if violation_amount > tol:
            record(3,
                   f"Day {day}: {num_routes} routes used, maximum allowed is {num_vehicles}",
                   num_routes, rhs, violation_amount)

    # =====================================================================
    # Route-level feasibility checks (Constraints 6-11)
    # These define which routes belong to R^p.
    # =====================================================================
    for day_str, day_routes in sol_routes.items():
        day = int(day_str)
        for ri, route in enumerate(day_routes):
            route_custs = route.get("customers", [])

            # -----------------------------------------------------------------
            # Constraint 11: Elementary routes — each customer at most once
            # -----------------------------------------------------------------
            cid_counts = {}
            for cid in route_custs:
                cid_counts[cid] = cid_counts.get(cid, 0) + 1
            for cid, cnt in cid_counts.items():
                if cnt > 1:
                    violation_amount = float(cnt - 1)
                    record(11,
                           f"Day {day}, route {ri}: customer {cid} visited {cnt} times (must be at most 1)",
                           cnt, 1.0, violation_amount)

            # -----------------------------------------------------------------
            # Constraint 6: Vehicle capacity
            # Total delivered demand ≤ Q
            # -----------------------------------------------------------------
            # Demand comes from the route's parts or the schedule assignment
            total_demand = 0
            route_parts = route.get("parts", {})
            if route_parts:
                for cid_key, part_info in route_parts.items():
                    total_demand += part_info.get("demand_per_visit", 0)
            else:
                # Fall back: compute from schedule assignments
                for cid in route_custs:
                    cid_s = str(cid)
                    if cid_s in sol_schedules:
                        sched_info = sol_schedules[cid_s]
                        cust = cust_by_id.get(cid)
                        if cust:
                            s_idx = sched_info.get("schedule_index", 0)
                            if 0 <= s_idx < len(cust["schedules"]):
                                sched = cust["schedules"][s_idx]
                                for part in sched["parts"]:
                                    if part["visit_day"] == day:
                                        total_demand += part["demand_per_visit"]

            rhs = float(capacity)
            violation_amount = max(0.0, total_demand - rhs)
            if violation_amount > tol:
                record(6,
                       f"Day {day}, route {ri}: total demand {total_demand} exceeds capacity {capacity}",
                       total_demand, rhs, violation_amount)

            # -----------------------------------------------------------------
            # Constraints 7, 8, 9: Duration, customer TW, depot TW
            # Using Tilk & Irnich (2017) duration resources:
            #   E_time = max(a_j, E_time_prev + t_ij)
            #   E_dur  = max(E_dur_prev + t_ij, E_help_prev + a_j)
            #   E_help = max(E_dur_prev + t_ij - b_j, E_help_prev)
            # -----------------------------------------------------------------
            E_time = depot_tw[0]
            E_dur = 0.0
            E_help = -depot_tw[1]
            prev_id = depot_id

            for cid in route_custs:
                if cid not in cust_by_id:
                    record(8,
                           f"Day {day}, route {ri}: customer {cid} not found in instance",
                           0, 0, 1.0)
                    continue
                cust = cust_by_id[cid]
                tw = cust["time_window"]
                t_ij = travel_time(prev_id, cid)

                E_time_new = max(tw[0], E_time + t_ij)
                E_dur_new = max(E_dur + t_ij, E_help + tw[0])
                E_help_new = max(E_dur + t_ij - tw[1], E_help)

                # Constraint 8: Customer time window
                if E_time_new > tw[1] + tol:
                    violation_amount = E_time_new - tw[1]
                    record(8,
                           f"Day {day}, route {ri}: arrival at customer {cid} at {E_time_new:.4f} exceeds time window [{tw[0]}, {tw[1]}]",
                           E_time_new, tw[1], violation_amount)

                # Constraint 7: Duration check at intermediate step
                if E_dur_new > max_duration + tol:
                    violation_amount = E_dur_new - max_duration
                    record(7,
                           f"Day {day}, route {ri}: route duration {E_dur_new:.4f} exceeds max {max_duration} at customer {cid}",
                           E_dur_new, float(max_duration), violation_amount)

                E_time = E_time_new
                E_dur = E_dur_new
                E_help = E_help_new
                prev_id = cid

            # Return to depot
            if route_custs:
                t_back = travel_time(prev_id, depot_id)
                E_time_final = E_time + t_back
                E_dur_final = max(E_dur + t_back, E_help + depot_tw[0])

                # Constraint 9: Depot time window on return
                if E_time_final > depot_tw[1] + tol:
                    violation_amount = E_time_final - depot_tw[1]
                    record(9,
                           f"Day {day}, route {ri}: return to depot at {E_time_final:.4f} exceeds depot window [{depot_tw[0]}, {depot_tw[1]}]",
                           E_time_final, float(depot_tw[1]), violation_amount)

                # Constraint 7: Final duration
                if E_dur_final > max_duration + tol:
                    violation_amount = E_dur_final - max_duration
                    record(7,
                           f"Day {day}, route {ri}: final route duration {E_dur_final:.4f} exceeds max {max_duration}",
                           E_dur_final, float(max_duration), violation_amount)

            # -----------------------------------------------------------------
            # Constraint 10: Schedule compatibility
            # Visit day and delivery amount must fit at least one offered schedule
            # -----------------------------------------------------------------
            for cid in route_custs:
                if cid not in cust_by_id:
                    continue
                cust = cust_by_id[cid]

                # Determine what this route delivers for this customer
                cid_str_parts = str(cid)
                route_part = route_parts.get(cid_str_parts) or route_parts.get(cid)
                if route_part:
                    visit_day = route_part.get("visit_day", day)
                    days_covered = route_part.get("days_covered")
                    demand_visit = route_part.get("demand_per_visit")
                else:
                    visit_day = day
                    days_covered = None
                    demand_visit = None

                # Check if any schedule of this customer has a matching part
                found_match = False
                for sched in cust["schedules"]:
                    for part in sched["parts"]:
                        if part["visit_day"] != day:
                            continue
                        # If the route specifies days_covered and demand, check them
                        if days_covered is not None and part["days_covered"] != days_covered:
                            continue
                        if demand_visit is not None and abs(part["demand_per_visit"] - demand_visit) > tol:
                            continue
                        found_match = True
                        break
                    if found_match:
                        break

                if not found_match:
                    record(10,
                           f"Day {day}, route {ri}: visit to customer {cid} "
                           f"(days_covered={days_covered}, demand={demand_visit}) "
                           f"does not match any offered schedule part for this day",
                           1, 0, 1.0)

    # =====================================================================
    # Constraint 2 (1c): Linking constraints
    # For each customer n, day p, length l where S_n^{p:l} ≠ ∅:
    #   Σ_{r ∈ R^p} a_{rn}^{p:l} λ_r^p = Σ_{s ∈ S_n} b_s^{p:l} z_n^s
    #
    # LHS: number of times routes on day p visit customer n with part length l
    # RHS: 1 if the selected schedule induces part (p, l), else 0
    # =====================================================================
    for cust in customers:
        cid = cust["id"]
        cid_str = str(cid)

        # Get the selected schedule
        sched_info = sol_schedules.get(cid_str)
        if sched_info is None:
            # Already flagged in Constraint 1
            continue
        s_idx = sched_info.get("schedule_index", 0)
        if s_idx < 0 or s_idx >= len(cust["schedules"]):
            continue
        selected_sched = cust["schedules"][s_idx]

        # For each day p and length l where S_n^{p:l} ≠ ∅
        for day in range(num_days):
            # Gather all (day, l) pairs from any schedule
            lengths_set = set()
            for sched in cust["schedules"]:
                for part in sched["parts"]:
                    if part["visit_day"] == day:
                        lengths_set.add(part["days_covered"])

            for l in lengths_set:
                # RHS: b_s^{p:l} for selected schedule
                rhs_val = 0
                for part in selected_sched["parts"]:
                    if part["visit_day"] == day and part["days_covered"] == l:
                        rhs_val = 1
                        break

                # LHS: count how many routes on day p include schedule part n^{p:l}.
                # A visit only counts toward a_{rn}^{p:l} when the route's parts
                # info explicitly records days_covered == l for this customer;
                # routes that list the customer without an explicit matching
                # part cannot be assumed to cover part (p:l).
                lhs_val = 0
                day_str = str(day)
                for route in sol_routes.get(day_str, []):
                    route_parts = route.get("parts", {})
                    rp = route_parts.get(str(cid)) or route_parts.get(cid)
                    if rp is not None and rp.get("days_covered") == l:
                        lhs_val += 1

                violation_amount = abs(lhs_val - rhs_val)
                if violation_amount > tol:
                    record(2,
                           f"Customer {cid}, day {day}, length {l}: "
                           f"routes cover {lhs_val} time(s) but selected schedule requires {rhs_val}",
                           lhs_val, rhs_val, violation_amount)

    # =====================================================================
    # Constraint 12: Objective consistency (Tier C defense)
    # The reported objective_value must equal Σ_{p,r} c_r λ_r^p, where c_r is
    # the Euclidean route cost (depot → c_1 → ... → c_k → depot).  Per the
    # math model (eq. 1a) and instance fields (distance_matrix_type=
    # 'euclidean', cost_type='euclidean_distance'), this is exactly the sum
    # of pairwise Euclidean distances along each route. We deliberately
    # recompute from the raw (depot, customer) coordinates rather than
    # trusting route['cost'], because a score-gaming candidate could
    # fabricate per-route cost values consistent with a fake total.
    # =====================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for day_str, day_routes in sol_routes.items():
                for route in day_routes:
                    route_custs = route.get("customers", [])
                    if not route_custs:
                        continue
                    prev = depot_id
                    seg = 0.0
                    for cid in route_custs:
                        if cid not in loc_by_id:
                            # Unknown customer is already flagged elsewhere; do
                            # not let it short-circuit obj recomputation.
                            continue
                        seg += dist(prev, cid)
                        prev = cid
                    seg += dist(prev, depot_id)
                    true_obj += seg

            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; the Gurobi
            # reference matches the recompute to ~1e-12, so this is loose
            # enough for floating-point but tight enough to catch any
            # exploit larger than a fractional currency unit.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record(12,
                       f"Objective consistency violated: reported objective_value="
                       f"{reported} differs from recomputed Σ c_r λ_r^p ="
                       f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})",
                       reported, true_obj, abs_diff)

    # =====================================================================
    # Build result
    # =====================================================================
    violated_list = sorted(violated_set)
    feasible = len(violated_list) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_list,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PVRPTW (Rothenbaecher 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    n_violations = len(result["violated_constraints"])
    print(f"Result: {status} ({n_violations} violated constraint type(s))")
    if not result["feasible"]:
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
