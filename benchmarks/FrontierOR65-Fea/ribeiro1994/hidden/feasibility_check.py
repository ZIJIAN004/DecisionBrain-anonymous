"""
Feasibility checker for the Multiple-Depot Vehicle Scheduling Problem (MDVSP).

Checks a candidate solution against the mathematical formulation from
Ribeiro & Soumis (1994).

Hard constraints checked (numbered as in the paper):
  Constraint (1): Trip coverage — each trip j in N covered exactly once.
  Constraint (2): Flow conservation — each route is a valid depot-out,
                   compatible-trip-sequence, depot-in path.
  Constraint (3): Depot capacity — vehicles from depot k <= r_k.
  Constraint (4): Nonnegativity — flow variables >= 0.
  Constraint (5): Integrality — flow variables are integer.
  Constraint (6): Objective consistency — reported objective_value matches
                   the recomputed sum of pull-out, trip-to-trip, and pull-in
                   arc costs across every route.
"""

import json
import argparse
import math
from collections import defaultdict


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    n_trips = instance["parameters"]["n_trips"]
    m_depots = instance["parameters"]["m_depots"]
    vehicle_caps = instance["vehicle_capacities"]
    trips_data = instance["trips"]

    # Build trip lookup: trip_id -> trip dict
    trip_by_id = {t["trip_id"]: t for t in trips_data}

    # Build compatible arc set for quick lookup
    compatible_set = set()
    for arc in instance["compatible_arcs"]:
        compatible_set.add((arc["from_trip"], arc["to_trip"]))

    routes = solution.get("routes", [])

    # ------------------------------------------------------------------
    # Constraint (1): Trip coverage — each trip must be covered exactly once
    # sum_{k=1}^{m} sum_{i in V^k} x^k_{ij} = 1  for all j in N
    # In the path formulation: each trip j appears in exactly one route.
    # ------------------------------------------------------------------
    trip_coverage = defaultdict(int)
    for route in routes:
        for t in route["trips"]:
            trip_coverage[t] += 1

    for j in range(1, n_trips + 1):
        count = trip_coverage.get(j, 0)
        if abs(count - 1) > tol:
            lhs = float(count)
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer

            violated_constraint_set.add(1)
            if count == 0:
                violations.append(f"Constraint (1): Trip {j} is not covered by any route")
            else:
                violations.append(
                    f"Constraint (1): Trip {j} is covered {count} times (expected 1)"
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
    # Constraint (2): Flow conservation — for each depot k and node j,
    # inflow = outflow. In the path formulation this means each route must
    # be a valid path: the vehicle leaves its depot, traverses a sequence
    # of compatible trips, and returns to the same depot.
    #
    # Specifically, consecutive trips (T_i, T_j) in a route must satisfy
    # compatibility: e_i + tau_{ij} <= s_j, i.e., (i, j) must be in the
    # compatible arc set.
    # ------------------------------------------------------------------
    for route_idx, route in enumerate(routes):
        trip_list = route["trips"]
        depot_k = route["depot"]

        # Check depot is valid
        if depot_k < 1 or depot_k > m_depots:
            violated_constraint_set.add(2)
            violations.append(
                f"Constraint (2): Route {route_idx + 1} uses invalid depot {depot_k}"
            )
            lhs = float(depot_k)
            rhs_val = float(m_depots)
            violation_amount = abs(lhs - rhs_val) if depot_k < 1 else max(lhs - rhs_val, 0.0)
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })
            continue

        # Check each consecutive pair in the trip sequence is compatible
        for pos in range(len(trip_list) - 1):
            ti = trip_list[pos]
            tj = trip_list[pos + 1]

            if (ti, tj) not in compatible_set:
                # Compute how much the compatibility constraint is violated
                trip_i = trip_by_id.get(ti)
                trip_j = trip_by_id.get(tj)
                if trip_i is not None and trip_j is not None:
                    # Compatibility requires e_i + tau_{ij} <= s_j
                    # We don't have tau_{ij} directly for incompatible arcs,
                    # but we know the arc is not in the compatible set,
                    # meaning e_i + tau_{ij} > s_j.
                    # For the violation, use the time gap: e_i - s_j
                    # (if positive, it means trip i ends after trip j starts).
                    e_i = trip_i["end_time"]
                    s_j = trip_j["start_time"]
                    # The flow conservation is violated: the arc doesn't exist
                    # in the graph, so the flow variable on a non-existent arc
                    # is being used. LHS (flow on non-arc) = 1, RHS = 0.
                    lhs = 1.0
                    rhs = 0.0
                    violation_amount = abs(lhs - rhs)
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                else:
                    lhs = 1.0
                    rhs = 0.0
                    violation_amount = 1.0
                    normalizer = eps
                    ratio = violation_amount / normalizer

                violated_constraint_set.add(2)
                violations.append(
                    f"Constraint (2): Route {route_idx + 1} has incompatible "
                    f"consecutive trips ({ti} -> {tj})"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

        # Check that all trip IDs in the route are valid
        for t in trip_list:
            if t < 1 or t > n_trips:
                violated_constraint_set.add(2)
                violations.append(
                    f"Constraint (2): Route {route_idx + 1} references invalid trip {t}"
                )
                lhs = float(t)
                rhs_val = float(n_trips)
                violation_amount = max(lhs - rhs_val, 0.0) if t > n_trips else abs(lhs)
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (3): Depot capacity — sum_{j in N} x^k_{n+k,j} <= r_k
    # In the path formulation: number of routes from depot k <= r_k.
    # ------------------------------------------------------------------
    depot_vehicle_count = defaultdict(int)
    for route in routes:
        depot_vehicle_count[route["depot"]] += 1

    for k in range(1, m_depots + 1):
        count = depot_vehicle_count.get(k, 0)
        r_k = vehicle_caps[k - 1]
        lhs = float(count)
        rhs = float(r_k)
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > tol:
            violated_constraint_set.add(3)
            violations.append(
                f"Constraint (3): Depot {k} uses {count} vehicles but capacity is {r_k}"
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

    # ------------------------------------------------------------------
    # Constraint (4): Nonnegativity — x^k_{ij} >= 0
    # In the path formulation, each route implicitly sets x = 1 on its arcs
    # and x = 0 elsewhere, so nonnegativity is automatically satisfied by
    # any route-based solution. We verify route counts are non-negative.
    # ------------------------------------------------------------------
    for route_idx, route in enumerate(routes):
        if len(route["trips"]) < 1:
            violated_constraint_set.add(4)
            lhs = 0.0
            rhs = 1.0
            violation_amount = max(rhs - lhs, 0.0)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint (4): Route {route_idx + 1} has no trips (empty route)"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint (5): Integrality — x^k_{ij} integer
    # In the path formulation each y_p should be 0 or 1. A route-based
    # solution inherently has integer y_p = 1 for selected routes. We check
    # that no trip appears fractionally (already caught by constraint 1).
    # We additionally check that the number of routes is a whole number
    # (which it always is by construction, but we verify).
    # ------------------------------------------------------------------
    total_vehicles = len(routes)
    if total_vehicles != int(total_vehicles):
        violated_constraint_set.add(5)
        lhs = float(total_vehicles)
        rhs = float(round(total_vehicles))
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Constraint (5): Number of vehicles {total_vehicles} is not integer"
        )
        violation_magnitudes.append({
            "constraint": 5,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint (6): Objective consistency — Tier C defence against
    # score-gaming exploits that lie about the reported objective_value.
    #
    # The MDVSP objective is fully determined by the routes: for each
    # route with depot k and trip sequence (t_1, ..., t_m),
    #   cost(route) = c_{n+k, t_1} + sum_p c_{t_p, t_{p+1}} + c_{t_m, n+k}
    # i.e. depot pull-out + trip-to-trip arc costs + depot pull-in.
    # All three cost components are provided in the instance, so we can
    # recompute the true objective exactly and compare to the value
    # reported in the solution. Skipped silently when any arc lookup is
    # missing (e.g. infeasible route — caught by constraints 1/2).
    # ------------------------------------------------------------------
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
    else:
        reported_obj = None

    if reported_obj is not None:
        depot_to_trip_cost = {}
        for entry in instance.get("depot_to_trip_costs", []):
            depot_to_trip_cost[(entry["depot"], entry["trip"])] = entry["cost"]
        trip_to_depot_cost = {}
        for entry in instance.get("trip_to_depot_costs", []):
            trip_to_depot_cost[(entry["trip"], entry["depot"])] = entry["cost"]
        arc_cost = {}
        for arc in instance["compatible_arcs"]:
            arc_cost[(arc["from_trip"], arc["to_trip"])] = arc["cost"]

        true_obj = 0.0
        lookup_failed = False
        for route in routes:
            trip_list = route.get("trips", [])
            depot_k = route.get("depot")
            if not trip_list or depot_k is None:
                lookup_failed = True
                break
            first_trip = trip_list[0]
            last_trip = trip_list[-1]
            pull_out = depot_to_trip_cost.get((depot_k, first_trip))
            pull_in = trip_to_depot_cost.get((last_trip, depot_k))
            if pull_out is None or pull_in is None:
                lookup_failed = True
                break
            true_obj += float(pull_out) + float(pull_in)
            for pos in range(len(trip_list) - 1):
                ti = trip_list[pos]
                tj = trip_list[pos + 1]
                c_ij = arc_cost.get((ti, tj))
                if c_ij is None:
                    lookup_failed = True
                    break
                true_obj += float(c_ij)
            if lookup_failed:
                break

        if not lookup_failed:
            abs_diff = abs(reported_obj - true_obj)
            # Costs are integer-valued (floor of mixed travel/idle terms),
            # so a 0.5 absolute tolerance catches any integer-magnitude
            # mismatch while staying robust to float-from-int round-trips.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                lhs = float(reported_obj)
                rhs = float(true_obj)
                violation_amount = abs_diff
                normalizer = max(abs(rhs), eps)
                violated_constraint_set.add(6)
                violations.append(
                    f"Constraint (6): Objective consistency violated: "
                    f"reported objective_value={reported_obj} differs from "
                    f"recomputed total arc cost={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    feasible = len(violated_constraint_set) == 0
    violated_constraints = sorted(violated_constraint_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MDVSP (Ribeiro & Soumis 1994)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result"
    )
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
