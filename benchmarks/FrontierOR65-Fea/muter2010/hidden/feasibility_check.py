#!/usr/bin/env python3
"""
Feasibility checker for VRPTW solutions modeled as a Set Covering Problem (SCP).

Based on: Muter, Birbil, Sahin (2010) - VRPTW reformulated as SCP.

Hard constraints checked (numbered top-to-bottom from the mathematical formulation):
  Constraint 1: Coverage — each customer must be covered by at least one selected route
                 sum_{p in P} a_{ip} * y_p >= 1,  for all i in C
  Constraint 2: Integrality — y_p in {0, 1}  (binary route selection)
  Constraint 3: Capacity — total demand on each route <= vehicle capacity
  Constraint 4: Time windows — vehicle arrives at each customer <= customer's due_date
  Constraint 5: Depot return — vehicle returns to depot <= depot's due_date
  Constraint 6: Variable domain — every customer id in any route must be valid
  Constraint 7: Objective consistency — reported objective_value must match
                recomputed total Euclidean travel distance over all routes
                (Tier C defense against score-gaming exploits)
"""

import argparse
import json
import math


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    depot = data["depot"]
    customers = {c["id"]: c for c in data["customers"]}
    capacity = data["vehicle_capacity"]
    # Build node lookup (depot + customers)
    nodes = {depot["id"]: depot}
    nodes.update(customers)
    return data, depot, customers, capacity, nodes


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_routes(solution):
    """Extract list of customer-id lists from either solution format."""
    routes = []
    for r in solution["routes"]:
        if "customers" in r:
            # efficient_algorithm format
            routes.append(list(r["customers"]))
        elif "customer_ids" in r:
            # gurobi format
            routes.append(list(r["customer_ids"]))
        else:
            routes.append([])
    return routes


def build_dist_lookup(nodes):
    """Return a function that computes distance between two node ids."""
    cache = {}

    def dist(i, j):
        key = (i, j)
        if key not in cache:
            cache[key] = euclidean_distance(
                nodes[i]["x"], nodes[i]["y"],
                nodes[j]["x"], nodes[j]["y"],
            )
        return cache[key]

    return dist


def check_feasibility(instance_data, solution_data):
    tol = 1e-5
    eps = 1e-5

    depot = instance_data["depot"]
    depot_id = depot["id"]
    customers = {c["id"]: c for c in instance_data["customers"]}
    capacity = instance_data["vehicle_capacity"]
    customer_ids = set(customers.keys())

    nodes = {depot_id: depot}
    nodes.update(customers)
    dist = build_dist_lookup(nodes)

    routes = extract_routes(solution_data)

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # ------------------------------------------------------------------
    # Constraint 6: Variable domain — every customer id in any route must
    # be a valid customer id in the instance.
    # ------------------------------------------------------------------
    for r_idx, route in enumerate(routes):
        for cid in route:
            if cid not in customer_ids:
                violated_constraint_set.add(6)
                violations.append(
                    f"Route {r_idx} contains invalid customer id {cid!r} "
                    f"(not in instance customers)"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": float(cid) if isinstance(cid, (int, float)) else 0.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": 1.0,
                    "ratio": 1.0,
                })

    # ------------------------------------------------------------------
    # Constraint 1: Coverage  (sum a_{ip} y_p >= 1 for each customer i)
    # ------------------------------------------------------------------
    covered = {}  # customer_id -> count
    for route in routes:
        for cid in route:
            covered[cid] = covered.get(cid, 0) + 1

    uncovered = customer_ids - set(covered.keys())
    if uncovered:
        violated_constraint_set.add(1)
        for cid in sorted(uncovered):
            lhs = 0.0
            rhs = 1.0
            violation_amount = rhs - lhs  # 1.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(f"Customer {cid} is not covered by any route")
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 2: Integrality  (y_p in {0, 1})
    # The solution is expressed as a set of selected routes, so y_p is
    # implicitly 1 for each listed route and 0 otherwise.  Always satisfied.
    # ------------------------------------------------------------------
    # (No violation possible given the solution format.)

    # ------------------------------------------------------------------
    # Constraint 3: Capacity  (total demand on route <= vehicle capacity)
    # For each route p: sum_{i in route_p} demand_i <= capacity
    # ------------------------------------------------------------------
    for r_idx, route in enumerate(routes):
        total_demand = sum(nodes[cid]["demand"] for cid in route)
        lhs = total_demand
        rhs = capacity
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > tol:
            violated_constraint_set.add(3)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Capacity exceeded on route {r_idx}: demand {total_demand} > capacity {capacity}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 4: Time windows  (arrival_i <= due_date_i for each customer)
    # Constraint 5: Depot return  (return time <= depot due_date)
    # ------------------------------------------------------------------
    for r_idx, route in enumerate(routes):
        if not route:
            continue
        t = 0.0  # depart depot at time 0
        prev = depot_id
        for cid in route:
            travel = dist(prev, cid)
            arrival = t + travel
            due = nodes[cid]["due_date"]
            # Constraint 4: arrival <= due_date
            lhs = arrival
            rhs = due
            violation_amount = max(lhs - rhs, 0.0)
            if violation_amount > tol:
                violated_constraint_set.add(4)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Time window violated for customer {cid} on route {r_idx}: "
                    f"arrival {arrival:.4f} > due_date {due}"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })
            # Wait if early
            t = max(arrival, nodes[cid]["ready_time"])
            t += nodes[cid]["service_time"]
            prev = cid

        # Constraint 5: return to depot <= depot due_date
        return_time = t + dist(prev, depot_id)
        lhs = return_time
        rhs = depot["due_date"]
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > tol:
            violated_constraint_set.add(5)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Depot return time violated on route {r_idx}: "
                f"return {return_time:.4f} > depot due_date {rhs}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 7: Objective consistency — recompute the total Euclidean
    # travel distance from the routes and reject if the reported value
    # disagrees beyond tolerance. Tier C defense against score-gaming
    # exploits that lie about objective_value while keeping the routes
    # constraint-feasible.
    #
    # The VRPTW objective is sum over arcs (i,j) selected with
    #   x_{ijk} = 1 of c_{ij} = euclidean(i, j). For each route, the
    #   arcs are depot -> r[0] -> r[1] -> ... -> r[-1] -> depot.
    # ------------------------------------------------------------------
    reported_obj = solution_data.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for route in routes:
                if not route:
                    continue
                prev = depot_id
                for cid in route:
                    if cid not in nodes:
                        # invalid customer id; constraint 6 will catch this.
                        # Skip the arc so distance lookup doesn't crash.
                        continue
                    true_obj += dist(prev, cid)
                    prev = cid
                true_obj += dist(prev, depot_id)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraint_set.add(7)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total Euclidean travel "
                    f"distance={true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    feasible = len(violated_constraint_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraint_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VRPTW / SCP solutions."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    instance_data, _, _, _, _ = load_instance(args.instance_path)
    solution_data = load_solution(args.solution_path)

    result = check_feasibility(instance_data, solution_data)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"{status}: {len(result['violated_constraints'])} constraint type(s) violated, "
          f"{len(result['violation_magnitudes'])} violation(s) total.")


if __name__ == "__main__":
    main()
