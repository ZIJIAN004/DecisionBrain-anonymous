#!/usr/bin/env python3
"""
Gurobi MIP implementation of the PVRPTW (Periodic Vehicle Routing Problem
with Time Windows) with Flexible Schedule Structures.

Based on: Rothenbächer (2019), "Branch-and-Price-and-Cut for the Periodic
Vehicle Routing Problem with Flexible Schedule Structures",
Transportation Science.

Model: Route-based Extended Set-Partitioning Formulation (Equations 1a-1f).

Since the model uses route-based variables (exponentially many), this
implementation enumerates all feasible elementary routes for the given
instance and then solves the MIP directly with Gurobi.

For small instances this is exact; for larger instances the enumeration
may be intractable within the time limit.
"""

import argparse
import json
import math
import time
import itertools
from typing import List, Dict, Tuple, Any, Optional

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass
def load_instance(path: str) -> dict:
    """Load a PVRPTW instance from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def euclidean_distance(x1, y1, x2, y2) -> float:
    """Compute Euclidean distance between two points.

    INFERRED ASSUMPTION: Euclidean distance as cost/distance metric,
    consistent with Cordeau et al. (2001) benchmark instances.
    NOT EXPLICITLY SPECIFIED IN PAPER (Comment 8 in math_model.txt).
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def build_distance_and_time_matrices(instance: dict):
    """
    Build distance/cost matrix and travel time matrix.

    Travel time t_{o1,o2} = Euclidean distance + service time at o1,
    as stated in the paper (Section 3): "t_{o1,o2} includes service time at o1".
    """
    depot = instance["depot"]
    customers = instance["customers"]
    all_locations = [depot] + customers
    n = len(all_locations)

    dist = {}
    travel_time = {}

    for i in range(n):
        loc_i = all_locations[i]
        for j in range(n):
            loc_j = all_locations[j]
            d = euclidean_distance(loc_i["x"], loc_i["y"],
                                   loc_j["x"], loc_j["y"])
            dist[(loc_i["id"], loc_j["id"])] = d
            # Travel time includes service time at origin (paper Section 3)
            travel_time[(loc_i["id"], loc_j["id"])] = d + loc_i["service_time"]

    return dist, travel_time


def enumerate_feasible_routes(instance: dict, dist: dict, travel_time: dict,
                              day: int, time_budget: float = float('inf')
                              ) -> List[dict]:
    """
    Enumerate all feasible elementary routes for a given day.

    Route feasibility (paper Section 3, Route Feasibility Conditions):
    1. Total delivered demand <= Q (vehicle capacity)
    2. Minimum route duration <= D (maximal route duration) - uses Tilk & Irnich (2017) resources
    3. Each customer visited within its time window [a_n, b_n]
    4. Route starts/ends at depot within [a_d, b_d]
    5. Visit day and delivery amount fit at least one offered schedule
    6. Each customer visited at most once per route (elementary)
    """
    start_enum = time.time()
    depot = instance["depot"]
    customers = instance["customers"]
    Q = instance["vehicle_capacity"]
    D = instance["max_route_duration"]
    num_days = instance["num_days"]
    depot_id = depot["id"]
    depot_tw = depot["time_window"]

    # Determine which customers can be visited on this day
    visitable = []
    for cust in customers:
        parts_for_day = []
        for sched in cust["schedules"]:
            for part in sched["parts"]:
                if part["visit_day"] == day:
                    parts_for_day.append(part)
        # Remove duplicate parts
        unique_parts = []
        seen = set()
        for p in parts_for_day:
            key = (p["visit_day"], p["days_covered"])
            if key not in seen:
                seen.add(key)
                unique_parts.append(p)
        if unique_parts:
            visitable.append((cust, unique_parts))

    routes = []

    def try_route(sequence: List[Tuple[dict, dict]]):
        """
        Check if a given sequence of (customer, schedule_part) is feasible.

        Uses the labeling resources from Tilk & Irnich (2017) as described
        in the paper (Section 4.1) for proper minimum route duration computation:
          E_j^time = max(a_{o(j)}, E_i^time + t_ij)
          E_j^dur  = max(E_i^dur + t_ij, E_i^help + a_{o(j)})
          E_j^help = max(E_i^dur + t_ij - b_{o(j)}, E_i^help)
        where t_ij = dist(o(i),o(j)) + service_time(o(i))
        """
        total_demand = sum(part["demand_per_visit"] for _, part in sequence)
        if total_demand > Q:
            return None

        # Forward labeling with duration resources (Tilk & Irnich 2017)
        E_time = depot_tw[0]   # a_d
        E_dur = 0.0
        E_help = -depot_tw[1]  # -b_d
        cost = 0.0
        prev_id = depot_id

        for cust, part in sequence:
            t_ij = travel_time[(prev_id, cust["id"])]
            E_time_new = max(cust["time_window"][0], E_time + t_ij)
            E_dur_new = max(E_dur + t_ij, E_help + cust["time_window"][0])
            E_help_new = max(E_dur + t_ij - cust["time_window"][1], E_help)

            if E_time_new > cust["time_window"][1]:
                return None
            if E_dur_new > D:
                return None

            cost += dist[(prev_id, cust["id"])]
            E_time, E_dur, E_help = E_time_new, E_dur_new, E_help_new
            prev_id = cust["id"]

        # Return to depot
        t_back = travel_time[(prev_id, depot_id)]
        E_time_final = E_time + t_back
        E_dur_final = max(E_dur + t_back, E_help + depot_tw[0])
        cost += dist[(prev_id, depot_id)]

        if E_time_final > depot_tw[1]:
            return None
        if E_dur_final > D:
            return None

        parts_info = {}
        for cust, part in sequence:
            parts_info[cust["id"]] = {
                "visit_day": part["visit_day"],
                "days_covered": part["days_covered"],
                "demand_per_visit": part["demand_per_visit"]
            }

        return {
            "cost": cost,
            "customers": [cust["id"] for cust, _ in sequence],
            "demand": total_demand,
            "duration": E_dur_final,
            "parts": parts_info
        }

    # Enumerate all subsets of visitable customers
    for size in range(1, len(visitable) + 1):
        if time.time() - start_enum > time_budget:
            break
        for subset in itertools.combinations(range(len(visitable)), size):
            if time.time() - start_enum > time_budget:
                break
            part_options = []
            for idx in subset:
                cust, parts = visitable[idx]
                part_options.append([(cust, p) for p in parts])

            for part_combo in itertools.product(*part_options):
                total_demand = sum(p["demand_per_visit"] for _, p in part_combo)
                if total_demand > Q:
                    continue

                # Enumerate all feasible orderings (each yields a distinct route in R^p)
                for perm in itertools.permutations(part_combo):
                    route = try_route(list(perm))
                    if route is not None:
                        route["day"] = day
                        routes.append(route)

    return routes


def solve_pvrptw(instance: dict, time_limit: int) -> dict:
    """Solve the PVRPTW using the route-based MIP formulation (Model 1)."""
    start_time = time.time()

    depot = instance["depot"]
    customers = instance["customers"]
    num_days = instance["num_days"]
    num_vehicles = instance["num_vehicles"]

    dist, travel_time = build_distance_and_time_matrices(instance)

    # Enumerate all feasible routes for each day
    print("Enumerating feasible routes...")
    all_routes = {}
    for day in range(num_days):
        elapsed = time.time() - start_time
        budget = max(1, (time_limit * 0.4) / num_days)
        routes = enumerate_feasible_routes(instance, dist, travel_time, day, budget)
        all_routes[day] = routes
        print(f"  Day {day}: {len(routes)} feasible routes")

    total_routes = sum(len(r) for r in all_routes.values())
    print(f"Total routes enumerated: {total_routes}")

    # Build the MIP model (Equations 1a-1f)
    model = gp.Model("PVRPTW")
    model.setParam("Threads", 1)
    remaining_time = max(1, time_limit - (time.time() - start_time))
    model.setParam("TimeLimit", remaining_time)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # λ_r^p ∈ {0,1}: route r performed on day p (Eq. 1f)
    lam = {}
    for day in range(num_days):
        for r_idx, route in enumerate(all_routes[day]):
            lam[(day, r_idx)] = model.addVar(
                vtype=GRB.BINARY, obj=route["cost"],
                name=f"lam_{day}_{r_idx}"
            )

    # z_n^s ∈ {0,1}: schedule s chosen for customer n (Eq. 1e)
    z = {}
    for cust in customers:
        for s_idx in range(len(cust["schedules"])):
            z[(cust["id"], s_idx)] = model.addVar(
                vtype=GRB.BINARY, obj=0,
                name=f"z_{cust['id']}_{s_idx}"
            )

    model.update()
    model.setAttr("ModelSense", GRB.MINIMIZE)  # (1a)

    # Constraint (1b): Exactly one schedule per customer
    for cust in customers:
        model.addConstr(
            gp.quicksum(z[(cust["id"], s_idx)]
                        for s_idx in range(len(cust["schedules"]))) == 1,
            name=f"schedule_select_{cust['id']}"
        )

    # Constraint (1c): Linking constraints
    # For each customer n, day p, length l where S_n^{p:l} ≠ ∅:
    #   Σ_{r∈R^p} a_{rn}^{p:l} λ_r^p = Σ_{s∈S_n} b_s^{p:l} z_n^s
    for cust in customers:
        n_id = cust["id"]
        for day in range(num_days):
            lengths_set = set()
            for sched in cust["schedules"]:
                for part in sched["parts"]:
                    if part["visit_day"] == day:
                        lengths_set.add(part["days_covered"])

            for l in lengths_set:
                lhs_terms = []
                for r_idx, route in enumerate(all_routes[day]):
                    if n_id in route["parts"]:
                        rp = route["parts"][n_id]
                        if rp["visit_day"] == day and rp["days_covered"] == l:
                            lhs_terms.append(lam[(day, r_idx)])

                rhs_terms = []
                for s_idx, sched in enumerate(cust["schedules"]):
                    for part in sched["parts"]:
                        if part["visit_day"] == day and part["days_covered"] == l:
                            rhs_terms.append(z[(n_id, s_idx)])
                            break

                if lhs_terms or rhs_terms:
                    lhs = gp.quicksum(lhs_terms) if lhs_terms else 0
                    rhs = gp.quicksum(rhs_terms) if rhs_terms else 0
                    model.addConstr(lhs == rhs,
                                    name=f"link_{n_id}_{day}_{l}")

    # Constraint (1d): Fleet constraint — at most m routes per day
    for day in range(num_days):
        model.addConstr(
            gp.quicksum(lam[(day, r_idx)]
                        for r_idx in range(len(all_routes[day]))
                        ) <= num_vehicles,
            name=f"fleet_{day}"
        )

    model.update()
    print(f"\nModel has {model.NumVars} variables and {model.NumConstrs} constraints")
    print("Solving...")

    model.optimize()

    # Extract solution
    solution = {
        "instance_id": instance.get("instance_id", 1),
        "problem_type": "PVRPTW",
        "solver": "Gurobi",
        "model": "Route-based Extended Set-Partitioning (Eq. 1a-1f)",
        "status": None,
        "objective_value": None,
        "solve_time": time.time() - start_time,
        "routes": {},
        "schedules": {}
    }

    if model.Status == GRB.OPTIMAL:
        solution["status"] = "optimal"
        solution["objective_value"] = model.ObjVal
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        solution["status"] = "time_limit_with_solution"
        solution["objective_value"] = model.ObjVal
    elif model.SolCount > 0:
        solution["status"] = "feasible"
        solution["objective_value"] = model.ObjVal
    elif model.Status == GRB.INFEASIBLE:
        solution["status"] = "infeasible"
        solution["objective_value"] = None
        print("\nModel is infeasible. The instance cannot be solved with the "
              "given number of vehicles and constraints.")
        return solution
    else:
        solution["status"] = "no_solution"
        solution["objective_value"] = None
        return solution

    # Extract chosen routes
    for day in range(num_days):
        day_routes = []
        for r_idx, route in enumerate(all_routes[day]):
            if lam[(day, r_idx)].X > 0.5:
                day_routes.append({
                    "customers": route["customers"],
                    "cost": route["cost"],
                    "demand": route["demand"],
                    "parts": {str(k): v for k, v in route["parts"].items()}
                })
        if day_routes:
            solution["routes"][str(day)] = day_routes

    # Extract chosen schedules
    for cust in customers:
        for s_idx, sched in enumerate(cust["schedules"]):
            if z[(cust["id"], s_idx)].X > 0.5:
                solution["schedules"][str(cust["id"])] = {
                    "schedule_index": s_idx,
                    "days": sched["days"],
                    "visit_frequency": sched["visit_frequency"]
                }
                break

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for PVRPTW with Flexible Schedule Structures"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    instance = load_instance(args.instance_path)

    print(f"Instance: {instance.get('num_customers', '?')} customers, "
          f"{instance.get('num_days', '?')} days, "
          f"{instance.get('num_vehicles', '?')} vehicles")
    print(f"Time limit: {args.time_limit}s")

    solution = solve_pvrptw(instance, args.time_limit)

    print(f"\nStatus: {solution['status']}")
    print(f"Objective value: {solution['objective_value']}")
    print(f"Solve time: {solution['solve_time']:.2f}s")

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)
    print(f"Solution saved to: {args.solution_path}")


if __name__ == "__main__":
    main()
