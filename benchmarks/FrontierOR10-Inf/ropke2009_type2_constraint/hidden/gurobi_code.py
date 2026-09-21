#!/usr/bin/env python3
"""
Gurobi MIP implementation of the Pickup and Delivery Problem with Time Windows (PDPTW).

Three-index formulation from:
    Ropke & Cordeau (2009), "Branch and Cut and Price for the Pickup and Delivery
    Problem with Time Windows", Transportation Science 43(3):267-286.

Implements constraints (1)-(12) with big-M linearization for (7)-(8).
"""

import argparse
import json
import math
import time
import sys
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

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy is not installed. Please install Gurobi.")
    sys.exit(1)


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(instance_path):
    """Load a PDPTW instance from JSON."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def solve_pdptw(data, time_limit):
    """
    Solve the PDPTW using the three-index formulation (1)-(12).

    The objective is hierarchical:
      Minimize total routing distance (paper Eq. 1: min sum c_ij * x_ij^k).
    """
    n = data["num_requests"]
    num_vehicles = data["num_vehicles"]
    Q = data["vehicle_capacity"]

    nodes = {node["id"]: node for node in data["nodes"]}
    num_nodes = 2 * n + 2  # 0, 1..n, n+1..2n, 2n+1

    # Node indices:
    # 0 = origin depot
    # 1..n = pickup nodes
    # n+1..2n = delivery nodes
    # 2n+1 = destination depot
    depot_origin = 0
    depot_dest = 2 * n + 1
    P = list(range(1, n + 1))       # pickup nodes
    D = list(range(n + 1, 2 * n + 1))  # delivery nodes
    N = list(range(num_nodes))      # all nodes
    K = list(range(num_vehicles))   # vehicles

    # Extract node data
    x_coord = {}
    y_coord = {}
    demand = {}
    service_time = {}
    tw_a = {}
    tw_b = {}

    for node in data["nodes"]:
        nid = node["id"]
        x_coord[nid] = node["x"]
        y_coord[nid] = node["y"]
        demand[nid] = node["demand"]
        service_time[nid] = node.get("service_time", 0)
        tw_a[nid] = node["time_window"][0]
        tw_b[nid] = node["time_window"][1]

    # Compute cost and travel time matrices
    # Per the paper: t_{ij} includes service time d_i at node i
    # c_{ij} = Euclidean distance
    # For depot arcs, add fixed cost to outgoing arcs from origin depot
    cost = {}
    travel_time = {}
    for i in N:
        for j in N:
            if i == j:
                continue
            dist = euclidean_distance(x_coord[i], y_coord[i], x_coord[j], y_coord[j])
            cost[i, j] = dist
            # Travel time includes service time at node i
            travel_time[i, j] = dist + service_time[i]

    # Arc elimination: remove arcs where time window is infeasible
    # Arc (i,j) is infeasible if a_i + t_{ij} > b_j
    A = []
    for i in N:
        for j in N:
            if i == j:
                continue
            # No arcs from destination depot
            if i == depot_dest:
                continue
            # No arcs to origin depot
            if j == depot_origin:
                continue
            # No arc from delivery i back to its own pickup
            # (self-loop on request pair)
            if i in D and j in P:
                req_i = i - n
                if j == req_i:
                    # This would be n+i -> i, which doesn't exist per paper
                    continue
            # Time window feasibility check
            if tw_a[i] + travel_time[i, j] > tw_b[j]:
                continue
            A.append((i, j))

    arc_set = set(A)

    # Build model
    model = gp.Model("PDPTW")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    # Focus on finding feasible solutions quickly (MIPFocus=1)
    model.setParam("MIPFocus", 1)

    # Decision variables
    # x[i,j,k] binary: vehicle k travels from i to j
    x = {}
    for (i, j) in A:
        for k in K:
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # B[i,k]: time vehicle k begins service at node i
    B = {}
    for i in N:
        for k in K:
            B[i, k] = model.addVar(lb=tw_a[i], ub=tw_b[i], vtype=GRB.CONTINUOUS,
                                    name=f"B_{i}_{k}")

    # Qvar[i,k]: load of vehicle k upon leaving node i
    Qvar = {}
    for i in N:
        for k in K:
            lb_q = max(0, demand[i])
            ub_q = min(Q, Q + demand[i])
            Qvar[i, k] = model.addVar(lb=lb_q, ub=ub_q, vtype=GRB.CONTINUOUS,
                                       name=f"Q_{i}_{k}")

    model.update()

    # Objective (1): minimize total routing cost
    model.setObjective(
        gp.quicksum(cost[i, j] * x[i, j, k] for (i, j) in A for k in K),
        GRB.MINIMIZE
    )

    # Constraint (2): Each request served exactly once
    for i in P:
        model.addConstr(
            gp.quicksum(x[i, j, k] for k in K for j in N
                        if (i, j) in arc_set) == 1,
            name=f"serve_{i}"
        )

    # Constraint (3): Pickup and delivery by same vehicle
    for i in P:
        for k in K:
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in N if (i, j) in arc_set) -
                gp.quicksum(x[n + i, j, k] for j in N if (n + i, j) in arc_set) == 0,
                name=f"pair_{i}_{k}"
            )

    # Constraint (4): Each vehicle leaves origin depot
    for k in K:
        model.addConstr(
            gp.quicksum(x[depot_origin, j, k] for j in N
                        if (depot_origin, j) in arc_set) == 1,
            name=f"leave_depot_{k}"
        )

    # Constraint (5): Flow conservation at pickup and delivery nodes
    for i in (P + D):
        for k in K:
            model.addConstr(
                gp.quicksum(x[j, i, k] for j in N if (j, i) in arc_set) -
                gp.quicksum(x[i, j, k] for j in N if (i, j) in arc_set) == 0,
                name=f"flow_{i}_{k}"
            )

    # Constraint (6): Each vehicle returns to destination depot
    for k in K:
        model.addConstr(
            gp.quicksum(x[i, depot_dest, k] for i in N
                        if (i, depot_dest) in arc_set) == 1,
            name=f"return_depot_{k}"
        )

    # Constraint (7) linearized: B_j >= B_i + t_{ij} - M*(1 - x_{ij}^k)
    # Big-M for time: M_{ij} = max(0, b_i + t_{ij} - a_j)
    for (i, j) in A:
        M_time = max(0, tw_b[i] + travel_time[i, j] - tw_a[j])
        for k in K:
            model.addConstr(
                B[j, k] >= B[i, k] + travel_time[i, j] - M_time * (1 - x[i, j, k]),
                name=f"time_{i}_{j}_{k}"
            )

    # Constraint (8) linearized: Q_j >= Q_i + q_j - M*(1 - x_{ij}^k)
    # Big-M for load: M = Q
    for (i, j) in A:
        for k in K:
            model.addConstr(
                Qvar[j, k] >= Qvar[i, k] + demand[j] - Q * (1 - x[i, j, k]),
                name=f"load_{i}_{j}_{k}"
            )

    # Constraint (9): Precedence - pickup before delivery
    for i in P:
        for k in K:
            model.addConstr(
                B[i, k] + travel_time[i, n + i] <= B[n + i, k],
                name=f"prec_{i}_{k}"
            )

    # Constraints (10) and (11) are handled by variable bounds

    # Symmetry breaking: vehicles are identical, so we can impose ordering
    # INFERRED ASSUMPTION: Not in paper; added to help solver performance.
    # This does not change the feasible solution set (just removes symmetry).
    # We order by the smallest pickup node index served by each vehicle.
    # Implemented as: if vehicle k doesn't serve any request, then vehicle k+1
    # doesn't either.
    for k in range(len(K) - 1):
        model.addConstr(
            gp.quicksum(x[depot_origin, j, k] for j in P + D
                        if (depot_origin, j) in arc_set) >=
            gp.quicksum(x[depot_origin, j, k + 1] for j in P + D
                        if (depot_origin, j) in arc_set),
            name=f"sym_{k}"
        )

    # Provide MIP start: assign each request to its own vehicle
    # This gives a trivial feasible solution (n vehicles, each serving one request)
    for k_idx, req in enumerate(P):
        if k_idx >= num_vehicles:
            break
        pickup = req
        delivery = req + n
        # Set the arcs for this route: depot->pickup->delivery->depot_dest
        if ((depot_origin, pickup) in arc_set and
                (pickup, delivery) in arc_set and
                (delivery, depot_dest) in arc_set):
            x[depot_origin, pickup, k_idx].Start = 1.0
            x[pickup, delivery, k_idx].Start = 1.0
            x[delivery, depot_dest, k_idx].Start = 1.0
    # Unused vehicles go directly depot_origin -> depot_dest
    for k_idx in range(n, num_vehicles):
        if (depot_origin, depot_dest) in arc_set:
            x[depot_origin, depot_dest, k_idx].Start = 1.0

    # Optimize
    model.optimize()

    # Extract solution
    result = {
        "problem": "PDPTW",
        "instance": data.get("instance_name", "unknown"),
        "solver": "Gurobi",
        "formulation": "three-index",
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        result["runtime_seconds"] = model.Runtime

        # Extract routes
        routes = []
        for k in K:
            route = []
            current = depot_origin
            visited = {depot_origin}
            while current != depot_dest:
                found_next = False
                for j in N:
                    if (current, j) in arc_set and j not in visited:
                        if x[current, j, k].X > 0.5:
                            if current != depot_origin or j != depot_dest:
                                route.append({
                                    "from": current,
                                    "to": j,
                                    "time": B[current, k].X if current in [nd["id"] for nd in data["nodes"]] else 0,
                                    "load": Qvar[current, k].X
                                })
                            visited.add(j)
                            current = j
                            found_next = True
                            break
                if not found_next:
                    # Try depot_dest
                    if (current, depot_dest) in arc_set and x[current, depot_dest, k].X > 0.5:
                        route.append({
                            "from": current,
                            "to": depot_dest,
                            "time": B[current, k].X,
                            "load": Qvar[current, k].X
                        })
                        current = depot_dest
                    else:
                        break

            # Only include non-trivial routes (not just depot->depot)
            if len(route) > 1 or (len(route) == 1 and route[0]["to"] != depot_dest):
                route_nodes = [depot_origin]
                cur = depot_origin
                vis = {depot_origin}
                while cur != depot_dest:
                    found = False
                    for j in N:
                        if (cur, j) in arc_set and j not in vis:
                            if x[cur, j, k].X > 0.5:
                                route_nodes.append(j)
                                vis.add(j)
                                cur = j
                                found = True
                                break
                    if not found:
                        if (cur, depot_dest) in arc_set and x[cur, depot_dest, k].X > 0.5:
                            route_nodes.append(depot_dest)
                            cur = depot_dest
                        else:
                            break

                # Only add if route visits at least one customer
                if len(route_nodes) > 2:  # more than just depot_origin -> depot_dest
                    routes.append({
                        "vehicle": k,
                        "nodes": route_nodes,
                    })

        result["num_vehicles_used"] = len(routes)
        result["routes"] = routes
    else:
        result["objective_value"] = None
        result["status"] = "infeasible" if model.Status == GRB.INFEASIBLE else "no_solution"
        result["runtime_seconds"] = model.Runtime

    return result


def main():
    parser = argparse.ArgumentParser(description="PDPTW Gurobi MIP Solver")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    data = load_instance(args.instance_path)

    print(f"Instance: {data.get('instance_name', 'unknown')}")
    print(f"  Requests: {data['num_requests']}")
    print(f"  Vehicles: {data['num_vehicles']}")
    print(f"  Capacity: {data['vehicle_capacity']}")
    print(f"  Time limit: {args.time_limit}s")

    result = solve_pdptw(data, args.time_limit)

    # Save solution
    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSolution saved to: {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Vehicles used: {result.get('num_vehicles_used', 'N/A')}")
        print(f"Status: {result['status']}")
    else:
        print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
