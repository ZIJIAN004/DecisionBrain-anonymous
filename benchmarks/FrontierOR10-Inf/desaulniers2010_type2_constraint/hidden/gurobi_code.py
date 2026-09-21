"""
Gurobi implementation of the arc-flow formulation for the Split-Delivery
Vehicle Routing Problem with Time Windows (SDVRPTW).

Based on: Desaulniers (2010), "Branch-and-Price-and-Cut for the Split-Delivery
Vehicle Routing Problem with Time Windows", Operations Research 58(1):179-192.

Implements equations (1)-(15) from the paper.
"""

import argparse
import json
import math
import time
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


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n = data["num_customers"]
    Q = data["vehicle_capacity"]
    depot = data["depot"]
    customers = data["customers"]
    dist = data["distance_matrix"]  # (n+1) x (n+1), indices 0..n for depot + customers

    # Node set: 0 = depot start, 1..n = customers, n+1 = depot end
    # The distance matrix is (n+1) x (n+1) with rows/cols for depot (0) and customers (1..n).
    # We need to add node n+1 (depot end) with same location as depot.

    # Build full distance/time matrix for nodes 0..n+1
    num_nodes = n + 2  # 0, 1, ..., n, n+1
    t_mat = [[0.0] * num_nodes for _ in range(num_nodes)]
    c_mat = [[0.0] * num_nodes for _ in range(num_nodes)]

    # Travel time includes service time at i (standard Solomon convention)
    service = [0] * num_nodes
    service[0] = depot["service_time"]
    for cust in customers:
        service[cust["id"]] = cust["service_time"]
    service[n + 1] = 0

    # Time windows
    e = [0] * num_nodes
    l = [0] * num_nodes
    e[0] = depot["time_window"][0]
    l[0] = depot["time_window"][1]
    e[n + 1] = depot["time_window"][0]
    l[n + 1] = depot["time_window"][1]
    for cust in customers:
        cid = cust["id"]
        e[cid] = cust["time_window"][0]
        l[cid] = cust["time_window"][1]

    # Demands
    demand = [0] * num_nodes
    for cust in customers:
        demand[cust["id"]] = cust["demand"]
    d_bar = [min(demand[i], Q) for i in range(num_nodes)]

    # Fill distance/cost matrix
    # dist matrix from JSON is (n+1) x (n+1) with index 0=depot, 1..n=customers
    for i in range(n + 1):
        for j in range(n + 1):
            c_mat[i][j] = dist[i][j]
            # t_{ij} = distance + service time at i (Solomon convention)
            t_mat[i][j] = dist[i][j] + service[i]

    # Node n+1 has same distances as depot (node 0)
    for i in range(n + 1):
        c_mat[i][n + 1] = dist[i][0]
        t_mat[i][n + 1] = dist[i][0] + service[i]
        c_mat[n + 1][i] = dist[0][i]
        t_mat[n + 1][i] = dist[0][i] + service[n + 1]
    c_mat[n + 1][n + 1] = 0.0
    t_mat[n + 1][n + 1] = 0.0

    # Build arc set A
    # Arc (i,j) exists if e_i + t_{ij} <= l_j, i != j
    # Include (0, n+1) but not (n+1, 0)
    N = list(range(1, n + 1))  # customer nodes
    V = list(range(0, n + 2))  # all nodes

    arcs = []
    for i in V:
        for j in V:
            if i == j:
                continue
            if i == n + 1 and j == 0:
                continue  # no arc (n+1, 0)
            if j == 0:
                continue  # no arcs into depot start
            if i == n + 1:
                continue  # no arcs from depot end (except handled above)
            if e[i] + t_mat[i][j] <= l[j]:
                arcs.append((i, j))

    arc_set = set(arcs)

    # Successor and predecessor sets
    V_plus = {i: [] for i in V}
    V_minus = {i: [] for i in V}
    for (i, j) in arcs:
        V_plus[i].append(j)
        V_minus[j].append(i)

    # Upper bound on number of vehicles
    F_size = sum(math.ceil(demand[i] / Q) for i in N)
    F = list(range(F_size))

    # k^C_i = ceil(d_i / Q)
    kC = {i: math.ceil(demand[i] / Q) for i in N}
    # k^C(N) = ceil(sum d_i / Q)
    kC_N = math.ceil(sum(demand[i] for i in N) / Q)

    print(f"Instance: {n} customers, {Q} capacity, {F_size} vehicles (upper bound)")
    print(f"Arcs: {len(arcs)}, kC(N): {kC_N}")

    # Build Gurobi model
    model = gp.Model("SDVRPTW")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[f,i,j] binary: vehicle f uses arc (i,j)
    x = {}
    for f in F:
        for (i, j) in arcs:
            x[f, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{f}_{i}_{j}")

    # delta[f,i] >= 0: quantity delivered by vehicle f to customer i
    delta = {}
    for f in F:
        for i in N:
            delta[f, i] = model.addVar(lb=0.0, ub=d_bar[i], vtype=GRB.CONTINUOUS,
                                        name=f"delta_{f}_{i}")

    # s[f,i]: visit start time at node i for vehicle f
    s = {}
    for f in F:
        for i in V:
            s[f, i] = model.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS,
                                    name=f"s_{f}_{i}")

    # H: total number of vehicles used
    H = model.addVar(lb=kC_N, ub=F_size, vtype=GRB.INTEGER, name="H")

    model.update()

    # Objective (1): minimize total travel cost
    model.setObjective(
        gp.quicksum(c_mat[i][j] * x[f, i, j] for f in F for (i, j) in arcs),
        GRB.MINIMIZE
    )

    # Constraint (2): Demand fulfillment
    for i in N:
        model.addConstr(
            gp.quicksum(delta[f, i] for f in F) >= demand[i],
            name=f"demand_{i}"
        )

    # Constraint (3): Minimum number of visits per customer (only when k^C_i >= 2)
    for i in N:
        if kC[i] >= 2:
            model.addConstr(
                gp.quicksum(x[f, i, j] for f in F for j in V_plus[i]) >= kC[i],
                name=f"min_visits_{i}"
            )

    # Constraint (4): Fleet size definition
    model.addConstr(
        gp.quicksum(x[f, 0, j] for f in F for j in V_plus[0] if j != n + 1) == H,
        name="fleet_size"
    )

    # Constraint (5): H bounds already set via variable bounds

    # Constraints (8)-(10): Path structure per vehicle
    for f in F:
        # (8): each vehicle leaves depot exactly once
        model.addConstr(
            gp.quicksum(x[f, 0, j] for j in V_plus[0]) == 1,
            name=f"depart_{f}"
        )
        # (9): flow conservation at each customer
        for i in N:
            model.addConstr(
                gp.quicksum(x[f, i, j] for j in V_plus[i] if (i, j) in arc_set)
                - gp.quicksum(x[f, j, i] for j in V_minus[i] if (j, i) in arc_set) == 0,
                name=f"flow_{f}_{i}"
            )
        # (10): each vehicle arrives at depot end exactly once
        model.addConstr(
            gp.quicksum(x[f, i, n + 1] for i in V_minus[n + 1] if (i, n + 1) in arc_set) == 1,
            name=f"arrive_{f}"
        )

    # Constraint (11) linearized: s_i + t_{ij} - s_j <= M*(1 - x_{ij})
    # Big-M: M = l_i + t_{ij} - e_j
    for f in F:
        for (i, j) in arcs:
            M_val = l[i] + t_mat[i][j] - e[j]
            model.addConstr(
                s[f, i] + t_mat[i][j] - s[f, j] <= M_val * (1 - x[f, i, j]),
                name=f"time_{f}_{i}_{j}"
            )

    # Constraint (12): Time windows already handled by variable bounds

    # Constraint (13): Vehicle capacity
    for f in F:
        model.addConstr(
            gp.quicksum(delta[f, i] for i in N) <= Q,
            name=f"capacity_{f}"
        )

    # Constraint (14): Delivery linking
    for f in F:
        for i in N:
            model.addConstr(
                delta[f, i] <= d_bar[i] * gp.quicksum(
                    x[f, i, j] for j in V_plus[i] if (i, j) in arc_set
                ),
                name=f"link_{f}_{i}"
            )

    # Constraint (15): Binary requirements already set via variable types

    # --- Symmetry breaking (inferred assumption): order vehicles ---
    # **INFERRED ASSUMPTION**: Add symmetry-breaking constraints to help Gurobi.
    # Vehicles are identical, so we can order them by usage.
    # Vehicle f is used only if vehicle f-1 is used.
    for f in range(1, len(F)):
        # If vehicle f leaves the depot to a customer, then vehicle f-1 must also
        model.addConstr(
            gp.quicksum(x[f, 0, j] for j in V_plus[0] if j != n + 1)
            <= gp.quicksum(x[f - 1, 0, j] for j in V_plus[0] if j != n + 1),
            name=f"sym_break_{f}"
        )

    print("Model built. Optimizing...")
    model.optimize()

    # Extract solution
    result = {
        "instance": instance_path,
        "solver": "Gurobi",
        "status": model.Status,
        "status_name": "",
        "objective_value": None,
        "num_vehicles_used": None,
        "routes": [],
        "deliveries": [],
        "time_limit": time_limit,
        "solve_time": model.Runtime
    }

    if model.Status == GRB.OPTIMAL:
        result["status_name"] = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        result["status_name"] = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        result["status_name"] = "INFEASIBLE"
    else:
        result["status_name"] = f"OTHER_{model.Status}"

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["num_vehicles_used"] = round(H.X)

        # Extract routes and deliveries
        for f in F:
            route_arcs = [(i, j) for (i, j) in arcs if x[f, i, j].X > 0.5]
            if not route_arcs:
                continue
            # Check if vehicle is used (goes to a customer)
            leaves_to_customer = any(i == 0 and j != n + 1 for (i, j) in route_arcs)
            if not leaves_to_customer:
                continue

            # Reconstruct route sequence
            adj = {}
            for (i, j) in route_arcs:
                adj[i] = j
            route_seq = [0]
            current = 0
            visited = set()
            while current != n + 1 and current in adj and current not in visited:
                visited.add(current)
                current = adj[current]
                route_seq.append(current)

            # Get deliveries for this vehicle
            vehicle_deliveries = {}
            for i in N:
                qty = delta[f, i].X
                if qty > 1e-6:
                    vehicle_deliveries[i] = round(qty, 2)

            result["routes"].append({
                "vehicle": f,
                "route": route_seq,
                "deliveries": vehicle_deliveries
            })

        # Also collect per-customer delivery summary
        delivery_summary = {}
        for i in N:
            total = sum(delta[f, i].X for f in F)
            delivery_summary[str(i)] = {
                "demand": demand[i],
                "total_delivered": round(total, 2)
            }
        result["deliveries"] = delivery_summary
    else:
        result["objective_value"] = None
        print("No feasible solution found.")

    # Write solution
    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Vehicles used: {result['num_vehicles_used']}")
    print(f"Solve time: {model.Runtime:.2f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for SDVRPTW (arc-flow formulation)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    build_and_solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
