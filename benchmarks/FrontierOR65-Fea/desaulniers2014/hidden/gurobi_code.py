"""
Gurobi MIP formulation for the Inventory-Routing Problem (IRP).

Based on: Desaulniers, Rakke, Coelho (2014) - "A Branch-Price-and-Cut Algorithm
for the Inventory-Routing Problem"

Since the paper's formulation (1)-(9) is a Dantzig-Wolfe decomposition (column generation)
that CANNOT be directly input into a MIP solver, we implement a compact arc-flow
formulation for the IRP that is equivalent and can be solved directly by Gurobi.

This compact formulation uses:
- Binary variables z_{ip} for whether customer i is visited in period p
- Binary arc variables x_{ijp} for routing in period p
- Continuous variables q_{ip} for quantity delivered to customer i in period p
- Continuous variables I_{ip} for inventory at node i at end of period p

Replenishment policy: Maximum-Level (ML) - any quantity can be delivered as long as
the maximum inventory capacity is not exceeded.

Assumptions (inferred):
- Travel costs are symmetric (c_{ij} = c_{ji}), based on Euclidean distances in the instance.
- The distance_matrix provided in the instance is used directly as travel cost.
- Demands are constant per period (demand_per_period).
- Production is constant per period (production_per_period).
"""

import argparse
import json
import math
import sys

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
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def solve_irp(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    # ------------------------------------------------------------------
    # Extract instance data
    # ------------------------------------------------------------------
    num_customers = data["num_customers"]
    num_periods = data["num_periods"]
    num_vehicles = data["num_vehicles"]
    Q = data["vehicle_capacity"]

    depot = data["depot"]
    customers = data["customers"]
    dist = data["distance_matrix"]

    N = list(range(1, num_customers + 1))  # customer indices
    V = [0] + N                             # all nodes (0 = depot)
    P = list(range(1, num_periods + 1))     # periods 1..rho

    # Depot parameters
    I0_0 = depot["initial_inventory"]
    C_0 = depot["max_inventory"]
    h_0 = depot["holding_cost"]
    prod = depot["production_per_period"]  # d^p_0

    # Customer parameters (indexed by customer id 1..n)
    demand = {}
    C = {}
    I0 = {}
    h = {}
    for c in customers:
        cid = c["id"]
        demand[cid] = c["demand_per_period"]
        C[cid] = c["max_inventory"]
        I0[cid] = c["initial_inventory"]
        h[cid] = c["holding_cost"]

    # Travel cost matrix
    cost = {}
    for i in V:
        for j in V:
            cost[i, j] = dist[i][j]

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("IRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision variables ---

    # x[i,j,p] = 1 if arc (i,j) is traversed in period p
    x = {}
    for p in P:
        for i in V:
            for j in V:
                if i != j:
                    x[i, j, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{p}")

    # z[i,p] = 1 if customer i is visited in period p
    z = {}
    for p in P:
        for i in N:
            z[i, p] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{p}")

    # q[i,p] = quantity delivered to customer i in period p
    q = {}
    for p in P:
        for i in N:
            q[i, p] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{p}")

    # I_cust[i,p] = inventory at customer i at end of period p
    I_cust = {}
    for i in N:
        for p in P:
            I_cust[i, p] = model.addVar(lb=0, ub=C[i], vtype=GRB.CONTINUOUS,
                                        name=f"Ic_{i}_{p}")

    # I_dep[p] = inventory at depot at end of period p
    I_dep = {}
    for p in P:
        I_dep[p] = model.addVar(lb=0, ub=C_0, vtype=GRB.CONTINUOUS,
                                name=f"Id_{p}")

    # u[i,j,p] = flow (number of units of load) on arc (i,j) in period p
    # Used for subtour elimination (MTZ-like flow formulation)
    f_var = {}
    for p in P:
        for i in V:
            for j in V:
                if i != j:
                    f_var[i, j, p] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                  name=f"f_{i}_{j}_{p}")

    model.update()

    # --- Objective function ---
    # Minimize routing costs + inventory holding costs
    obj = gp.LinExpr()

    # Routing costs
    for p in P:
        for i in V:
            for j in V:
                if i != j:
                    obj += cost[i, j] * x[i, j, p]

    # Holding costs at customers
    for p in P:
        for i in N:
            obj += h[i] * I_cust[i, p]

    # Holding costs at depot
    for p in P:
        obj += h_0 * I_dep[p]

    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (C1) Depot inventory balance
    for p in P:
        total_delivered = gp.quicksum(q[i, p] for i in N)
        if p == 1:
            model.addConstr(I0_0 + prod - total_delivered == I_dep[p],
                            name=f"depot_inv_{p}")
        else:
            model.addConstr(I_dep[p - 1] + prod - total_delivered == I_dep[p],
                            name=f"depot_inv_{p}")

    # (C2) Customer inventory balance (no stockouts)
    for i in N:
        for p in P:
            if p == 1:
                model.addConstr(I0[i] + q[i, p] - demand[i] == I_cust[i, p],
                                name=f"cust_inv_{i}_{p}")
            else:
                model.addConstr(I_cust[i, p - 1] + q[i, p] - demand[i] == I_cust[i, p],
                                name=f"cust_inv_{i}_{p}")

    # (C3) Customer inventory capacity: inventory after delivery <= C_i
    # Inventory right after delivery (before consumption) = I_{i,p-1} + q_{i,p}
    # This must be <= C_i
    for i in N:
        for p in P:
            if p == 1:
                model.addConstr(I0[i] + q[i, p] <= C[i],
                                name=f"cust_cap_{i}_{p}")
            else:
                model.addConstr(I_cust[i, p - 1] + q[i, p] <= C[i],
                                name=f"cust_cap_{i}_{p}")

    # (C4) Delivery only if visited
    # q[i,p] <= C_i * z[i,p]  (if not visited, no delivery)
    for i in N:
        for p in P:
            model.addConstr(q[i, p] <= C[i] * z[i, p],
                            name=f"link_qz_{i}_{p}")

    # (C5) Each customer visited at most once per period
    # (already implied by z being binary, but we link z to routing)
    # z[i,p] = 1 iff customer i is visited in period p
    for i in N:
        for p in P:
            model.addConstr(
                gp.quicksum(x[j, i, p] for j in V if j != i) == z[i, p],
                name=f"visit_{i}_{p}")

    # (C6) Flow conservation for routing: each visited node has in-degree = out-degree
    for p in P:
        for i in N:
            model.addConstr(
                gp.quicksum(x[j, i, p] for j in V if j != i) ==
                gp.quicksum(x[i, j, p] for j in V if j != i),
                name=f"flow_{i}_{p}")

    # (C7) At most K vehicles leave the depot in each period
    for p in P:
        model.addConstr(
            gp.quicksum(x[0, j, p] for j in N) <= num_vehicles,
            name=f"vehicles_{p}")

    # (C8) Depot out-degree = depot in-degree (balanced routes)
    for p in P:
        model.addConstr(
            gp.quicksum(x[0, j, p] for j in N) ==
            gp.quicksum(x[j, 0, p] for j in N),
            name=f"depot_balance_{p}")

    # (C9) Subtour elimination via commodity flow
    # Flow on arcs leaving depot = total delivery on corresponding route
    # f[0,j,p] <= Q * x[0,j,p]
    for p in P:
        for j in N:
            model.addConstr(f_var[0, j, p] <= Q * x[0, j, p],
                            name=f"flow_depot_out_{j}_{p}")

    # Flow conservation at customer nodes
    for p in P:
        for i in N:
            model.addConstr(
                gp.quicksum(f_var[j, i, p] for j in V if j != i) -
                gp.quicksum(f_var[i, j, p] for j in V if j != i) == q[i, p],
                name=f"flow_cons_{i}_{p}")

    # Flow on arcs bounded by capacity times arc usage
    for p in P:
        for i in V:
            for j in V:
                if i != j:
                    model.addConstr(f_var[i, j, p] <= Q * x[i, j, p],
                                    name=f"flow_cap_{i}_{j}_{p}")

    # (C10) Non-negativity of customer inventory (no stockouts)
    # Already handled by lb=0 on I_cust variables

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    result = {}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val

        # Extract routes and deliveries
        solution_details = {"periods": {}}
        for p in P:
            period_info = {"routes": [], "deliveries": {}}

            # Find routes by tracing arcs from depot
            visited_arcs = []
            for i in V:
                for j in V:
                    if i != j and x[i, j, p].X > 0.5:
                        visited_arcs.append((i, j))

            # Trace routes from depot
            routes = []
            depot_successors = [j for (i, j) in visited_arcs if i == 0]
            used_arcs = set()
            for start in depot_successors:
                route = [0, start]
                used_arcs.add((0, start))
                current = start
                while current != 0:
                    next_node = None
                    for (i, j) in visited_arcs:
                        if i == current and (i, j) not in used_arcs:
                            next_node = j
                            used_arcs.add((i, j))
                            break
                    if next_node is None:
                        break
                    route.append(next_node)
                    current = next_node
                routes.append(route)

            period_info["routes"] = routes

            # Deliveries
            deliveries = {}
            for i in N:
                qval = q[i, p].X
                if qval > 1e-6:
                    deliveries[str(i)] = round(qval, 4)
            period_info["deliveries"] = deliveries

            # Inventories
            inventories = {}
            inventories["depot"] = round(I_dep[p].X, 4)
            for i in N:
                inventories[str(i)] = round(I_cust[i, p].X, 4)
            period_info["inventories"] = inventories

            solution_details["periods"][str(p)] = period_info

        result["solution_details"] = solution_details
        result["status"] = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible"
        result["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None

    else:
        result["objective_value"] = None
        result["status"] = "Infeasible or no solution found"

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Gurobi MIP for Inventory-Routing Problem")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_irp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
