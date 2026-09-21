"""
Gurobi implementation of the Multiple-Depot Vehicle Scheduling Problem (MDVSP)
using the integer multicommodity flow formulation from:

Ribeiro & Soumis (1994), "A Column Generation Approach to the Multiple-Depot
Vehicle Scheduling Problem", Operations Research 42(1):41-52.

Formulation (Problem MDVSP):
    Minimize  sum_{k=1}^{m} sum_{(i,j) in A^k} c_{ij} * x^k_{ij}
    s.t.
      (1) sum_{k=1}^{m} sum_{i in V^k} x^k_{ij} = 1       for all j in N
      (2) sum_{i in V^k} x^k_{ij} - sum_{i in V^k} x^k_{ji} = 0
                                                              for all k in K, j in V^k
      (3) sum_{j in N} x^k_{n+k,j} <= r_k                   for all k in K
          x^k_{ij} >= 0, integer                              for all k, (i,j) in A^k
"""

import json
import argparse
import math
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


def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_mdvsp(data, time_limit):
    n_trips = data["parameters"]["n_trips"]
    m_depots = data["parameters"]["m_depots"]
    trips = data["trips"]
    depots = data["depots"]
    relief_points = data["relief_points"]
    vehicle_capacities = data["vehicle_capacities"]

    # Build lookup for compatible arcs: (from_trip, to_trip) -> cost
    compatible_arc_costs = {}
    for arc in data["compatible_arcs"]:
        i = arc["from_trip"]
        j = arc["to_trip"]
        compatible_arc_costs[(i, j)] = arc["cost"]

    # Build lookup for depot-to-trip costs: (depot_1indexed, trip) -> cost
    depot_to_trip_cost = {}
    for entry in data["depot_to_trip_costs"]:
        depot_to_trip_cost[(entry["depot"], entry["trip"])] = entry["cost"]

    # Build lookup for trip-to-depot costs: (trip, depot_1indexed) -> cost
    trip_to_depot_cost = {}
    for entry in data["trip_to_depot_costs"]:
        trip_to_depot_cost[(entry["trip"], entry["depot"])] = entry["cost"]

    # Create model
    model = gp.Model("MDVSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x[k, i, j] = flow of type k through arc (i, j)
    # Nodes: trips are 1..n, depot k is represented as node (n+k)
    # k is 1-indexed (1..m), trips are 1-indexed (1..n)

    x = {}

    for k in range(1, m_depots + 1):
        depot_node = n_trips + k

        # Depot-to-trip arcs
        for j in range(1, n_trips + 1):
            cost = depot_to_trip_cost[(k, j)]
            x[k, depot_node, j] = model.addVar(
                vtype=GRB.INTEGER, lb=0, obj=cost,
                name=f"x_{k}_{depot_node}_{j}"
            )

        # Trip-to-depot arcs
        for j in range(1, n_trips + 1):
            cost = trip_to_depot_cost[(j, k)]
            x[k, j, depot_node] = model.addVar(
                vtype=GRB.INTEGER, lb=0, obj=cost,
                name=f"x_{k}_{j}_{depot_node}"
            )

        # Trip-to-trip arcs (only compatible pairs)
        for (i, j), cost in compatible_arc_costs.items():
            x[k, i, j] = model.addVar(
                vtype=GRB.INTEGER, lb=0, obj=cost,
                name=f"x_{k}_{i}_{j}"
            )

    model.update()

    # Constraint (1): Each trip j is covered exactly once
    for j in range(1, n_trips + 1):
        expr = gp.LinExpr()
        for k in range(1, m_depots + 1):
            depot_node = n_trips + k
            # Arcs entering trip j: from depot k, or from other trips
            if (k, depot_node, j) in x:
                expr += x[k, depot_node, j]
            for i in range(1, n_trips + 1):
                if (k, i, j) in x:
                    expr += x[k, i, j]
        model.addConstr(expr == 1, name=f"cover_{j}")

    # Constraint (2): Flow conservation for each depot k and each node j in V^k
    for k in range(1, m_depots + 1):
        depot_node = n_trips + k

        # Flow conservation at each trip node j
        for j in range(1, n_trips + 1):
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()

            # Inflow to j: from depot_node or from other trips
            if (k, depot_node, j) in x:
                inflow += x[k, depot_node, j]
            for i in range(1, n_trips + 1):
                if (k, i, j) in x:
                    inflow += x[k, i, j]

            # Outflow from j: to depot_node or to other trips
            if (k, j, depot_node) in x:
                outflow += x[k, j, depot_node]
            for i in range(1, n_trips + 1):
                if (k, j, i) in x:
                    outflow += x[k, j, i]

            model.addConstr(inflow - outflow == 0,
                            name=f"flow_{k}_{j}")

        # Flow conservation at depot node:
        # The paper notes that x_{n+k,n+k} represents idle vehicles,
        # and equations (9)/(10) follow from (2).
        # We enforce: outflow from depot = inflow to depot
        depot_outflow = gp.LinExpr()
        depot_inflow = gp.LinExpr()
        for j in range(1, n_trips + 1):
            if (k, depot_node, j) in x:
                depot_outflow += x[k, depot_node, j]
            if (k, j, depot_node) in x:
                depot_inflow += x[k, j, depot_node]
        model.addConstr(depot_outflow - depot_inflow == 0,
                        name=f"flow_depot_{k}")

    # Constraint (3): Depot capacity
    for k in range(1, m_depots + 1):
        depot_node = n_trips + k
        expr = gp.LinExpr()
        for j in range(1, n_trips + 1):
            if (k, depot_node, j) in x:
                expr += x[k, depot_node, j]
        model.addConstr(expr <= vehicle_capacities[k - 1],
                        name=f"capacity_{k}")

    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = model.Status
        result["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0

        # Extract vehicle routes
        routes = []
        for k in range(1, m_depots + 1):
            depot_node = n_trips + k
            # Find trips that start from this depot
            for j in range(1, n_trips + 1):
                if (k, depot_node, j) in x and x[k, depot_node, j].X > 0.5:
                    # Trace the route from depot through trips
                    route = {"depot": k, "trips": [j]}
                    current = j
                    while True:
                        next_trip = None
                        # Check trip-to-trip arcs
                        for jj in range(1, n_trips + 1):
                            if (k, current, jj) in x and x[k, current, jj].X > 0.5:
                                next_trip = jj
                                break
                        if next_trip is None:
                            break
                        route["trips"].append(next_trip)
                        current = next_trip
                    routes.append(route)
        result["routes"] = routes
        result["num_vehicles"] = len(routes)
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["routes"] = []
        result["num_vehicles"] = 0

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for MDVSP (Ribeiro & Soumis 1994)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = solve_mdvsp(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Number of vehicles: {result['num_vehicles']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
