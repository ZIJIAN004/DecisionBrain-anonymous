"""
Gurobi implementation of the Origin-Destination Integer Multicommodity Flow (ODIMCF) problem.

Based on: Barnhart, Hane, and Vance (2000), "Using Branch-and-Price-and-Cut to Solve
Origin-Destination Integer Multicommodity Flow Problems", Operations Research 48(2), 318-326.

This implements the Node-Arc (Formulation 1) from the paper:
  min  sum_{k in K} sum_{ij in A} c^k_{ij} * q^k * x^k_{ij}
  s.t. sum_{k in K} q^k * x^k_{ij} <= d_{ij},      for all ij in A      (capacity)
       sum_{ij} x^k_{ij} - sum_{ji} x^k_{ji} = b^k_i, for all i, k      (flow conservation)
       x^k_{ij} in {0,1}                                                  (binary)

Each commodity has an artificial arc from origin to destination with cost = revenue,
representing rejection. Original arcs have cost 0.
"""

import argparse
import json
import os
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
    """Load the problem instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """Build and solve the ODIMCF model using Gurobi."""

    nodes = data["network"]["nodes"]
    arcs = data["network"]["arcs"]
    commodities = data["commodities"]["commodity_list"]

    num_nodes = data["network"]["num_nodes"]
    num_arcs = data["network"]["num_arcs"]
    num_commodities = data["commodities"]["num_commodities"]

    # Build adjacency: arc index -> (from, to, capacity, cost)
    arc_list = []
    for arc in arcs:
        arc_list.append((arc["from_node"], arc["to_node"], arc["capacity"], arc["cost"]))

    model = gp.Model("ODIMCF")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # x[k][a] = 1 if commodity k uses arc a (original arcs)
    x = {}
    for k_idx, comm in enumerate(commodities):
        for a_idx, (i, j, cap, cost) in enumerate(arc_list):
            x[k_idx, a_idx] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{k_idx}_{a_idx}",
                obj=cost * comm["demand"]  # c^k_{ij} * q^k
            )

    # x_art[k] = 1 if commodity k is rejected (uses artificial arc)
    x_art = {}
    for k_idx, comm in enumerate(commodities):
        x_art[k_idx] = model.addVar(
            vtype=GRB.BINARY,
            name=f"x_art_{k_idx}",
            obj=comm["artificial_arc_cost"] * comm["demand"]  # revenue * q^k
        )

    model.update()

    # --- Capacity Constraints ---
    # sum_{k in K} q^k * x^k_{ij} <= d_{ij}, for all ij in A
    for a_idx, (i, j, cap, cost) in enumerate(arc_list):
        model.addConstr(
            gp.quicksum(
                commodities[k_idx]["demand"] * x[k_idx, a_idx]
                for k_idx in range(num_commodities)
            ) <= cap,
            name=f"cap_{a_idx}"
        )

    # --- Flow Conservation Constraints ---
    # For each commodity k, for each node i:
    #   sum_{ij in A} x^k_{ij} - sum_{ji in A} x^k_{ji} = b^k_i
    # The artificial arc goes from origin to destination directly.
    for k_idx, comm in enumerate(commodities):
        origin = comm["origin"]
        destination = comm["destination"]
        for node in nodes:
            # Compute b^k_i
            if node == origin:
                b_ki = 1
            elif node == destination:
                b_ki = -1
            else:
                b_ki = 0

            # Outflow: arcs leaving node (including artificial if node == origin)
            outflow = gp.quicksum(
                x[k_idx, a_idx]
                for a_idx, (i, j, cap, cost) in enumerate(arc_list)
                if i == node
            )
            if node == origin:
                outflow += x_art[k_idx]

            # Inflow: arcs entering node (including artificial if node == destination)
            inflow = gp.quicksum(
                x[k_idx, a_idx]
                for a_idx, (i, j, cap, cost) in enumerate(arc_list)
                if j == node
            )
            if node == destination:
                inflow += x_art[k_idx]

            model.addConstr(
                outflow - inflow == b_ki,
                name=f"flow_{k_idx}_{node}"
            )

    # --- Optimize ---
    model.optimize()

    # --- Extract Solution ---
    solution = {
        "objective_value": None,
        "status": None,
        "commodities": []
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        for k_idx, comm in enumerate(commodities):
            comm_sol = {
                "commodity_id": comm["commodity_id"],
                "origin": comm["origin"],
                "destination": comm["destination"],
                "demand": comm["demand"],
                "rejected": False,
                "path_arcs": []
            }

            if x_art[k_idx].X > 0.5:
                comm_sol["rejected"] = True
            else:
                for a_idx, (i, j, cap, cost) in enumerate(arc_list):
                    if x[k_idx, a_idx].X > 0.5:
                        comm_sol["path_arcs"].append({"from": i, "to": j, "arc_id": a_idx})

            solution["commodities"].append(comm_sol)

        # Compute gap
        if model.Status == GRB.OPTIMAL:
            solution["gap"] = 0.0
        else:
            solution["gap"] = model.MIPGap
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible_or_no_solution"

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve ODIMCF using Gurobi (Node-Arc formulation)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    solution = build_and_solve(data, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
