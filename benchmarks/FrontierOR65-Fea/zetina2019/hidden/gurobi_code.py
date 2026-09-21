"""
Gurobi solver for the Multicommodity Uncapacitated Fixed-charge Network Design (MUFND) problem.

Model (P):
    min  sum_{(i,j) in A} f_{ij} y_{ij} + sum_{k in K} sum_{(i,j) in A} W^k c^k_{ij} x^k_{ij}

    subject to:
        Flow conservation:
            sum_j x^k_{ji} - sum_j x^k_{ij} = b^k_i   for all i in N, k in K
            where b^k_i = -1 if i = o_k, 1 if i = d_k, 0 otherwise

        Linking constraints:
            x^k_{ij} <= y_{ij}   for all (i,j) in A, k in K

        x^k_{ij} >= 0
        y_{ij} in {0, 1}
"""

import argparse
import json
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


def solve_mufnd(instance_path, solution_path, time_limit):
    with open(instance_path, "r") as f:
        data = json.load(f)

    num_nodes = data["num_nodes"]
    num_commodities = data["num_commodities"]
    arcs = [tuple(a) for a in data["arcs"]]
    fixed_costs = data["fixed_costs"]
    variable_costs = data["variable_costs"]
    commodities = data["commodities"]

    # Build adjacency structures for flow conservation
    # For each node, which arcs go into it and out of it
    arcs_out = {n: [] for n in range(num_nodes)}
    arcs_in = {n: [] for n in range(num_nodes)}
    for i, j in arcs:
        arcs_out[i].append((i, j))
        arcs_in[j].append((i, j))

    model = gp.Model("MUFND")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Decision variables
    # y_{ij} in {0,1}: whether arc (i,j) is opened
    y = {}
    for i, j in arcs:
        key = f"{i}_{j}"
        f_ij = fixed_costs[key]
        y[i, j] = model.addVar(vtype=GRB.BINARY, obj=f_ij, name=f"y_{i}_{j}")

    # x^k_{ij} >= 0: fraction of commodity k routed on arc (i,j)
    x = {}
    for k in range(num_commodities):
        wk = commodities[k]["demand"]
        for i, j in arcs:
            key = f"{i}_{j}"
            c_k_ij = variable_costs[key][k]
            x[k, i, j] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                obj=wk * c_k_ij,
                name=f"x_{k}_{i}_{j}",
            )

    model.update()

    # Flow conservation constraints
    for k in range(num_commodities):
        ok = commodities[k]["origin"]
        dk = commodities[k]["destination"]
        for n in range(num_nodes):
            if n == ok:
                rhs = -1.0
            elif n == dk:
                rhs = 1.0
            else:
                rhs = 0.0

            inflow = gp.quicksum(x[k, i, j] for i, j in arcs_in[n])
            outflow = gp.quicksum(x[k, i, j] for i, j in arcs_out[n])

            model.addConstr(inflow - outflow == rhs, name=f"flow_{k}_{n}")

    # Linking constraints: x^k_{ij} <= y_{ij}
    for k in range(num_commodities):
        for i, j in arcs:
            model.addConstr(x[k, i, j] <= y[i, j], name=f"link_{k}_{i}_{j}")

    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    # Build solution output
    solution = {
        "instance_path": instance_path,
        "solver": "gurobi",
        "time_limit": time_limit,
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["best_bound"] = model.ObjBound
        solution["mip_gap"] = model.MIPGap
        solution["runtime"] = model.Runtime
        solution["status"] = model.Status
        solution["status_name"] = (
            "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME_LIMIT"
        )

        # Extract open arcs
        open_arcs = {}
        for i, j in arcs:
            if y[i, j].X > 0.5:
                open_arcs[f"{i}_{j}"] = 1

        # Extract commodity routings
        routings = {}
        for k in range(num_commodities):
            flows = {}
            for i, j in arcs:
                val = x[k, i, j].X
                if val > 1e-6:
                    flows[f"{i}_{j}"] = val
            if flows:
                routings[str(k)] = flows

        solution["open_arcs"] = open_arcs
        solution["routings"] = routings
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["status_name"] = "INFEASIBLE_OR_NO_SOLUTION"
        solution["runtime"] = model.Runtime

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found.")

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve MUFND using Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path to output solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Gurobi time limit in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_mufnd(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
