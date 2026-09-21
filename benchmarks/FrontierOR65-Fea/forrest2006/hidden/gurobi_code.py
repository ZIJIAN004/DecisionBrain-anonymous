"""
Gurobi implementation of the Multiple Knapsack Problem with Color Constraints (MKCP).

Based on: Forrest, Kalagnanam, and Ladanyi (2006)
"A Column-Generation Approach to the Multiple Knapsack Problem with Color Constraints"
INFORMS Journal on Computing 18(1), pp. 129-134.

This implements the "natural" (original) formulation using the linear objective form (1').
"""

import argparse
import json
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


def load_instance(instance_path):
    """Load a MKCP instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build and solve the natural MKCP formulation.

    Sets:
      N = {0, ..., n-1}: orders (items)
      M = {0, ..., m-1}: slabs (knapsacks)
      M_i: eligible slabs for order i
      N_j: orders eligible for slab j (derived from M_i)
      C_j: distinct colors among orders in N_j

    Variables:
      x[i,j] in {0,1}: 1 if order i assigned to slab j  (only for j in M_i)
      y[c,j] in {0,1}: 1 if color c is used on slab j    (only for c in C_j)
      z[j]   in {0,1}: 1 if slab j is used

    Objective (linear form 1'):
      max sum_{i in N} sum_{j in M_i} 2*w_i*x[i,j] - sum_{j in M} W_j*z[j]

    Constraints:
      (2) sum_{i in N_j} w_i * x[i,j] <= W_j * z[j],  for all j in M
      (3) sum_{j in M_i} x[i,j] <= 1,                  for all i in N
      (4) sum_{c in C_j} y[c,j] <= 2,                   for all j in M
      (5a) x[i,j] <= y[c_i, j],                         for all i in N, j in M_i
    """
    n = data["num_orders"]
    m = data["num_slabs"]
    w = data["order_weights"]
    W = data["slab_weights"]
    colors = data["order_colors"]
    eligible = data["eligible_slabs_per_order"]  # M_i for each order i
    K = data["max_colors_per_knapsack"]  # typically 2

    # Derive N_j: orders incident to slab j
    N_j = [[] for _ in range(m)]
    for i in range(n):
        for j in eligible[i]:
            N_j[j].append(i)

    # Derive C_j: distinct colors incident on slab j
    C_j = [set() for _ in range(m)]
    for j in range(m):
        for i in N_j[j]:
            C_j[j].add(colors[i])

    # Build model
    model = gp.Model("MKCP_Natural")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,j] only for incident pairs
    x = {}
    for i in range(n):
        for j in eligible[i]:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # y[c,j] only for colors in C_j
    y = {}
    for j in range(m):
        for c in C_j[j]:
            y[c, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{j}")

    # z[j] for all slabs
    z = {}
    for j in range(m):
        z[j] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}")

    model.update()

    # Objective: max sum 2*w_i*x[i,j] - sum W_j*z[j]  (linear form 1')
    obj = gp.LinExpr()
    for i in range(n):
        for j in eligible[i]:
            obj += 2.0 * w[i] * x[i, j]
    for j in range(m):
        obj -= W[j] * z[j]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint (2): capacity
    for j in range(m):
        lhs = gp.LinExpr()
        for i in N_j[j]:
            lhs += w[i] * x[i, j]
        model.addConstr(lhs <= W[j] * z[j], name=f"capacity_{j}")

    # Constraint (3): each order assigned at most once
    for i in range(n):
        lhs = gp.LinExpr()
        for j in eligible[i]:
            lhs += x[i, j]
        model.addConstr(lhs <= 1, name=f"assign_{i}")

    # Constraint (4): at most K distinct colors per slab
    for j in range(m):
        if C_j[j]:
            lhs = gp.LinExpr()
            for c in C_j[j]:
                lhs += y[c, j]
            model.addConstr(lhs <= K, name=f"color_limit_{j}")

    # Constraint (5a): linking x and y
    for i in range(n):
        for j in eligible[i]:
            c_i = colors[i]
            model.addConstr(x[i, j] <= y[c_i, j], name=f"link_{i}_{j}")

    # Solve
    model.optimize()

    # Extract solution
    result = {
        "status": model.Status,
        "objective_value": None,
        "assignments": [],  # list of (order, slab) pairs
        "slabs_used": [],
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        for i in range(n):
            for j in eligible[i]:
                if x[i, j].X > 0.5:
                    result["assignments"].append({"order": i, "slab": j})
                    break

        for j in range(m):
            if z[j].X > 0.5:
                result["slabs_used"].append(j)

        # Verify color constraints in solution
        slab_colors = {}
        for a in result["assignments"]:
            j = a["slab"]
            c = colors[a["order"]]
            if j not in slab_colors:
                slab_colors[j] = set()
            slab_colors[j].add(c)

        result["solution_details"] = {
            "num_orders_assigned": len(result["assignments"]),
            "num_slabs_used": len(result["slabs_used"]),
            "colors_per_slab": {str(j): list(cs) for j, cs in slab_colors.items()},
        }

        if model.Status == GRB.OPTIMAL:
            result["solution_status"] = "optimal"
        elif model.Status == GRB.TIME_LIMIT:
            result["solution_status"] = "time_limit_feasible"
        else:
            result["solution_status"] = "feasible"
    else:
        result["objective_value"] = None
        result["solution_status"] = "no_feasible_solution_found"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve MKCP using the natural formulation with Gurobi."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to the instance JSON file."
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path for the output solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver runtime in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    print(f"Loaded instance: {data.get('problem_name', 'MKCP')}")
    print(f"  Orders: {data['num_orders']}, Slabs: {data['num_slabs']}, Colors: {data['num_colors']}")
    print(f"  Max colors per knapsack: {data['max_colors_per_knapsack']}")
    print(f"  Time limit: {args.time_limit}s")

    result = build_and_solve(data, args.time_limit)

    print(f"\nSolution status: {result['solution_status']}")
    print(f"Objective value: {result['objective_value']}")

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
