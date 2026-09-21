#!/usr/bin/env python3
"""
Gurobi implementation of the Linearized Compromise Assignment Problem (LCAP).

Based on: Zheng et al. (2014), multi-objective assignment problem scalarized
using the weighted Tchebychev (achievement) function.

The LCAP minimizes the maximum weighted deviation from a reference point:

    min  mu
    s.t. mu >= lambda_k * (sum_{i,j} c^k_{ij} * x_{ij} - z_bar_k)  for k = 1,...,p
         sum_j x_{ij} = 1   for i = 1,...,n
         sum_i x_{ij} = 1   for j = 1,...,n
         x_{ij} in {0,1}
         mu unrestricted
"""

import argparse
import json

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
    """Load an LCAP instance from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def solve_lcap(instance_path, solution_path, time_limit):
    """Build and solve the LCAP model using Gurobi."""
    data = load_instance(instance_path)

    n = data["n"]
    p = data["p"]
    cost_matrices = data["cost_matrices"]          # p matrices, each n x n
    z_bar = data["reference_point"]                 # length p
    lam = data["search_direction_lambda"]           # length p

    print(f"LCAP Instance: n={n}, p={p}")
    print(f"Reference point: {z_bar}")
    print(f"Search direction (lambda): {lam}")
    print(f"Time limit: {time_limit} seconds")

    # ---- Build model ----
    model = gp.Model("LCAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,j] binary assignment variables
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # mu: unrestricted continuous variable (the Tchebychev objective)
    mu = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu")

    model.update()

    # Objective: minimize mu
    model.setObjective(mu, GRB.MINIMIZE)

    # Assignment constraints: each row assigned to exactly one column
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n)) == 1,
            name=f"row_{i}"
        )

    # Assignment constraints: each column assigned to exactly one row
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == 1,
            name=f"col_{j}"
        )

    # Linearization constraints:
    # mu >= lambda_k * (sum_{i,j} c^k_{ij} * x_{ij} - z_bar_k)  for each k
    for k in range(p):
        c_k = cost_matrices[k]
        model.addConstr(
            mu >= lam[k] * (
                gp.quicksum(c_k[i][j] * x[i, j] for i in range(n) for j in range(n))
                - z_bar[k]
            ),
            name=f"tcheby_{k}"
        )

    model.update()

    # ---- Solve ----
    model.optimize()

    # ---- Extract solution ----
    result = {
        "problem": "LCAP",
        "instance": instance_path,
        "instance_id": data.get("instance_id"),
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["gap"] = model.MIPGap if hasattr(model, "MIPGap") else 0.0

        # Extract the assignment matrix and permutation
        assignment_matrix = [[0] * n for _ in range(n)]
        assignment = []  # assignment[i] = j means row i assigned to column j
        for i in range(n):
            for j in range(n):
                if x[i, j].X > 0.5:
                    assignment_matrix[i][j] = 1
                    assignment.append(j)

        result["assignment"] = assignment
        result["assignment_matrix"] = assignment_matrix

        # Compute objective value for each criterion
        objective_values_per_criterion = []
        for k in range(p):
            c_k = cost_matrices[k]
            val = sum(
                c_k[i][assignment[i]] for i in range(n)
            )
            objective_values_per_criterion.append(val)
        result["objective_values_per_criterion"] = objective_values_per_criterion

        # Compute weighted deviations for verification
        weighted_deviations = []
        for k in range(p):
            dev = lam[k] * (objective_values_per_criterion[k] - z_bar[k])
            weighted_deviations.append(dev)
        result["weighted_deviations"] = weighted_deviations
    else:
        result["objective_value"] = None
        result["status"] = "infeasible_or_no_solution"

    result["solve_time"] = model.Runtime

    # Write solution
    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSolution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value (mu): {result['objective_value']}")
        print(f"Status: {result['status']}")
        print(f"Assignment: {result['assignment']}")
        print(f"Objective values per criterion: {result['objective_values_per_criterion']}")
        print(f"Weighted deviations: {result['weighted_deviations']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Linearized Compromise Assignment Problem (LCAP) using Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file"
    )
    parser.add_argument(
        "--solution_path", type=str, default="gurobi_solution_1.json",
        help="Path for the output solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Maximum solver runtime in seconds"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_lcap(args.instance_path, args.solution_path, args.time_limit)
