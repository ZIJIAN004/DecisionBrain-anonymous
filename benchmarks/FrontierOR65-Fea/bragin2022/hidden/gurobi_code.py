#!/usr/bin/env python3
"""
Gurobi implementation of the Generalized Assignment Problem (GAP)
from Bragin & Tucker (2022), "Surrogate 'Level-Based' Lagrangian Relaxation
for Mixed-Integer Linear Programming," Scientific Reports 12:22417.

The GAP formulation:
  min  sum_i sum_j c[i][j] * x[i][j]
  s.t. sum_i x[i][j] = 1          for all j  (each job assigned to exactly one machine)
       sum_j a[i][j] * x[i][j] <= b[i]  for all i  (machine capacity)
       x[i][j] in {0, 1}
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
def main():
    parser = argparse.ArgumentParser(description="Solve GAP with Gurobi")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write solution JSON")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_machines = data["num_machines"]  # I
    num_jobs = data["num_jobs"]          # J
    cost = data["cost_matrix"]           # c[i][j], shape (I, J)
    resource = data["resource_matrix"]   # a[i][j], shape (I, J)
    capacity = data["capacities"]        # b[i], length I

    # Create model
    model = gp.Model("GAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # Decision variables: x[i][j] binary
    x = {}
    for i in range(num_machines):
        for j in range(num_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective: minimize sum_i sum_j c[i][j] * x[i][j]
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j]
                     for i in range(num_machines)
                     for j in range(num_jobs)),
        GRB.MINIMIZE
    )

    # Constraint 1: Each job assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_machines)) == 1,
            name=f"assign_{j}"
        )

    # Constraint 2: Machine capacity
    for i in range(num_machines):
        model.addConstr(
            gp.quicksum(resource[i][j] * x[i, j] for j in range(num_jobs)) <= capacity[i],
            name=f"capacity_{i}"
        )

    # Solve
    model.optimize()

    # Extract solution
    solution = {}
    if model.SolCount > 0:
        objective_value = model.ObjVal
        assignments = {}
        for i in range(num_machines):
            for j in range(num_jobs):
                if x[i, j].X > 0.5:
                    assignments[str(j)] = i
        solution["objective_value"] = objective_value
        solution["assignments"] = assignments
        solution["status"] = model.Status
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["assignments"] = {}

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Status: {model.Status}")
    if model.SolCount > 0:
        print(f"Objective value: {objective_value}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
