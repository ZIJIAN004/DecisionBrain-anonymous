"""
Gurobi implementation of the Set Partitioning Problem (SPP).

Based on: "A Unified Column Generation and Elimination Method for Solving
           Large-Scale Set Partitioning Problems" (Zhang et al., 2025)

Formulation (Equation 2):
  min  c^T x
  s.t. A x = 1_M
       x_j in {0, 1}  for all j in J
"""

import argparse
import json
import numpy as np
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
def load_instance(instance_path: str) -> dict:
    with open(instance_path, "r") as f:
        return json.load(f)


def build_and_solve(instance: dict, time_limit: int) -> dict:
    M = instance["parameters"]["num_trips_M"]
    N = instance["parameters"]["num_duties_N"]
    duties = instance["duties"]  # list of lists of trip_ids
    costs = instance["costs"]    # list of floats, length N

    # Build binary matrix A (M x N)
    # a_{i,j} = 1 if trip i is in duty j
    A = np.zeros((M, N), dtype=int)
    for j, duty in enumerate(duties):
        for trip_id in duty:
            A[trip_id, j] = 1

    # Create Gurobi model
    model = gp.Model("SetPartitioning")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x_j in {0, 1}
    x = model.addVars(N, vtype=GRB.BINARY, name="x")

    # Objective: min c^T x
    model.setObjective(
        gp.quicksum(costs[j] * x[j] for j in range(N)),
        GRB.MINIMIZE,
    )

    # Constraints: A x = 1_M (each trip covered exactly once)
    for i in range(M):
        model.addConstr(
            gp.quicksum(A[i, j] * x[j] for j in range(N) if A[i, j] == 1) == 1,
            name=f"cover_{i}",
        )

    model.optimize()

    # Extract solution
    result = {"objective_value": None, "selected_duties": [], "status": None}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["selected_duties"] = [j for j in range(N) if x[j].X > 0.5]
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
    else:
        result["status"] = "infeasible_or_no_solution"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve Set Partitioning Problem with Gurobi"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Objective value: {solution['objective_value']}")
    print(f"Status: {solution['status']}")


if __name__ == "__main__":
    main()
