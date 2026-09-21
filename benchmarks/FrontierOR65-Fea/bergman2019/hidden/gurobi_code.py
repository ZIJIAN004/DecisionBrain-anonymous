"""
Gurobi implementation of the Quadratic Multiknapsack Problem (QMKP-QP).
Source: Bergman (2019), "An Exact Algorithm for the Quadratic Multiknapsack Problem
        with an Application to Event Seating", INFORMS Journal on Computing.

Model: QMKP-QP (Section 3.1)
  maximize   sum_{i,k} p_i * x_{i,k}
             + sum_{i<j, k} p_{i,j} * x_{i,k} * x_{j,k}
  subject to:
    sum_i w_i * x_{i,k} <= C_k,   for all k
    sum_k x_{i,k} <= 1,            for all i
    x_{i,k} in {0,1}

Gurobi solver settings (from reproduction-critical notes, Section 6.1):
  - PreQLinearize = 1
  - Threads = 1
  - MIPGap = 1e-4 (default), MIPGapAbs = 1e-10 (default)
"""

import argparse
import json
import os
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
    with open(path, "r") as f:
        return json.load(f)


def solve_qmkp(instance, time_limit):
    n = instance["n"]
    m = instance["m"]
    profits = instance["profits"]           # p_i, length n (0-indexed)
    pairwise = instance["pairwise_profits"] # p_{i,j}, n x n matrix (0-indexed)
    weights = instance["weights"]           # w_i, length n
    capacities = instance["capacities"]    # C_k, length m

    model = gp.Model("QMKP")

    # Solver settings from paper (Section 6.1 / reproduction-critical notes)
    model.setParam("PreQLinearize", 1)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # MIPGap and MIPGapAbs use Gurobi defaults (1e-4 and 1e-10)

    # Decision variables: x[i][k] = 1 iff item i assigned to knapsack k
    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    model.update()

    # Objective: linear profits + quadratic pairwise profits
    obj = gp.QuadExpr()
    # Linear part
    for i in range(n):
        for k in range(m):
            obj += profits[i] * x[i, k]
    # Quadratic part: sum over i < j, k of p_{i,j} * x_{i,k} * x_{j,k}
    for i in range(n):
        for j in range(i + 1, n):
            p_ij = pairwise[i][j]
            if p_ij != 0:
                for k in range(m):
                    obj += p_ij * x[i, k] * x[j, k]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraints: sum_i w_i * x_{i,k} <= C_k
    for k in range(m):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k],
            name=f"cap_{k}"
        )

    # Assignment constraints: sum_k x_{i,k} <= 1 (each item in at most one knapsack)
    # NOTE: For RQMKP this would be equality; for standard QMKP it is <=
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, k] for k in range(m)) <= 1,
            name=f"assign_{i}"
        )

    model.optimize()

    # Extract solution
    result = {}
    status = model.Status

    if status == GRB.OPTIMAL or (status == GRB.TIME_LIMIT and model.SolCount > 0):
        obj_val = model.ObjVal
        # Fix_7: emit list-of-pairs so the feasibility checker can detect
        # multi-assignment violations; the prior dict keying collapsed
        # duplicate item keys, making constraint 2 unreachable.
        assignment = []
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment.append([i, k])

        result["objective_value"] = obj_val
        result["assignment"] = assignment
        result["status"] = "optimal" if status == GRB.OPTIMAL else "time_limit_feasible"
        result["gap"] = model.MIPGap if model.SolCount > 0 else None
    elif status == GRB.INFEASIBLE:
        result["objective_value"] = None
        result["status"] = "infeasible"
    else:
        result["objective_value"] = None
        result["status"] = "no_solution_found"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for the Quadratic Multiknapsack Problem (QMKP-QP)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_qmkp(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status']}")
    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
