"""
Quadratic Knapsack Problem (QKP) solver using Gurobi.

Implements the linearized ILP formulation from:
  Caprara, Pisinger, and Toth (1999),
  "Exact Solution of the Quadratic Knapsack Problem"

Linearized ILP:
  maximize  sum_{j in N} sum_{i in N\{j}} p_{ij} y_{ij} + sum_{j in N} q_j x_j
  subject to:
    sum_{j in N} w_j x_j <= c
    sum_{i in N\{j}} w_i y_{ij} <= (c - w_j) x_j,  for all j in N
    0 <= y_{ij} <= x_j,                              for all i,j in N, j != i
    y_{ij} = y_{ji},                                 for all i,j in N, j > i
    x_j, y_{ij} in {0,1}
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


def solve_qkp(instance_path: str, solution_path: str, time_limit: float) -> None:
    # Load instance
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    capacity = data["capacity"]
    weights = data["weights"]
    P = data["profit_matrix"]

    N = range(n)

    # Diagonal entries are the individual item profits q_j
    q = [P[j][j] for j in N]

    # Build model
    model = gp.Model("QKP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # Decision variables
    x = model.addVars(N, vtype=GRB.BINARY, name="x")

    # y_{ij} for i != j
    y = {}
    for i in N:
        for j in N:
            if i != j:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.update()

    # Objective (2): sum_{j} sum_{i != j} p_{ij} y_{ij} + sum_{j} q_j x_j
    obj = gp.quicksum(P[i][j] * y[i, j] for i in N for j in N if i != j)
    obj += gp.quicksum(q[j] * x[j] for j in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint (3): capacity constraint
    model.addConstr(
        gp.quicksum(weights[j] * x[j] for j in N) <= capacity,
        name="capacity"
    )

    # Constraint (4): surrogate knapsack constraints for each j
    for j in N:
        model.addConstr(
            gp.quicksum(weights[i] * y[i, j] for i in N if i != j)
            <= (capacity - weights[j]) * x[j],
            name=f"surrogate_{j}"
        )

    # Constraint (5): y_{ij} <= x_j
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(y[i, j] <= x[j], name=f"link_{i}_{j}")

    # Constraint (6): symmetry y_{ij} = y_{ji} for j > i
    for i in N:
        for j in N:
            if j > i:
                model.addConstr(y[i, j] == y[j, i], name=f"sym_{i}_{j}")

    # Solve
    model.optimize()

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        selected_items = [int(x[j].X > 0.5) for j in N]
    else:
        objective_value = None
        selected_items = [0] * n

    # Write solution
    solution = {
        "objective_value": objective_value,
        "selected_items": selected_items,
    }

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Solve QKP using Gurobi (Caprara et al. 1999 linearization)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to write solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=int, default=300,
        help="Gurobi time limit in seconds (default: 300)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_qkp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
