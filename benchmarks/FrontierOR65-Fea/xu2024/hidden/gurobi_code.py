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


def solve(instance_path: str, solution_path: str, time_limit: float) -> None:
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    k = data["k"]
    b = data["b"]
    r = data["r"]
    Q = data["Q"]

    model = gp.Model("CCMV")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")

    # Binary indicator variables for cardinality constraint.
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # Objective: min (1/2) x^T Q x
    obj = 0.5 * gp.quicksum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # Return constraint: r^T x >= b
    model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b, name="return")

    # Budget constraint: e^T x = 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1, name="budget")

    # Indicator constraints: z[i] = 0 forces x[i] = 0, exactly representing delta(x_i).
    for i in range(n):
        model.addGenConstrIndicator(z[i], False, x[i] == 0, name=f"indicator_{i}")

    # Cardinality constraint: sum z_i <= k
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, name="cardinality")

    model.optimize()

    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["x"] = [x[i].X for i in range(n)]
    else:
        result["objective_value"] = None

    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cardinality Constrained Mean-Variance Portfolio Optimization (Gurobi)"
    )
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json", help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)
