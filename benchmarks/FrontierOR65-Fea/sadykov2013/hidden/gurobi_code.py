"""
Bin Packing Problem with Conflicts (BPPC) - Compact Formulation 1.

Solves the BPPC using a standard compact MIP formulation with Gurobi.

Reference: Sadykov & Vanderbeck (2013)
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


def solve_bppc(instance_path: str, solution_path: str, time_limit: float) -> None:
    # Load instance
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["num_items"]
    W = data["bin_capacity"]
    items = data["items"]
    conflict_edges = data["conflict_edges"]

    weights = {item["id"]: item["weight"] for item in items}
    K = n  # upper bound on number of bins

    # Create model
    model = gp.Model("BPPC")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Decision variables
    # y[k]: 1 if bin k is used
    y = model.addVars(K, vtype=GRB.BINARY, name="y")
    # x[i,k]: 1 if item i is assigned to bin k
    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")

    # Objective: minimize number of bins used
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    # Constraints

    # 1. Each item must be assigned to at least one bin
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, k] for k in range(K)) >= 1,
            name=f"assign_{i}",
        )

    # 2. Bin capacity constraints
    for k in range(K):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= W * y[k],
            name=f"capacity_{k}",
        )

    # 3. Conflict constraints
    for i, j in conflict_edges:
        for k in range(K):
            model.addConstr(
                x[i, k] + x[j, k] <= y[k],
                name=f"conflict_{i}_{j}_{k}",
            )

    # 4. Symmetry breaking: y[k] >= y[k+1]
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], name=f"symmetry_{k}")

    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # Build solution
    solution = {
        "instance_path": instance_path,
        "solver": "gurobi",
        "formulation": "compact_1",
        "solve_time_seconds": solve_time,
        "time_limit": time_limit,
    }

    if model.SolCount > 0:
        obj_val = round(model.ObjVal)
        solution["objective_value"] = obj_val
        solution["best_bound"] = model.ObjBound
        solution["mip_gap"] = model.MIPGap
        solution["status"] = model.Status

        # Extract bin assignments
        bins = {}
        for k in range(K):
            if y[k].X > 0.5:
                bin_items = [i for i in range(n) if x[i, k].X > 0.5]
                if bin_items:
                    bins[str(k)] = {
                        "items": bin_items,
                        "total_weight": sum(weights[i] for i in bin_items),
                    }
        solution["bins"] = bins
        solution["num_bins_used"] = len(bins)
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["bins"] = {}

    # Write solution
    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value (bins used): {obj_val}")
        print(f"MIP gap: {model.MIPGap:.4f}")
    else:
        print(f"No feasible solution found. Status: {model.Status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve BPPC using compact MIP formulation with Gurobi."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to instance JSON file."
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path to output solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Time limit in seconds for Gurobi.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_bppc(args.instance_path, args.solution_path, args.time_limit)
