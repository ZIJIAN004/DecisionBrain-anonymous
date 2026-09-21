"""
Gurobi implementation of the Minimum Hyperplanes Clustering Problem (Min-HCP).

Based on: Amaldi, Dhyani, and Ceselli (2013),
"Column Generation for the Minimum Hyperplanes Clustering Problem",
INFORMS Journal on Computing.

This implements the MINLP formulation (Eqs. 2-9) with the unit-norm
simplification (Eqs. 11-13), yielding constraints that are linear
except for the quadratic norm constraint ||w_j||_2 = 1.

Gurobi handles the nonconvex quadratic constraint via NonConvex=2.
"""

import argparse
import json
import math
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
def load_instance(path):
    """Load a Min-HCP instance from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compute_big_M(points):
    """Compute big-M as the maximum inter-point Euclidean distance (Eq. 10)."""
    n = len(points)
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((points[i][l] - points[j][l]) ** 2 for l in range(len(points[i]))))
            if dist > max_dist:
                max_dist = dist
    return max_dist


def solve_min_hcp(instance_path, solution_path, time_limit):
    """Solve Min-HCP using Gurobi with the MINLP formulation."""
    data = load_instance(instance_path)

    points = np.array(data["points"])
    n = data["n"]
    d = data["d"]
    epsilon = data["epsilon"]

    # Upper bound on number of hyperplanes: K = ceil(n/d)
    K = math.ceil(n / d)

    # Big-M constant (Eq. 10): largest inter-point Euclidean distance
    # Use precomputed value if available, otherwise compute
    if "big_M" in data:
        M = data["big_M"]
    else:
        M = compute_big_M(points.tolist())

    print(f"Instance: n={n}, d={d}, epsilon={epsilon:.6f}, K={K}, M={M:.6f}")

    # Create model
    model = gp.Model("Min-HCP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Allow nonconvex quadratic constraints (for ||w_j||_2 = 1)
    model.setParam("NonConvex", 2)
    # Reduce output verbosity
    model.setParam("OutputFlag", 1)

    # ---- Decision Variables ----
    # w[j,l]: normal vector component l of hyperplane j (Eq. 7)
    w = {}
    for j in range(K):
        for l in range(d):
            w[j, l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                   vtype=GRB.CONTINUOUS, name=f"w_{j}_{l}")

    # w0[j]: offset of hyperplane j (Eq. 7)
    w0 = {}
    for j in range(K):
        w0[j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                              vtype=GRB.CONTINUOUS, name=f"w0_{j}")

    # D[i,j]: assignment of point i to hyperplane j (Eq. 8)
    D = {}
    for i in range(n):
        for j in range(K):
            D[i, j] = model.addVar(vtype=GRB.BINARY, name=f"D_{i}_{j}")

    # y[j]: whether hyperplane j is used (Eq. 9)
    y = {}
    for j in range(K):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

    model.update()

    # ---- Objective: minimize number of hyperplanes (Eq. 2) ----
    model.setObjective(gp.quicksum(y[j] for j in range(K)), GRB.MINIMIZE)

    # ---- Constraints ----

    # Constraint (11): -(a_i * w_j - w_j^0) <= epsilon + M*(1 - D_ij)
    # Equivalent to: w_j^0 - a_i * w_j <= epsilon + M*(1 - D_ij)
    for i in range(n):
        for j in range(K):
            lhs = w0[j] - gp.quicksum(points[i, l] * w[j, l] for l in range(d))
            model.addConstr(lhs <= epsilon + M * (1 - D[i, j]),
                            name=f"dist_neg_{i}_{j}")

    # Constraint (12): (a_i * w_j - w_j^0) <= epsilon + M*(1 - D_ij)
    for i in range(n):
        for j in range(K):
            lhs = gp.quicksum(points[i, l] * w[j, l] for l in range(d)) - w0[j]
            model.addConstr(lhs <= epsilon + M * (1 - D[i, j]),
                            name=f"dist_pos_{i}_{j}")

    # Constraint (5): each point assigned to at least one hyperplane
    for i in range(n):
        model.addConstr(gp.quicksum(D[i, j] for j in range(K)) >= 1,
                        name=f"cover_{i}")

    # Constraint (6): D_ij <= y_j
    for i in range(n):
        for j in range(K):
            model.addConstr(D[i, j] <= y[j], name=f"link_{i}_{j}")

    # Constraint (13): ||w_j||_2 = 1 (unit norm)
    # This is a nonconvex quadratic constraint: sum_l w[j,l]^2 = 1
    for j in range(K):
        model.addConstr(
            gp.quicksum(w[j, l] * w[j, l] for l in range(d)) == 1.0,
            name=f"norm_{j}"
        )

    # ---- Symmetry breaking: order hyperplanes by index of first assigned point ----
    # (helps solver performance)
    for j in range(K - 1):
        model.addConstr(y[j] >= y[j + 1], name=f"symbreak_{j}")

    print(f"Model has {model.NumVars} variables, solving...")

    # ---- Solve ----
    model.optimize()

    # ---- Extract solution ----
    result = {
        "problem": "Min-HCP",
        "instance": instance_path,
        "solver": "Gurobi",
        "status": model.Status,
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val

        # Extract hyperplane parameters and assignments
        hyperplanes = []
        assignments = [[] for _ in range(n)]

        for j in range(K):
            if y[j].X > 0.5:
                wj = [w[j, l].X for l in range(d)]
                w0j = w0[j].X
                assigned_points = [i for i in range(n) if D[i, j].X > 0.5]
                hyperplanes.append({
                    "w": wj,
                    "w0": w0j,
                    "assigned_points": assigned_points
                })
                for i in assigned_points:
                    assignments[i].append(len(hyperplanes) - 1)

        result["num_hyperplanes"] = len(hyperplanes)
        result["hyperplanes"] = hyperplanes
        result["point_assignments"] = assignments

        print(f"\nSolution found: {len(hyperplanes)} hyperplanes (objective = {obj_val})")
        print(f"Solver status: {model.Status}")
        if model.Status == GRB.TIME_LIMIT:
            print("(Time limit reached - returning best feasible solution)")
        if hasattr(model, 'MIPGap'):
            try:
                print(f"MIP Gap: {model.MIPGap * 100:.2f}%")
            except Exception:
                pass
    else:
        result["objective_value"] = None
        result["num_hyperplanes"] = None
        print("No feasible solution found.")

    # Write solution
    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve Min-HCP using Gurobi (MINLP with unit-norm constraint)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")

    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_min_hcp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
