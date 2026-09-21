"""
Gurobi implementation of Model 1 (IP formulation) for the Selective Graph Coloring
problem (SEL-COL) from Seker, Ekim, Taskin (2021).

Model 1 uses binary variables y_k (color k used) and w_ik (vertex i gets color k),
with symmetry-breaking constraints y_k <= y_{k-1}.
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
    with open(path, "r") as f:
        return json.load(f)


def solve(instance, time_limit):
    num_vertices = instance["graph"]["num_vertices"]
    edges = [tuple(e) for e in instance["graph"]["edges"]]
    clusters = instance["partition"]["clusters"]
    P = instance["partition"]["num_clusters"]  # number of clusters = max colors needed

    V = list(range(num_vertices))

    model = gp.Model("SEL_COL_Model1")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # y_k: 1 if color k is used (k in 0..P-1)
    y = model.addVars(P, vtype=GRB.BINARY, name="y")
    # w_ik: 1 if vertex i is selected and assigned color k
    w = model.addVars(num_vertices, P, vtype=GRB.BINARY, name="w")

    # Objective (1a): minimize number of colors used
    model.setObjective(gp.quicksum(y[k] for k in range(P)), GRB.MINIMIZE)

    # Constraint (1b): w_ik <= y_k
    for i in V:
        for k in range(P):
            model.addConstr(w[i, k] <= y[k], name=f"link_{i}_{k}")

    # Constraint (1c): w_ik + w_jk <= 1 for each edge {i,j} and each color k
    for (i, j) in edges:
        for k in range(P):
            model.addConstr(w[i, k] + w[j, k] <= 1, name=f"edge_{i}_{j}_{k}")

    # Constraint (1d): exactly one vertex selected per cluster
    for p_idx, cluster in enumerate(clusters):
        model.addConstr(
            gp.quicksum(w[i, k] for i in cluster for k in range(P)) == 1,
            name=f"cluster_{p_idx}",
        )

    # Symmetry-breaking constraints (2): y_k <= y_{k-1}
    for k in range(1, P):
        model.addConstr(y[k] <= y[k - 1], name=f"symbreak_{k}")

    model.optimize()

    # Extract solution
    result = {
        "objective_value": None,
        "status": model.Status,
        "selected_vertices": {},
        "coloring": {},
    }

    if model.SolCount > 0:
        # Extract which vertex is selected per cluster and its color
        for p_idx, cluster in enumerate(clusters):
            for i in cluster:
                for k in range(P):
                    if w[i, k].X > 0.5:
                        result["selected_vertices"][str(p_idx)] = i
                        result["coloring"][str(i)] = k
                        break

        # Report objective as DISTINCT colors actually assigned (not sum y_k).
        # The model's sum y_k can exceed this when symmetry-breaking forces
        # y[k]=1 at low indices that aren't actually used by any vertex.
        # Checker reconstructs y[k] only from `coloring`, so report obj that
        # matches checker's view to avoid spurious obj-mismatch violations.
        # NOTE 2026-05-19: was `model.ObjVal` (= sum y_k including symbreak
        # padding), causing obj=59 reported vs obj=34 used = checker reject.
        distinct_colors = len(set(result["coloring"].values()))
        result["objective_value"] = float(distinct_colors)
    else:
        # No feasible solution found
        result["objective_value"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve SEL-COL using Model 1 (IP) with Gurobi"
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

    instance = load_instance(args.instance_path)
    result = solve(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
