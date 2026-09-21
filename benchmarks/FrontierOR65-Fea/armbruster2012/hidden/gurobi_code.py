"""
Minimum Graph Bisection ILP solver using Gurobi.

Implements the edge-based ILP formulation (Equation 1) from:
  Armbruster et al. (2012) - "LP and SDP branch-and-cut algorithms
  for the minimum graph bisection problem"

Triangle (cycle) inequalities on the augmented star graph are added upfront.
Since the augmented graph contains a star K_{1,n-1} from node 0, every pair
of nodes (i, j) with i,j != 0 that share an edge forms a triangle (0, i, j).
Triangle inequalities on this augmented graph, combined with binary variables,
are sufficient to ensure valid cut solutions.
"""

import argparse
import json
from collections import defaultdict

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


def solve_bisection(instance, time_limit):
    n = instance["num_nodes"]
    edges_list = [tuple(e) for e in instance["edges"]]
    edge_weights = instance["edge_weights"]
    node_weights = instance["node_weights"]
    F = instance["bisection_capacity_F"]

    # Build edge set and adjacency structure from original graph
    edge_set = set()
    edge_cost = {}
    for idx, (i, j) in enumerate(edges_list):
        key = (min(i, j), max(i, j))
        edge_set.add(key)
        edge_cost[key] = edge_weights[idx]

    # Add star edges from node 0 to all other nodes (zero cost if not present)
    for j in range(1, n):
        key = (0, j)
        if key not in edge_set:
            edge_set.add(key)
            edge_cost[key] = 0

    # Build adjacency list for the augmented graph
    adj = defaultdict(set)
    for i, j in edge_set:
        adj[i].add(j)
        adj[j].add(i)

    # Build neighbor-pair index for triangle enumeration
    # For each node, store its neighbors as a set for O(1) lookup
    neighbor_set = {v: set(adj[v]) for v in range(n)}

    # Pre-compute all triangles (triples of mutually adjacent nodes)
    triangles = []
    for i, j in edge_set:
        common = neighbor_set[i] & neighbor_set[j]
        for k in common:
            tri = tuple(sorted([i, j, k]))
            triangles.append(tri)
    # Deduplicate
    triangles = list(set(triangles))

    # Create model
    model = gp.Model("MinGraphBisection")
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1

    # Create binary edge variables
    all_edges = sorted(edge_set)
    y = {}
    for i, j in all_edges:
        y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}",
                                obj=edge_cost[(i, j)])

    model.setAttr("ModelSense", GRB.MINIMIZE)
    model.update()

    # Helper to get variable for edge (a, b) in canonical order
    def yvar(a, b):
        if a > b:
            a, b = b, a
        return y[a, b]

    # Capacity constraint 1: sum_{i=2}^{n} f_i * y_{0,i} <= F
    # (weight of cluster separated from node 0)
    model.addConstr(
        gp.quicksum(node_weights[j] * yvar(0, j) for j in range(1, n)) <= F,
        name="cap_separated"
    )

    # Capacity constraint 2: f_0 + sum_{i=2}^{n} f_i * (1 - y_{0,i}) <= F
    # (weight of cluster containing node 0)
    model.addConstr(
        node_weights[0] + gp.quicksum(
            node_weights[j] * (1 - yvar(0, j)) for j in range(1, n)
        ) <= F,
        name="cap_containing"
    )

    # Add all triangle inequalities upfront as regular constraints
    for idx_t, (a, b, c) in enumerate(triangles):
        vab = yvar(a, b)
        vbc = yvar(b, c)
        vac = yvar(a, c)
        # |D|=1: y_ab - y_ac - y_bc <= 0 (and permutations)
        model.addConstr(vab - vac - vbc <= 0, name=f"tri_{idx_t}_D1a")
        model.addConstr(vac - vab - vbc <= 0, name=f"tri_{idx_t}_D1b")
        model.addConstr(vbc - vab - vac <= 0, name=f"tri_{idx_t}_D1c")
        # |D|=3: y_ab + y_ac + y_bc <= 2
        model.addConstr(vab + vac + vbc <= 2, name=f"tri_{idx_t}_D3")
    model.update()

    # Optimize (triangle inequalities are already added upfront as regular
    # constraints; no lazy constraint callback is needed since the augmented
    # graph's triangle inequalities plus binary variables fully characterize
    # valid cuts)
    model.optimize()

    # Extract solution
    result = {
        "objective_value": None,
        "partition_S": [],
        "partition_complement": [],
        "status": model.Status,
        "num_nodes": n,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Determine partition from star edge variables y_{0,i}
        # y_{0,i} = 1 means node i is in opposite cluster from node 0
        set_with_0 = [0]  # Node 0 is always in its own cluster
        set_without_0 = []

        for j in range(1, n):
            val = yvar(0, j).X
            if val > 0.5:
                set_without_0.append(j)
            else:
                set_with_0.append(j)

        result["partition_S"] = set_with_0
        result["partition_complement"] = set_without_0

        # Also compute the actual cut value from original edges only
        cut_value = 0.0
        for idx, (i, j) in enumerate(edges_list):
            key = (min(i, j), max(i, j))
            if key in y:
                val = y[key].X
                if val > 0.5:
                    cut_value += edge_weights[idx]
        result["cut_value_original_edges"] = cut_value

    if model.Status == GRB.OPTIMAL:
        result["status_str"] = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        result["status_str"] = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        result["status_str"] = "INFEASIBLE"
    else:
        result["status_str"] = f"OTHER_{model.Status}"

    if hasattr(model, "MIPGap") and model.SolCount > 0:
        try:
            result["mip_gap"] = model.MIPGap
        except Exception:
            pass

    if hasattr(model, "ObjBound"):
        try:
            result["best_bound"] = model.ObjBound
        except Exception:
            pass

    result["runtime"] = model.Runtime

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve Minimum Graph Bisection via ILP (Armbruster et al. 2012)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to output solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Gurobi time limit in seconds (default: 3600)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_bisection(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {result['status_str']}")
    print(f"Objective: {result['objective_value']}")
    if result["objective_value"] is not None:
        print(f"Partition S size: {len(result['partition_S'])}")
        print(f"Partition complement size: {len(result['partition_complement'])}")
    print(f"Runtime: {result['runtime']:.2f}s")


if __name__ == "__main__":
    main()
