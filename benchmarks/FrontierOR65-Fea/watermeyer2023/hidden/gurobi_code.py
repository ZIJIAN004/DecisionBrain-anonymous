#!/usr/bin/env python3
"""
Solves the RCPSP/max-pi (Resource-Constrained Project Scheduling Problem
with generalized precedence relations and partially renewable resources)
using a time-indexed MIP formulation solved by Gurobi.

Reference: Watermeyer et al. (2023)
"""

import argparse
import json
import math
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


def load_instance(path):
    """Load a problem instance from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_distance_matrix(num_nodes, temporal_constraints):
    """
    Build the arc set and run Floyd-Warshall to compute longest-path
    distances between all pairs of nodes.

    Each temporal constraint gives an arc (from, to) with weight delta,
    representing S_to >= S_from + delta.

    Returns the distance matrix d where d[i][j] is the longest-path
    distance from i to j (or -inf if no path exists).
    """
    NEG_INF = -float("inf")
    d = [[NEG_INF] * num_nodes for _ in range(num_nodes)]

    # Distance from a node to itself is 0
    for i in range(num_nodes):
        d[i][i] = 0

    # Add arcs from temporal constraints
    for tc in temporal_constraints:
        i = tc["from"]
        j = tc["to"]
        delta = tc["delta"]
        # Arc (i, j) with weight delta: S_j >= S_i + delta
        # Keep the maximum weight if multiple arcs exist
        if delta > d[i][j]:
            d[i][j] = delta

    # Floyd-Warshall for longest paths
    for k in range(num_nodes):
        for i in range(num_nodes):
            if d[i][k] == NEG_INF:
                continue
            for j in range(num_nodes):
                if d[k][j] == NEG_INF:
                    continue
                candidate = d[i][k] + d[k][j]
                if candidate > d[i][j]:
                    d[i][j] = candidate

    return d


def compute_resource_usage(start_time, processing_time, time_periods_set):
    """
    Compute r^u_{ik}(t) = |{p in Pi_k : t < p <= t + p_i}|

    This counts periods in Pi_k that fall in the half-open interval
    (start_time, start_time + processing_time].
    """
    count = 0
    for p in time_periods_set:
        if start_time < p <= start_time + processing_time:
            count += 1
    return count


def solve(instance, time_limit):
    """
    Build and solve the time-indexed MIP formulation for RCPSP/max-pi.

    Returns a dict with solution information.
    """
    num_nodes = instance["num_nodes"]
    deadline = instance["deadline"]
    processing_times = instance["processing_times"]
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]

    n_plus_1 = num_nodes - 1  # index of the dummy end activity

    # --- Step 1: Floyd-Warshall for earliest/latest start times ---
    dist = build_distance_matrix(num_nodes, temporal_constraints)

    NEG_INF = -float("inf")

    # ES_i = d[0][i], LS_i = deadline - d[i][n+1] (or use -d[i][0] with care)
    # More robust: ES_i = max(0, d[0][i])
    #              LS_i = min(deadline, deadline - d[i][n_plus_1]) if path exists
    # But the standard approach:
    #   ES_i = d[0][i]  (longest path from source to i)
    #   LS_i = deadline + d[0][n+1] - d[i][n+1]  ... no.
    # Actually: LS_i = -d[i][0] if we consider the reverse.
    # Simpler: LS_i = deadline - d[i][n+1] ... but that doesn't account for
    # the processing time of the end node (which is 0).
    #
    # Standard formulation:
    #   ES_i = d[0][i]
    #   LS_i = d[0][n+1] - d[i][n+1]  ... no, this doesn't work either for
    #   max_lag constraints.
    #
    # The correct way using Floyd-Warshall:
    #   ES_i = d[0][i]
    #   LS_i = -d[i][0]  (since d[i][0] represents the longest path from i
    #          back to 0, and S_0 >= S_i + d[i][0] => S_i <= -d[i][0])
    # But we also need LS_i <= deadline - p_i to ensure the activity finishes
    # by the deadline.
    #
    # Actually, we should bound LS_i using the end node too:
    #   From the constraint S_{n+1} <= deadline and S_{n+1} >= S_i + d[i][n+1]:
    #   S_i <= deadline - d[i][n+1]

    es = [0] * num_nodes
    ls = [0] * num_nodes

    for i in range(num_nodes):
        # Earliest start
        if dist[0][i] != NEG_INF:
            es[i] = max(0, dist[0][i])
        else:
            es[i] = 0

        # Latest start: minimum of bounds from source and sink
        ls_candidates = []

        # From S_0 = 0 and S_0 >= S_i + d[i][0]: S_i <= -d[i][0]
        if dist[i][0] != NEG_INF:
            ls_candidates.append(-dist[i][0])

        # From S_{n+1} <= deadline and S_{n+1} >= S_i + d[i][n+1]:
        # S_i <= deadline - d[i][n+1]
        if dist[i][n_plus_1] != NEG_INF:
            ls_candidates.append(deadline - dist[i][n_plus_1])

        # Fallback
        if ls_candidates:
            ls[i] = min(ls_candidates)
        else:
            ls[i] = deadline

        # Ensure activity finishes by deadline (real upper bound from the
        # deadline; ls must not exceed this).
        ls[i] = min(ls[i], deadline - processing_times[i])

        # NOTE: previously this block had `ls[i] = max(ls[i], es[i])` which
        # *silently overrode* a too-tight deadline by raising ls back up to
        # the critical-path es. That hides genuine infeasibility — the end
        # node would get x[end, es[end]] = 1 even when es[end] > deadline,
        # so makespan was free to exceed the deadline (the deadline became a
        # no-op). The correct behavior when ls < es is to flag the model as
        # infeasible (no feasible start time given the temporal/deadline
        # bounds), letting Gurobi confirm via the empty-range == 1 constraint
        # rather than papering over it here.
        if ls[i] < es[i]:
            # Leave ls[i] strictly less than es[i] so that the convexity
            # constraint sum_t x[i,t] = 1 over the empty range becomes 0 = 1
            # → Gurobi reports INFEASIBLE. We do NOT clamp ls up to es here.
            pass

    # Force source to start at 0
    es[0] = 0
    ls[0] = 0

    # --- Step 2: Build MIP ---
    model = gp.Model("RCPSP_max_pi")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 1)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # Binary variables x[i][t]: activity i starts at time t
    x = {}
    for i in range(num_nodes):
        for t in range(es[i], ls[i] + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # --- Constraint 1: Assignment ---
    # Each activity starts exactly once
    for i in range(num_nodes):
        model.addConstr(
            gp.quicksum(x[i, t] for t in range(es[i], ls[i] + 1)) == 1,
            name=f"assign_{i}",
        )

    # --- Constraint 2: Temporal constraints ---
    # For each arc (i, j) with weight delta: S_j >= S_i + delta
    # i.e., sum_t t*x[j,t] >= sum_t t*x[i,t] + delta
    for tc in temporal_constraints:
        i = tc["from"]
        j = tc["to"]
        delta = tc["delta"]
        model.addConstr(
            gp.quicksum(t * x[j, t] for t in range(es[j], ls[j] + 1))
            >= gp.quicksum(t * x[i, t] for t in range(es[i], ls[i] + 1)) + delta,
            name=f"temp_{i}_{j}",
        )

    # --- Constraint 3: Resource constraints ---
    # For each partially renewable resource k:
    # sum_i sum_t r^u_{ik}(t) * r^d_{ik} * x[i,t] <= R_k
    for res in resources:
        k = res["resource_id"]
        capacity = res["capacity"]
        time_periods = res["time_periods"]
        time_periods_set = set(time_periods)
        demands = res["demands"]  # demands[i] = r^d_{ik}

        terms = []
        for i in range(num_nodes):
            r_d = demands[i]
            if r_d == 0:
                continue
            p_i = processing_times[i]
            for t in range(es[i], ls[i] + 1):
                r_u = compute_resource_usage(t, p_i, time_periods_set)
                if r_u == 0:
                    continue
                coeff = r_u * r_d
                terms.append(coeff * x[i, t])

        if terms:
            model.addConstr(
                gp.quicksum(terms) <= capacity,
                name=f"resource_{k}",
            )

    # --- Constraint 4: Fix source start ---
    model.addConstr(x[0, 0] == 1, name="fix_source")

    # --- Objective: minimize makespan (start time of end node) ---
    model.setObjective(
        gp.quicksum(t * x[n_plus_1, t] for t in range(es[n_plus_1], ls[n_plus_1] + 1)),
        GRB.MINIMIZE,
    )

    # --- Solve ---
    wall_start = time.time()
    model.optimize()
    wall_elapsed = time.time() - wall_start

    # --- Extract solution ---
    result = {
        "solver": "gurobi",
        "wall_time_seconds": round(wall_elapsed, 3),
    }

    if model.SolCount > 0:
        # Extract start times
        start_times = [0] * num_nodes
        for i in range(num_nodes):
            for t in range(es[i], ls[i] + 1):
                if x[i, t].X > 0.5:
                    start_times[i] = t
                    break

        obj_val = round(model.ObjVal)
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["objective_value"] = obj_val
        result["start_times"] = start_times
        result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
    else:
        status_map = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            GRB.UNBOUNDED: "unbounded",
            GRB.TIME_LIMIT: "time_limit",
        }
        result["status"] = status_map.get(model.Status, f"unknown_{model.Status}")
        result["objective_value"] = None
        result["start_times"] = None

    result["num_variables"] = model.NumVars
    result["num_constraints"] = model.NumConstrs

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve RCPSP/max-pi using Gurobi MIP"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path for the output solution JSON (default: gurobi_solution_1.json)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Maximum solver time in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve(instance, args.time_limit)

    # Add instance metadata to output
    result["instance_path"] = args.instance_path
    if "instance_id" in instance:
        result["instance_id"] = instance["instance_id"]

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Status: {result['status']}")
    print(f"Objective value: {result['objective_value']}")
    if result.get("start_times"):
        print(f"Start times: {result['start_times']}")


if __name__ == "__main__":
    main()
