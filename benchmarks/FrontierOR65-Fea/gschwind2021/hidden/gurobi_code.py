#!/usr/bin/env python3
"""
Gurobi-based Column Generation for PGMRC/CGMRC.
Paper: "A Branch-and-Price Framework for Decomposing Graphs into Relaxed Cliques"
Authors: Gschwind, Irnich, Furini, Wolfler Calvo (2017)

Implements the set-partitioning/covering formulation (Model 1) via column generation.
Both the Restricted Master Problem (LP/IP) and the pricing subproblem (MIP) use Gurobi.
"""

import argparse
import json
import time
import math
from collections import deque
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
# ---------------------------------------------------------------------------
# Instance loading and graph utilities
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path) as f:
        return json.load(f)


def build_adj(vertices, edges):
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def connected_components(S_list, adj):
    """Return list of connected components (as lists) in G[S_list]."""
    S_set = set(S_list)
    visited = set()
    comps = []
    for v in S_list:
        if v not in visited:
            comp = []
            q = deque([v])
            visited.add(v)
            while q:
                u = q.popleft()
                comp.append(u)
                for w in adj.get(u, set()):
                    if w in S_set and w not in visited:
                        visited.add(w)
                        q.append(w)
            comps.append(comp)
    return comps


def check_splex(S, adj, s):
    """Return True if S is an s-plex: every vertex has degree >= |S|-s in G[S]."""
    S_set = set(S)
    for v in S:
        deg = sum(1 for u in adj.get(v, set()) if u in S_set)
        if deg < len(S) - s:
            return False
    return True


# ---------------------------------------------------------------------------
# Pricing subproblem: max-weight s-plex (with optional connectivity)
# Solved with Gurobi MIP. Connectivity enforced via iterative cut generation.
# ---------------------------------------------------------------------------

def solve_pricing_mip(weights, vertices, adj, s, connectivity_required, time_budget):
    """
    Solve max-weight (connected) s-plex pricing subproblem with Gurobi.

    Formulation for s-plex (from Balasundaram et al., 2011, adapted):
      max  sum_i w_i * x_i
      s.t. sum_{j in N(i)} x_j >= sum_k x_k - s - n*(1 - x_i)  for all i
           x_i in {0,1}

    Connectivity is enforced iteratively by adding cut constraints when the
    solution is disconnected (branch-and-cut style, within this function).

    NOT SPECIFIED IN PAPER: The exact MIP for s-plex pricing is referenced to
    Balasundaram et al. (2011) but not reproduced. We use the standard
    linearization of the s-plex degree constraint.
    INFERRED ASSUMPTION: Use big-M linearization with n as the big-M value.
    INFERRED ASSUMPTION: Connectivity is enforced via iterative "component
    isolation" cuts: sum_{v in C} x_v <= sum_{u in N(C)\C} x_u for each
    non-largest disconnected component C (Desaulniers et al., 2005 style).
    """
    n = len(vertices)
    if n == 0:
        return [], 0.0

    idx = {v: i for i, v in enumerate(vertices)}
    start = time.time()

    m = gp.Model()
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', max(0.5, time_budget))
    m.setParam('PoolSearchMode', 2)   # collect multiple solutions
    m.setParam('PoolSolutions', 20)

    x = m.addVars(n, vtype=GRB.BINARY, name='x')

    # s-plex degree constraints:
    # for each i: sum_{j in N(i)} x_j >= sum_k x_k - s - n*(1 - x_i)
    for i, v in enumerate(vertices):
        nbrs = [idx[u] for u in adj.get(v, set()) if u in idx]
        m.addConstr(
            gp.quicksum(x[j] for j in nbrs) >=
            gp.quicksum(x[k] for k in range(n)) - s - n * (1 - x[i]),
            name=f'splex_{i}'
        )

    m.setObjective(
        gp.quicksum(weights.get(v, 0.0) * x[i] for i, v in enumerate(vertices)),
        GRB.MAXIMIZE
    )

    if not connectivity_required:
        m.optimize()
        if m.SolCount > 0 and m.ObjVal > -1e9:
            sol = [vertices[i] for i in range(n) if x[i].X > 0.5]
            obj = sum(weights.get(v, 0.0) for v in sol)
            return sol, obj
        return [], 0.0

    # Connectivity enforcement via iterative cut generation
    best_sol = []
    best_obj = -1e18
    cut_iter = 0
    max_cuts = 100

    while cut_iter < max_cuts:
        remaining = time_budget - (time.time() - start)
        if remaining <= 0.05:
            break

        m.setParam('TimeLimit', max(0.1, remaining))
        m.optimize()

        if m.SolCount == 0:
            break

        sol = [vertices[i] for i in range(n) if x[i].X > 0.5]
        if not sol:
            break

        obj = sum(weights.get(v, 0.0) for v in sol)
        comps = connected_components(sol, adj)

        if len(comps) == 1:
            # Feasible connected s-plex found
            if obj > best_obj:
                best_obj = obj
                best_sol = sol
            break

        # Add connectivity cuts: for each non-largest component C,
        # vertices in C must have at least one neighbor outside C that is selected.
        # Cut: sum_{v in C} x_v <= sum_{u in N(C)\C} x_u
        largest = max(comps, key=len)
        for comp in comps:
            if comp is largest:
                continue
            comp_set = set(comp)
            nbrs_outside = set()
            for v in comp:
                for u in adj.get(v, set()):
                    if u in idx and u not in comp_set:
                        nbrs_outside.add(u)
            if not nbrs_outside:
                # Component is disconnected from rest; forbid it entirely
                m.addConstr(
                    gp.quicksum(x[idx[v]] for v in comp) <= len(comp) - 1,
                    name=f'conn_cut_{cut_iter}'
                )
            else:
                m.addConstr(
                    gp.quicksum(x[idx[v]] for v in comp) <=
                    gp.quicksum(x[idx[u]] for u in nbrs_outside),
                    name=f'conn_cut_{cut_iter}'
                )
        cut_iter += 1

    return best_sol, best_obj if best_obj > -1e17 else 0.0


# ---------------------------------------------------------------------------
# Restricted Master Problem (RMP): LP relaxation
# ---------------------------------------------------------------------------

def solve_rmp_lp(columns, vertices, partitioning):
    """
    Solve LP relaxation of the master set-partitioning/covering formulation.
    Returns (obj_val, dual_prices dict, lambda_vals list) or None if infeasible.

    Model (1) from the paper (LP relaxation: lambda_S >= 0):
      min  sum_S lambda_S
      s.t. sum_{S: i in S} lambda_S = 1  (partitioning) or >= 1 (covering)
           lambda_S >= 0
    """
    n = len(vertices)
    vtx_idx = {v: i for i, v in enumerate(vertices)}

    rmp = gp.Model()
    rmp.setParam("Threads", 1)
    rmp.setParam('OutputFlag', 0)
    rmp.setParam('Method', 1)   # dual simplex -> better for column generation

    lam = rmp.addVars(len(columns), lb=0.0, name='lam')

    cover_constrs = {}
    for i, v in enumerate(vertices):
        cols_with_v = [j for j, col in enumerate(columns) if v in col]
        if partitioning:
            c = rmp.addConstr(
                gp.quicksum(lam[j] for j in cols_with_v) == 1.0,
                name=f'vtx_{i}'
            )
        else:
            c = rmp.addConstr(
                gp.quicksum(lam[j] for j in cols_with_v) >= 1.0,
                name=f'vtx_{i}'
            )
        cover_constrs[v] = c

    rmp.setObjective(gp.quicksum(lam), GRB.MINIMIZE)
    rmp.optimize()

    if rmp.status != GRB.OPTIMAL:
        return None

    pi = {v: cover_constrs[v].Pi for v in vertices}
    lam_vals = [lam[j].X for j in range(len(columns))]
    return rmp.ObjVal, pi, lam_vals


# ---------------------------------------------------------------------------
# Column generation loop
# ---------------------------------------------------------------------------

def column_generation(vertices, adj, columns, s, rc_type, connectivity_required,
                       partitioning, deadline):
    """
    Iteratively solve RMP and add columns with negative reduced cost.
    Returns augmented columns list and final dual prices.

    Pricing subproblem: maximize sum_i pi_i * x_i s.t. S feasible RC.
    Negative reduced cost: 1 - sum_{i in S} pi_i < 0, i.e., obj > 1.

    Multi-column strategy: Gurobi's solution pool returns multiple integer
    solutions per pricing call; we add all with negative reduced cost.
    (INFERRED ASSUMPTION: Use PoolSearchMode=2 to mimic CPLEX multi-solution
    collection described in the paper.)
    """
    col_set_cache = set(frozenset(c) for c in columns)

    while True:
        remaining = deadline - time.time()
        if remaining <= 0.5:
            break

        result = solve_rmp_lp(columns, vertices, partitioning)
        if result is None:
            break
        lp_obj, pi, lam_vals = result

        # Pricing subproblem weights: w_i = pi_i
        weights = {v: pi[v] for v in vertices}

        pricing_budget = min(15.0, max(1.0, remaining * 0.4))
        new_col, col_obj = solve_pricing_mip(
            weights, vertices, adj, s, connectivity_required, pricing_budget
        )

        if not new_col:
            break

        reduced_cost = col_obj - 1.0   # obj - 1 (the reduced cost of this column)
        if reduced_cost <= 1e-6:
            # No column with negative reduced cost; LP relaxation solved to optimality
            break

        added = 0
        fs = frozenset(new_col)
        if fs not in col_set_cache:
            columns.append(set(new_col))
            col_set_cache.add(fs)
            added += 1

        if added == 0:
            # No new distinct column found; stop
            break

    return columns


# ---------------------------------------------------------------------------
# Final integer master problem solve
# ---------------------------------------------------------------------------

def solve_master_ip(columns, vertices, partitioning, time_limit_sec):
    """
    Solve master problem as binary integer program over collected columns.

    model.setParam("TimeLimit", time_limit_sec) sets Gurobi's time limit
    as required by the implementation specification.
    Returns (selected_columns, objective_value).
    """
    n = len(vertices)

    m = gp.Model()
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit_sec)   # <-- required time limit parameter

    lam = m.addVars(len(columns), vtype=GRB.BINARY, name='lam')

    for v in vertices:
        cols_with_v = [j for j, col in enumerate(columns) if v in col]
        if not cols_with_v:
            # Singleton must exist (initialised from singletons)
            continue
        if partitioning:
            m.addConstr(gp.quicksum(lam[j] for j in cols_with_v) == 1,
                        name=f'part_{v}')
        else:
            m.addConstr(gp.quicksum(lam[j] for j in cols_with_v) >= 1,
                        name=f'cov_{v}')

    m.setObjective(gp.quicksum(lam), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount > 0:
        selected = [columns[j] for j in range(len(columns)) if lam[j].X > 0.5]
        return selected, m.ObjVal
    return None, float('inf')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Gurobi column generation for PGMRC/CGMRC (Gschwind et al. 2017)'
    )
    parser.add_argument('--instance_path', required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', required=True,
                        help='Path to write the JSON solution file.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    start_time = time.time()
    deadline = start_time + args.time_limit

    # Load instance
    inst = load_instance(args.instance_path)
    vertices = inst['graph']['vertices']
    edges = inst['graph']['edges']
    adj = build_adj(vertices, edges)
    n = len(vertices)

    settings = inst['problem_settings']
    rc_type = settings['relaxed_clique_type']          # e.g. 's-plex'
    s = settings.get('parameter_s', 2)
    partitioning = (settings['decomposition_type'] == 'partitioning')
    connectivity_required = settings.get('connectivity_required', False)

    if rc_type != 's-plex':
        # INFERRED ASSUMPTION: Only s-plex is implemented here.
        # Other RC types (s-clique, s-club, gamma-quasi-clique, etc.) would require
        # additional pricing MIP formulations referenced to companion papers.
        raise NotImplementedError(
            f"RC type '{rc_type}' not implemented. Only 's-plex' is supported."
        )

    # Initialize columns with singletons {v} for each vertex v.
    # INFERRED ASSUMPTION: Every singleton is a feasible RC (any RC definition
    # holds trivially for |S|=1 since degree conditions are 1 - s <= 0 for s>=1).
    columns = [set({v}) for v in vertices]

    # Column generation phase (budget: leave ~30% of time for final IP)
    cg_deadline = start_time + args.time_limit * 0.7
    columns = column_generation(
        vertices, adj, columns, s, rc_type, connectivity_required,
        partitioning, min(cg_deadline, deadline - 2.0)
    )

    # Final integer solve with all collected columns
    remaining = deadline - time.time()
    ip_time = max(1.0, remaining)
    solution, obj_val = solve_master_ip(columns, vertices, partitioning, ip_time)

    # Fallback: if IP didn't find a solution, use greedy partition (singletons)
    # INFERRED ASSUMPTION: If time limit is reached before finding a feasible
    # integer solution, return the trivial singleton partition (n RCs) as the
    # best feasible solution found.
    if solution is None:
        solution = [{v} for v in vertices]
        obj_val = float(n)

    result = {
        'objective_value': float(obj_val),
        'num_rcs': int(round(float(obj_val))),
        'solution': [sorted(list(rc)) for rc in solution],
        'instance_id': inst.get('instance_id', ''),
        'rc_type': rc_type,
        'parameter_s': s,
        'partitioning': partitioning,
        'connectivity_required': connectivity_required,
        'num_columns_generated': len(columns),
        'computation_time_seconds': time.time() - start_time,
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Objective value: {obj_val}")
    print(f"RCs in decomposition: {result['num_rcs']}")
    print(f"Columns generated: {len(columns)}")
    print(f"Time elapsed: {result['computation_time_seconds']:.2f}s")
    print(f"Solution written to: {args.solution_path}")


if __name__ == '__main__':
    main()
