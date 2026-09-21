"""
Gurobi implementation of the BIP formulation for Approximately Submodular
Function Maximization (ASFM) from:
  "An efficient branch-and-cut algorithm for approximately submodular
   function maximization" (Uematsu, Umetani, Kawahara, 2019)

Uses the Modified Constraint Generation (MCG) algorithm (Section 4.1)
to solve BIP(Q) (Eq. 11) iteratively.
"""

import argparse
import json
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
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
        pass# ===========================================================================
# Set function oracle
# ===========================================================================

def build_set_function(instance):
    """Return a callable f(S) based on instance type."""
    inst_type = instance["instance_type"]
    data = np.array(instance["data_matrix"])  # m x n
    m, n = data.shape

    if inst_type == "LOC":
        # Facility Location: f(S) = sum_{i in M} max_{j in S} g_{ij}
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            return float(np.sum(np.max(data[:, cols], axis=1)))
        return f, n, m

    elif inst_type == "COV":
        # Weighted Coverage: f(S) = sum_{i in M} w_i * max_{j in S} a_{ij}
        # data_matrix stores a_{ij} (binary coverage), node_weights stores w_i
        weights = np.array(instance["node_weights"])
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            covered = np.max(data[:, cols], axis=1)
            return float(np.dot(weights, covered))
        return f, n, m

    elif inst_type == "INF":
        # Bipartite Influence: f(S) = sum_{i in M} (1 - prod_{j in S} (1 - q_{ij}))
        # data_matrix stores q_{ij}
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            survival = np.prod(1.0 - data[:, cols], axis=1)
            return float(np.sum(1.0 - survival))
        return f, n, m

    else:
        # **NOT SPECIFIED IN PAPER**: Unknown instance type.
        # Inferred assumption: treat data_matrix as LOC-style.
        def f(S):
            if len(S) == 0:
                return 0.0
            cols = list(S)
            return float(np.sum(np.max(data[:, cols], axis=1)))
        return f, n, m


def marginal_gain(f, S, i):
    """f({i} | S) = f(S ∪ {i}) - f(S)"""
    return f(S | {i}) - f(S)


# ===========================================================================
# Greedy algorithm
# ===========================================================================

def greedy(f, n, k):
    """
    Standard greedy algorithm (Minoux 1978, Nemhauser et al. 1978).
    Returns (S, order) where order is the sequence of elements added.
    """
    S = set()
    order = []
    for _ in range(k):
        best_i, best_gain = -1, -float('inf')
        for i in range(n):
            if i not in S:
                g = marginal_gain(f, S, i)
                if g > best_gain:
                    best_gain = g
                    best_i = i
        S.add(best_i)
        order.append(best_i)
    return S, order


# ===========================================================================
# BIP(Q) solver using Gurobi — Modified Constraint Generation (MCG)
# ===========================================================================

def solve_bip_q(f, n, k, gamma_bar, Q_plus, time_remaining):
    """
    Solve BIP(Q^+) (Eq. 11) using Gurobi.

    For each S in Q_plus:
        j = argmax_{i in N\\S} f({i}|S)
        z <= f(S) + f({j}|S)*y_j + sum_{i in N\\(S∪{j})} (1/gamma_bar)*f({i}|S)*y_i

    Returns (solution_set, objective_value, y_values)
    """
    model = gp.Model("BIP_Q")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", max(1, time_remaining))

    # Variables
    z = model.addVar(lb=-GRB.INFINITY, name="z")
    y = {}
    for i in range(n):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    model.setObjective(z, GRB.MAXIMIZE)

    # Cardinality constraint
    model.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, "cardinality")

    # Constraints for each S in Q_plus
    N_set = set(range(n))
    for idx, S in enumerate(Q_plus):
        S_frozen = frozenset(S)
        fS = f(S_frozen)

        # j = argmax_{i in N\S} f({i}|S)
        complement = N_set - S_frozen
        if len(complement) == 0:
            # S = N, constraint is z <= f(N)
            model.addConstr(z <= fS, f"cut_{idx}")
            continue

        best_j, best_mg = -1, -float('inf')
        marginals = {}
        for i in complement:
            mg = marginal_gain(f, S_frozen, i)
            marginals[i] = mg
            if mg > best_mg:
                best_mg = mg
                best_j = i

        # Build constraint
        expr = fS + marginals[best_j] * y[best_j]
        for i in complement:
            if i != best_j:
                expr += (1.0 / gamma_bar) * marginals[i] * y[i]

        model.addConstr(z <= expr, f"cut_{idx}")

    model.optimize()

    if model.SolCount > 0:
        sol = set()
        y_vals = {}
        for i in range(n):
            y_vals[i] = y[i].X
            if y[i].X > 0.5:
                sol.add(i)
        return sol, model.ObjVal, y_vals
    else:
        return None, -float('inf'), {}


def solve_mcg(f, n, k, gamma_bar, time_limit):
    """
    Modified Constraint Generation Algorithm (MCG) from Section 4.1.
    Iteratively solves BIP(Q) and adds violated constraints.
    """
    start_time = time.time()

    # Step 1: Greedy initial solution
    S0, order = greedy(f, n, k)
    S_star = set(S0)
    f_star = f(frozenset(S_star))

    # Q = {S^(0)}, Q^+ includes greedy sub-solutions
    Q = [set(S0)]
    Q_plus_list = []
    for i in range(k + 1):
        Q_plus_list.append(set(order[:i]))
    # Q_plus includes the greedy sub-solutions
    Q_plus = list(Q_plus_list)

    t = 1
    while True:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining <= 1:
            break

        # Step 2: Solve BIP(Q^+)
        sol, z_t, y_vals = solve_bip_q(f, n, k, gamma_bar, Q_plus, remaining)

        if sol is None:
            break

        # Step 3: Update incumbent
        f_sol = f(frozenset(sol))
        if f_sol > f_star:
            S_star = set(sol)
            f_star = f_sol

        # Step 4: Check convergence
        if abs(z_t - f_star) < 1e-9:
            break

        if z_t <= f_star + 1e-9:
            break

        # Add S^(t) to Q and Q^+
        Q.append(set(sol))
        Q_plus.append(set(sol))

        t += 1

        # Safety: avoid infinite loops
        # The number of iterations is bounded by C(n,k)
        if t > 10000:
            break

    return S_star, f_star


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for ASFM via Modified Constraint Generation (MCG)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    with open(args.instance_path, "r") as fp:
        instance = json.load(fp)

    n = instance["n"]
    k = instance["k"]
    gamma_bar = instance["gamma"]  # lower bound on submodular ratio used as gamma_bar

    f, n_actual, m = build_set_function(instance)
    assert n_actual == n, f"Mismatch: instance n={n}, data columns={n_actual}"

    S_star, f_star = solve_mcg(f, n, k, gamma_bar, args.time_limit)

    solution = {
        "objective_value": f_star,
        "selected_elements": sorted(list(S_star)),
        "instance_id": instance.get("instance_id", None),
    }

    with open(args.solution_path, "w") as fp:
        json.dump(solution, fp, indent=2)

    print(f"Objective value: {f_star}")
    print(f"Selected elements: {sorted(list(S_star))}")


if __name__ == "__main__":
    main()
