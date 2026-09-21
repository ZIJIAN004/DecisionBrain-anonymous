"""
Gurobi-based exact solver for Multi-Objective Integer Programming (MOIP).

Implements the AIRA algorithm (Algorithm 3 from Pettersson & Ozlen 2020) using
the identity permutation to find all non-dominated objective vectors for a
multi-objective knapsack problem.

The algorithm recursively solves constrained lexicographic subproblems
OIP_s^n(k, (a_{s(k+1)}, ..., a_{s(n)})) using Gurobi as the single-objective
IP solver.
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


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_base_model(instance):
    """Build the base Gurobi model for the knapsack problem (constraints only)."""
    n_items = instance["num_items"]
    weights = instance["weights"]
    capacity = instance["capacity"]

    model = gp.Model("moip_knapsack")
    model.setParam("OutputFlag", 0)

    # Binary decision variables
    x = model.addVars(n_items, vtype=GRB.BINARY, name="x")

    # Knapsack capacity constraint: sum(w_i * x_i) <= capacity
    model.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n_items)) <= capacity,
        name="capacity"
    )

    model.update()
    return model, x


def solve_single_objective_ip(instance, obj_index, upper_bounds, perm, time_limit_remaining):
    """
    Solve a single-objective IP:
      min f_{perm[0]}(x)
      s.t. Ax <= C
           f_{perm[i]}(x) <= upper_bounds[perm[i]] for i in {1, ..., n-1}
           x in {0,1}^c

    Returns the solution x and objective values, or None if infeasible.
    """
    if time_limit_remaining <= 0:
        return None

    n_items = instance["num_items"]
    obj_coeffs = instance["objective_coefficients"]
    n_obj = instance["num_objectives"]

    model, x = build_base_model(instance)
    model.setParam("TimeLimit", max(1, time_limit_remaining))
    model.setParam("Threads", 1)

    # Set objective: minimize f_{perm[0]}
    # Since this is a knapsack where we MINIMIZE, and the paper says minimize,
    # we minimize the sum of selected item coefficients.
    primary_obj = perm[obj_index]
    model.setObjective(
        gp.quicksum(obj_coeffs[primary_obj][i] * x[i] for i in range(n_items)),
        GRB.MINIMIZE
    )

    # Add upper bound constraints for constrained objectives
    for idx in range(n_obj):
        obj_id = perm[idx]
        if obj_id in upper_bounds and upper_bounds[obj_id] < float('inf'):
            model.addConstr(
                gp.quicksum(obj_coeffs[obj_id][i] * x[i] for i in range(n_items)) <= upper_bounds[obj_id],
                name=f"ub_obj_{obj_id}"
            )

    model.optimize()

    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        sol = [int(round(x[i].X)) for i in range(n_items)]
        obj_vals = []
        for o in range(n_obj):
            val = sum(obj_coeffs[o][i] * sol[i] for i in range(n_items))
            obj_vals.append(val)
        return {"solution": sol, "obj_vals": obj_vals}
    else:
        return None


def is_dominated(obj_a, obj_b):
    """Check if obj_a is dominated by obj_b (all minimize)."""
    all_leq = all(b <= a for a, b in zip(obj_a, obj_b))
    any_lt = any(b < a for a, b in zip(obj_a, obj_b))
    return all_leq and any_lt


def filter_dominated(nd_set):
    """Remove dominated solutions from the set."""
    filtered = []
    for i, sol_i in enumerate(nd_set):
        dominated = False
        for j, sol_j in enumerate(nd_set):
            if i != j and is_dominated(sol_i["obj_vals"], sol_j["obj_vals"]):
                dominated = True
                break
        if not dominated:
            filtered.append(sol_i)
    return filtered


def aira_worker(instance, perm, k, upper_bounds, start_time, time_limit, prev_solutions=None):
    """
    Algorithm 3: Worker thread (sequential version for Gurobi baseline).

    Solves OIP_s^n(k, (a_{s(k+1)}, ..., a_{s(n)})) where:
      - perm defines the ordering s
      - k is the number of free objectives (1-indexed)
      - upper_bounds maps objective index -> upper bound value

    Returns list of non-dominated solutions.
    """
    elapsed = time.time() - start_time
    if elapsed >= time_limit:
        return []

    n_obj = instance["num_objectives"]
    nd_k = []

    # Step 2: Relaxation reuse check
    # If previous solutions all satisfy current bounds, reuse them
    if prev_solutions is not None:
        all_satisfy = True
        for sol in prev_solutions:
            for idx in range(k, n_obj):
                obj_id = perm[idx]
                if obj_id in upper_bounds and sol["obj_vals"][obj_id] > upper_bounds[obj_id]:
                    all_satisfy = False
                    break
            if not all_satisfy:
                break
        if all_satisfy and len(prev_solutions) > 0:
            return prev_solutions

    # Step 3: Base case (k == 1)
    if k == 1:
        time_remaining = time_limit - (time.time() - start_time)
        result = solve_single_objective_ip(instance, 0, upper_bounds, perm, time_remaining)
        if result is not None:
            nd_k = [result]
        return nd_k

    # Step 4: Recursive case (k > 1)
    # Let a_{s(k)} = infinity (index k-1 in 0-based perm)
    current_obj = perm[k - 1]  # The k-th objective in permuted order (0-based: k-1)

    # Create bounds for subproblem: add current_obj with bound = infinity
    sub_bounds = dict(upper_bounds)
    sub_bounds[current_obj] = float('inf')

    # Solve P = OIP_s^n(k-1, (a_{s(k)}, a_{s(k+1)}, ..., a_{s(n)}))
    prev_sol = None
    Y = aira_worker(instance, perm, k - 1, sub_bounds, start_time, time_limit, prev_sol)

    while len(Y) > 0:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        # Step 4a: Collect solutions
        nd_k.extend(Y)

        # Step 4b: Update bound on f_{s(k)}
        # a_{s(k)} = max{f_{s(k)}(x) | x in Y} - 1
        # We subtract 1 because variables are integer, so we need strictly less
        max_val = max(sol["obj_vals"][current_obj] for sol in Y)
        new_bound = max_val - 1  # Integer variables: strict < is equivalent to <= (val-1)

        if current_obj in sub_bounds and sub_bounds[current_obj] != float('inf'):
            # If we already have a bound, take the minimum (tighter)
            sub_bounds[current_obj] = min(sub_bounds[current_obj], new_bound)
        else:
            sub_bounds[current_obj] = new_bound

        # Step 4d: Re-solve with updated bound
        prev_sol = Y
        Y = aira_worker(instance, perm, k - 1, sub_bounds, start_time, time_limit, prev_sol)

    # Filter dominated solutions
    nd_k = filter_dominated(nd_k)
    return nd_k


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi-based AIRA solver for Multi-Objective Integer Programming"
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
    n_obj = instance["num_objectives"]

    start_time = time.time()

    # Use identity permutation: s = (0, 1, 2, ..., n-1) (0-indexed)
    perm = list(range(n_obj))
    upper_bounds = {}  # No initial upper bounds

    # Run AIRA (Algorithm 3) with k = n (all objectives free)
    nd_solutions = aira_worker(instance, perm, n_obj, upper_bounds, start_time, args.time_limit)

    # Filter dominated across all found solutions
    nd_solutions = filter_dominated(nd_solutions)

    elapsed = time.time() - start_time

    # Compute objective_value as the number of non-dominated solutions found
    # **INFERRED ASSUMPTION**: Since this is a multi-objective problem with no single
    # scalar objective, we report the count of non-dominated solutions as the
    # objective_value. This is a reasonable proxy for solution quality.
    result = {
        "objective_value": len(nd_solutions),
        "num_non_dominated_solutions": len(nd_solutions),
        "non_dominated_objective_vectors": [sol["obj_vals"] for sol in nd_solutions],
        "solutions": [sol["solution"] for sol in nd_solutions],
        "elapsed_time": elapsed,
        "time_limit_reached": elapsed >= args.time_limit
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Found {len(nd_solutions)} non-dominated solutions in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
