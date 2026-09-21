"""
Gurobi implementation of the Workload Smoothing Problem (WSP) from:
"A Branch and Cut Approach for Workload Smoothing on Assembly Lines"
by Anulark Pinnoi and Wilbert F. Wilhelm (1997).

The WSP assigns tasks to S* stations (optimal number) while minimizing
the maximum idle time on any station to balance workloads.
"""

import argparse
import json
import math
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
        data = json.load(f)
    return data


def compute_predecessors_successors(tasks, arcs):
    """Compute immediate and transitive predecessor/successor sets."""
    # M(t): immediate predecessors
    M = defaultdict(set)
    # R(t): immediate successors
    R = defaultdict(set)
    for (i, j) in arcs:
        M[j].add(i)
        R[i].add(j)

    # A(t): all predecessors (transitive closure)
    A = defaultdict(set)
    # B(t): all successors (transitive closure)
    B = defaultdict(set)

    # Topological order for predecessor computation
    # BFS-based transitive closure
    for t in tasks:
        # Compute A(t) by BFS backward
        visited = set()
        stack = list(M[t])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(M[node])
        A[t] = visited

    for t in tasks:
        visited = set()
        stack = list(R[t])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(R[node])
        B[t] = visited

    return M, R, A, B


def compute_E_L(tasks, p, c, S_star, A, B):
    """
    Compute earliest and latest station bounds for each task.

    E_i = max(1, ceil((sum of p_k for k in A(i) + p_i) / c))     -- eq (28)
    L_i = min(S*, S* + 1 - ceil((p_i + sum of p_k for k in B(i)) / c))  -- eq (29)
    """
    E = {}
    L = {}
    for i in tasks:
        pred_sum = sum(p[k] for k in A[i])
        E[i] = max(1, math.ceil((pred_sum + p[i]) / c))

        succ_sum = sum(p[k] for k in B[i])
        L[i] = min(S_star, S_star + 1 - math.ceil((p[i] + succ_sum) / c))

    return E, L


def solve_wsp(data, time_limit):
    """Build and solve the WSP model using Gurobi."""
    # Extract data
    tasks = [int(t) for t in data["task_processing_times"].keys()]
    p = {int(t): v for t, v in data["task_processing_times"].items()}
    c = data["cycle_time"]
    S_star = data["num_stations"]
    arcs = [(a[0], a[1]) for a in data["precedence_arcs"]]

    # Use provided E, L if available; otherwise compute them
    M, R, A, B = compute_predecessors_successors(tasks, arcs)

    # Compute E_i and L_i from the formulas in the paper
    E_computed, L_computed = compute_E_L(tasks, p, c, S_star, A, B)

    # Use instance-provided bounds if they are tighter
    if "earliest_station" in data:
        E_provided = {int(k): v for k, v in data["earliest_station"].items()}
    else:
        E_provided = {}
    if "latest_station" in data:
        L_provided = {int(k): v for k, v in data["latest_station"].items()}
    else:
        L_provided = {}

    E = {}
    L = {}
    for i in tasks:
        E[i] = max(E_computed[i], E_provided.get(i, 1))
        L[i] = min(L_computed[i], L_provided.get(i, S_star))

    # Validate bounds
    for i in tasks:
        if E[i] > L[i]:
            # Relax to computed bounds if instance bounds are inconsistent
            E[i] = E_computed[i]
            L[i] = L_computed[i]

    # T(s): tasks that can be assigned to station s
    T_s = defaultdict(list)
    for i in tasks:
        for s in range(E[i], L[i] + 1):
            T_s[s].append(i)

    # Build the model
    model = gp.Model("WSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x[s,i] = 1 if task i assigned to station s
    x = {}
    for i in tasks:
        for s in range(E[i], L[i] + 1):
            x[s, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}")

    # z_max: maximum idle time (continuous variable)
    z_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z_max")

    # Objective: minimize z_max
    model.setObjective(z_max, GRB.MINIMIZE)

    # Constraint (2): Assignment - each task assigned to exactly one station
    for i in tasks:
        model.addConstr(
            gp.quicksum(x[s, i] for s in range(E[i], L[i] + 1)) == 1,
            name=f"assign_{i}"
        )

    # Constraint (3): Precedence - for each arc (i,j), station(i) <= station(j)
    for (i, j) in arcs:
        model.addConstr(
            gp.quicksum(s * x[s, i] for s in range(E[i], L[i] + 1))
            - gp.quicksum(s * x[s, j] for s in range(E[j], L[j] + 1)) <= 0,
            name=f"prec_{i}_{j}"
        )

    # Constraint (4): Cycle time / capacity constraint
    for s in range(1, S_star + 1):
        if T_s[s]:
            model.addConstr(
                gp.quicksum(p[i] * x[s, i] for i in T_s[s] if (s, i) in x) <= c,
                name=f"capacity_{s}"
            )

    # Constraint (6): z_max >= c - sum_i p_i * x_{si} for each station
    for s in range(1, S_star + 1):
        if T_s[s]:
            model.addConstr(
                gp.quicksum(p[i] * x[s, i] for i in T_s[s] if (s, i) in x) + z_max >= c,
                name=f"idle_{s}"
            )
        else:
            # If no task can be assigned to station s, idle time = c
            model.addConstr(z_max >= c, name=f"idle_empty_{s}")

    # Lower bound on z_max (from the paper, Section 5.2):
    # z_max >= floor((c * S* - sum p_t) / c)
    total_p = sum(p[i] for i in tasks)
    z_lb = math.floor((c * S_star - total_p) / c)
    if z_lb > 0:
        model.addConstr(z_max >= z_lb, name="z_lower_bound")

    # Optimize
    model.optimize()

    # Extract solution
    result = {
        "objective_value": None,
        "status": None,
        "task_assignments": {},
        "station_workloads": {},
        "station_idle_times": {}
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract task assignments
        for i in tasks:
            for s in range(E[i], L[i] + 1):
                if (s, i) in x and x[s, i].X > 0.5:
                    result["task_assignments"][str(i)] = s
                    break

        # Compute station workloads and idle times
        for s in range(1, S_star + 1):
            workload = 0
            for i in T_s[s]:
                if (s, i) in x and x[s, i].X > 0.5:
                    workload += p[i]
            result["station_workloads"][str(s)] = workload
            result["station_idle_times"][str(s)] = c - workload

        result["z_max"] = z_max.X
        result["num_stations"] = S_star
        result["cycle_time"] = c
    else:
        result["status"] = "infeasible"
        result["objective_value"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the Workload Smoothing Problem (WSP) using Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = solve_wsp(data, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value (z_max): {result['objective_value']}")
        print(f"Status: {result['status']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
