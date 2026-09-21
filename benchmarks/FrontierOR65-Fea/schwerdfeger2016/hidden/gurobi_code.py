"""
Gurobi implementation of P || NSSWD
Minimizes the normalized sum of squared workload deviations on identical parallel machines.
Source: Schwerdfeger & Walter (2016), Computers & Operations Research 73 (2016) 84-91
"""

import argparse
import json
import math

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
def compute_nsswd(completions, mu):
    """Compute NSSWD value given machine completion times and average mu."""
    if mu == 0:
        return 0.0
    return math.sqrt(sum((c - mu) ** 2 for c in completions)) / mu


def solve(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, "r") as f:
        inst = json.load(f)

    processing_times = inst["processing_times"]  # already sorted descending per paper
    n_orig = len(processing_times)
    m_orig = inst["num_machines"]

    # Sort jobs descending (p_1 >= p_2 >= ... >= p_n)
    processing_times = sorted(processing_times, reverse=True)

    # --- Preprocessing / Reduction (Fig. 1 in paper) ---
    # If p_1 >= mu, assign that job alone to a machine and remove both.
    jobs = list(range(n_orig))       # indices into processing_times
    p = {j: processing_times[j] for j in jobs}
    active_jobs = list(jobs)
    removed_assignments = []         # list of (machine_label, job_index)

    machine_counter = 0
    P_total = sum(p[j] for j in active_jobs)
    m_active = m_orig

    i = 0
    while i < len(active_jobs):
        if m_active <= 0:
            break
        mu_current = P_total / m_active
        job_i = active_jobs[i]
        if p[job_i] >= mu_current:
            removed_assignments.append((machine_counter, job_i))
            machine_counter += 1
            active_jobs.pop(i)
            P_total -= p[job_i]
            m_active -= 1
            # do not increment i; next job slides into position i
        else:
            break  # jobs are sorted; if p[i] < mu then all remaining < mu

    # Remaining jobs and machines after reduction
    n = len(active_jobs)
    m = m_active

    mu = P_total / m if m > 0 else 0.0

    # Build Gurobi model
    model = gp.Model("P_NSSWD")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)

    # Decision variables: x[i,j] = 1 if active job j assigned to active machine i
    # i in 0..m-1, j indexes into active_jobs list
    x = {}
    for i in range(m):
        for jidx, j in enumerate(active_jobs):
            x[i, jidx] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{jidx}")

    # Completion time auxiliary variables C[i]
    C = {}
    for i in range(m):
        C[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"C_{i}")

    model.update()

    # Constraint (3): sum_j p_j * x[i,j] = C[i]
    for i in range(m):
        model.addConstr(
            gp.quicksum(p[active_jobs[jidx]] * x[i, jidx] for jidx in range(n)) == C[i],
            name=f"completion_{i}"
        )

    # Constraint (4): each job assigned to exactly one machine
    for jidx in range(n):
        model.addConstr(
            gp.quicksum(x[i, jidx] for i in range(m)) == 1,
            name=f"assign_{jidx}"
        )

    # Objective: minimize sum_i (C_i - mu)^2  (equivalent to minimizing NSSWD, Eq. 6)
    # This is a quadratic objective (binary quadratic program).
    # NOTE: Since mu is constant for a given instance, minimizing sum (C_i - mu)^2
    # is equivalent to minimizing NSSWD (see Eq. 6 in paper).
    obj = gp.quicksum((C[i] - mu) * (C[i] - mu) for i in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    # Extract solution
    status = model.Status
    has_solution = status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0

    if has_solution:
        # Build assignment: machine -> list of original job indices
        assignment = {}
        completions = []

        # Pre-fill removed (preprocessed) machines
        for mac_label, job_idx in removed_assignments:
            assignment[mac_label] = [job_idx]

        for i in range(m):
            mac_label = machine_counter + i
            assignment[mac_label] = []
            c_val = 0.0
            for jidx, j in enumerate(active_jobs):
                if x[i, jidx].X > 0.5:
                    assignment[mac_label].append(j)
                    c_val += p[j]
            completions.append(c_val)

        # Add removed machines' completion times
        removed_completions = [p[job_idx] for _, job_idx in removed_assignments]
        all_completions = removed_completions + completions

        mu_full = sum(processing_times[j] for j in range(n_orig)) / m_orig
        nsswd_val = compute_nsswd(all_completions, mu_full)

        # Convert assignment keys to strings for JSON
        assignment_out = {str(k): v for k, v in assignment.items()}

        solution = {
            "objective_value": nsswd_val,
            "status": "optimal" if status == GRB.OPTIMAL else "feasible",
            "num_machines": m_orig,
            "num_jobs": n_orig,
            "assignment": assignment_out,
            "machine_completion_times": {str(k): sum(processing_times[j] for j in v)
                                          for k, v in assignment.items()},
        }
    else:
        solution = {
            "objective_value": None,
            "status": "infeasible_or_no_solution",
            "num_machines": m_orig,
            "num_jobs": n_orig,
        }

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Status: {solution['status']}")
    print(f"Objective (NSSWD): {solution['objective_value']}")
    print(f"Solution written to: {solution_path}")


def main():
    parser = argparse.ArgumentParser(description="Gurobi solver for P || NSSWD")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
