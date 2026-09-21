"""
Gurobi implementation of the MILP model for Parallel Batch Scheduling
to Minimize Total Flow Time.

Paper: Ozturk (2020) - "A truncated column generation algorithm for the
       parallel batch scheduling problem to minimize total flow time"

Problem: P/p-Batch, r_j, p_j, v_j, Cap / sum(C_j)

MILP Model (Section 3):
  min sum C_j
  s.t.
    (1) sum_k sum_m x_{jkm} = 1,                   for all j
    (2) sum_j x_{jkm} * v_j <= Cap,                 for all k, m
    (3) p_{km} >= x_{jkm} * p_j,                    for all j, k, m
    (4) S_{km} >= x_{jkm} * r_j,                    for all j, k, m
    (5) S_{km} >= S_{k-1,m} + p_{k-1,m},            for k>=2, all m
    (6) C_j >= (S_{km} + p_{km}) - Q(1 - x_{jkm}),  for all j, k, m
    (7) x_{jkm} in {0,1}, S_{km} >= 0, p_{km} >= 0
"""

import json
import argparse
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


def solve(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_jobs']
    M = data['num_machines']
    Cap = data['batch_capacity']
    jobs = data['jobs']

    # Extract job parameters (0-indexed internally)
    r = [jobs[j]['release_date'] for j in range(N)]
    p = [jobs[j]['processing_time'] for j in range(N)]
    v = [jobs[j]['size'] for j in range(N)]

    # Big-M: upper bound on any completion time
    # ASSUMPTION (NOT SPECIFIED IN PAPER): Q = sum of all release dates + sum of all processing times + max release date
    # This is a safe upper bound on the latest possible completion time.
    Q = sum(r) + sum(p) + max(r) + max(p)

    model = gp.Model("BatchScheduling")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # x[j,k,m]: 1 if job j is in batch k on machine m
    x = {}
    for j in range(N):
        for k in range(N):
            for m in range(M):
                x[j, k, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}_{m}")

    # p_batch[k,m]: processing duration of batch k on machine m
    p_batch = {}
    for k in range(N):
        for m in range(M):
            p_batch[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"p_{k}_{m}")

    # S[k,m]: start time of batch k on machine m
    S = {}
    for k in range(N):
        for m in range(M):
            S[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S_{k}_{m}")

    # C[j]: completion time (flow time) of job j
    C = {}
    for j in range(N):
        C[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"C_{j}")

    model.update()

    # --- Objective: minimize sum of completion times ---
    model.setObjective(gp.quicksum(C[j] for j in range(N)), GRB.MINIMIZE)

    # --- Constraints ---

    # (1) Each job assigned to exactly one batch on one machine
    for j in range(N):
        model.addConstr(
            gp.quicksum(x[j, k, m] for k in range(N) for m in range(M)) == 1,
            name=f"assign_{j}"
        )

    # (2) Capacity constraint
    for k in range(N):
        for m in range(M):
            model.addConstr(
                gp.quicksum(x[j, k, m] * v[j] for j in range(N)) <= Cap,
                name=f"cap_{k}_{m}"
            )

    # (3) Batch processing duration >= processing time of any assigned job
    for j in range(N):
        for k in range(N):
            for m in range(M):
                model.addConstr(
                    p_batch[k, m] >= x[j, k, m] * p[j],
                    name=f"proc_{j}_{k}_{m}"
                )

    # (4) Batch start time >= release date of any assigned job
    for j in range(N):
        for k in range(N):
            for m in range(M):
                model.addConstr(
                    S[k, m] >= x[j, k, m] * r[j],
                    name=f"release_{j}_{k}_{m}"
                )

    # (5) Consecutive batches on the same machine don't overlap
    for k in range(1, N):
        for m in range(M):
            model.addConstr(
                S[k, m] >= S[k - 1, m] + p_batch[k - 1, m],
                name=f"seq_{k}_{m}"
            )

    # (6) Flow time (completion time) of each job
    for j in range(N):
        for k in range(N):
            for m in range(M):
                model.addConstr(
                    C[j] >= (S[k, m] + p_batch[k, m]) - Q * (1 - x[j, k, m]),
                    name=f"flow_{j}_{k}_{m}"
                )

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    result = {"objective_value": None, "status": None, "batches": [], "job_assignments": {}}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract job assignments
        for j in range(N):
            for k in range(N):
                for m in range(M):
                    if x[j, k, m].X > 0.5:
                        job_id = jobs[j]['job_id']
                        result["job_assignments"][str(job_id)] = {
                            "batch": k + 1,
                            "machine": m + 1,
                            "completion_time": C[j].X
                        }
                        break
                else:
                    continue
                break

        # Extract batch info
        for k in range(N):
            for m in range(M):
                if p_batch[k, m].X > 1e-6:
                    batch_jobs = []
                    for j in range(N):
                        if x[j, k, m].X > 0.5:
                            batch_jobs.append(jobs[j]['job_id'])
                    if batch_jobs:
                        result["batches"].append({
                            "batch_id": k + 1,
                            "machine": m + 1,
                            "start_time": S[k, m].X,
                            "processing_time": p_batch[k, m].X,
                            "jobs": batch_jobs
                        })
    else:
        result["status"] = "infeasible_or_no_solution"
        result["objective_value"] = None

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gurobi MILP for Parallel Batch Scheduling (Ozturk 2020)"
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

    solve(args.instance_path, args.solution_path, args.time_limit)
