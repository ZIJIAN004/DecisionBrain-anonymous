"""
Gurobi MIP solver for the Job Shop Scheduling Problem with Total Weighted
Tardiness (JSPTWT), following the formulation from Bierwirth & Kuhpfahl (2017).

Mathematical model:
    min  sum_j  w_j * T_j

    s.t.
    (C1) T_j >= c_j - d_j                          for all j
         T_j >= 0                                   (implicit in variable bound)
    (C2) c_j  = s_{last_machine_j, j} + p_{last_machine_j, j}   for all j
    (C3) s_{sigma_j(k+1), j} >= s_{sigma_j(k), j} + p_{sigma_j(k), j}
                                                    for all j, k=1..m-1
    (C4) s_{sigma_j(1), j} >= r_j                   for all j
    (C5) Disjunctive (machine capacity):
         s_{i,k} >= s_{i,j} + p_{i,j} - V*(1 - y_{i,j,k})
         s_{i,j} >= s_{i,k} + p_{i,k} - V*y_{i,j,k}
                                                    for all i, j<k sharing machine i
    (C6) s_{i,j} >= 0, y binary

    Big-M value V = sum of all processing times + max release date.
    This is a safe upper bound on the makespan (and hence any start time),
    since even if all operations were serialised, the total time cannot exceed
    this value.  The paper does not specify V; this is a standard choice.
"""

import argparse
import json
import os
import time
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


def solve_jsptwt(instance_path: str, solution_path: str, time_limit: float) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load instance
    # ------------------------------------------------------------------ #
    with open(instance_path, "r") as f:
        data = json.load(f)

    jobs = data["jobs"]
    num_jobs = data["num_jobs"]
    num_machines = data["num_machines"]

    # Build helper structures
    # p[i][j]  = processing time of job j on machine i
    # order[j] = list of machines in technological order for job j
    p = {}
    order = {}
    weights = {}
    due_dates = {}
    release_dates = {}

    for job in jobs:
        j = job["job_id"]
        weights[j] = job["weight"]
        due_dates[j] = job["due_date"]
        release_dates[j] = job["release_date"]
        order[j] = []
        for op in job["operations"]:
            m = op["machine"]
            order[j].append(m)
            p[(m, j)] = op["processing_time"]

    # Compute Big-M: sum of all processing times + max release date
    total_processing = sum(p.values())
    max_release = max(release_dates.values()) if release_dates else 0
    V = total_processing + max_release

    # Group jobs by machine for disjunctive constraints
    # machine_jobs[i] = list of job ids that visit machine i
    machine_jobs = defaultdict(list)
    for j in range(num_jobs):
        for m in order[j]:
            machine_jobs[m].append(j)

    # ------------------------------------------------------------------ #
    # 2. Build Gurobi model
    # ------------------------------------------------------------------ #
    model = gp.Model("JSPTWT")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    # Suppress verbose output; summary will still print at end
    model.Params.OutputFlag = 1

    # -- Decision variables --
    # s[i,j]: start time of operation (i/j)
    s = {}
    for j in range(num_jobs):
        for m in order[j]:
            s[(m, j)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"s_{m}_{j}")

    # c[j]: completion time of job j
    c = {}
    for j in range(num_jobs):
        c[j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"c_{j}")

    # T[j]: tardiness of job j
    T = {}
    for j in range(num_jobs):
        T[j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{j}")

    # y[i,j,k]: binary, 1 if operation (i/j) precedes (i/k) on machine i
    y = {}
    for m in range(num_machines):
        job_list = machine_jobs[m]
        for idx_a in range(len(job_list)):
            for idx_b in range(idx_a + 1, len(job_list)):
                ja = job_list[idx_a]
                jb = job_list[idx_b]
                y[(m, ja, jb)] = model.addVar(vtype=GRB.BINARY,
                                              name=f"y_{m}_{ja}_{jb}")

    model.update()

    # -- Objective --
    model.setObjective(
        gp.quicksum(weights[j] * T[j] for j in range(num_jobs)),
        GRB.MINIMIZE,
    )

    # -- Constraints --

    # (C1) Tardiness: T_j >= c_j - d_j  (T_j >= 0 by lb)
    for j in range(num_jobs):
        model.addConstr(T[j] >= c[j] - due_dates[j], name=f"tard_{j}")

    # (C2) Completion time: c_j = s_{last_op} + p_{last_op}
    for j in range(num_jobs):
        last_m = order[j][-1]
        model.addConstr(c[j] == s[(last_m, j)] + p[(last_m, j)],
                        name=f"compl_{j}")

    # (C3) Precedence within each job's technological sequence
    for j in range(num_jobs):
        for k in range(len(order[j]) - 1):
            m_curr = order[j][k]
            m_next = order[j][k + 1]
            model.addConstr(
                s[(m_next, j)] >= s[(m_curr, j)] + p[(m_curr, j)],
                name=f"prec_{j}_{k}",
            )

    # (C4) Release date: first operation of each job starts no earlier than r_j
    for j in range(num_jobs):
        first_m = order[j][0]
        model.addConstr(s[(first_m, j)] >= release_dates[j],
                        name=f"release_{j}")

    # (C5) Disjunctive constraints for each machine
    for m in range(num_machines):
        job_list = machine_jobs[m]
        for idx_a in range(len(job_list)):
            for idx_b in range(idx_a + 1, len(job_list)):
                ja = job_list[idx_a]
                jb = job_list[idx_b]
                yvar = y[(m, ja, jb)]
                # If y=1: ja before jb  =>  s[m,jb] >= s[m,ja] + p[m,ja]
                model.addConstr(
                    s[(m, jb)] >= s[(m, ja)] + p[(m, ja)] - V * (1 - yvar),
                    name=f"disj1_{m}_{ja}_{jb}",
                )
                # If y=0: jb before ja  =>  s[m,ja] >= s[m,jb] + p[m,jb]
                model.addConstr(
                    s[(m, ja)] >= s[(m, jb)] + p[(m, jb)] - V * yvar,
                    name=f"disj2_{m}_{ja}_{jb}",
                )

    # ------------------------------------------------------------------ #
    # 3. Solve
    # ------------------------------------------------------------------ #
    wall_start = time.time()
    model.optimize()
    wall_elapsed = time.time() - wall_start

    # ------------------------------------------------------------------ #
    # 4. Extract solution and write JSON
    # ------------------------------------------------------------------ #
    solution = {
        "instance_path": instance_path,
        "solver": "Gurobi",
        "time_limit": time_limit,
        "wall_time": round(wall_elapsed, 3),
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution["objective_value"] = round(obj_val, 6)
        solution["best_bound"] = round(model.ObjBound, 6)
        solution["mip_gap"] = round(model.MIPGap, 6) if model.MIPGap < GRB.INFINITY else None
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        solution["num_solutions"] = model.SolCount

        # Build schedule details
        schedule = []
        for job in jobs:
            j = job["job_id"]
            job_schedule = {
                "job_id": j,
                "weight": weights[j],
                "due_date": due_dates[j],
                "release_date": release_dates[j],
                "completion_time": round(c[j].X, 4),
                "tardiness": round(T[j].X, 4),
                "weighted_tardiness": round(weights[j] * T[j].X, 4),
                "operations": [],
            }
            for m in order[j]:
                start_val = round(s[(m, j)].X, 4)
                job_schedule["operations"].append({
                    "machine": m,
                    "start_time": start_val,
                    "processing_time": p[(m, j)],
                    "end_time": round(start_val + p[(m, j)], 4),
                })
            schedule.append(job_schedule)

        solution["schedule"] = schedule
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible_or_no_solution"
        solution["schedule"] = []

    # Determine output path
    if solution_path is None:
        # Derive from instance filename
        basename = os.path.splitext(os.path.basename(instance_path))[0]
        # e.g. instance_1 -> gurobi_solution_1.json
        idx = basename.replace("instance_", "")
        solution_path = os.path.join(
            os.path.dirname(instance_path), f"gurobi_solution_{idx}.json"
        )

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective (TWT): {solution['objective_value']}")
        print(f"Status: {solution['status']}")
    else:
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Solve JSPTWT using Gurobi MIP (Bierwirth & Kuhpfahl 2017 formulation)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance JSON file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default=None,
        help="Path for the output solution JSON file "
             "(default: gurobi_solution_{i}.json in the same directory as the instance)",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=3600.0,
        help="Gurobi time limit in seconds (default: 3600)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_jsptwt(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
