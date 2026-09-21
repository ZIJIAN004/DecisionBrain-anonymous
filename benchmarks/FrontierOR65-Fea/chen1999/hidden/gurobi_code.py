"""
Gurobi MIP implementation for the Parallel Machine Scheduling Problem
from Chen & Powell (1999), "Solving Parallel Machine Scheduling Problems
by Column Generation", INFORMS Journal on Computing, 11(1):78-94.

This implements the IP2 formulation (for identical machines P||sum w_j C_j)
with Big-M linearization for the bilinear completion time constraints.

For non-identical machines (Q, R), it implements the IP1 formulation.

The paper's formulation has bilinear terms C_i * x_{ij} in constraint (5)/(11).
We linearize these using McCormick envelopes with auxiliary variables L_{ij}.
"""

import argparse
import json
import math
import sys
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


def load_instance(instance_path):
    """Load problem instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_weighted_completion_time(data, time_limit):
    """
    Solve the total weighted completion time problem: P||sum w_j C_j,
    Q||sum w_j C_j, or R||sum w_j C_j.

    Uses IP2 for identical machines, IP1 for non-identical machines.
    """
    n = data["num_jobs"]
    m = data["num_machines"]
    machine_type = data.get("machine_type", "identical")
    weights = data["jobs"]["weights"]
    processing_times = data["jobs"]["processing_times"]  # p[j][k] for job j, machine k

    # For identical machines, use base_processing_times
    if machine_type == "identical":
        base_p = data["jobs"]["base_processing_times"]
    else:
        base_p = None

    # Determine SWPT order for feasible predecessor sets
    # SWPT: p_j/w_j non-decreasing. Ties broken by smaller index first.
    jobs = list(range(n))

    if machine_type == "identical":
        # Single SWPT order for all machines
        swpt_order = sorted(jobs, key=lambda j: (base_p[j] / weights[j], j))
        swpt_rank = [0] * n
        for rank, j in enumerate(swpt_order):
            swpt_rank[j] = rank

        # B_j = {i in N | i precedes j in SWPT order}
        B = {}
        for j in jobs:
            B[j] = [i for i in jobs if swpt_rank[i] < swpt_rank[j]]

        # A_j = {i in N | i succeeds j in SWPT order}
        A = {}
        for j in jobs:
            A[j] = [i for i in jobs if swpt_rank[i] > swpt_rank[j]]
    else:
        # For non-identical machines, SWPT order may differ per machine
        # B_j^k and A_j^k defined per machine
        B_k = {}
        A_k = {}
        for k in range(m):
            swpt_order_k = sorted(jobs, key=lambda j: (processing_times[j][k] / weights[j], j))
            swpt_rank_k = [0] * n
            for rank, j in enumerate(swpt_order_k):
                swpt_rank_k[j] = rank
            for j in jobs:
                B_k[(j, k)] = [i for i in jobs if swpt_rank_k[i] < swpt_rank_k[j]]
                A_k[(j, k)] = [i for i in jobs if swpt_rank_k[i] > swpt_rank_k[j]]

    # Total processing time (upper bound for completion times)
    if machine_type == "identical":
        P_total = sum(base_p)
    else:
        P_total = max(sum(processing_times[j][k] for j in jobs) for k in range(m))

    # Big-M value for linearization
    M_val = P_total

    # Create Gurobi model
    model = gp.Model("PMAC_WCT")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    if machine_type == "identical":
        # ============================================================
        # IP2 formulation (identical machines)
        # ============================================================
        # Variables: x_{ij} for i in B_j union {0}, j in N
        #   x_{0j} = 1 if job j is first on some machine
        #   x_{ij} = 1 if job i immediately precedes job j
        # Also x_{j,n+1} for flow conservation

        model.remove(model.getVars())
        model = gp.Model("PMAC_WCT")
        model.setParam("Threads", 1)
        model.setParam("TimeLimit", time_limit)
        model.setParam("OutputFlag", 1)

        # x[i][j]: i is the predecessor of j. i=-1 means j is first on a machine.
        # j=-1 means j is last (dummy sink n+1)
        x = {}
        DUMMY_START = -1
        DUMMY_END = n

        # x_{0,j}: job j is first on some machine
        for j in jobs:
            x[(DUMMY_START, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_start_{j}")

        # x_{i,j}: job i immediately before job j, for i in B_j
        for j in jobs:
            for i in B[j]:
                x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # x_{j, n+1}: job j is last on some machine
        for j in jobs:
            x[(j, DUMMY_END)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_end")

        # Completion time variables
        C = {}
        for j in jobs:
            C[j] = model.addVar(lb=0, ub=M_val, vtype=GRB.CONTINUOUS, name=f"C_{j}")

        # Linearization variables L_{ij} = C_i * x_{ij}
        L = {}
        for j in jobs:
            for i in B[j]:
                L[(i, j)] = model.addVar(lb=0, ub=M_val, vtype=GRB.CONTINUOUS,
                                         name=f"L_{i}_{j}")

        model.update()

        # Objective: minimize sum w_j * C_j (Eq. 7)
        model.setObjective(
            gp.quicksum(weights[j] * C[j] for j in jobs),
            GRB.MINIMIZE
        )

        # Constraint (8): each job has exactly one predecessor
        # sum_{i in B_j union {0}} x_{ij} = 1, for all j in N
        for j in jobs:
            model.addConstr(
                x[(DUMMY_START, j)] + gp.quicksum(x[(i, j)] for i in B[j]) == 1,
                name=f"pred_{j}"
            )

        # Constraint (9): at most m machines start
        # sum_j x_{0j} <= m
        model.addConstr(
            gp.quicksum(x[(DUMMY_START, j)] for j in jobs) <= m,
            name="machine_limit"
        )

        # Constraint (10): flow conservation
        # sum_{i in B_j union {0}} x_{ij} = sum_{i in A_j union {n+1}} x_{ji}
        for j in jobs:
            lhs = x[(DUMMY_START, j)] + gp.quicksum(x[(i, j)] for i in B[j])
            rhs = x[(j, DUMMY_END)] + gp.quicksum(x[(j, i)] for i in A[j])
            model.addConstr(lhs == rhs, name=f"flow_{j}")

        # Constraint (11) linearized: C_j = p_j * x_{0j} + sum_{i in B_j} (L_{ij} + p_j * x_{ij})
        # where L_{ij} = C_i * x_{ij} (linearized)
        for j in jobs:
            p_j = base_p[j]
            model.addConstr(
                C[j] == p_j * x[(DUMMY_START, j)] +
                gp.quicksum(L[(i, j)] + p_j * x[(i, j)] for i in B[j]),
                name=f"completion_{j}"
            )

        # McCormick linearization for L_{ij} = C_i * x_{ij}:
        # L_{ij} <= C_i
        # L_{ij} <= M * x_{ij}
        # L_{ij} >= C_i - M * (1 - x_{ij})
        # L_{ij} >= 0 (already set as lb)
        for j in jobs:
            for i in B[j]:
                model.addConstr(L[(i, j)] <= C[i], name=f"mc1_{i}_{j}")
                model.addConstr(L[(i, j)] <= M_val * x[(i, j)], name=f"mc2_{i}_{j}")
                model.addConstr(L[(i, j)] >= C[i] - M_val * (1 - x[(i, j)]),
                                name=f"mc3_{i}_{j}")

    else:
        # ============================================================
        # IP1 formulation (non-identical machines: Q or R)
        # ============================================================
        DUMMY_START = -1
        DUMMY_END = n

        x = {}
        for k in range(m):
            for j in jobs:
                x[(DUMMY_START, j, k)] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_start_{j}_{k}")
            for j in jobs:
                for i in B_k[(j, k)]:
                    x[(i, j, k)] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
            for j in jobs:
                x[(j, DUMMY_END, k)] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_{j}_end_{k}")

        C = {}
        for j in jobs:
            C[j] = model.addVar(lb=0, ub=M_val, vtype=GRB.CONTINUOUS, name=f"C_{j}")

        L = {}
        for k in range(m):
            for j in jobs:
                for i in B_k[(j, k)]:
                    L[(i, j, k)] = model.addVar(
                        lb=0, ub=M_val, vtype=GRB.CONTINUOUS,
                        name=f"L_{i}_{j}_{k}")

        model.update()

        # Objective: minimize sum w_j * C_j
        model.setObjective(
            gp.quicksum(weights[j] * C[j] for j in jobs),
            GRB.MINIMIZE
        )

        # Constraint (2): each job assigned exactly once
        for j in jobs:
            model.addConstr(
                gp.quicksum(
                    x[(DUMMY_START, j, k)] +
                    gp.quicksum(x[(i, j, k)] for i in B_k[(j, k)])
                    for k in range(m)
                ) == 1,
                name=f"assign_{j}"
            )

        # Constraint (3): at most one job starts on each machine
        for k in range(m):
            model.addConstr(
                gp.quicksum(x[(DUMMY_START, j, k)] for j in jobs) <= 1,
                name=f"machine_start_{k}"
            )

        # Constraint (4): flow conservation per machine
        for k in range(m):
            for j in jobs:
                lhs = x[(DUMMY_START, j, k)] + gp.quicksum(
                    x[(i, j, k)] for i in B_k[(j, k)])
                rhs = x[(j, DUMMY_END, k)] + gp.quicksum(
                    x[(j, i, k)] for i in A_k[(j, k)])
                model.addConstr(lhs == rhs, name=f"flow_{j}_{k}")

        # Constraint (5) linearized
        for j in jobs:
            model.addConstr(
                C[j] == gp.quicksum(
                    processing_times[j][k] * x[(DUMMY_START, j, k)] +
                    gp.quicksum(
                        L[(i, j, k)] + processing_times[j][k] * x[(i, j, k)]
                        for i in B_k[(j, k)]
                    )
                    for k in range(m)
                ),
                name=f"completion_{j}"
            )

        # McCormick linearization
        for k in range(m):
            for j in jobs:
                for i in B_k[(j, k)]:
                    model.addConstr(L[(i, j, k)] <= C[i],
                                    name=f"mc1_{i}_{j}_{k}")
                    model.addConstr(L[(i, j, k)] <= M_val * x[(i, j, k)],
                                    name=f"mc2_{i}_{j}_{k}")
                    model.addConstr(
                        L[(i, j, k)] >= C[i] - M_val * (1 - x[(i, j, k)]),
                        name=f"mc3_{i}_{j}_{k}")

    # Optimize
    model.optimize()

    # Extract solution
    result = {
        "problem_type": "weighted_completion_time",
        "machine_type": machine_type,
        "num_jobs": n,
        "num_machines": m,
        "status": model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(model.Status, f"OTHER_{model.Status}"),
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["gap"] = model.MIPGap

        # Extract schedule
        schedule = {k: [] for k in range(m)}
        completion_times = {}
        for j in jobs:
            completion_times[j] = C[j].X

        if machine_type == "identical":
            # Reconstruct schedule from x variables
            # Find which jobs start on a machine
            machine_assignments = _reconstruct_schedule_identical(
                x, B, A, jobs, n, m, DUMMY_START, DUMMY_END)
            result["schedule"] = machine_assignments
        else:
            machine_assignments = _reconstruct_schedule_nonidentical(
                x, B_k, A_k, jobs, n, m, DUMMY_START, DUMMY_END)
            result["schedule"] = machine_assignments

        result["completion_times"] = {str(j): completion_times[j] for j in jobs}
    else:
        result["objective_value"] = None

    return result


def _reconstruct_schedule_identical(x, B, A, jobs, n, m, DUMMY_START, DUMMY_END):
    """Reconstruct the machine schedule from x-variable solution (identical machines)."""
    schedules = []

    # Find jobs that start a machine (x_{0,j} = 1)
    start_jobs = [j for j in jobs if x[(DUMMY_START, j)].X > 0.5]

    for start_j in start_jobs:
        machine_schedule = [start_j]
        current = start_j
        while True:
            # Find successor
            next_job = None
            for succ in A[current]:
                if (current, succ) in x and x[(current, succ)].X > 0.5:
                    next_job = succ
                    break
            if next_job is None:
                break
            machine_schedule.append(next_job)
            current = next_job
        schedules.append(machine_schedule)

    return {str(i): sched for i, sched in enumerate(schedules)}


def _reconstruct_schedule_nonidentical(x, B_k, A_k, jobs, n, m, DUMMY_START, DUMMY_END):
    """Reconstruct the machine schedule from x-variable solution (non-identical machines)."""
    schedules = {}
    for k in range(m):
        # Find the starting job on machine k
        start_job = None
        for j in jobs:
            if x[(DUMMY_START, j, k)].X > 0.5:
                start_job = j
                break
        if start_job is None:
            schedules[str(k)] = []
            continue

        machine_schedule = [start_job]
        current = start_job
        while True:
            next_job = None
            for succ in A_k[(current, k)]:
                if (current, succ, k) in x and x[(current, succ, k)].X > 0.5:
                    next_job = succ
                    break
            if next_job is None:
                break
            machine_schedule.append(next_job)
            current = next_job
        schedules[str(k)] = machine_schedule

    return schedules


def solve_weighted_tardy_jobs(data, time_limit):
    """
    Solve the weighted number of tardy jobs problem: P||sum w_j U_j,
    Q||sum w_j U_j, or R||sum w_j U_j.

    Uses the modified IP1' formulation from the paper (Section 3.1).
    """
    n = data["num_jobs"]
    m = data["num_machines"]
    machine_type = data.get("machine_type", "identical")
    weights = data["jobs"]["weights"]
    processing_times = data["jobs"]["processing_times"]
    due_dates = data["jobs"]["due_dates"]

    if machine_type == "identical":
        base_p = data["jobs"]["base_processing_times"]
    else:
        base_p = None

    jobs = list(range(n))

    # EDD order: sort by due date, ties broken by smaller index
    edd_order = sorted(jobs, key=lambda j: (due_dates[j], j))
    edd_rank = [0] * n
    for rank, j in enumerate(edd_order):
        edd_rank[j] = rank

    # B_j = {i in N | i precedes j in EDD order}
    B = {}
    A = {}
    for j in jobs:
        B[j] = [i for i in jobs if edd_rank[i] < edd_rank[j]]
        A[j] = [i for i in jobs if edd_rank[i] > edd_rank[j]]

    # Upper bound on time
    if machine_type == "identical":
        P_total = sum(base_p)
    else:
        P_total = max(sum(processing_times[j][k] for j in jobs) for k in range(m))

    M_val = P_total

    model = gp.Model("PMAC_TARDY")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    DUMMY_START = -1
    DUMMY_END = n

    # z_j: 1 if job j is tardy
    z = {}
    for j in jobs:
        z[j] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}")

    # x_{ij}^k variables (on-time jobs only)
    x = {}
    for k in range(m):
        for j in jobs:
            x[(DUMMY_START, j, k)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_start_{j}_{k}")
        for j in jobs:
            for i in B[j]:
                x[(i, j, k)] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        for j in jobs:
            x[(j, DUMMY_END, k)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_{j}_end_{k}")

    # Completion time for on-time jobs
    C = {}
    for j in jobs:
        C[j] = model.addVar(lb=0, ub=M_val, vtype=GRB.CONTINUOUS, name=f"C_{j}")

    # Linearization variables
    L = {}
    for k in range(m):
        for j in jobs:
            for i in B[j]:
                L[(i, j, k)] = model.addVar(
                    lb=0, ub=M_val, vtype=GRB.CONTINUOUS,
                    name=f"L_{i}_{j}_{k}")

    model.update()

    # Objective (32): minimize sum w_j z_j
    model.setObjective(
        gp.quicksum(weights[j] * z[j] for j in jobs),
        GRB.MINIMIZE
    )

    # Constraint (33): sum_k sum_{i in B_j union {0}} x_{ij}^k + z_j = 1
    for j in jobs:
        model.addConstr(
            gp.quicksum(
                x[(DUMMY_START, j, k)] +
                gp.quicksum(x[(i, j, k)] for i in B[j])
                for k in range(m)
            ) + z[j] == 1,
            name=f"cover_{j}"
        )

    # Constraint (34): sum_j x_{0j}^k <= 1
    for k in range(m):
        model.addConstr(
            gp.quicksum(x[(DUMMY_START, j, k)] for j in jobs) <= 1,
            name=f"machine_start_{k}"
        )

    # Constraint (35): flow conservation
    for k in range(m):
        for j in jobs:
            lhs = x[(DUMMY_START, j, k)] + gp.quicksum(
                x[(i, j, k)] for i in B[j])
            rhs = x[(j, DUMMY_END, k)] + gp.quicksum(
                x[(j, i, k)] for i in A[j])
            model.addConstr(lhs == rhs, name=f"flow_{j}_{k}")

    # Constraint (36) linearized: completion time
    for j in jobs:
        p_j_terms = []
        for k in range(m):
            p_jk = processing_times[j][k]
            p_j_terms.append(
                p_jk * x[(DUMMY_START, j, k)] +
                gp.quicksum(
                    L[(i, j, k)] + p_jk * x[(i, j, k)]
                    for i in B[j]
                )
            )
        model.addConstr(C[j] == gp.quicksum(p_j_terms), name=f"completion_{j}")

    # Constraint (37): 0 <= C_j <= d_j (for on-time jobs)
    # If z_j = 1 (tardy), C_j = 0 (all x's are 0)
    # If z_j = 0 (on-time), C_j <= d_j
    for j in jobs:
        model.addConstr(C[j] <= due_dates[j] * (1 - z[j]), name=f"due_{j}")

    # McCormick linearization
    for k in range(m):
        for j in jobs:
            for i in B[j]:
                model.addConstr(L[(i, j, k)] <= C[i],
                                name=f"mc1_{i}_{j}_{k}")
                model.addConstr(L[(i, j, k)] <= M_val * x[(i, j, k)],
                                name=f"mc2_{i}_{j}_{k}")
                model.addConstr(
                    L[(i, j, k)] >= C[i] - M_val * (1 - x[(i, j, k)]),
                    name=f"mc3_{i}_{j}_{k}")

    model.optimize()

    result = {
        "problem_type": "weighted_tardy_jobs",
        "machine_type": machine_type,
        "num_jobs": n,
        "num_machines": m,
        "status": model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
        }.get(model.Status, f"OTHER_{model.Status}"),
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["gap"] = model.MIPGap

        tardy = [j for j in jobs if z[j].X > 0.5]
        on_time = [j for j in jobs if z[j].X < 0.5]
        result["tardy_jobs"] = tardy
        result["on_time_jobs"] = on_time
        result["total_tardy_weight"] = sum(weights[j] for j in tardy)
    else:
        result["objective_value"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for Parallel Machine Scheduling (Chen & Powell 1999)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    problem_type = data.get("problem_type", "weighted_completion_time")

    if problem_type == "weighted_completion_time":
        result = solve_weighted_completion_time(data, args.time_limit)
    elif problem_type == "weighted_tardy_jobs":
        result = solve_weighted_tardy_jobs(data, args.time_limit)
    else:
        print(f"Error: Unknown problem type '{problem_type}'")
        sys.exit(1)

    # Ensure objective_value is at the top level
    if result.get("objective_value") is not None:
        # Round to avoid floating point noise for integer-valued objectives
        result["objective_value"] = round(result["objective_value"], 6)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSolution written to: {args.solution_path}")
    if result.get("objective_value") is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
