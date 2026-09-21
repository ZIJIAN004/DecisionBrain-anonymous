"""
Deterministic Equivalent MILP for Stochastic Planning and Scheduling
Paper: "Stochastic Planning and Scheduling with Logic-Based Benders Decomposition"
       Elci & Hooker (2020)
Model: Eq. (22) – Minimum Makespan variant

Usage:
    python gurobi_code.py --instance_path instance_1.json \
                          --solution_path gurobi_solution_1.json \
                          --time_limit 3600
"""

import argparse
import json
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
# Helpers
# ---------------------------------------------------------------------------

def compute_time_horizon(n, m, S, r, p):
    """
    Upper bound on the scheduling horizon.
    NOT SPECIFIED IN PAPER exactly.
    INFERRED ASSUMPTION: T_max = max_j(r_j) + sum_j(max_{i,omega}(p^omega_ij)).
    This is a valid upper bound because even if all jobs run sequentially
    starting from the latest release time, the makespan cannot exceed this value.
    """
    max_release = max(r)
    # For each job, the largest processing time across all facilities and scenarios
    max_proc_per_job = [
        max(p[omega][i][j] for i in range(m) for omega in range(S))
        for j in range(n)
    ]
    return max_release + sum(max_proc_per_job)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(instance_path, solution_path, time_limit):
    # ---- Load instance ----
    with open(instance_path) as f:
        inst = json.load(f)

    n   = inst["n"]                        # number of jobs
    m   = inst["m"]                        # number of facilities
    S   = inst["num_scenarios"]
    K   = inst["facility_capacity_limits"] # K[i]
    c   = inst["capacity_requirements"]    # c[i][j]
    r   = inst["release_times"]            # r[j]
    scenarios = inst["scenarios"]
    pi  = [sc["probability"] for sc in scenarios]
    p   = [sc["processing_times"] for sc in scenarios]  # p[omega][i][j]

    T_max = compute_time_horizon(n, m, S, r, p)

    # ---- Build model ----
    model = gp.Model("DetEquivMILP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))

    # ----------------------------------------------------------------
    # Variables
    # ----------------------------------------------------------------

    # x[i,j] in {0,1}: 1 if job j assigned to facility i  (Eq. 22d)
    x = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # beta[omega] >= 0: scenario-level makespan surrogate  (Eq. 22a)
    beta = {
        omega: model.addVar(lb=0.0, name=f"beta_{omega}")
        for omega in range(S)
    }

    # beta_iw[i,omega] >= 0: facility–scenario makespan surrogate  (Eq. 22c)
    beta_iw = {
        (i, omega): model.addVar(lb=0.0, name=f"biw_{i}_{omega}")
        for i in range(m) for omega in range(S)
    }

    # z[omega,i,j,t] in {0,1}: 1 if job j starts at time t on facility i, scenario omega
    # Only create for valid (omega, i, j, t):  r[j] <= t <= T_max - p[omega][i][j]
    z = {}
    for omega in range(S):
        for i in range(m):
            for j in range(n):
                p_ij = p[omega][i][j]
                t_end = T_max - p_ij
                for t in range(r[j], max(r[j], t_end) + 1):
                    z[omega, i, j, t] = model.addVar(
                        vtype=GRB.BINARY, name=f"z_{omega}_{i}_{j}_{t}"
                    )

    # ----------------------------------------------------------------
    # Objective  (Eq. 22a): minimize sum_omega pi_omega * beta_omega
    # ----------------------------------------------------------------
    model.setObjective(
        gp.quicksum(pi[omega] * beta[omega] for omega in range(S)),
        GRB.MINIMIZE
    )

    # ----------------------------------------------------------------
    # Constraint (22b): each job assigned to exactly one facility
    # ----------------------------------------------------------------
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(m)) == 1,
            name=f"assign_{j}"
        )

    # ----------------------------------------------------------------
    # Constraint (22c): beta_omega >= beta_iw[i,omega]
    # ----------------------------------------------------------------
    for omega in range(S):
        for i in range(m):
            model.addConstr(
                beta[omega] >= beta_iw[i, omega],
                name=f"betadecomp_{omega}_{i}"
            )

    # ----------------------------------------------------------------
    # Constraint (22e): beta_iw[i,omega] >= finish time of job j on facility i
    # beta_iw[i,omega] >= sum_t (t + p_ij) * z_ijt   for all i, j, omega
    # Note: z_ijt = 0 when job j is NOT assigned to facility i (from constraint f),
    # so this properly bounds the facility makespan.
    # ----------------------------------------------------------------
    for omega in range(S):
        for i in range(m):
            for j in range(n):
                p_ij = p[omega][i][j]
                valid = [
                    (t, z[omega, i, j, t])
                    for t in range(r[j], T_max - p_ij + 1)
                    if (omega, i, j, t) in z
                ]
                if valid:
                    model.addConstr(
                        beta_iw[i, omega] >= gp.quicksum(
                            (t + p_ij) * zv for t, zv in valid
                        ),
                        name=f"mksp_{omega}_{i}_{j}"
                    )

    # ----------------------------------------------------------------
    # Constraint (22f): z_ijt <= x_ij
    # ----------------------------------------------------------------
    for (omega, i, j, t), zv in z.items():
        model.addConstr(zv <= x[i, j], name=f"link_{omega}_{i}_{j}_{t}")

    # ----------------------------------------------------------------
    # Constraint (22g): each job scheduled in exactly one (facility, time) slot
    #                   per scenario
    # ----------------------------------------------------------------
    for omega in range(S):
        for j in range(n):
            model.addConstr(
                gp.quicksum(
                    z[omega, i, j, t]
                    for i in range(m)
                    for t in range(r[j], T_max - p[omega][i][j] + 1)
                    if (omega, i, j, t) in z
                ) == 1,
                name=f"sched_{omega}_{j}"
            )

    # ----------------------------------------------------------------
    # Constraint (22h): cumulative resource capacity at each time point.
    #
    # Paper states: T^omega_tij = {t' | 0 <= t' <= t - p^omega_ij}
    # However, applying the standard cumulative constraint interpretation,
    # job j (started at t') is ACTIVE at time t if t' <= t < t' + p_ij,
    # i.e., t - p_ij + 1 <= t' <= t.
    #
    # INFERRED ASSUMPTION: We use the standard cumulative formulation where
    # job j is active at time t iff its start time t' satisfies
    # max(r_j, t - p_ij + 1) <= t' <= t. This is the correct formulation
    # for a cumulative resource constraint and matches the CP subproblem (11).
    # ----------------------------------------------------------------
    for omega in range(S):
        for i in range(m):
            for t in range(T_max + 1):
                parts = []
                for j in range(n):
                    p_ij = p[omega][i][j]
                    t_lo = max(r[j], t - p_ij + 1)
                    t_hi = t
                    for t_prime in range(t_lo, t_hi + 1):
                        if (omega, i, j, t_prime) in z:
                            parts.append(c[i][j] * z[omega, i, j, t_prime])
                if parts:
                    model.addConstr(
                        gp.quicksum(parts) <= K[i],
                        name=f"cap_{omega}_{i}_{t}"
                    )

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # Extract and write solution
    # ----------------------------------------------------------------
    result = {
        "objective_value": None,
        "status": "no_solution",
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.status == GRB.OPTIMAL else "time_limit"

        assignment = {}
        for j in range(n):
            for i in range(m):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
        result["assignment"] = assignment

        schedule = {}
        for omega in range(S):
            schedule[str(omega)] = {}
            for j in range(n):
                for i in range(m):
                    for t in range(r[j], T_max + 1):
                        if (omega, i, j, t) in z and z[omega, i, j, t].X > 0.5:
                            schedule[str(omega)][str(j)] = {
                                "facility": i,
                                "start": t,
                                "finish": t + p[omega][i][j],
                            }
        result["schedule"] = schedule

    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Objective value : {result['objective_value']}")
    print(f"Status          : {result['status']}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve stochastic planning & scheduling via deterministic "
                    "equivalent MILP (Eq. 22, Elci & Hooker 2020)."
    )
    parser.add_argument(
        "--instance_path", required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", required=True,
        help="Path where the solution JSON will be written."
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Maximum solver time in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)
