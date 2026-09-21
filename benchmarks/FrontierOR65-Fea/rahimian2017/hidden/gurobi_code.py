#!/usr/bin/env python3
"""
Gurobi IP formulation for the Nurse Rostering Problem (NRP).
Based on: Rahimian, Akartunali, and Levine (2017)
"A Hybrid Integer Programming and Variable Neighbourhood Search Algorithm
 to Solve Nurse Rostering Problems"

Implements the full IP model from Section 2 of the paper.
"""

import argparse
import json
import os
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
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def build_and_solve(instance, time_limit):
    """
    Build and solve the NRP IP model as described in Section 2 of the paper.
    """
    num_days = instance["num_days"]
    num_nurses = instance["num_nurses"]
    num_shift_types = instance["num_shift_types"]
    num_weekends = instance["num_weekends"]

    # Sets
    D = list(range(num_days))           # days 0..num_days-1
    I = list(range(num_nurses))         # nurses
    T = list(range(num_shift_types))    # shift types
    W = list(range(1, num_weekends + 1))  # weekends 1-indexed as per paper

    # Shift type data
    shift_lengths = {}
    for st in instance["shift_types"]:
        shift_lengths[st["id"]] = st["length_minutes"]

    # Forbidden shift rotations: R_t = set of shift types that cannot follow t
    R = {t: set() for t in T}
    for rot in instance["forbidden_shift_rotations"]:
        preceding = rot["shift_type"]
        following = rot["cannot_be_followed_by"]
        R[preceding].add(following)

    # Nurse-specific data
    nurses = instance["nurses"]
    max_shifts_per_type = {}  # m_{it}^{max}
    min_total_minutes = {}     # b_i^{min}
    max_total_minutes = {}     # b_i^{max}
    min_consecutive = {}       # c_i^{min}
    max_consecutive = {}       # c_i^{max}
    min_consecutive_off = {}   # o_i^{min}
    max_weekends_nurse = {}    # a_i^{max}
    days_off = {}              # N_i

    for nurse in nurses:
        i = nurse["id"]
        max_shifts_per_type[i] = {}
        for t_str, val in nurse["max_shifts_per_type"].items():
            max_shifts_per_type[i][int(t_str)] = val
        min_total_minutes[i] = nurse["min_total_minutes"]
        max_total_minutes[i] = nurse["max_total_minutes"]
        min_consecutive[i] = nurse["min_consecutive_shifts"]
        max_consecutive[i] = nurse["max_consecutive_shifts"]
        min_consecutive_off[i] = nurse["min_consecutive_days_off"]
        max_weekends_nurse[i] = nurse["max_weekends"]
        days_off[i] = set(nurse["day_off_requests"])

    # Shift on/off request penalties
    # q_{idt}: penalty if shift t is NOT assigned to nurse i on day d (shift-on request)
    # p_{idt}: penalty if shift t IS assigned to nurse i on day d (shift-off request)
    q = {}
    p = {}
    for i in I:
        for d in D:
            for t in T:
                q[i, d, t] = 0
                p[i, d, t] = 0

    for req in instance["shift_on_requests"]:
        q[req["nurse_id"], req["day"], req["shift_type"]] = req["penalty"]

    for req in instance["shift_off_requests"]:
        p[req["nurse_id"], req["day"], req["shift_type"]] = req["penalty"]

    # Coverage requirements
    u = {}   # u_{dt}: preferred number of nurses
    w_min = {}  # w_{dt}^{min}: under-coverage weight
    w_max = {}  # w_{dt}^{max}: over-coverage weight
    for cov in instance["coverage_requirements"]:
        d = cov["day"]
        t = cov["shift_type"]
        u[d, t] = cov["preferred"]
        w_min[d, t] = cov["under_weight"]
        w_max[d, t] = cov["over_weight"]

    # Ensure all (d,t) pairs have coverage data; default to 0 if missing
    for d in D:
        for t in T:
            if (d, t) not in u:
                u[d, t] = 0
                w_min[d, t] = 0
                w_max[d, t] = 0

    # =========================================================================
    # Model
    # =========================================================================
    model = gp.Model("NRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x_{idt}: binary, nurse i assigned shift t on day d
    x = {}
    for i in I:
        for d in D:
            for t in T:
                x[i, d, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{d}_{t}")

    # k_{iw}: binary, nurse i works on weekend w
    k = {}
    for i in I:
        for w in W:
            k[i, w] = model.addVar(vtype=GRB.BINARY, name=f"k_{i}_{w}")

    # y_{dt}: under-coverage (integer >= 0)
    y = {}
    for d in D:
        for t in T:
            y[d, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{d}_{t}")

    # z_{dt}: over-coverage (integer >= 0)
    z = {}
    for d in D:
        for t in T:
            z[d, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"z_{d}_{t}")

    # v_{idt}: penalty from shift on/off requests (integer >= 0)
    v = {}
    for i in I:
        for d in D:
            for t in T:
                v[i, d, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"v_{i}_{d}_{t}")

    model.update()

    # =========================================================================
    # Objective function
    # =========================================================================
    # min sum_{i,d,t} v_{idt} + sum_{d,t} w_min_{dt} * y_{dt} + sum_{d,t} w_max_{dt} * z_{dt}
    obj = gp.LinExpr()
    for i in I:
        for d in D:
            for t in T:
                obj += v[i, d, t]
    for d in D:
        for t in T:
            obj += w_min[d, t] * y[d, t]
            obj += w_max[d, t] * z[d, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # =========================================================================
    # Constraints
    # =========================================================================

    # HC1: At most one shift per day per nurse
    for i in I:
        for d in D:
            model.addConstr(
                gp.quicksum(x[i, d, t] for t in T) <= 1,
                name=f"HC1_{i}_{d}"
            )

    # HC2: Forbidden shift rotations
    # x_{idt} + x_{i(d+1)u} <= 1 for all forbidden pairs
    # Paper assumes last day of previous period and first day of next period are days off,
    # so we only need d in {0..num_days-2}
    for i in I:
        for d in range(num_days - 1):
            for t in T:
                for u_shift in R[t]:
                    model.addConstr(
                        x[i, d, t] + x[i, d + 1, u_shift] <= 1,
                        name=f"HC2_{i}_{d}_{t}_{u_shift}"
                    )

    # HC3: Maximum number of shifts per type
    for i in I:
        for t in T:
            if t in max_shifts_per_type[i]:
                model.addConstr(
                    gp.quicksum(x[i, d, t] for d in D) <= max_shifts_per_type[i][t],
                    name=f"HC3_{i}_{t}"
                )

    # HC4 & HC5: Min and max total minutes
    for i in I:
        total_minutes = gp.quicksum(
            shift_lengths[t] * x[i, d, t] for d in D for t in T
        )
        model.addConstr(total_minutes >= min_total_minutes[i], name=f"HC5_{i}")
        model.addConstr(total_minutes <= max_total_minutes[i], name=f"HC4_{i}")

    # HC6: Maximum consecutive shifts
    # sum_{j=d}^{d+c_max} sum_t x_{ijt} <= c_max
    # for d in {0..num_days - c_max - 1} (0-indexed)
    for i in I:
        c_max = max_consecutive[i]
        for d in range(num_days - c_max):
            model.addConstr(
                gp.quicksum(
                    x[i, j, t] for j in range(d, d + c_max + 1) for t in T
                ) <= c_max,
                name=f"HC6_{i}_{d}"
            )

    # HC7: Minimum consecutive shifts
    # Paper formulation (HC7):
    # sum_t x_{idt} + (c - 1 - sum_{j=d+1}^{d+c} sum_t x_{ijt}) + sum_t x_{i(d+c+1)t} >= 0
    # for c in {1..c_min-1}, d in {0..num_days-(c+2)} (0-indexed)
    #
    # Paper comment 3: "infinite consecutive shifts at boundaries" means the constraint
    # is not violated at the start/end of the horizon. We only generate constraints
    # for interior positions where d and d+c+1 are valid days.
    for i in I:
        c_min = min_consecutive[i]
        for c in range(1, c_min):
            for d in range(num_days - c - 1):
                # day d is the day before the gap, d+1..d+c is the potential short sequence,
                # d+c+1 is the day after
                lhs = gp.LinExpr()
                # sum_t x_{idt}  (day before)
                lhs += gp.quicksum(x[i, d, t] for t in T)
                # + (c - 1 - sum_{j=d+1}^{d+c} sum_t x_{ijt})
                lhs += c - 1
                lhs -= gp.quicksum(x[i, j, t] for j in range(d + 1, d + c + 1) for t in T)
                # + sum_t x_{i(d+c+1)t}
                lhs += gp.quicksum(x[i, d + c + 1, t] for t in T)
                model.addConstr(lhs >= 0, name=f"HC7_{i}_{c}_{d}")

    # HC8: Minimum consecutive days off
    # (1 - sum_t x_{idt}) + sum_{j=d+1}^{d+b} sum_t x_{ijt} + sum_t x_{i(d+b+1)t} >= 0
    # for b in {1..o_min-1}, d in {0..num_days-(b+2)}
    # Paper comment 4: infinite consecutive days off at boundaries
    for i in I:
        o_min = min_consecutive_off[i]
        for b in range(1, o_min):
            for d in range(num_days - b - 1):
                lhs = gp.LinExpr()
                # (1 - sum_t x_{idt})
                lhs += 1
                lhs -= gp.quicksum(x[i, d, t] for t in T)
                # + sum_{j=d+1}^{d+b} sum_t x_{ijt}
                lhs += gp.quicksum(x[i, j, t] for j in range(d + 1, d + b + 1) for t in T)
                # + sum_t x_{i(d+b+1)t}
                lhs += gp.quicksum(x[i, d + b + 1, t] for t in T)
                model.addConstr(lhs >= 0, name=f"HC8_{i}_{b}_{d}")

    # HC9: Maximum weekends
    # Weekend w: Saturday = day (7w-2) and Sunday = day (7w-1) in 0-indexed
    # Paper uses 1-indexed days: Saturday = 7w-1, Sunday = 7w
    # In 0-indexed: Saturday = 7*w - 2, Sunday = 7*w - 1
    for i in I:
        for w in W:
            sat = 7 * w - 2  # 0-indexed Saturday
            sun = 7 * w - 1  # 0-indexed Sunday
            if sat < num_days and sun < num_days:
                # k_{iw} <= sum_t x_{i,sat,t} + sum_t x_{i,sun,t} <= 2*k_{iw}
                weekend_shifts = gp.quicksum(x[i, sat, t] for t in T) + \
                                 gp.quicksum(x[i, sun, t] for t in T)
                model.addConstr(k[i, w] <= weekend_shifts, name=f"HC9a_{i}_{w}")
                model.addConstr(weekend_shifts <= 2 * k[i, w], name=f"HC9b_{i}_{w}")

        # sum_w k_{iw} <= a_max
        model.addConstr(
            gp.quicksum(k[i, w] for w in W) <= max_weekends_nurse[i],
            name=f"HC9c_{i}"
        )

    # HC10: Requested days off
    for i in I:
        for d_off in days_off[i]:
            for t in T:
                model.addConstr(x[i, d_off, t] == 0, name=f"HC10_{i}_{d_off}_{t}")

    # SC1: Shift on/off requests
    # q_{idt}(1 - x_{idt}) + p_{idt} * x_{idt} = v_{idt}
    # => v_{idt} = q_{idt} - q_{idt}*x_{idt} + p_{idt}*x_{idt}
    # => v_{idt} = q_{idt} + (p_{idt} - q_{idt})*x_{idt}
    for i in I:
        for d in D:
            for t in T:
                model.addConstr(
                    v[i, d, t] == q[i, d, t] + (p[i, d, t] - q[i, d, t]) * x[i, d, t],
                    name=f"SC1_{i}_{d}_{t}"
                )

    # SC2: Coverage
    # sum_i x_{idt} - z_{dt} + y_{dt} = u_{dt}
    for d in D:
        for t in T:
            model.addConstr(
                gp.quicksum(x[i, d, t] for i in I) - z[d, t] + y[d, t] == u[d, t],
                name=f"SC2_{d}_{t}"
            )

    # =========================================================================
    # Solve
    # =========================================================================
    model.optimize()

    # Extract solution
    objective_value = None
    solution = {}

    if model.SolCount > 0:
        objective_value = model.ObjVal
        solution["schedule"] = {}
        for i in I:
            nurse_schedule = {}
            for d in D:
                assigned_shift = None
                for t in T:
                    if x[i, d, t].X > 0.5:
                        assigned_shift = t
                        break
                nurse_schedule[str(d)] = assigned_shift
            solution["schedule"][str(i)] = nurse_schedule

        solution["coverage_under"] = {}
        solution["coverage_over"] = {}
        for d in D:
            for t in T:
                if y[d, t].X > 0.5:
                    solution["coverage_under"][f"{d}_{t}"] = round(y[d, t].X)
                if z[d, t].X > 0.5:
                    solution["coverage_over"][f"{d}_{t}"] = round(z[d, t].X)

    result = {
        "objective_value": objective_value,
        "status": model.Status,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "solution": solution
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi IP solver for the Nurse Rostering Problem (Rahimian et al. 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
