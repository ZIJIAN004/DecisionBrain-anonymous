"""
Gurobi implementation of the Execution-Interval (EI) formulation for the
Resource Loading Problem (RLP).

Paper: Song, Kis & Leus (2020), "Polyhedral Results and Branch-and-Cut for
       the Resource Loading Problem", INFORMS Journal on Computing.

This implements the Execution-Interval formulation (Section 4.1, Eqs 14-17 + 6-9):
  - Binary variables a^j_{kl} for choosing execution intervals
  - Continuous variables y_{jt} for intensity assignments
  - Continuous variables z_t for nonregular capacity

The formulation is provably stronger than the three time-indexed formulations
(Proposition 4 of the paper).
"""

import json
import argparse
import math
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


def load_instance(instance_path):
    """Load problem instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def enumerate_execution_intervals(order, H):
    """
    Enumerate all feasible execution intervals for an order.

    A feasible execution interval (k, l) for order j must satisfy:
      r_j <= k <= l <= H
      (l - k + 1) * LB_j <= 1
      (l - k + 1) * UB_j >= 1
    """
    r_j = order['release_date_r']
    LB_j = order['intensity_lower_bound_LB']
    UB_j = order['intensity_upper_bound_UB']

    intervals = []
    for k in range(r_j, H + 1):
        for l in range(k, H + 1):
            length = l - k + 1
            if length * LB_j <= 1.0 + 1e-9 and length * UB_j >= 1.0 - 1e-9:
                intervals.append((k, l))
    return intervals


def compute_interval_bounds(order, k, l):
    """
    Compute tighter per-interval bounds (Eqs 25-26).

    UB_{kl} = min{ UB_j,  1 - LB_j * (l - k) }
    LB_{kl} = max{ LB_j,  1 - UB_j * (l - k) }
    """
    LB_j = order['intensity_lower_bound_LB']
    UB_j = order['intensity_upper_bound_UB']

    UB_kl = min(UB_j, 1.0 - LB_j * (l - k))
    LB_kl = max(LB_j, 1.0 - UB_j * (l - k))

    return LB_kl, UB_kl


def solve_rlp(instance_path, solution_path, time_limit):
    """Build and solve the EI formulation using Gurobi."""
    data = load_instance(instance_path)

    H = data['planning_horizon_H']
    n = data['num_orders_n']
    sigma = data['unit_cost_nonregular_capacity_sigma']
    capacities = data['capacities']
    orders = data['orders']

    # Preprocess: enumerate execution intervals and compute parameters
    E = {}  # E[j] = list of (k, l) tuples
    T_tard = {}  # T_tard[j][(k,l)] = tardiness for interval
    LB_kl = {}  # LB_kl[j][(k,l)] = tighter lower bound
    UB_kl = {}  # UB_kl[j][(k,l)] = tighter upper bound

    for order in orders:
        j = order['order_id']
        d_j = order['due_date_d']

        E[j] = enumerate_execution_intervals(order, H)
        T_tard[j] = {}
        LB_kl[j] = {}
        UB_kl[j] = {}

        for (k, l) in E[j]:
            T_tard[j][(k, l)] = max(0, l - d_j)
            lb, ub = compute_interval_bounds(order, k, l)
            LB_kl[j][(k, l)] = lb
            UB_kl[j][(k, l)] = ub

    # Build Gurobi model
    model = gp.Model("RLP_EI")
    model.setParam("TimeLimit", time_limit)
    # Paper uses single thread
    model.setParam("Threads", 1)

    # Decision variables
    # a^j_{kl} in {0,1}: binary execution interval selection
    a = {}
    for order in orders:
        j = order['order_id']
        for (k, l) in E[j]:
            a[j, k, l] = model.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}_{l}")

    # y_{jt} >= 0: intensity of order j in period t
    y = {}
    for order in orders:
        j = order['order_id']
        r_j = order['release_date_r']
        for t in range(r_j, H + 1):
            y[j, t] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                   name=f"y_{j}_{t}")

    # z_t >= 0: nonregular capacity in period t
    z = {}
    for t in range(1, H + 1):
        z[t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{t}")

    model.update()

    # Objective function (Eq. 14):
    # min sum_j sum_{(k,l) in E_j} w_j * T^j_{kl} * a^j_{kl} + sigma * sum_t z_t
    obj = gp.LinExpr()
    for order in orders:
        j = order['order_id']
        w_j = order['tardiness_cost_w']
        for (k, l) in E[j]:
            if T_tard[j][(k, l)] > 0:
                obj += w_j * T_tard[j][(k, l)] * a[j, k, l]
    for t in range(1, H + 1):
        obj += sigma * z[t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (16): Exactly one execution interval per order
    # sum_{(k,l) in E_j} a^j_{kl} = 1   for all j
    for order in orders:
        j = order['order_id']
        model.addConstr(
            gp.quicksum(a[j, k, l] for (k, l) in E[j]) == 1,
            name=f"one_interval_{j}"
        )

    # Constraint (15): Intensity bounds linking y and a variables
    # LB_j * sum_{(k,l) in E_j: k<=t<=l} a^j_{kl} <= y_{jt}
    #   <= UB_j * sum_{(k,l) in E_j: k<=t<=l} a^j_{kl}
    for order in orders:
        j = order['order_id']
        r_j = order['release_date_r']
        LB_j = order['intensity_lower_bound_LB']
        UB_j = order['intensity_upper_bound_UB']

        for t in range(r_j, H + 1):
            covering = [(k, l) for (k, l) in E[j] if k <= t <= l]
            if covering:
                sum_a = gp.quicksum(a[j, k, l] for (k, l) in covering)
                model.addConstr(y[j, t] >= LB_j * sum_a,
                                name=f"lb_intensity_{j}_{t}")
                model.addConstr(y[j, t] <= UB_j * sum_a,
                                name=f"ub_intensity_{j}_{t}")
            else:
                # No execution interval covers this period, intensity must be 0
                model.addConstr(y[j, t] == 0,
                                name=f"zero_intensity_{j}_{t}")

    # Constraint (6): Total intensity equals 1
    # sum_{t=r_j}^{H} y_{jt} = 1   for all j
    for order in orders:
        j = order['order_id']
        r_j = order['release_date_r']
        model.addConstr(
            gp.quicksum(y[j, t] for t in range(r_j, H + 1)) == 1,
            name=f"total_intensity_{j}"
        )

    # Constraint (7): Nonregular capacity
    # z_t >= sum_j y_{jt} * p_j - C_t   for all t
    for t in range(1, H + 1):
        C_t = capacities[t - 1]  # 0-indexed in the array
        resource_usage = gp.LinExpr()
        for order in orders:
            j = order['order_id']
            r_j = order['release_date_r']
            p_j = order['work_content_p']
            if t >= r_j:
                resource_usage += y[j, t] * p_j
        model.addConstr(z[t] >= resource_usage - C_t,
                        name=f"nonregular_cap_{t}")

    # Solve
    print(f"Model has {model.NumVars} variables and {model.NumConstrs} constraints")
    model.optimize()

    # Extract solution
    result = {
        "problem_name": data.get("problem_name", "RLP"),
        "solver": "Gurobi",
        "formulation": "Execution-Interval (EI)",
        "status": model.Status,
        "status_description": "",
        "objective_value": None,
        "solve_time_seconds": model.Runtime,
        "gap": None,
        "orders_solution": []
    }

    if model.Status == GRB.OPTIMAL:
        result["status_description"] = "Optimal"
        result["objective_value"] = model.ObjVal
        result["gap"] = 0.0
    elif model.Status == GRB.TIME_LIMIT:
        result["status_description"] = "Time limit reached"
        if model.SolCount > 0:
            result["objective_value"] = model.ObjVal
            result["gap"] = model.MIPGap
        else:
            result["objective_value"] = None
            result["gap"] = None
    elif model.SolCount > 0:
        result["status_description"] = f"Status code {model.Status}"
        result["objective_value"] = model.ObjVal
        result["gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None
    else:
        result["status_description"] = f"No feasible solution found (status {model.Status})"
        result["objective_value"] = None

    # Extract detailed solution if available
    if model.SolCount > 0:
        for order in orders:
            j = order['order_id']
            r_j = order['release_date_r']
            d_j = order['due_date_d']
            w_j = order['tardiness_cost_w']

            # Find chosen execution interval
            chosen_interval = None
            for (k, l) in E[j]:
                if a[j, k, l].X > 0.5:
                    chosen_interval = (k, l)
                    break

            # Extract intensities
            intensities = {}
            for t in range(r_j, H + 1):
                val = y[j, t].X
                if val > 1e-8:
                    intensities[str(t)] = round(val, 8)

            tardiness = 0
            if chosen_interval:
                tardiness = max(0, chosen_interval[1] - d_j)

            order_sol = {
                "order_id": j,
                "chosen_interval": list(chosen_interval) if chosen_interval else None,
                "tardiness": tardiness,
                "tardiness_cost": w_j * tardiness,
                "intensities": intensities
            }
            result["orders_solution"].append(order_sol)

        # Compute nonregular capacity details
        nonregular_capacity = {}
        total_nonregular = 0.0
        for t in range(1, H + 1):
            val = z[t].X
            if val > 1e-8:
                nonregular_capacity[str(t)] = round(val, 8)
                total_nonregular += val
        result["total_nonregular_capacity"] = round(total_nonregular, 8)
        result["nonregular_capacity_by_period"] = nonregular_capacity

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSolution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status_description']}")
    if result["gap"] is not None:
        print(f"Gap: {result['gap']:.4%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Resource Loading Problem using the Execution-Interval "
                    "formulation with Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_rlp(args.instance_path, args.solution_path, args.time_limit)
