#!/usr/bin/env python3
"""
Monolithic Gurobi MIP for the IMSOAN model from:
Wang & Jacquillat (2020), "A Stochastic Integer Programming Approach to
Air Traffic Scheduling and Operations," Operations Research.

Two-stage stochastic integer program for air traffic scheduling.
"""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="IMSOAN monolithic Gurobi MIP")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Time limit in seconds")
    parser.add_argument("--rho", type=float, default=0.67,
                        help="Weight parameter rho (default: 0.67)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ----------------------------------------------------------------
    # Load instance
    # ----------------------------------------------------------------
    with open(args.instance_path) as f:
        inst = json.load(f)

    T = inst["num_time_periods"]  # 87
    delta = inst["maximum_displacement_delta"]  # 1
    flights = inst["flights"]
    num_flights = len(flights)
    scenarios = inst["scenarios"]
    num_scenarios = len(scenarios)
    airports = inst["airports"]
    capacity_envelopes = inst["capacity_envelopes"]
    rho = args.rho

    # Build flight lookup by id (ids are 1-based)
    flt = {}  # id -> flight dict
    for fl in flights:
        flt[fl["id"]] = fl

    # Build E^2 (aircraft connections) with tau^(2)
    E2 = []  # list of (i, j, tau)
    E2_set = {}  # (i,j) -> tau
    for ac in inst["aircraft_connections"]:
        i, j, tau = ac["flight_i"], ac["flight_j"], ac["min_connection_time"]
        E2.append((i, j, tau))
        E2_set[(i, j)] = tau

    # Build E^1 = union of aircraft + passenger connections
    # For pairs in both, tau^(1) = max of connection times
    E1_dict = {}  # (i,j) -> tau^(1)
    for ac in inst["aircraft_connections"]:
        i, j, tau = ac["flight_i"], ac["flight_j"], ac["min_connection_time"]
        E1_dict[(i, j)] = max(E1_dict.get((i, j), 0), tau)
    for pc in inst["passenger_connections"]:
        i, j, tau = pc["flight_i"], pc["flight_j"], pc["min_connection_time"]
        E1_dict[(i, j)] = max(E1_dict.get((i, j), 0), tau)
    E1 = [(i, j, tau) for (i, j), tau in E1_dict.items()]

    # Departing/arriving flights by airport
    dep_at = {k: [] for k in airports}
    arr_at = {k: [] for k in airports}
    for fl in flights:
        dep_at[fl["dep_airport"]].append(fl["id"])
        arr_at[fl["arr_airport"]].append(fl["id"])

    # ----------------------------------------------------------------
    # Helper: time period ranges
    # ----------------------------------------------------------------
    def bar_T_dep(i):
        """First-stage w_dep variable range for flight i."""
        S = flt[i]["dep_period"]
        return range(S - delta + 1, S + delta + 1)

    def bar_T_arr(i):
        """First-stage w_arr variable range for flight i."""
        S = flt[i]["arr_period"]
        return range(S - delta + 1, S + delta + 1)

    def T_dep(i):
        """Second-stage x_dep variable range for flight i."""
        S = flt[i]["dep_period"]
        l_dep = flt[i]["max_dep_delay"]
        return range(S - delta + 1, S + delta + l_dep + 1)

    def T_arr(i):
        """Second-stage x_arr variable range for flight i."""
        S_dep = flt[i]["dep_period"]
        S_arr = flt[i]["arr_period"]
        delta_min = flt[i]["delta_min"]
        l_arr = flt[i]["max_arr_delay"]
        return range(S_dep - delta + delta_min + 1, S_arr + delta + l_arr + 1)

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    model = gp.Model("IMSOAN")
    model.setParam("Threads", 1)

    # --- First-stage variables ---
    w_dep = {}  # (i, t) -> var
    w_arr = {}  # (i, t) -> var
    for fl in flights:
        i = fl["id"]
        for t in bar_T_dep(i):
            w_dep[i, t] = model.addVar(vtype=GRB.BINARY, name=f"w_dep_{i}_{t}")
        for t in bar_T_arr(i):
            w_arr[i, t] = model.addVar(vtype=GRB.BINARY, name=f"w_arr_{i}_{t}")

    # --- Second-stage variables ---
    x_dep = {}  # (i, t, s_idx) -> var
    x_arr = {}  # (i, t, s_idx) -> var
    v_dep = {}  # (i, s_idx) -> var
    v_arr = {}  # (i, s_idx) -> var

    for s_idx in range(num_scenarios):
        for fl in flights:
            i = fl["id"]
            for t in T_dep(i):
                x_dep[i, t, s_idx] = model.addVar(vtype=GRB.BINARY,
                                                   name=f"x_dep_{i}_{t}_{s_idx}")
            for t in T_arr(i):
                x_arr[i, t, s_idx] = model.addVar(vtype=GRB.BINARY,
                                                   name=f"x_arr_{i}_{t}_{s_idx}")
            v_dep[i, s_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                           name=f"v_dep_{i}_{s_idx}")
            v_arr[i, s_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                           name=f"v_arr_{i}_{s_idx}")

    model.update()

    # ----------------------------------------------------------------
    # Helpers: get variable or constant value for w/x at any time t
    # ----------------------------------------------------------------
    def get_w_dep(i, t):
        """Return w_dep variable or constant for flight i at time t."""
        S = flt[i]["dep_period"]
        if t <= S - delta:
            return 1  # convention: fixed to 1
        elif t > S + delta:
            return 0  # convention: fixed to 0
        else:
            return w_dep[i, t]

    def get_w_arr(i, t):
        S = flt[i]["arr_period"]
        if t <= S - delta:
            return 1
        elif t > S + delta:
            return 0
        else:
            return w_arr[i, t]

    def get_x_dep(i, t, s_idx):
        S = flt[i]["dep_period"]
        l_dep = flt[i]["max_dep_delay"]
        if t <= S - delta:
            return 1
        elif t > S + delta + l_dep:
            return 0
        else:
            return x_dep[i, t, s_idx]

    def get_x_arr(i, t, s_idx):
        S_dep = flt[i]["dep_period"]
        S_arr = flt[i]["arr_period"]
        delta_min = flt[i]["delta_min"]
        l_arr = flt[i]["max_arr_delay"]
        if t <= S_dep - delta + delta_min:
            return 1
        elif t > S_arr + delta + l_arr:
            return 0
        else:
            return x_arr[i, t, s_idx]

    # Full sum helpers (sum over all T periods)
    def full_sum_w_dep(i):
        S = flt[i]["dep_period"]
        const = S - delta  # sum of w=1 for t=1..S-delta
        return const + gp.quicksum(w_dep[i, t] for t in bar_T_dep(i))

    def full_sum_w_arr(i):
        S = flt[i]["arr_period"]
        const = S - delta
        return const + gp.quicksum(w_arr[i, t] for t in bar_T_arr(i))

    def full_sum_x_dep(i, s_idx):
        S = flt[i]["dep_period"]
        const = S - delta
        return const + gp.quicksum(x_dep[i, t, s_idx] for t in T_dep(i))

    def full_sum_x_arr(i, s_idx):
        S_dep = flt[i]["dep_period"]
        delta_min = flt[i]["delta_min"]
        const = S_dep - delta + delta_min
        return const + gp.quicksum(x_arr[i, t, s_idx] for t in T_arr(i))

    # ----------------------------------------------------------------
    # Objective (1a)
    # ----------------------------------------------------------------
    obj = gp.LinExpr()

    # First-stage: displacement cost
    for fl in flights:
        i = fl["id"]
        S = fl["dep_period"]
        for t in bar_T_dep(i):
            g_it = abs(t - S)
            # (w_{i,t-1}^dep - w_{it}^dep) is the probability flight departs at t
            w_prev = get_w_dep(i, t - 1)
            w_cur = w_dep[i, t]
            # g_it * (w_prev - w_cur)
            if isinstance(w_prev, (int, float)):
                obj += rho * g_it * (w_prev - w_cur)
            else:
                obj += rho * g_it * (w_prev - w_cur)

    # Second-stage: expected delay cost
    for s_idx in range(num_scenarios):
        p_s = scenarios[s_idx]["probability"]
        for fl in flights:
            i = fl["id"]
            c_dep = fl["cost_dep_delay"]
            c_arr = fl["cost_arr_delay"]
            obj += (1 - rho) * p_s * (c_dep * v_dep[i, s_idx] + c_arr * v_arr[i, s_idx])

    model.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # First-stage constraints
    # ----------------------------------------------------------------

    # (1b): w_dep non-increasing
    for fl in flights:
        i = fl["id"]
        for t in bar_T_dep(i):
            w_prev = get_w_dep(i, t - 1)
            if isinstance(w_prev, (int, float)):
                # w_dep[i,t] <= constant
                model.addConstr(w_dep[i, t] <= w_prev, name=f"1b_{i}_{t}")
            else:
                model.addConstr(w_dep[i, t] <= w_prev, name=f"1b_{i}_{t}")

    # (1c): w_arr non-increasing
    for fl in flights:
        i = fl["id"]
        for t in bar_T_arr(i):
            w_prev = get_w_arr(i, t - 1)
            if isinstance(w_prev, (int, float)):
                model.addConstr(w_arr[i, t] <= w_prev, name=f"1c_{i}_{t}")
            else:
                model.addConstr(w_arr[i, t] <= w_prev, name=f"1c_{i}_{t}")

    # (1d): scheduled en-route time
    for fl in flights:
        i = fl["id"]
        delta_sch = fl["delta_sch"]
        model.addConstr(full_sum_w_arr(i) - full_sum_w_dep(i) == delta_sch,
                        name=f"1d_{i}")

    # (1e): first-stage connection constraints
    for (fi, fj, tau) in E1:
        model.addConstr(full_sum_w_dep(fj) - full_sum_w_arr(fi) >= tau,
                        name=f"1e_{fi}_{fj}")

    # ----------------------------------------------------------------
    # Second-stage constraints (for each scenario)
    # ----------------------------------------------------------------
    for s_idx in range(num_scenarios):
        # (2b): x_dep non-increasing
        for fl in flights:
            i = fl["id"]
            for t in T_dep(i):
                x_prev = get_x_dep(i, t - 1, s_idx)
                if isinstance(x_prev, (int, float)):
                    model.addConstr(x_dep[i, t, s_idx] <= x_prev,
                                    name=f"2b_{i}_{t}_{s_idx}")
                else:
                    model.addConstr(x_dep[i, t, s_idx] <= x_prev,
                                    name=f"2b_{i}_{t}_{s_idx}")

        # (2c): x_arr non-increasing
        for fl in flights:
            i = fl["id"]
            for t in T_arr(i):
                x_prev = get_x_arr(i, t - 1, s_idx)
                if isinstance(x_prev, (int, float)):
                    model.addConstr(x_arr[i, t, s_idx] <= x_prev,
                                    name=f"2c_{i}_{t}_{s_idx}")
                else:
                    model.addConstr(x_arr[i, t, s_idx] <= x_prev,
                                    name=f"2c_{i}_{t}_{s_idx}")

        # (2d): departure delay definition
        for fl in flights:
            i = fl["id"]
            model.addConstr(
                full_sum_x_dep(i, s_idx) - full_sum_w_dep(i) == v_dep[i, s_idx],
                name=f"2d_{i}_{s_idx}")

        # (2e): arrival delay definition (inequality)
        for fl in flights:
            i = fl["id"]
            model.addConstr(
                full_sum_x_arr(i, s_idx) - full_sum_w_arr(i) <= v_arr[i, s_idx],
                name=f"2e_{i}_{s_idx}")

        # (2f): departure delay bound
        for fl in flights:
            i = fl["id"]
            l_dep = fl["max_dep_delay"]
            model.addConstr(v_dep[i, s_idx] <= l_dep, name=f"2f_{i}_{s_idx}")

        # (2g): arrival delay bound
        for fl in flights:
            i = fl["id"]
            l_arr = fl["max_arr_delay"]
            model.addConstr(v_arr[i, s_idx] <= l_arr, name=f"2g_{i}_{s_idx}")

        # (2h): aircraft connection constraints (second-stage)
        for (fi, fj, tau) in E2:
            model.addConstr(
                full_sum_x_dep(fj, s_idx) - full_sum_x_arr(fi, s_idx) >= tau,
                name=f"2h_{fi}_{fj}_{s_idx}")

        # (2i): minimum en-route time
        for fl in flights:
            i = fl["id"]
            delta_min = fl["delta_min"]
            model.addConstr(
                full_sum_x_arr(i, s_idx) - full_sum_x_dep(i, s_idx) >= delta_min,
                name=f"2i_{i}_{s_idx}")

        # (2j): maximum en-route time
        for fl in flights:
            i = fl["id"]
            delta_max = fl["delta_max"]
            model.addConstr(
                full_sum_x_arr(i, s_idx) - full_sum_x_dep(i, s_idx) <= delta_max,
                name=f"2j_{i}_{s_idx}")

        # (2k): capacity envelope constraints
        for k in airports:
            env_vmc = capacity_envelopes[k]["VMC"]
            env_imc = capacity_envelopes[k]["IMC"]
            oc = scenarios[s_idx]["operating_conditions"][k]

            for t in range(1, T + 1):
                # Operating condition at airport k, time t, scenario s
                phi = oc[t - 1]  # 0-indexed
                if phi == "VMC":
                    segments = env_vmc
                else:
                    segments = env_imc

                # Compute departure and arrival counts at time t
                # dep_count = sum over dep flights at k: (x_dep[i,t-1,s] - x_dep[i,t,s])
                # arr_count = sum over arr flights at k: (x_arr[i,t-1,s] - x_arr[i,t,s])
                dep_terms = []
                for i in dep_at[k]:
                    xp = get_x_dep(i, t - 1, s_idx)
                    xc = get_x_dep(i, t, s_idx)
                    # (xp - xc) can be var expression, constant, or mix
                    dep_terms.append((xp, xc))

                arr_terms = []
                for i in arr_at[k]:
                    xp = get_x_arr(i, t - 1, s_idx)
                    xc = get_x_arr(i, t, s_idx)
                    arr_terms.append((xp, xc))

                for seg in segments:
                    a_kq = seg["a"]
                    b_kq = seg["b"]
                    Q_kq = seg["Q"]

                    lhs = gp.LinExpr()
                    if a_kq != 0:
                        for (xp, xc) in dep_terms:
                            lhs += a_kq * (xp - xc)
                    if b_kq != 0:
                        for (xp, xc) in arr_terms:
                            lhs += b_kq * (xp - xc)

                    model.addConstr(lhs <= Q_kq,
                                    name=f"2k_{k}_{t}_{seg['a']}_{seg['b']}_{s_idx}")

        # (2l): valid inequality - no early departure
        for fl in flights:
            i = fl["id"]
            # For t in bar_T_dep (where w_dep is a variable and x_dep is also a variable)
            for t in bar_T_dep(i):
                if (i, t, s_idx) in x_dep:
                    model.addConstr(x_dep[i, t, s_idx] >= w_dep[i, t],
                                    name=f"2l_{i}_{t}_{s_idx}")

        # (2m): valid inequality - leverages en-route time savings
        for fl in flights:
            i = fl["id"]
            delta_sch = fl["delta_sch"]
            delta_min = fl["delta_min"]
            shift = delta_sch - delta_min
            # For t in bar_T_arr where w_arr is a variable
            for t in bar_T_arr(i):
                t_shifted = t - shift
                # Check if x_arr at t_shifted is a variable
                if (i, t_shifted, s_idx) in x_arr:
                    model.addConstr(x_arr[i, t_shifted, s_idx] >= w_arr[i, t],
                                    name=f"2m_{i}_{t}_{s_idx}")

    # ----------------------------------------------------------------
    # Optimize
    # ----------------------------------------------------------------
    model.setParam("TimeLimit", args.time_limit)
    model.optimize()

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # First-stage `schedule`: per flight, recover scheduled_dep / scheduled_arr
        # from the non-increasing w_dep / w_arr indicator sums plus the constant
        # head (S - delta) of 1's that lie outside bar_T.
        schedule = {}
        for fl in flights:
            i = fl["id"]
            S_dep = fl["dep_period"]
            S_arr = fl["arr_period"]
            head_dep = S_dep - delta
            head_arr = S_arr - delta
            sd = head_dep + sum(int(round(w_dep[i, t].X)) for t in bar_T_dep(i))
            sa = head_arr + sum(int(round(w_arr[i, t].X)) for t in bar_T_arr(i))
            schedule[str(i)] = {
                "scheduled_dep": int(sd),
                "scheduled_arr": int(sa),
            }
        result["schedule"] = schedule

        # Second-stage `scenario_solutions`: one entry per scenario containing
        # x_dep, x_arr (binary indicator dicts keyed "{fid}_{t}") and v_dep,
        # v_arr (continuous delays keyed by str(fid)).
        scenario_solutions = []
        for s_idx in range(num_scenarios):
            sc = scenarios[s_idx]
            x_dep_d = {}
            x_arr_d = {}
            v_dep_d = {}
            v_arr_d = {}
            for fl in flights:
                fid = fl["id"]
                for t in T_dep(fid):
                    x_dep_d[f"{fid}_{t}"] = int(round(x_dep[fid, t, s_idx].X))
                for t in T_arr(fid):
                    x_arr_d[f"{fid}_{t}"] = int(round(x_arr[fid, t, s_idx].X))
                v_dep_d[str(fid)] = float(v_dep[fid, s_idx].X)
                v_arr_d[str(fid)] = float(v_arr[fid, s_idx].X)
            scenario_solutions.append({
                "scenario_id": sc["id"],
                "x_dep": x_dep_d,
                "x_arr": x_arr_d,
                "v_dep": v_dep_d,
                "v_arr": v_arr_d,
            })
        result["scenario_solutions"] = scenario_solutions
    else:
        result["objective_value"] = None

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {model.ObjVal}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
