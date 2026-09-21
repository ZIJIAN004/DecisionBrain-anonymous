#!/usr/bin/env python3
"""
TCAPCP_Win MIP formulation solved with Gurobi.

Implements the time-window-based MIP (MIP TCAPCP_Win) for the Trade-Container
Assignment Problem in Construction Projects (TCAPCP) from:
    Dienstknecht & Briskorn (2024), EJOR 315(1), 324-337.

The TCAPCP_Win formulation exploits Property 1: re-assignments only occur in
critical periods P'. This reduces the number of binary variables compared to
the period-based formulation (TCAPCP_Per).

The paper shows TCAPCP_Win is superior to TCAPCP_Per and recommends it
(Section 6.2: "We will, therefore, abandon MIP TCAPCP_Per in favor of MIP
TCAPCP_Win subsequently.").
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
def load_instance(path):
    """Load a TCAPCP instance from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def compute_time_windows(P, containers, trades):
    """
    Compute the set P' of critical periods and derive time windows W.

    P' := {p | p = 2, ..., P : exists j in J such that s_j = p
           OR exists c in C such that f_c + 1 = p}

    Time windows partition the horizon into non-overlapping intervals.
    Each window w_i = [s_{w_i}, f_{w_i}) covers periods s_{w_i} to f_{w_i}-1.
    """
    P_prime = set()
    for t in trades:
        s_j = t['start_period']
        if 2 <= s_j <= P:
            P_prime.add(s_j)
    for c in containers:
        f_c = c['availability_end']
        p = f_c + 1
        if 2 <= p <= P:
            P_prime.add(p)

    # Critical start periods always include period 1
    critical_starts = sorted({1} | P_prime)

    # Derive time windows: w_i = [critical_starts[i], critical_starts[i+1])
    # Last window ends at P+1
    windows = []
    for k in range(len(critical_starts)):
        s_w = critical_starts[k]
        if k + 1 < len(critical_starts):
            f_w = critical_starts[k + 1]
        else:
            f_w = P + 1
        windows.append((s_w, f_w))

    return windows


def precompute_window_data(windows, containers, trades):
    """
    Precompute sets C^{w_i}, J^{w_i}, and adjacency pairs for each window.
    """
    m = len(windows)

    # Build adjacency and neighbor lookup
    neighbors = {}
    for c in containers:
        neighbors[c['id']] = set(c['adjacent_containers'])

    # Ordered adjacency pairs (c1, c2) with c1 < c2
    adj_pairs = set()
    for c in containers:
        for c2_id in c['adjacent_containers']:
            if c['id'] < c2_id:
                adj_pairs.add((c['id'], c2_id))

    trade_by_id = {t['id']: t for t in trades}

    C_w = []  # C_w[i] = set of container IDs available in window i
    J_w = []  # J_w[i] = set of trade IDs active in window i
    adj_pairs_w = []  # adj_pairs_w[i] = adjacency pairs within window i

    for i, (s_w, f_w) in enumerate(windows):
        # C^{w_i} = {c in C : s_c <= s_w AND f_c >= f_w - 1}
        c_set = set()
        for c in containers:
            if c['availability_start'] <= s_w and c['availability_end'] >= f_w - 1:
                c_set.add(c['id'])
        C_w.append(c_set)

        # J^{w_i} = {j in J : s_j <= s_w <= f_j}
        j_set = set()
        for t in trades:
            if t['start_period'] <= s_w and s_w <= t['end_period']:
                j_set.add(t['id'])
        J_w.append(j_set)

        # Adjacency pairs restricted to this window
        pairs = set()
        for (c1, c2) in adj_pairs:
            if c1 in c_set and c2 in c_set:
                pairs.add((c1, c2))
        adj_pairs_w.append(pairs)

    return C_w, J_w, adj_pairs_w, neighbors, trade_by_id


def solve_tcapcp_win(instance_data, time_limit):
    """
    Build and solve the full MIP TCAPCP_Win formulation using Gurobi.
    """
    P = instance_data['problem_parameters']['num_periods']
    containers = instance_data['containers']
    trades = instance_data['trades']

    # Compute time windows
    windows = compute_time_windows(P, containers, trades)
    m = len(windows)

    # Precompute window data
    C_w, J_w, adj_pairs_w, neighbors, trade_by_id = precompute_window_data(
        windows, containers, trades
    )

    # Build Gurobi model
    model = gp.Model("TCAPCP_Win")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # =========================================================================
    # Decision variables
    # =========================================================================

    # x[c, j, i]: binary - 1 if trade j assigned to container c in window i
    x = {}
    for i in range(m):
        for j in J_w[i]:
            for c in C_w[i]:
                x[c, j, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{j}_{i}")

    # y[(c1,c2), j, i]: binary - 1 if c1 and c2 in same cluster, c1 < c2
    y = {}
    for i in range(m):
        for j in J_w[i]:
            for (c1, c2) in adj_pairs_w[i]:
                y[c1, c2, j, i] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{c1}_{c2}_{j}_{i}"
                )

    # z[c, j, i]: binary - 1 if c is source of flow for trade j in window i
    z = {}
    for i in range(m):
        for j in J_w[i]:
            for c in C_w[i]:
                z[c, j, i] = model.addVar(vtype=GRB.BINARY, name=f"z_{c}_{j}_{i}")

    # f[c1, c2, j, i]: continuous >= 0 - flow on directed arc (c1,c2)
    # Created for both directions of each adjacency pair
    f = {}
    for i in range(m):
        for j in J_w[i]:
            for (c1, c2) in adj_pairs_w[i]:
                f[c1, c2, j, i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"f_{c1}_{c2}_{j}_{i}"
                )
                f[c2, c1, j, i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"f_{c2}_{c1}_{j}_{i}"
                )

    # r[j, i]: continuous [0,1] - 1 if trade j re-assigned in window i
    # Relaxed from binary per eq. (34) / paper's eq. (19)
    r = {}
    for i in range(1, m):
        s_w_i = windows[i][0]
        for j in J_w[i]:
            s_j = trade_by_id[j]['start_period']
            if s_j < s_w_i:
                r[j, i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"r_{j}_{i}"
                )

    model.update()

    # =========================================================================
    # Objective function (20): minimize total re-assignments
    # =========================================================================
    model.setObjective(
        gp.quicksum(r[j, i] for (j, i) in r), GRB.MINIMIZE
    )

    # =========================================================================
    # Constraints
    # =========================================================================

    # (21) Demand satisfaction
    for i in range(m):
        for j in J_w[i]:
            n_j = trade_by_id[j]['container_demand']
            model.addConstr(
                gp.quicksum(x[c, j, i] for c in C_w[i]) == n_j,
                name=f"demand_{j}_{i}"
            )

    # (22) Container capacity - at most one trade per container per window
    for i in range(m):
        for c in C_w[i]:
            active_trades = [j for j in J_w[i]]
            if active_trades:
                model.addConstr(
                    gp.quicksum(x[c, j, i] for j in active_trades) <= 1,
                    name=f"capacity_{c}_{i}"
                )

    # (23) Cluster source linking: z <= x
    for i in range(m):
        for j in J_w[i]:
            for c in C_w[i]:
                model.addConstr(
                    z[c, j, i] <= x[c, j, i],
                    name=f"source_{c}_{j}_{i}"
                )

    # (24) y <= x (first endpoint)
    for i in range(m):
        for j in J_w[i]:
            for (c1, c2) in adj_pairs_w[i]:
                model.addConstr(
                    y[c1, c2, j, i] <= x[c1, j, i],
                    name=f"y_x1_{c1}_{c2}_{j}_{i}"
                )

    # (25) y <= x (second endpoint)
    for i in range(m):
        for j in J_w[i]:
            for (c1, c2) in adj_pairs_w[i]:
                model.addConstr(
                    y[c1, c2, j, i] <= x[c2, j, i],
                    name=f"y_x2_{c1}_{c2}_{j}_{i}"
                )

    # (26) Symmetry breaking: 1 - y >= z_{c'} (for c < c')
    # Forces the container with larger index not to be source when adjacent
    # to a same-cluster container with smaller index
    for i in range(m):
        for j in J_w[i]:
            for (c1, c2) in adj_pairs_w[i]:
                model.addConstr(
                    1 - y[c1, c2, j, i] >= z[c2, j, i],
                    name=f"symbreak_{c1}_{c2}_{j}_{i}"
                )

    # (27) Flow conservation / cluster identification
    for i in range(m):
        for j in J_w[i]:
            n_j = trade_by_id[j]['container_demand']
            for c in C_w[i]:
                # Incoming flow to c from adjacent containers in this window
                in_flow = gp.quicksum(
                    f[c2, c, j, i]
                    for c2 in neighbors[c]
                    if c2 in C_w[i]
                    and (min(c, c2), max(c, c2)) in adj_pairs_w[i]
                )
                # Outgoing flow from c to adjacent containers in this window
                out_flow = gp.quicksum(
                    f[c, c2, j, i]
                    for c2 in neighbors[c]
                    if c2 in C_w[i]
                    and (min(c, c2), max(c, c2)) in adj_pairs_w[i]
                )
                model.addConstr(
                    z[c, j, i] * n_j + in_flow - out_flow >= x[c, j, i],
                    name=f"flowcons_{c}_{j}_{i}"
                )

    # (28) Flow capacity (both directions bounded by y * n_j)
    for i in range(m):
        for j in J_w[i]:
            n_j = trade_by_id[j]['container_demand']
            for (c1, c2) in adj_pairs_w[i]:
                # Forward direction
                model.addConstr(
                    f[c1, c2, j, i] <= y[c1, c2, j, i] * n_j,
                    name=f"fcap_fwd_{c1}_{c2}_{j}_{i}"
                )
                # Reverse direction
                model.addConstr(
                    f[c2, c1, j, i] <= y[c1, c2, j, i] * n_j,
                    name=f"fcap_rev_{c2}_{c1}_{j}_{i}"
                )

    # (29) Dispersion limit
    for i in range(m):
        for j in J_w[i]:
            d_max_j = trade_by_id[j]['max_dispersion']
            model.addConstr(
                gp.quicksum(z[c, j, i] for c in C_w[i]) <= d_max_j,
                name=f"dispersion_{j}_{i}"
            )

    # (31) Re-assignment detection: container available in both windows
    for i in range(1, m):
        s_w_i = windows[i][0]
        for j in J_w[i]:
            s_j = trade_by_id[j]['start_period']
            if s_j < s_w_i:
                for c in C_w[i]:
                    if c in C_w[i - 1]:
                        # j must also be in J_w[i-1] (proven by construction)
                        model.addConstr(
                            x[c, j, i] - x[c, j, i - 1] <= r[j, i],
                            name=f"reassign_same_{c}_{j}_{i}"
                        )

    # (32) Re-assignment detection: container newly available in window i
    for i in range(1, m):
        s_w_i = windows[i][0]
        for j in J_w[i]:
            s_j = trade_by_id[j]['start_period']
            if s_j < s_w_i:
                for c in C_w[i]:
                    if c not in C_w[i - 1]:
                        model.addConstr(
                            x[c, j, i] <= r[j, i],
                            name=f"reassign_new_{c}_{j}_{i}"
                        )

    # =========================================================================
    # Solve
    # =========================================================================
    model.optimize()

    # =========================================================================
    # Extract solution
    # =========================================================================
    solution = {"objective_value": None}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal

        # Extract assignments: for each trade, for each period, the containers
        assignments = {}
        for t in trades:
            j = t['id']
            trade_assignments = {}
            for i in range(m):
                if j in J_w[i]:
                    s_w, f_w = windows[i]
                    assigned_containers = []
                    for c in sorted(C_w[i]):
                        if (c, j, i) in x and x[c, j, i].X > 0.5:
                            assigned_containers.append(c)
                    # Apply assignment to all periods in this window
                    for p in range(s_w, min(f_w, P + 1)):
                        trade_assignments[str(p)] = assigned_containers
            assignments[str(j)] = trade_assignments

        solution["assignments"] = assignments
        solution["status"] = (
            "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        )
        try:
            solution["mip_gap"] = model.MIPGap
        except Exception:
            solution["mip_gap"] = None
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible"

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve TCAPCP using MIP TCAPCP_Win formulation with Gurobi"
    )
    parser.add_argument(
        '--instance_path', type=str, required=True,
        help='Path to the JSON file containing the problem instance'
    )
    parser.add_argument(
        '--solution_path', type=str, required=True,
        help='Path where the solution JSON file will be written'
    )
    parser.add_argument(
        '--time_limit', type=int, required=True,
        help='Maximum solver runtime in seconds'
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance_data = load_instance(args.instance_path)
    solution = solve_tcapcp_win(instance_data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == '__main__':
    main()
