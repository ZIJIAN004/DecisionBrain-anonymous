#!/usr/bin/env python3
"""
MIP formulation for the PCSP-SL (Preemptive Crane Scheduling Problem with
Seaside and Landside containers) from Appendix A of:

    Kress, Dornseifer & Jaehn (2019)
    "An Exact Solution Approach for Scheduling Cooperative Gantry Cranes"
    European Journal of Operational Research, 273(1), 82-101.

Solver: Gurobi (gurobipy)

Usage:
    python gurobi_code.py --instance_path instance_1.json \
                          --solution_path gurobi_solution_1.json \
                          --time_limit 3600
"""

import argparse
import json
import time
from collections import Counter

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
def load_instance(path: str) -> dict:
    """Load a PCSP-SL instance from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compute_T(S: int, n: int, m: int, p: int) -> int:
    """
    Compute a conservative upper bound T on the time horizon.

    T must be large enough so that an optimal schedule fits within [0, T].
    We use:
        T = 2*p*n + (S+1)*n + 2*p*m + (S+1)*m + S + 1
    which accounts for every container being lifted, moved the maximum
    distance, and dropped, plus initial repositioning.
    """
    return 2 * p * n + (S + 1) * n + 2 * p * m + (S + 1) * m + S + 1


def compute_lambda(landside_containers: list, m: int) -> dict:
    """
    Compute lambda_j for each landside container l_j.

    lambda_j = number of landside jobs with the same deadline as l_j.
    lambda_{m+1} = 0  (by convention, used in constraint A.27).

    Landside containers are assumed to be 1-indexed (id = 1..m).
    """
    if m == 0:
        return {}

    deadlines = [lc["deadline"] for lc in landside_containers]
    deadline_counts = Counter(deadlines)

    lam = {}
    for lc in landside_containers:
        j = lc["id"]
        lam[j] = deadline_counts[lc["deadline"]]

    # lambda_{m+1} = 0
    lam[m + 1] = 0
    return lam


def solve_pcsp_sl(instance: dict, time_limit: int) -> dict:
    """
    Build and solve the MIP formulation from Appendix A of Kress et al. (2019).

    Returns a dictionary with at least the 'objective_value' field.
    """
    # =========================================================================
    # Extract instance data
    # =========================================================================
    S = instance["S"]
    n = instance["n"]
    m = instance["m"]
    p = instance["p"]
    sigma_w = instance["sigma_w"]
    sigma_l = instance["sigma_l"]

    # Seaside containers (1-indexed): target slots s_i
    seaside = instance["seaside_containers"]  # list of dicts with 'id', 'target_slot'
    s = {}  # s[i] = target slot for seaside container w_i
    for sc in seaside:
        s[sc["id"]] = sc["target_slot"]

    # Landside containers (1-indexed): source slots a_j, earliest finish time r_j, deadline d_j
    landside = instance["landside_containers"]  # list of dicts
    a = {}  # a[j] = source slot for landside container l_j
    r = {}  # r[j] = earliest finish time
    d = {}  # d[j] = deadline
    for lc in landside:
        j = lc["id"]
        a[j] = lc["source_slot"]
        r[j] = lc["earliest_finish_time"]
        d[j] = lc["deadline"]

    # Compute lambda values
    lam = compute_lambda(landside, m)

    # Compute time horizon upper bound
    T = compute_T(S, n, m, p)

    # Index sets
    I_set = list(range(1, n + 1))       # seaside containers {w_1, ..., w_n}
    J_set = list(range(1, m + 1))       # landside containers {l_1, ..., l_m}
    T_set = list(range(0, T + 1))       # time instants {0, ..., T}
    S_slots = list(range(1, S + 1))     # intermediate storage slots {1, ..., S}

    # Big-M constant for constraint (A.23)
    bigM = T * T  # sufficiently large

    print(f"Instance: S={S}, n={n}, m={m}, p={p}, T={T}")
    print(f"  sigma_w={sigma_w}, sigma_l={sigma_l}")
    print(f"  Seaside targets: {s}")
    if m > 0:
        print(f"  Landside sources: {a}, r: {r}, d: {d}")
        print(f"  Lambda: {lam}")

    # =========================================================================
    # Create Gurobi model
    # =========================================================================
    model = gp.Model("PCSP_SL")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # =========================================================================
    # Decision Variables (A.31 - A.38)
    # =========================================================================

    # x_{c,t}: position of crane c at time t (continuous, non-negative) (A.35)
    x_w = {}  # seaside crane positions
    x_l = {}  # landside crane positions
    for t in T_set:
        x_w[t] = model.addVar(lb=0.0, ub=S + 1, vtype=GRB.CONTINUOUS,
                               name=f"x_w_{t}")
        x_l[t] = model.addVar(lb=0.0, ub=S + 1, vtype=GRB.CONTINUOUS,
                               name=f"x_l_{t}")

    # C: makespan (continuous, non-negative) (A.36)
    C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    # l^I_t: seaside crane starts lifting any w_i at time t (A.31)
    lI = {}
    for t in T_set:
        lI[t] = model.addVar(vtype=GRB.BINARY, name=f"lI_{t}")

    # l^J_{t,i,s}: landside crane starts lifting w_i at time t in slot s (A.32)
    lJ_tis = {}
    for t in T_set:
        for i in I_set:
            for ss in S_slots:
                lJ_tis[t, i, ss] = model.addVar(vtype=GRB.BINARY,
                                                  name=f"lJ_{t}_{i}_{ss}")

    # d^I_{t,i,s}: seaside crane starts dropping w_i at time t in slot s (A.32)
    dI_tis = {}
    for t in T_set:
        for i in I_set:
            for ss in S_slots:
                dI_tis[t, i, ss] = model.addVar(vtype=GRB.BINARY,
                                                  name=f"dI_{t}_{i}_{ss}")

    # l^J_{t,j}: landside crane starts lifting l_j at time t (A.33)
    lJ_tj = {}
    for t in T_set:
        for j in J_set:
            lJ_tj[t, j] = model.addVar(vtype=GRB.BINARY,
                                         name=f"lJj_{t}_{j}")

    # d^J_{t,j}: landside crane starts dropping l_j at time t (A.33)
    dJ_tj = {}
    for t in T_set:
        for j in J_set:
            dJ_tj[t, j] = model.addVar(vtype=GRB.BINARY,
                                         name=f"dJj_{t}_{j}")

    # d^I_{t,i}: landside crane starts dropping w_i at time t (A.34)
    dI_ti = {}
    for t in T_set:
        for i in I_set:
            dI_ti[t, i] = model.addVar(vtype=GRB.BINARY,
                                         name=f"dIi_{t}_{i}")

    # u_j, v_j, q_j: auxiliary binary variables for landside scheduling (A.37)
    u = {}
    v = {}
    q = {}
    for j in J_set:
        u[j] = model.addVar(vtype=GRB.BINARY, name=f"u_{j}")
        v[j] = model.addVar(vtype=GRB.BINARY, name=f"v_{j}")
        q[j] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}")

    # (A.38) v_j = 0 for j in {m+1, ..., m + lambda_m}
    # These are extra v variables that must be fixed to 0.
    # lambda_m is the number of landside containers with the same deadline
    # as l_m (the last landside container).
    v_extra = {}
    if m > 0:
        lambda_m = lam[m]
        for j in range(m + 1, m + lambda_m + 1):
            v_extra[j] = model.addVar(vtype=GRB.BINARY, name=f"v_{j}")
            model.addConstr(v_extra[j] == 0, name=f"A38_v_{j}_eq_0")

    model.update()

    # =========================================================================
    # Objective Function (A.1): min C
    # =========================================================================
    model.setObjective(C, GRB.MINIMIZE)

    # =========================================================================
    # Constraints
    # =========================================================================

    # ---- (A.2) Makespan: last seaside container dropped by seaside crane ----
    # t * d^I_{t,n,s_n} + p <= C,  for all t in {1,...,T}
    # Note: s_n is the target slot of the last seaside container w_n.
    # d^I_{t,n,s_n} is only defined for s_n in {1,...,S}.
    # If s_n = S+1, then w_n goes directly to landside I/O and the seaside
    # crane drops it there; but d^I_{t,i,s} is only for s in {1,...,S}.
    # For s_n = S+1, the container is dropped at the target by the landside
    # crane (via d^I_{t,n} for the landside drop), so A.2 may not directly
    # apply via dI_tis. However, constraint A.21 ensures that either
    # d^I_{t,i,s_i} or d^I_{t,i} is used. When s_i = S+1, d^I_{t,i,s_i}
    # is not in {1,...,S}, so d^I_{t,i} must be used (landside drops it).
    # In that case, A.3 covers the makespan.
    #
    # We add the constraint for s_n in {1,...,S} only:
    s_n = s[n]
    if s_n <= S:
        for t in range(1, T + 1):
            model.addConstr(
                t * dI_tis[t, n, s_n] + p <= C,
                name=f"A2_t{t}"
            )

    # ---- (A.3) Makespan: last seaside container dropped by landside crane ----
    # t * sum_{w_i in I} d^I_{t,i} + p <= C,  for all t in {1,...,T}
    for t in range(1, T + 1):
        model.addConstr(
            t * gp.quicksum(dI_ti[t, i] for i in I_set) + p <= C,
            name=f"A3_t{t}"
        )

    # ---- (A.4) Initial crane positions ----
    # x_{w,0} = sigma_w,  x_{l,0} = sigma_l
    model.addConstr(x_w[0] == sigma_w, name="A4_w")
    model.addConstr(x_l[0] == sigma_l, name="A4_l")

    # ---- (A.5) Crane movement: at most 1 slot per time unit ----
    # x_{c,t-1} - 1 <= x_{c,t} <= x_{c,t-1} + 1
    # for c in {w,l}, t in {1,...,T}
    for t in range(1, T + 1):
        model.addConstr(x_w[t] >= x_w[t - 1] - 1, name=f"A5_w_lb_t{t}")
        model.addConstr(x_w[t] <= x_w[t - 1] + 1, name=f"A5_w_ub_t{t}")
        model.addConstr(x_l[t] >= x_l[t - 1] - 1, name=f"A5_l_lb_t{t}")
        model.addConstr(x_l[t] <= x_l[t - 1] + 1, name=f"A5_l_ub_t{t}")

    # ---- (A.6) Non-crossing: seaside crane always left of landside crane ----
    # x_{w,t} <= x_{l,t} - 1,  for all t in {0,...,T}
    for t in T_set:
        model.addConstr(x_w[t] <= x_l[t] - 1, name=f"A6_t{t}")

    # ---- (A.7) Cranes stay within block ----
    # x_{c,t} <= S+1,  for c in {w,l}, t in {1,...,T}
    # (Already enforced by variable upper bounds, but add explicitly for clarity)
    for t in range(1, T + 1):
        model.addConstr(x_w[t] <= S + 1, name=f"A7_w_t{t}")
        model.addConstr(x_l[t] <= S + 1, name=f"A7_l_t{t}")

    # ---- (A.8) Seaside container drop ordering ----
    # sum_t sum_s t*d^I_{t,i,s} <= sum_t sum_s t*d^I_{t,j,s}
    # for all w_i, w_j in I with i < j
    for i in I_set:
        for j in I_set:
            if i < j:
                lhs = gp.quicksum(
                    t * dI_tis[t, i, ss]
                    for t in T_set for ss in S_slots
                )
                rhs = gp.quicksum(
                    t * dI_tis[t, j, ss]
                    for t in T_set for ss in S_slots
                )
                model.addConstr(lhs <= rhs, name=f"A8_i{i}_j{j}")

    # ---- (A.9) Seaside crane at slot 0 during lifting ----
    # x_{w,t'} <= (1 - l^I_t) * S
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    for t in range(0, T - p + 1):
        for tp in range(t, t + p + 1):
            model.addConstr(
                x_w[tp] <= (1 - lI[t]) * S,
                name=f"A9_t{t}_tp{tp}"
            )

    # ---- (A.10) Seaside crane position during dropping by seaside crane ----
    # (1 - sum_i sum_s d^I_{t,i,s}) * S + sum_i sum_s s*d^I_{t,i,s} >= x_{w,t'}
    # x_{w,t'} >= sum_i sum_s s*d^I_{t,i,s}
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    for t in range(0, T - p + 1):
        sum_dI = gp.quicksum(dI_tis[t, i, ss] for i in I_set for ss in S_slots)
        sum_s_dI = gp.quicksum(
            ss * dI_tis[t, i, ss] for i in I_set for ss in S_slots
        )
        for tp in range(t, t + p + 1):
            # Upper bound on x_w[tp]
            model.addConstr(
                (1 - sum_dI) * S + sum_s_dI >= x_w[tp],
                name=f"A10_ub_t{t}_tp{tp}"
            )
            # Lower bound on x_w[tp]
            model.addConstr(
                x_w[tp] >= sum_s_dI,
                name=f"A10_lb_t{t}_tp{tp}"
            )

    # ---- (A.11) Landside crane position during lifting of seaside container ----
    # (1 - sum_i sum_s l^J_{t,i,s}) * (S+1) + sum_i sum_s s*l^J_{t,i,s} >= x_{l,t'}
    # x_{l,t'} >= sum_i sum_s s*l^J_{t,i,s}
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    for t in range(0, T - p + 1):
        sum_lJ = gp.quicksum(
            lJ_tis[t, i, ss] for i in I_set for ss in S_slots
        )
        sum_s_lJ = gp.quicksum(
            ss * lJ_tis[t, i, ss] for i in I_set for ss in S_slots
        )
        for tp in range(t, t + p + 1):
            model.addConstr(
                (1 - sum_lJ) * (S + 1) + sum_s_lJ >= x_l[tp],
                name=f"A11_ub_t{t}_tp{tp}"
            )
            model.addConstr(
                x_l[tp] >= sum_s_lJ,
                name=f"A11_lb_t{t}_tp{tp}"
            )

    # ---- (A.12) Landside crane position during dropping of seaside container ----
    # (1 - sum_i d^I_{t,i}) * (S+1) + sum_i s_i*d^I_{t,i} >= x_{l,t'}
    # x_{l,t'} >= sum_i s_i*d^I_{t,i}
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    for t in range(0, T - p + 1):
        sum_dIi = gp.quicksum(dI_ti[t, i] for i in I_set)
        sum_si_dIi = gp.quicksum(s[i] * dI_ti[t, i] for i in I_set)
        for tp in range(t, t + p + 1):
            model.addConstr(
                (1 - sum_dIi) * (S + 1) + sum_si_dIi >= x_l[tp],
                name=f"A12_ub_t{t}_tp{tp}"
            )
            model.addConstr(
                x_l[tp] >= sum_si_dIi,
                name=f"A12_lb_t{t}_tp{tp}"
            )

    # ---- (A.13) Landside crane position during lifting of landside container ----
    # (1 - sum_j l^J_{t,j}) * (S+1) + sum_j a_j*l^J_{t,j} >= x_{l,t'}
    # x_{l,t'} >= sum_j a_j*l^J_{t,j}
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    if m > 0:
        for t in range(0, T - p + 1):
            sum_lJj = gp.quicksum(lJ_tj[t, j] for j in J_set)
            sum_aj_lJj = gp.quicksum(a[j] * lJ_tj[t, j] for j in J_set)
            for tp in range(t, t + p + 1):
                model.addConstr(
                    (1 - sum_lJj) * (S + 1) + sum_aj_lJj >= x_l[tp],
                    name=f"A13_ub_t{t}_tp{tp}"
                )
                model.addConstr(
                    x_l[tp] >= sum_aj_lJj,
                    name=f"A13_lb_t{t}_tp{tp}"
                )

    # ---- (A.14) Landside crane at slot S+1 during dropping of landside container ----
    # x_{l,t'} >= sum_j (S+1)*d^J_{t,j}
    # for t in {0,...,T-p}, t' in {t,...,t+p}
    if m > 0:
        for t in range(0, T - p + 1):
            sum_dJj = gp.quicksum((S + 1) * dJ_tj[t, j] for j in J_set)
            for tp in range(t, t + p + 1):
                model.addConstr(
                    x_l[tp] >= sum_dJj,
                    name=f"A14_t{t}_tp{tp}"
                )

    # ---- (A.15) Landside crane does not simultaneously lift and drop ----
    # sum_i sum_s l^J_{t',i,s} + sum_j l^J_{t',j} <= 1 - sum_i d^I_{t,i}
    # for t in {0,...,T-p}, t' in {t,...,t+p-1}
    #
    # Note: The original paper writes d^I_{t,s} which appears to be d^I_{t,i}.
    # This constraint ensures the landside crane cannot start lifting while it
    # is in the process of dropping a seaside container.
    for t in range(0, T - p + 1):
        sum_dIi_t = gp.quicksum(dI_ti[t, i] for i in I_set)
        for tp in range(t, t + p):  # t' in {t, ..., t+p-1}
            sum_lift_tp = gp.quicksum(
                lJ_tis[tp, i, ss] for i in I_set for ss in S_slots
            )
            if m > 0:
                sum_lift_tp += gp.quicksum(lJ_tj[tp, j] for j in J_set)
            model.addConstr(
                sum_lift_tp <= 1 - sum_dIi_t,
                name=f"A15_t{t}_tp{tp}"
            )

    # ---- (A.16) Seaside crane: at most one container in transit at any time ----
    # 0 <= sum_{t'=0}^{t} (l^I_{t'} - sum_i sum_s d^I_{t',i,s}) <= 1
    # for all t in {0,...,T}
    for t in T_set:
        cumul = gp.quicksum(
            lI[tp] - gp.quicksum(
                dI_tis[tp, i, ss] for i in I_set for ss in S_slots
            )
            for tp in range(0, t + 1)
        )
        model.addConstr(cumul >= 0, name=f"A16_lb_t{t}")
        model.addConstr(cumul <= 1, name=f"A16_ub_t{t}")

    # ---- (A.17) Landside crane: at most one container in transit at any time ----
    # 0 <= sum_{t'=0}^{t} (sum_i sum_s l^J_{t',i,s} + sum_j l^J_{t',j}
    #       - sum_i d^I_{t',i} - sum_j d^J_{t',j}) <= 1
    # for all t in {0,...,T}
    for t in T_set:
        cumul = gp.quicksum(
            gp.quicksum(lJ_tis[tp, i, ss] for i in I_set for ss in S_slots)
            + gp.quicksum(lJ_tj[tp, j] for j in J_set)
            - gp.quicksum(dI_ti[tp, i] for i in I_set)
            - gp.quicksum(dJ_tj[tp, j] for j in J_set)
            for tp in range(0, t + 1)
        )
        model.addConstr(cumul >= 0, name=f"A17_lb_t{t}")
        model.addConstr(cumul <= 1, name=f"A17_ub_t{t}")

    # ---- (A.18) Landside crane: seaside container lift-drop balance ----
    # 0 <= sum_{t'=0}^{t} sum_i (sum_s l^J_{t',i,s} - d^I_{t',i}) <= 1
    # for all t in {0,...,T}
    for t in T_set:
        cumul = gp.quicksum(
            gp.quicksum(lJ_tis[tp, i, ss] for ss in S_slots) - dI_ti[tp, i]
            for tp in range(0, t + 1) for i in I_set
        )
        model.addConstr(cumul >= 0, name=f"A18_lb_t{t}")
        model.addConstr(cumul <= 1, name=f"A18_ub_t{t}")

    # ---- (A.19) Landside crane: landside container lift-drop balance ----
    # 0 <= sum_{t'=0}^{t} sum_j (l^J_{t',j} - d^J_{t',j}) <= 1
    # for all t in {0,...,T}
    if m > 0:
        for t in T_set:
            cumul = gp.quicksum(
                lJ_tj[tp, j] - dJ_tj[tp, j]
                for tp in range(0, t + 1) for j in J_set
            )
            model.addConstr(cumul >= 0, name=f"A19_lb_t{t}")
            model.addConstr(cumul <= 1, name=f"A19_ub_t{t}")

    # ---- (A.20) Each seaside container reaches target: either landside picks ----
    #             it up or seaside drops it at s_i directly.
    # sum_t (sum_s l^J_{t,i,s} + d^I_{t,i,s_i}) = 1,  for all w_i in I
    #
    # Note: d^I_{t,i,s_i} is only defined for s_i in {1,...,S}.
    # If s_i = S+1, the container must be handed over (landside picks up),
    # so the d^I_{t,i,s_i} term effectively does not exist for s_i = S+1.
    for i in I_set:
        expr = gp.quicksum(
            gp.quicksum(lJ_tis[t, i, ss] for ss in S_slots)
            for t in T_set
        )
        if s[i] <= S:
            expr += gp.quicksum(dI_tis[t, i, s[i]] for t in T_set)
        model.addConstr(expr == 1, name=f"A20_i{i}")

    # ---- (A.21) Each seaside container: landside drop or seaside direct drop ----
    # sum_t (d^I_{t,i} + d^I_{t,i,s_i}) = 1,  for all w_i in I
    #
    # d^I_{t,i} is the landside crane's drop of w_i. d^I_{t,i,s_i} is the
    # seaside crane's direct drop at the target slot.
    # Again, d^I_{t,i,s_i} only applies when s_i in {1,...,S}.
    for i in I_set:
        expr = gp.quicksum(dI_ti[t, i] for t in T_set)
        if s[i] <= S:
            expr += gp.quicksum(dI_tis[t, i, s[i]] for t in T_set)
        model.addConstr(expr == 1, name=f"A21_i{i}")

    # ---- (A.22) Handover: landside can only pick up where seaside dropped ----
    # sum_t l^J_{t,i,s} <= sum_t d^I_{t,i,s}
    # for all w_i in I, s in {1,...,S}
    for i in I_set:
        for ss in S_slots:
            model.addConstr(
                gp.quicksum(lJ_tis[t, i, ss] for t in T_set)
                <= gp.quicksum(dI_tis[t, i, ss] for t in T_set),
                name=f"A22_i{i}_s{ss}"
            )

    # ---- (A.23) Handover: landside picks up after seaside drops ----
    # (1 - sum_t l^J_{t,i,s}) * bigM + sum_t t*l^J_{t,i,s}
    #   >= sum_t t*d^I_{t,i,s}
    # for all w_i in I, s in {1,...,S}
    for i in I_set:
        for ss in S_slots:
            sum_lJ_active = gp.quicksum(lJ_tis[t, i, ss] for t in T_set)
            sum_t_lJ = gp.quicksum(t * lJ_tis[t, i, ss] for t in T_set)
            sum_t_dI = gp.quicksum(t * dI_tis[t, i, ss] for t in T_set)
            model.addConstr(
                (1 - sum_lJ_active) * bigM + sum_t_lJ >= sum_t_dI,
                name=f"A23_i{i}_s{ss}"
            )

    # ---- (A.24) Landside drop after landside lift for seaside containers ----
    # sum_t t*d^I_{t,i} >= sum_t sum_s t*l^J_{t,i,s}
    # for all w_i in I
    for i in I_set:
        lhs = gp.quicksum(t * dI_ti[t, i] for t in T_set)
        rhs = gp.quicksum(
            t * lJ_tis[t, i, ss] for t in T_set for ss in S_slots
        )
        model.addConstr(lhs >= rhs, name=f"A24_i{i}")

    # ---- (A.25) Landside drop after landside lift for landside containers ----
    # sum_t t*d^J_{t,j} >= sum_t t*l^J_{t,j}
    # for all l_j in J
    for j in J_set:
        lhs = gp.quicksum(t * dJ_tj[t, j] for t in T_set)
        rhs = gp.quicksum(t * lJ_tj[t, j] for t in T_set)
        model.addConstr(lhs >= rhs, name=f"A25_j{j}")

    # ---- (A.26)-(A.30) Landside container time window enforcement ----
    if m > 0:
        # Merge v and v_extra into a single lookup
        v_all = {}
        for j in J_set:
            v_all[j] = v[j]
        for j in v_extra:
            v_all[j] = v_extra[j]

        for j in J_set:
            # (A.26) (T+1)*u_j >= C - d_j + 0.5
            model.addConstr(
                (T + 1) * u[j] >= C - d[j] + 0.5,
                name=f"A26_j{j}"
            )

            # (A.27) u_j <= sum_{k=j+1}^{j+lambda_{j+1}} v_k
            # lambda_{j+1}: for j < m, this is lam[j+1]; for j = m, lam[m+1] = 0
            if j < m:
                lam_jp1 = lam[j + 1]
            else:
                lam_jp1 = lam[m + 1]  # = 0

            if lam_jp1 > 0:
                sum_v = gp.quicksum(
                    v_all[k] for k in range(j + 1, j + lam_jp1 + 1)
                    if k in v_all
                )
                model.addConstr(u[j] <= sum_v, name=f"A27_j{j}")
            else:
                # lambda_{j+1} = 0 => u_j <= 0
                model.addConstr(u[j] <= 0, name=f"A27_j{j}")

            # (A.28) q_j >= 0.5*(u_j + v_j)
            model.addConstr(
                q[j] >= 0.5 * (u[j] + v[j]),
                name=f"A28_j{j}"
            )

            # (A.29) sum_t d^J_{t,j} >= q_j
            model.addConstr(
                gp.quicksum(dJ_tj[t, j] for t in T_set) >= q[j],
                name=f"A29_j{j}"
            )

            # (A.30) q_j * r_j <= sum_t t*d^J_{t,j} + p <= d_j
            sum_t_dJ = gp.quicksum(t * dJ_tj[t, j] for t in T_set)
            model.addConstr(
                q[j] * r[j] <= sum_t_dJ + p,
                name=f"A30_lb_j{j}"
            )
            model.addConstr(
                sum_t_dJ + p <= d[j],
                name=f"A30_ub_j{j}"
            )

    # ---- Additional: total number of lifts by seaside crane equals n ----
    # (Implied by A.16 + A.20 + A.21, but can help solver)
    model.addConstr(
        gp.quicksum(lI[t] for t in T_set) == n,
        name="total_seaside_lifts"
    )

    # =========================================================================
    # Solve
    # =========================================================================
    print(f"\nModel has {model.NumVars} variables and {model.NumConstrs} constraints.")
    print(f"Solving with time limit = {time_limit} seconds...\n")

    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start

    # =========================================================================
    # Extract solution
    # =========================================================================
    result = {}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["makespan"] = obj_val
        result["solve_time"] = solve_time
        result["status"] = model.Status
        result["status_name"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, f"OTHER({model.Status})")

        if model.Status == GRB.OPTIMAL:
            result["optimal"] = True
            result["gap"] = 0.0
        else:
            result["optimal"] = False
            result["gap"] = model.MIPGap

        # Extract crane positions for the solution
        crane_schedule_w = {}
        crane_schedule_l = {}
        for t in T_set:
            crane_schedule_w[t] = x_w[t].X
            crane_schedule_l[t] = x_l[t].X
        result["crane_w_positions"] = crane_schedule_w
        result["crane_l_positions"] = crane_schedule_l

        # Extract seaside container events
        seaside_events = []
        for i in I_set:
            event = {"container_id": i, "target_slot": s[i]}
            # Find when seaside crane lifts (the lift time for container i is
            # determined by the cumulative count: the i-th lift start)
            for t in T_set:
                for ss in S_slots:
                    if dI_tis[t, i, ss].X > 0.5:
                        event["seaside_drop_time"] = t
                        event["seaside_drop_slot"] = ss
                if dI_ti[t, i].X > 0.5:
                    event["landside_drop_time"] = t
                for ss in S_slots:
                    if lJ_tis[t, i, ss].X > 0.5:
                        event["landside_lift_time"] = t
                        event["landside_lift_slot"] = ss
            seaside_events.append(event)
        result["seaside_events"] = seaside_events

        # Extract landside container events
        if m > 0:
            landside_events = []
            for j in J_set:
                event = {"container_id": j, "source_slot": a[j]}
                for t in T_set:
                    if lJ_tj[t, j].X > 0.5:
                        event["lift_time"] = t
                    if dJ_tj[t, j].X > 0.5:
                        event["drop_time"] = t
                landside_events.append(event)
            result["landside_events"] = landside_events

        print(f"\nSolution found: C (makespan) = {obj_val}")
        print(f"  Solve time: {solve_time:.2f}s")
        if model.Status == GRB.OPTIMAL:
            print("  Status: OPTIMAL")
        else:
            print(f"  Status: {result['status_name']} (gap = {result.get('gap', 'N/A')})")

    else:
        result["objective_value"] = None
        result["makespan"] = None
        result["solve_time"] = solve_time
        result["status"] = model.Status
        result["status_name"] = "NO_SOLUTION"
        result["optimal"] = False
        print(f"\nNo feasible solution found. Status = {model.Status}")
        print(f"  Solve time: {solve_time:.2f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve PCSP-SL using the MIP formulation from "
                    "Kress, Dornseifer & Jaehn (2019), Appendix A."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default=None,
        help="Path for the output solution JSON. "
             "Default: gurobi_solution_{instance_index}.json"
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum solver runtime in seconds (default: 3600)."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    instance = load_instance(args.instance_path)

    # Determine output path
    if args.solution_path is None:
        idx = instance.get("instance_index", 0)
        args.solution_path = f"gurobi_solution_{idx}.json"

    # Solve
    result = solve_pcsp_sl(instance, args.time_limit)

    # Convert any non-serializable keys (int dict keys) to strings for JSON
    serializable_result = {}
    for key, val in result.items():
        if isinstance(val, dict):
            serializable_result[key] = {str(k): v for k, v in val.items()}
        else:
            serializable_result[key] = val

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(serializable_result, f, indent=2)

    print(f"\nSolution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
