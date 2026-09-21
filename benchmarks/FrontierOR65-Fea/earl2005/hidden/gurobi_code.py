#!/usr/bin/env python3
"""
Gurobi implementation of Defensive Drill 1 MILP.

Paper: "Multi-vehicle Cooperative Control Using Mixed Integer Linear Programming"
Authors: Matthew G. Earl and Raffaello D'Andrea (2005)

Implements the full MILP formulation (Eq. 44) with both epsilon > 0 and epsilon = 0
objective options, as described in Sections II–V of the paper.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_AB(T_u):
    """Compute state-transition matrices A[k] and B[k] for a given step time T_u.
    Eq. (6) from the paper.
    """
    e = math.exp(-T_u)
    A = [
        [1, 0, 1 - e, 0],
        [0, 1, 0,     1 - e],
        [0, 0, e,     0],
        [0, 0, 0,     e],
    ]
    B = [
        [T_u - 1 + e, 0],
        [0,           T_u - 1 + e],
        [1 - e,       0],
        [0,           1 - e],
    ]
    return A, B


def defender_pos_at_time(t, T_u, N_u, x_var, y_var, xdot_var, ydot_var, ux_var, uy_var, i):
    """Return linear (Gurobi) expressions for the defender i's (x, y) position
    at continuous time t, computed via Eq. (7).

    Eq. (7):
        x(t) = x_u[k] + (1 - exp(t_u[k] - t)) * xdot_u[k]
                       + (t - t_u[k] - 1 + exp(t_u[k] - t)) * u_x[k]
    where k satisfies t_u[k] <= t <= t_u[k+1], t_u[k] = k * T_u.

    Since A[k] and B[k] are pre-computed constants, the resulting expressions
    are linear in the decision variables.
    """
    k_u = int(t / T_u)
    if k_u >= N_u:
        k_u = N_u - 1          # clamp to last control interval
    tau = t - k_u * T_u        # time elapsed since start of interval k_u
    e_tau = math.exp(-tau)

    # x_{a,i}(t) = x_i[k_u] + (1 - e^{-tau}) * xdot_i[k_u]
    #                        + (tau - 1 + e^{-tau}) * ux_i[k_u]
    xa = (x_var[i, k_u]
          + (1.0 - e_tau) * xdot_var[i, k_u]
          + (tau - 1.0 + e_tau) * ux_var[i, k_u])
    ya = (y_var[i, k_u]
          + (1.0 - e_tau) * ydot_var[i, k_u]
          + (tau - 1.0 + e_tau) * uy_var[i, k_u])
    return xa, ya


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(instance_path, solution_path, time_limit):
    # ------------------------------------------------------------------
    # Load instance
    # ------------------------------------------------------------------
    with open(instance_path) as f:
        inst = json.load(f)

    N_D   = inst['N_D']
    N_A   = inst['N_A']
    eps   = inst['epsilon']       # epsilon in objective (0 or 0.1)
    M_u   = inst['M_u']
    M_I   = inst['M_I']
    M_dz  = inst['M_dz']
    N_u   = inst['N_u']
    N_a   = inst['N_a']
    R_dz  = inst['R_dz']
    R_I   = inst['R_I']
    T_a   = inst['T_a']
    T_u   = inst['T_u']
    defenders = inst['defenders']
    attackers = inst['attackers']

    # ------------------------------------------------------------------
    # Inferred / assumed parameters (NOT SPECIFIED IN PAPER)
    # ------------------------------------------------------------------
    # H: big-M constant.  Paper says H must be "larger than the maximum
    #    dimension of the vehicle's operating environment plus the radius
    #    of the obstacle."  Playing field radius R_f = 15, R_dz = 2, so
    #    H = 100 is a safe conservative choice.
    H = 100.0

    # eps_c: small tolerance for strict-inequality big-M constraints (Eq. 14, 23, 29).
    # Paper describes it as "a small positive number"; we use 1e-4.
    eps_c = 1e-4

    # M_o: sides of polygon approximating the obstacle (Defense Zone) for
    # defender avoidance.  NOT SPECIFIED IN PAPER for the computational study.
    # We use M_o = 4 (consistent with the other polygon parameters).
    M_o = 4

    # N_o: number of obstacle-avoidance check steps.  NOT SPECIFIED IN PAPER.
    # We set N_o = N_u and place checks at t_o[k] = k * T_u (the defender
    # control-step boundaries), so x_{o,i}[k] = x_i[k] exactly.
    N_o = N_u

    # ------------------------------------------------------------------
    # Pre-compute dynamics matrices (uniform T_u)
    # ------------------------------------------------------------------
    A, B = compute_AB(T_u)

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("DefensiveDrill1")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))

    # ---- Decision variables ------------------------------------------

    # Defender state x_{u,i}[k] = (x, y, xdot, ydot), k in {0,...,N_u}
    x_v    = {}
    y_v    = {}
    xdot_v = {}
    ydot_v = {}
    for i in range(N_D):
        for k in range(N_u + 1):
            x_v[i, k]    = model.addVar(lb=-GRB.INFINITY, name=f"x_{i}_{k}")
            y_v[i, k]    = model.addVar(lb=-GRB.INFINITY, name=f"y_{i}_{k}")
            xdot_v[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"xdot_{i}_{k}")
            ydot_v[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"ydot_{i}_{k}")

    # Defender control u_i[k] and auxiliary |u| variables, k in {0,...,N_u-1}
    ux_v = {}
    uy_v = {}
    zx_v = {}
    zy_v = {}
    for i in range(N_D):
        for k in range(N_u):
            ux_v[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"ux_{i}_{k}")
            uy_v[i, k] = model.addVar(lb=-GRB.INFINITY, name=f"uy_{i}_{k}")
            zx_v[i, k] = model.addVar(lb=0.0, name=f"zx_{i}_{k}")
            zy_v[i, k] = model.addVar(lb=0.0, name=f"zy_{i}_{k}")

    # Attacker position p_j[k], q_j[k], k in {0,...,N_a}
    p_v = {}
    q_v = {}
    for j in range(N_A):
        for k in range(N_a + 1):
            p_v[j, k] = model.addVar(lb=-GRB.INFINITY, name=f"p_{j}_{k}")
            q_v[j, k] = model.addVar(lb=-GRB.INFINITY, name=f"q_{j}_{k}")

    # Attacker mode a_j[k] in {0,1}, k in {0,...,N_a}
    a_v = {}
    for j in range(N_A):
        for k in range(N_a + 1):
            a_v[j, k] = model.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}")

    # gamma_j[k]: 1 iff attacker j is inside Defense Zone at step k
    gamma_v = {}
    for j in range(N_A):
        for k in range(1, N_a + 1):
            gamma_v[j, k] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{j}_{k}")

    # g_{mj}[k]: auxiliary binary for gamma, m in {1,...,M_dz}
    g_v = {}
    for j in range(N_A):
        for k in range(1, N_a + 1):
            for m in range(1, M_dz + 1):
                g_v[m, j, k] = model.addVar(vtype=GRB.BINARY, name=f"g_{m}_{j}_{k}")

    # delta_{ij}[k]: 1 iff attacker j is in intercept region of defender i
    delta_v = {}
    for i in range(N_D):
        for j in range(N_A):
            for k in range(1, N_a + 1):
                delta_v[i, j, k] = model.addVar(vtype=GRB.BINARY,
                                                  name=f"delta_{i}_{j}_{k}")

    # d_{mij}[k]: auxiliary binary for delta, m in {1,...,M_I}
    d_v = {}
    for i in range(N_D):
        for j in range(N_A):
            for k in range(1, N_a + 1):
                for m in range(1, M_I + 1):
                    d_v[m, i, j, k] = model.addVar(vtype=GRB.BINARY,
                                                     name=f"d_{m}_{i}_{j}_{k}")

    # delta_comb_j[k]: combined interception indicator for attacker j at step k.
    # NOT SPECIFIED IN PAPER for N_D > 1.
    # Inferred assumption: attacker j is "intercepted" if delta_{ij}[k] = 1
    # for at least one defender i.  This variable encodes the OR over i.
    delta_comb_v = {}
    for j in range(N_A):
        for k in range(1, N_a + 1):
            delta_comb_v[j, k] = model.addVar(vtype=GRB.BINARY,
                                               name=f"delta_comb_{j}_{k}")

    # b_{mi}[k]: auxiliary binary for defender i obstacle avoidance.
    # The paper mentions b_{mij}[k] with a j-index for the multi-defender case
    # but states it "follows a similar trend" without specifying the role of j.
    # Inferred assumption: obstacle avoidance depends only on defender i, not
    # on attacker j, so b is indexed only by (m, i, k).
    b_v = {}
    for i in range(N_D):
        for k in range(1, N_o + 1):
            for m in range(1, M_o + 1):
                b_v[m, i, k] = model.addVar(vtype=GRB.BINARY, name=f"b_{m}_{i}_{k}")

    model.update()

    # ---- Objective (Eq. 44) ------------------------------------------
    obj = gp.quicksum(gamma_v[j, k] for j in range(N_A) for k in range(1, N_a + 1))
    if eps > 0.0:
        obj = obj + eps * gp.quicksum(
            zx_v[i, k] + zy_v[i, k]
            for i in range(N_D) for k in range(N_u)
        )
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraint 1: Defender initial conditions -------------------
    for i in range(N_D):
        d = defenders[i]
        model.addConstr(x_v[i, 0]    == d['x_s'],    name=f"ic_x_{i}")
        model.addConstr(y_v[i, 0]    == d['y_s'],    name=f"ic_y_{i}")
        model.addConstr(xdot_v[i, 0] == d['xdot_s'], name=f"ic_xd_{i}")
        model.addConstr(ydot_v[i, 0] == d['ydot_s'], name=f"ic_yd_{i}")

    # ---- Constraint 2: Defender dynamics (Eq. 6) ---------------------
    # x_u[k+1] = A[k] * x_u[k] + B[k] * u[k]
    # With uniform T_u, A and B are the same for all k.
    e_Tu = math.exp(-T_u)
    for i in range(N_D):
        for k in range(N_u):
            model.addConstr(
                x_v[i, k+1] == x_v[i, k] + (1.0 - e_Tu) * xdot_v[i, k]
                                + (T_u - 1.0 + e_Tu) * ux_v[i, k],
                name=f"dyn_x_{i}_{k}")
            model.addConstr(
                y_v[i, k+1] == y_v[i, k] + (1.0 - e_Tu) * ydot_v[i, k]
                                + (T_u - 1.0 + e_Tu) * uy_v[i, k],
                name=f"dyn_y_{i}_{k}")
            model.addConstr(
                xdot_v[i, k+1] == e_Tu * xdot_v[i, k] + (1.0 - e_Tu) * ux_v[i, k],
                name=f"dyn_xd_{i}_{k}")
            model.addConstr(
                ydot_v[i, k+1] == e_Tu * ydot_v[i, k] + (1.0 - e_Tu) * uy_v[i, k],
                name=f"dyn_yd_{i}_{k}")

    # ---- Constraint 3: Control-input polygon (Eq. 8) -----------------
    cos_pi_Mu = math.cos(math.pi / M_u)
    for i in range(N_D):
        for k in range(N_u):
            for m in range(1, M_u + 1):
                sm = math.sin(2.0 * math.pi * m / M_u)
                cm = math.cos(2.0 * math.pi * m / M_u)
                model.addConstr(
                    sm * ux_v[i, k] + cm * uy_v[i, k] <= cos_pi_Mu,
                    name=f"ctrl_{i}_{k}_{m}")

    # ---- Constraint 4: Auxiliary |u| constraints (Eq. 10) -----------
    for i in range(N_D):
        for k in range(N_u):
            model.addConstr( ux_v[i, k] <=  zx_v[i, k], name=f"zx_p_{i}_{k}")
            model.addConstr(-ux_v[i, k] <=  zx_v[i, k], name=f"zx_n_{i}_{k}")
            model.addConstr( uy_v[i, k] <=  zy_v[i, k], name=f"zy_p_{i}_{k}")
            model.addConstr(-uy_v[i, k] <=  zy_v[i, k], name=f"zy_n_{i}_{k}")

    # ---- Constraint 5: Attacker initial conditions (Eq. 19) ----------
    for j in range(N_A):
        att = attackers[j]
        model.addConstr(p_v[j, 0] == att['p_s'], name=f"ic_p_{j}")
        model.addConstr(q_v[j, 0] == att['q_s'], name=f"ic_q_{j}")
        model.addConstr(a_v[j, 0] == 1,          name=f"ic_a_{j}")

    # ---- Constraint 6: Attacker dynamics (Eq. 17) --------------------
    # p[k+1] = p[k] + v_p * T_a * a[k]   (linear: v_p, T_a are constants)
    for j in range(N_A):
        att = attackers[j]
        v_p = att['v_p']
        v_q = att['v_q']
        for k in range(N_a):
            model.addConstr(
                p_v[j, k+1] == p_v[j, k] + v_p * T_a * a_v[j, k],
                name=f"att_p_{j}_{k}")
            model.addConstr(
                q_v[j, k+1] == q_v[j, k] + v_q * T_a * a_v[j, k],
                name=f"att_q_{j}_{k}")

    # ---- Constraint 7: Defense Zone indicator (Eq. 23 & 25) ----------
    for j in range(N_A):
        for k in range(1, N_a + 1):
            for m in range(1, M_dz + 1):
                sm = math.sin(2.0 * math.pi * m / M_dz)
                cm = math.cos(2.0 * math.pi * m / M_dz)
                lhs = sm * p_v[j, k] + cm * q_v[j, k]
                # Eq. 23a
                model.addConstr(
                    lhs <= R_dz + H * (1.0 - g_v[m, j, k]),
                    name=f"dz23a_{m}_{j}_{k}")
                # Eq. 23b
                model.addConstr(
                    lhs >= R_dz + eps_c - (H + eps_c) * g_v[m, j, k],
                    name=f"dz23b_{m}_{j}_{k}")
            # Eq. 25a: g_{mj}[k] >= gamma_j[k]  for all m
            for m in range(1, M_dz + 1):
                model.addConstr(
                    g_v[m, j, k] >= gamma_v[j, k],
                    name=f"dz25a_{m}_{j}_{k}")
            # Eq. 25b: sum_{l}(1 - g_{lj}[k]) + gamma_j[k] >= 1
            model.addConstr(
                gp.quicksum(1.0 - g_v[l, j, k] for l in range(1, M_dz + 1))
                + gamma_v[j, k] >= 1.0,
                name=f"dz25b_{j}_{k}")

    # ---- Constraint 8: Defender position at attacker time steps (Eq. 7) -
    # Pre-compute linear expressions for x_{a,i}[k], y_{a,i}[k].
    xa_expr = {}
    ya_expr = {}
    for k in range(1, N_a + 1):
        t_ak = k * T_a
        for i in range(N_D):
            xa_expr[i, k], ya_expr[i, k] = defender_pos_at_time(
                t_ak, T_u, N_u,
                x_v, y_v, xdot_v, ydot_v, ux_v, uy_v, i)

    # ---- Constraint 9: Intercept region indicator (Eq. 29 & 31) ------
    for i in range(N_D):
        for j in range(N_A):
            for k in range(1, N_a + 1):
                for m in range(1, M_I + 1):
                    sm = math.sin(2.0 * math.pi * m / M_I)
                    cm = math.cos(2.0 * math.pi * m / M_I)
                    lhs = (sm * (p_v[j, k] - xa_expr[i, k])
                           + cm * (q_v[j, k] - ya_expr[i, k]))
                    # Eq. 29a
                    model.addConstr(
                        lhs <= R_I + H * (1.0 - d_v[m, i, j, k]),
                        name=f"int29a_{m}_{i}_{j}_{k}")
                    # Eq. 29b
                    model.addConstr(
                        lhs >= R_I + eps_c - (H + eps_c) * d_v[m, i, j, k],
                        name=f"int29b_{m}_{i}_{j}_{k}")
                # Eq. 31a: d_{mij}[k] >= delta_{ij}[k]  for all m
                for m in range(1, M_I + 1):
                    model.addConstr(
                        d_v[m, i, j, k] >= delta_v[i, j, k],
                        name=f"int31a_{m}_{i}_{j}_{k}")
                # Eq. 31b: sum_{l}(1 - d_{lij}[k]) + delta_{ij}[k] >= 1
                model.addConstr(
                    gp.quicksum(1.0 - d_v[l, i, j, k] for l in range(1, M_I + 1))
                    + delta_v[i, j, k] >= 1.0,
                    name=f"int31b_{i}_{j}_{k}")

    # ---- Constraint 10: Combined interception indicator --------------
    # NOT SPECIFIED IN PAPER for N_D > 1.
    # Inferred assumption: delta_comb_j[k] = 1 iff any defender i intercepts
    # attacker j at step k (OR over i).
    # Implementation via standard OR-linearisation:
    #   delta_comb_j[k] >= delta_{ij}[k]  for all i   (if any is 1, combined >= 1)
    #   delta_comb_j[k] <= sum_i delta_{ij}[k]          (if none is 1, combined = 0)
    for j in range(N_A):
        for k in range(1, N_a + 1):
            for i in range(N_D):
                model.addConstr(
                    delta_comb_v[j, k] >= delta_v[i, j, k],
                    name=f"comb_lb_{i}_{j}_{k}")
            model.addConstr(
                delta_comb_v[j, k] <= gp.quicksum(delta_v[i, j, k] for i in range(N_D)),
                name=f"comb_ub_{j}_{k}")

    # ---- Constraint 11: Attacker state machine (Eq. 34) --------------
    # One-on-one form from the paper, generalized using delta_comb_v for N_D > 1.
    # The exact multi-defender generalisation is NOT SPECIFIED IN PAPER;
    # we replace delta[k] with delta_comb_v[j,k].
    for j in range(N_A):
        for k in range(1, N_a):          # k+1 ranges up to N_a
            dc = delta_comb_v[j, k]
            gm = gamma_v[j, k]
            # Eq. 34a: a[k+1] + delta[k] <= 1
            model.addConstr(a_v[j, k+1] + dc <= 1.0, name=f"sm34a_{j}_{k}")
            # Eq. 34b: a[k+1] - a[k] <= 0
            model.addConstr(a_v[j, k+1] - a_v[j, k] <= 0.0, name=f"sm34b_{j}_{k}")
            # Eq. 34c: a[k+1] + gamma[k] <= 1
            model.addConstr(a_v[j, k+1] + gm <= 1.0, name=f"sm34c_{j}_{k}")
            # Eq. 34d: a[k] - delta[k] - gamma[k] - a[k+1] <= 0
            model.addConstr(a_v[j, k] - dc - gm - a_v[j, k+1] <= 0.0,
                            name=f"sm34d_{j}_{k}")

    # ---- Constraint 12: Defender avoidance of Defense Zone (Eq. 14 & 15) -
    # t_o[k] = k * T_u; at these times x_{o,i}[k] = x_i[k] (defender state).
    # Obstacle centered at origin, R_obst = R_dz.
    for i in range(N_D):
        for k in range(1, N_o + 1):
            for m in range(1, M_o + 1):
                sm = math.sin(2.0 * math.pi * m / M_o)
                cm = math.cos(2.0 * math.pi * m / M_o)
                # Eq. 14 (strict > replaced by >= + eps_c):
                # (x_o - 0)*s + (y_o - 0)*c >= R_obst + eps_c - H * b_{mi}[k]
                model.addConstr(
                    sm * x_v[i, k] + cm * y_v[i, k]
                    >= R_dz + eps_c - H * b_v[m, i, k],
                    name=f"avoid14_{m}_{i}_{k}")
            # Eq. 15: sum_m b_{mi}[k] <= M_o - 1
            model.addConstr(
                gp.quicksum(b_v[m, i, k] for m in range(1, M_o + 1)) <= M_o - 1,
                name=f"avoid15_{i}_{k}")

    # ---- Solve -------------------------------------------------------
    model.optimize()

    # ---- Extract solution --------------------------------------------
    solution = {}
    if model.SolCount > 0:
        solution['objective_value'] = model.objVal
        solution['status'] = model.status
        solution['mip_gap'] = model.MIPGap

        sol_defenders = []
        for i in range(N_D):
            sol_defenders.append({
                'x':    [x_v[i, k].X    for k in range(N_u + 1)],
                'y':    [y_v[i, k].X    for k in range(N_u + 1)],
                'xdot': [xdot_v[i, k].X for k in range(N_u + 1)],
                'ydot': [ydot_v[i, k].X for k in range(N_u + 1)],
                'ux':   [ux_v[i, k].X   for k in range(N_u)],
                'uy':   [uy_v[i, k].X   for k in range(N_u)],
            })
        solution['defenders'] = sol_defenders

        # Original Earl & D'Andrea formulation has per-(defender, attacker, step)
        # intercept indicator delta_{ij}[k]. The scalar delta_comb is a
        # benchmark-added auxiliary (OR_i delta_{ij}) only needed for the
        # N_D-on-N_A state machine linearization; it is NOT part of the paper's
        # original variables and is therefore NOT exported.
        sol_attackers = []
        for j in range(N_A):
            sol_attackers.append({
                'p':     [p_v[j, k].X     for k in range(N_a + 1)],
                'q':     [q_v[j, k].X     for k in range(N_a + 1)],
                'a':     [a_v[j, k].X     for k in range(N_a + 1)],
                'gamma': [0.0] + [gamma_v[j, k].X for k in range(1, N_a + 1)],
                # delta[i][k]: whether defender i intercepts attacker j at step k.
                # Padded at k=0 with 0.0 to match other per-step arrays.
                'delta': [
                    [0.0] + [delta_v[i, j, k].X for k in range(1, N_a + 1)]
                    for i in range(N_D)
                ],
            })
        solution['attackers'] = sol_attackers
    else:
        solution['objective_value'] = None
        solution['status'] = model.status
        solution['mip_gap'] = None
        print("WARNING: No feasible solution found within the time limit.",
              file=sys.stderr)

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to: {solution_path}")
    print(f"Objective value:     {solution['objective_value']}")
    return solution


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve Defensive Drill 1 (Earl & D'Andrea 2005) with Gurobi.")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON problem instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file '
                             '(e.g. gurobi_solution_1.json).')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == '__main__':
    main()
