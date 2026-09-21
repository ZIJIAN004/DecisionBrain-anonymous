#!/usr/bin/env python3
"""
Feasibility checker for Defensive Drill 1 solutions (with objective recompute).

Paper: "Multi-vehicle Cooperative Control Using Mixed Integer Linear Programming"
       Earl & D'Andrea (2005)

Checks constraints at the ORIGINAL PHYSICAL level (circle / distance semantics),
not at the MILP polygon-approximation level used by the paper's formulation.
This allows a generated algorithm (MIP, physics simulation, trajectory
optimization, etc.) to be checked regardless of its internal encoding.

Constraints:
  1. Defender Dynamics + IC (Eq. 6) - linear discrete-time state transition
  2. Control-Input Unit-Disk Feasibility - ||u_i[k]||_2 <= 1
  3. Auxiliary |u| constraints (Eq. 10) - checked only if zx/zy present
  4. Attacker Dynamics (Eq. 17) - linear motion while active
  5. Attacker Initial Conditions (Eq. 19) - start in attack mode at given pos
  6. Attacker State Machine - mode[k+1]=1 iff mode[k]=1 AND not in-zone AND not intercepted
  7. Defense Zone containment - gamma[k]=1 iff ||attacker_pos[k]|| <= R_dz
  8. Interception - defender i intercepts attacker j at step k iff
       ||attacker_pos[k] - defender_pos(k*T_a)||_2 <= R_I
  9. Defender Zone Avoidance - at each obstacle-check time t_o,
       ||defender_pos(t_o)||_2 >= R_dz
 10. Objective Consistency (Tier C) - reported objective_value must equal
       the recomputed objective J = sum_{j,k>=1} gamma_j[k]
                                  + epsilon * sum_{i,k}(|u_xi[k]| + |u_yi[k]|)
       (when zx/zy are absent, the optimal z = |u| is used as the canonical
       interpretation; when present, the stored z is used directly).
"""

import argparse
import json
import math


def check_feasibility(inst, sol):
    """Run all feasibility checks; return the result dict (no I/O)."""
    tol = 1e-5
    eps = 1e-5

    # Check if solution has variable data
    if 'defenders' not in sol or 'attackers' not in sol:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution contains no variable data (solver may have failed)."],
            "violation_magnitudes": []
        }

    # Instance parameters
    N_D = inst['N_D']
    N_A = inst['N_A']
    M_u = inst['M_u']
    M_I = inst['M_I']
    M_dz = inst['M_dz']
    N_u = inst['N_u']
    N_a = inst['N_a']
    R_dz = inst['R_dz']
    R_I = inst['R_I']
    T_a_val = inst['T_a']
    T_u_val = inst['T_u']
    defenders_inst = inst['defenders']
    attackers_inst = inst['attackers']

    # Inferred parameters (matching solver code, NOT SPECIFIED IN PAPER)
    H = 100.0
    eps_c = 1e-4
    M_o = 4
    N_o = N_u

    def _poly_gap(M, R):
        if M <= 0 or R <= 0:
            return 0.0
        return R * (1.0 / math.cos(math.pi / M) - 1.0)
    zone_tol = _poly_gap(M_dz, R_dz) + tol
    intercept_tol = _poly_gap(M_I, R_I) + tol
    avoid_tol = _poly_gap(M_o, R_dz) + tol

    defenders_sol = sol['defenders']
    attackers_sol = sol['attackers']

    # Structural validation: the constraint checks below index defenders_sol[i]
    # for i in range(N_D), attackers_sol[j] for j in range(N_A), and trajectory
    # arrays at [k]/[k+1] up to N_u/N_a. A solution that omits agents or supplies
    # short arrays would otherwise raise IndexError/KeyError and crash the
    # checker (reported upstream as checker_error). Validate up front and return
    # a clean infeasible verdict instead.
    struct_errors = []
    if not isinstance(defenders_sol, list) or len(defenders_sol) < N_D:
        struct_errors.append(
            f"defenders: expected {N_D} entries, got "
            f"{len(defenders_sol) if isinstance(defenders_sol, list) else type(defenders_sol).__name__}")
    if not isinstance(attackers_sol, list) or len(attackers_sol) < N_A:
        struct_errors.append(
            f"attackers: expected {N_A} entries, got "
            f"{len(attackers_sol) if isinstance(attackers_sol, list) else type(attackers_sol).__name__}")

    if not struct_errors:
        # Per-agent key/length checks (only over the indices actually accessed).
        def_arr_full = ['x', 'y', 'xdot', 'ydot']   # indexed up to k+1 => need N_u+1
        def_arr_ctrl = ['ux', 'uy']                 # indexed up to k   => need N_u
        for i in range(N_D):
            ds = defenders_sol[i]
            if not isinstance(ds, dict):
                struct_errors.append(f"Defender {i}: expected object, got {type(ds).__name__}")
                continue
            for key in def_arr_full:
                v = ds.get(key)
                if not isinstance(v, list) or len(v) < N_u + 1:
                    struct_errors.append(
                        f"Defender {i} field '{key}': need >= {N_u + 1} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
            for key in def_arr_ctrl:
                v = ds.get(key)
                if not isinstance(v, list) or len(v) < N_u:
                    struct_errors.append(
                        f"Defender {i} field '{key}': need >= {N_u} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
        for j in range(N_A):
            asol = attackers_sol[j]
            if not isinstance(asol, dict):
                struct_errors.append(f"Attacker {j}: expected object, got {type(asol).__name__}")
                continue
            for key in ('p', 'q'):
                v = asol.get(key)
                if not isinstance(v, list) or len(v) < N_a + 1:
                    struct_errors.append(
                        f"Attacker {j} field '{key}': need >= {N_a + 1} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
            v = asol.get('a')
            if not isinstance(v, list) or len(v) < N_a:
                struct_errors.append(
                    f"Attacker {j} field 'a': need >= {N_a} values, "
                    f"got {len(v) if isinstance(v, list) else 'missing'}")

    if struct_errors:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Malformed solution structure: " + "; ".join(struct_errors[:10])],
            "violation_magnitudes": []
        }

    e_Tu = math.exp(-T_u_val)

    violation_magnitudes = []
    constraint_violations = {}

    def record(cidx, msg, lhs_val, rhs_val, raw_excess):
        normalizer = max(abs(rhs_val), eps)
        ratio = raw_excess / normalizer
        violation_magnitudes.append({
            "constraint": cidx,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(raw_excess),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })
        constraint_violations.setdefault(cidx, []).append(msg)

    def check_eq(cidx, msg, actual, expected):
        diff = abs(actual - expected)
        if diff > tol:
            record(cidx, msg, actual, expected, diff)

    def check_leq(cidx, msg, lhs_val, rhs_val):
        excess = lhs_val - rhs_val
        if excess > tol:
            record(cidx, msg, lhs_val, rhs_val, excess)

    def check_geq(cidx, msg, lhs_val, rhs_val):
        excess = rhs_val - lhs_val
        if excess > tol:
            record(cidx, msg, lhs_val, rhs_val, excess)

    def rb(v):
        return round(float(v))

    def def_pos(d_sol, t):
        k_u = int(t / T_u_val)
        if k_u >= N_u:
            k_u = N_u - 1
        tau = t - k_u * T_u_val
        e_tau = math.exp(-tau)
        xa = (d_sol['x'][k_u]
              + (1.0 - e_tau) * d_sol['xdot'][k_u]
              + (tau - 1.0 + e_tau) * d_sol['ux'][k_u])
        ya = (d_sol['y'][k_u]
              + (1.0 - e_tau) * d_sol['ydot'][k_u]
              + (tau - 1.0 + e_tau) * d_sol['uy'][k_u])
        return xa, ya

    # ================================================================
    # Constraint 1: Defender Dynamics (Eq. 6) + Initial Conditions
    # ================================================================
    for i in range(N_D):
        di = defenders_inst[i]
        ds = defenders_sol[i]
        check_eq(1, f"Defender {i} IC x: {ds['x'][0]} != {di['x_s']}",
                 ds['x'][0], di['x_s'])
        check_eq(1, f"Defender {i} IC y: {ds['y'][0]} != {di['y_s']}",
                 ds['y'][0], di['y_s'])
        check_eq(1, f"Defender {i} IC xdot: {ds['xdot'][0]} != {di['xdot_s']}",
                 ds['xdot'][0], di['xdot_s'])
        check_eq(1, f"Defender {i} IC ydot: {ds['ydot'][0]} != {di['ydot_s']}",
                 ds['ydot'][0], di['ydot_s'])
        for k in range(N_u):
            exp_x = (ds['x'][k] + (1.0 - e_Tu) * ds['xdot'][k]
                     + (T_u_val - 1.0 + e_Tu) * ds['ux'][k])
            check_eq(1, f"Defender {i} dynamics x k={k}: {ds['x'][k+1]:.8f} != {exp_x:.8f}",
                     ds['x'][k+1], exp_x)
            exp_y = (ds['y'][k] + (1.0 - e_Tu) * ds['ydot'][k]
                     + (T_u_val - 1.0 + e_Tu) * ds['uy'][k])
            check_eq(1, f"Defender {i} dynamics y k={k}: {ds['y'][k+1]:.8f} != {exp_y:.8f}",
                     ds['y'][k+1], exp_y)
            exp_xd = e_Tu * ds['xdot'][k] + (1.0 - e_Tu) * ds['ux'][k]
            check_eq(1, f"Defender {i} dynamics xdot k={k}: {ds['xdot'][k+1]:.8f} != {exp_xd:.8f}",
                     ds['xdot'][k+1], exp_xd)
            exp_yd = e_Tu * ds['ydot'][k] + (1.0 - e_Tu) * ds['uy'][k]
            check_eq(1, f"Defender {i} dynamics ydot k={k}: {ds['ydot'][k+1]:.8f} != {exp_yd:.8f}",
                     ds['ydot'][k+1], exp_yd)

    # ================================================================
    # Constraint 2: Control-Input Unit-Disk Feasibility
    # ================================================================
    for i in range(N_D):
        ds = defenders_sol[i]
        for k in range(N_u):
            ux, uy = ds['ux'][k], ds['uy'][k]
            norm = math.sqrt(ux * ux + uy * uy)
            check_leq(2,
                      f"Defender {i} control unit-disk k={k}: ||u||={norm:.6f} > 1",
                      norm, 1.0)

    # ================================================================
    # Constraint 3: Auxiliary |u| constraints (Eq. 10)
    # ================================================================
    has_z = len(defenders_sol) > 0 and 'zx' in defenders_sol[0]
    if has_z:
        for i in range(N_D):
            ds = defenders_sol[i]
            for k in range(N_u):
                zx, zy = ds['zx'][k], ds['zy'][k]
                ux, uy = ds['ux'][k], ds['uy'][k]
                check_leq(3, f"Defender {i} ux<=zx k={k}", ux, zx)
                check_leq(3, f"Defender {i} -ux<=zx k={k}", -ux, zx)
                check_leq(3, f"Defender {i} uy<=zy k={k}", uy, zy)
                check_leq(3, f"Defender {i} -uy<=zy k={k}", -uy, zy)
                if zx < -tol:
                    record(3, f"Defender {i} zx<0 k={k}", zx, 0.0, -zx)
                if zy < -tol:
                    record(3, f"Defender {i} zy<0 k={k}", zy, 0.0, -zy)

    # ================================================================
    # Constraint 4: Attacker Dynamics (Eq. 17)
    # ================================================================
    for j in range(N_A):
        ai = attackers_inst[j]
        asol = attackers_sol[j]
        vp, vq = ai['v_p'], ai['v_q']
        for k in range(N_a):
            ak = rb(asol['a'][k])
            exp_p = asol['p'][k] + vp * T_a_val * ak
            check_eq(4, f"Attacker {j} dynamics p k={k}: {asol['p'][k+1]:.8f} != {exp_p:.8f}",
                     asol['p'][k + 1], exp_p)
            exp_q = asol['q'][k] + vq * T_a_val * ak
            check_eq(4, f"Attacker {j} dynamics q k={k}: {asol['q'][k+1]:.8f} != {exp_q:.8f}",
                     asol['q'][k + 1], exp_q)

    # ================================================================
    # Constraint 5: Attacker Initial Conditions (Eq. 19)
    # ================================================================
    for j in range(N_A):
        ai = attackers_inst[j]
        asol = attackers_sol[j]
        check_eq(5, f"Attacker {j} IC p: {asol['p'][0]} != {ai['p_s']}",
                 asol['p'][0], ai['p_s'])
        check_eq(5, f"Attacker {j} IC q: {asol['q'][0]} != {ai['q_s']}",
                 asol['q'][0], ai['q_s'])
        check_eq(5, f"Attacker {j} IC a: {rb(asol['a'][0])} != 1",
                 float(rb(asol['a'][0])), 1.0)

    def in_zone_physical(px, py):
        return math.sqrt(px * px + py * py) <= R_dz + tol

    def intercepts_physical(px, py, dx, dy):
        return math.sqrt((px - dx) ** 2 + (py - dy) ** 2) <= R_I + tol

    # ================================================================
    # Constraint 6: Attacker State Machine (self-consistency)
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        gamma_arr = asol.get('gamma')
        delta_mat = asol.get('delta')
        for k in range(1, N_a):
            ak = rb(asol['a'][k])
            ak1 = rb(asol['a'][k + 1])
            gm_stored = rb(gamma_arr[k]) if gamma_arr is not None else (
                1 if in_zone_physical(asol['p'][k], asol['q'][k]) else 0)
            if delta_mat is not None:
                dc_stored = 1 if any(rb(delta_mat[i][k]) == 1 for i in range(N_D)) else 0
            else:
                dc_stored = 0
                pk, qk = asol['p'][k], asol['q'][k]
                for i in range(N_D):
                    ds = defenders_sol[i]
                    xa_i, ya_i = def_pos(ds, k * T_a_val)
                    if intercepts_physical(pk, qk, xa_i, ya_i):
                        dc_stored = 1
                        break
            expected_ak1 = 1 if (ak == 1 and gm_stored == 0 and dc_stored == 0) else 0
            if ak1 != expected_ak1:
                record(6,
                       f"Attacker {j} state-machine k={k}: a[k]={ak}, "
                       f"gamma={gm_stored}, delta_comb={dc_stored} -> expected a[k+1]={expected_ak1}, got {ak1}",
                       float(ak1), float(expected_ak1), abs(ak1 - expected_ak1))

    # ================================================================
    # Constraint 7: Zone consistency (stored gamma vs physical)
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        gamma_arr = asol.get('gamma')
        if gamma_arr is None:
            continue
        for k in range(1, N_a + 1):
            pk, qk = asol['p'][k], asol['q'][k]
            dist = math.sqrt(pk * pk + qk * qk)
            gm = rb(gamma_arr[k])
            if gm == 1 and dist > R_dz + zone_tol:
                record(7,
                       f"Attacker {j} zone k={k}: stored gamma=1 but dist={dist:.6f} > R_dz+tol={R_dz + zone_tol:.6f}",
                       float(gm), 0.0, dist - R_dz - zone_tol)
            elif gm == 0 and dist < R_dz - zone_tol:
                record(7,
                       f"Attacker {j} zone k={k}: stored gamma=0 but dist={dist:.6f} < R_dz-tol={R_dz - zone_tol:.6f}",
                       float(gm), 1.0, R_dz - zone_tol - dist)

    # ================================================================
    # Constraint 8: Interception consistency (stored delta vs physical)
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        delta_mat = asol.get('delta')
        if delta_mat is None:
            continue
        for k in range(1, N_a + 1):
            pk, qk = asol['p'][k], asol['q'][k]
            for i in range(N_D):
                ds = defenders_sol[i]
                t_ak = k * T_a_val
                xa_i, ya_i = def_pos(ds, t_ak)
                dist = math.sqrt((pk - xa_i) ** 2 + (qk - ya_i) ** 2)
                d = rb(delta_mat[i][k])
                if d == 1 and dist > R_I + intercept_tol:
                    record(8,
                           f"Intercept i={i} j={j} k={k}: stored delta=1 but dist={dist:.6f} > R_I+tol={R_I + intercept_tol:.6f}",
                           float(d), 0.0, dist - R_I - intercept_tol)
                elif d == 0 and dist < R_I - intercept_tol:
                    record(8,
                           f"Intercept i={i} j={j} k={k}: stored delta=0 but dist={dist:.6f} < R_I-tol={R_I - intercept_tol:.6f}",
                           float(d), 1.0, R_I - intercept_tol - dist)

    # ================================================================
    # Constraint 9: Defender Zone Avoidance (physical circle check)
    # ================================================================
    for i in range(N_D):
        ds = defenders_sol[i]
        for k in range(1, N_o + 1):
            xk, yk = ds['x'][k], ds['y'][k]
            dist = math.sqrt(xk * xk + yk * yk)
            if dist < R_dz - avoid_tol:
                record(9,
                       f"Defender {i} enters zone at k={k}: dist={dist:.6f} < R_dz-tol={R_dz - avoid_tol:.6f}",
                       dist, R_dz, R_dz - avoid_tol - dist)

    # ================================================================
    # Constraint 10: Objective Consistency (Tier C anti-exploit defense)
    # Recompute J = sum_{j,k>=1} gamma_j[k]
    #             + epsilon * sum_{i,k}(z_xi[k] + z_yi[k])
    # When zx/zy are absent from the solution, use z = |u| (the minimum
    # value of z given fixed u, which is what the solver picks at optimum
    # since z appears only in the obj with positive coefficient and the
    # only constraint is z >= |u|). This makes the recomputation a unique
    # deterministic function of the displayed variables, suitable for
    # strict equality checking against the reported objective_value.
    # ================================================================
    reported_raw = sol.get('objective_value')
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            epsilon = float(inst.get('epsilon', 0.0))
            sum_gamma = 0
            gamma_available = True
            for j in range(N_A):
                asol = attackers_sol[j]
                g = asol.get('gamma')
                if g is None:
                    gamma_available = False
                    break
                for k in range(1, N_a + 1):
                    if k < len(g):
                        sum_gamma += rb(g[k])
            if gamma_available:
                sum_z = 0.0
                z_available = True
                for i in range(N_D):
                    ds = defenders_sol[i]
                    ux_list = ds.get('ux')
                    uy_list = ds.get('uy')
                    if ux_list is None or uy_list is None:
                        z_available = False
                        break
                    zx_list = ds.get('zx')
                    zy_list = ds.get('zy')
                    for k in range(N_u):
                        if zx_list is not None and k < len(zx_list):
                            sum_z += float(zx_list[k])
                        elif k < len(ux_list):
                            sum_z += abs(float(ux_list[k]))
                        if zy_list is not None and k < len(zy_list):
                            sum_z += float(zy_list[k])
                        elif k < len(uy_list):
                            sum_z += abs(float(uy_list[k]))
                if z_available:
                    true_obj = float(sum_gamma) + epsilon * sum_z
                    diff = abs(reported - true_obj)
                    tol_obj = max(1e-3, 1e-3 * abs(true_obj))
                    if diff > tol_obj:
                        record(10,
                               f"Objective consistency violated: reported objective_value="
                               f"{reported} differs from recomputed "
                               f"sum_gamma + epsilon*sum(z|=|u|)={true_obj} "
                               f"(sum_gamma={sum_gamma}, epsilon={epsilon}, sum_z={sum_z}, "
                               f"|diff|={diff:.3g}, tol={tol_obj:.3g})",
                               reported, true_obj, diff)
        elif reported is not None and not math.isfinite(reported):
            # reported is +/-inf or NaN: cannot match any finite recomputed obj.
            # Compute a finite recomputed obj for the message; if all parts
            # are accessible, this still flags constraint 10.
            epsilon = float(inst.get('epsilon', 0.0))
            sum_gamma = 0
            for j in range(N_A):
                asol = attackers_sol[j]
                g = asol.get('gamma')
                if g is None:
                    sum_gamma = None
                    break
                for k in range(1, N_a + 1):
                    if k < len(g):
                        sum_gamma += rb(g[k])
            sum_z = 0.0
            if sum_gamma is not None:
                for i in range(N_D):
                    ds = defenders_sol[i]
                    ux_list = ds.get('ux')
                    uy_list = ds.get('uy')
                    if ux_list is None or uy_list is None:
                        sum_gamma = None
                        break
                    zx_list = ds.get('zx')
                    zy_list = ds.get('zy')
                    for k in range(N_u):
                        if zx_list is not None and k < len(zx_list):
                            sum_z += float(zx_list[k])
                        elif k < len(ux_list):
                            sum_z += abs(float(ux_list[k]))
                        if zy_list is not None and k < len(zy_list):
                            sum_z += float(zy_list[k])
                        elif k < len(uy_list):
                            sum_z += abs(float(uy_list[k]))
            if sum_gamma is not None:
                true_obj = float(sum_gamma) + epsilon * sum_z
                record(10,
                       f"Objective consistency violated: reported objective_value="
                       f"{reported} is not finite; recomputed obj={true_obj}",
                       reported, true_obj, float('inf'))

    # ================================================================
    # Aggregate and output
    # ================================================================
    violated_constraints = sorted(constraint_violations.keys())

    agg_violations = []
    for cidx in violated_constraints:
        msgs = constraint_violations[cidx]
        count = len(msgs)
        if count == 1:
            agg_violations.append(f"Constraint {cidx}: {msgs[0]}")
        else:
            agg_violations.append(
                f"Constraint {cidx}: {count} violations, e.g., {msgs[0]}")

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": agg_violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a candidate solution for Defensive Drill 1.")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON solution file.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON feasibility result.')
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)

    result = check_feasibility(inst, sol)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    feasible = result["feasible"]
    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == '__main__':
    main()
