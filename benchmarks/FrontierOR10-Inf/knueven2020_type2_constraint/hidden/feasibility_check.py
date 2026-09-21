#!/usr/bin/env python3
"""
Feasibility checker for the Unit Commitment (UC) problem.

Paper: Knueven, Ostrowski, Watson (2020)
       "On Mixed Integer Programming Formulations for the Unit Commitment Problem"

Checks ORIGINAL 1-bin formulation constraints (the first concrete model the
paper writes, Slide 4, with decision variables u, p, pb). The 3-bin
auxiliaries v, w (startup/shutdown indicators) are NOT part of the
original and are derived internally from u only when needed for
min-up / min-down verification. Constraints:

  1.  Power balance: sum_g p[g,t] = demand[t]
  2.  [SKIPPED — 3-bin definitional linking u(t)-u(t-1)=v(t)-w(t)
        is a reformulation artifact, not an original-1-bin constraint]
  3.  Minimum up-time: via u consecutive-on-runs (derived v)
  4.  Minimum down-time: via u consecutive-off-runs (derived w)
  5.  Generation lower bound: p[g,t] >= p_min * u[g,t]
  6.  Generation upper bound: pb[g,t] <= p_max * u[g,t]
  7.  Power <= available: p[g,t] <= pb[g,t]
  8.  Ramp-up: p[g,t] - p[g,t-1] <= RU * u[g,t-1] + SU * v[g,t]
  9.  Ramp-down: p[g,t-1] - p[g,t] <= RD * u[g,t] + SD * w[g,t]
  10. VUB inequality (Slide 10, when UT >= TRU + TRD + 2)
  11. Piecewise cost sum feasibility: 0 <= p[g,t] - p_min*u[g,t] <= (p_max-p_min)*u[g,t]
  12. Piecewise segment bound (requires pl variables - not in solution, skip)
  13. Startup tier partition (requires delta variables - not in solution, skip)
  14. Startup tier window (requires delta variables - not in solution, skip)
  15. Reserve requirement: sum_g pb[g,t] >= demand[t] + reserve[t]
  16. Binary constraints: u, v, w in {0, 1}
  17. Non-negativity: p >= 0, pb >= 0
  18. Initial up/down time obligations
  19. Objective consistency (Tier C defense against score-gaming): the
        self-reported objective_value must equal the cost recomputed from
        the committed schedule. The objective is
          min sum_g sum_t [ NL_g*u + sum_l f^l*pl + sum_k SC_k*delta ].
        Since the piecewise-segment variables pl and the startup-tier
        variables delta are cost-minimised auxiliaries (no constraint
        binds them other than their definitions), they are fully
        determined by (u, p): pl is the cheapest segment fill of
        p-p_min*u, and delta picks the cheapest eligible startup tier.
        This makes a FULL exact recompute possible from the solution's
        u and p alone. A reported objective that disagrees with the
        recompute (beyond a tight relative tolerance) is rejected.

NOTE: feasibility_check_new.py adds only constraint 19 on top of the
original feasibility_check.py; constraints 1-18 are byte-for-byte the
original checks. The eval-pipeline JSON shape (feasible /
violated_constraints / violations / violation_magnitudes) is unchanged.
"""

import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a UC candidate solution."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    return parser.parse_args()


def compute_startup_windows(startup_costs):
    """
    Compute [lo, hi] downtime windows for each startup cost tier.

    Mirrors gurobi_code.compute_startup_windows exactly: consecutive tiers
    partition the downtime axis, lo_k = prev_hi + 1, hi_k = max_downtime_hours
    of tier k. The last (cold) tier has hi=None (no upper bound).
    """
    windows = []
    prev_max = 0
    for tier in startup_costs:
        lo = prev_max + 1
        hi = tier["max_downtime_hours"]  # None for the last (cold) tier
        windows.append((lo, hi))
        if hi is not None:
            prev_max = hi
    return windows


def recompute_objective(data, sol):
    """
    Recompute the UC objective exactly from the reported schedule (u, p).

    Objective (gurobi_code.py): for every generator g and period t,
        NL_g * u[g,t]                              (no-load cost)
      + sum_l marginal_cost_l * pl[g,t,l]          (piecewise production cost)
      + sum_k SC_k * delta[g,t,k]                  (downtime-dependent startup)

    pl and delta are cost-minimised LP auxiliaries with no binding
    constraint other than their definitions, so they are determined by
    (u, p):
      * production cost = cheapest segment fill of p-p_min*u; segments are
        contiguous and monotone in marginal cost, so the greedy positional
        fill is optimal.
      * startup cost = cheapest eligible tier for each startup event,
        where tier k (< cold) is eligible iff a shutdown lies in its
        downtime window [lo_k, hi_k] (in-horizon w events or, for an
        initially-off generator, the pre-horizon shutdown). The cold tier
        is always eligible. This replicates the matching-formulation delta
        constraints in gurobi_code.py.

    Returns the recomputed objective (float).
    """
    generators = data["generators"]
    demand = data["demand"]
    T = len(demand)
    schedule = sol["schedule"]

    total = 0.0
    for gd in generators:
        g_id = gd["id"]
        s = schedule[g_id]
        u_arr = s["u"]
        p_arr = s["p"]

        p_min = gd["p_min"]
        NL = gd["no_load_cost"]
        segs = gd["production_cost_segments"]
        tiers = gd["startup_costs"]
        windows = compute_startup_windows(tiers)
        n_tiers = len(tiers)
        init = int(gd["initial_status"])

        # Derive v, w from u exactly as constraints 3/4/8/9 do above.
        u0 = 1 if init > 0 else 0
        u_prev_arr = [u0] + list(u_arr[:-1])
        v_arr = [max(0, u_arr[t] - u_prev_arr[t]) for t in range(T)]
        w_arr = [max(0, u_prev_arr[t] - u_arr[t]) for t in range(T)]

        for t in range(T):
            ut = u_arr[t]

            # --- No-load cost --------------------------------------------
            total += NL * ut

            # --- Piecewise production cost (greedy / cheapest fill) ------
            # When the generator is off, pl is pinned to 0 in the MIP, so
            # there is no production cost regardless of p.
            if ut >= 0.5:
                remaining = p_arr[t] - p_min * ut
                for seg in segs:
                    width = seg["output_mw_end"] - seg["output_mw_start"]
                    take = min(max(remaining, 0.0), width)
                    total += take * seg["marginal_cost_per_mwh"]
                    remaining -= take

            # --- Downtime-dependent startup cost -------------------------
            if v_arr[t] >= 0.5:
                best_cost = None
                for k in range(n_tiers):
                    if k == n_tiers - 1:
                        eligible = True  # cold tier: residual, no window UB
                    else:
                        lo, hi = windows[k]
                        eligible = False
                        for i in range(lo, hi + 1):
                            prev_t = t - i
                            if prev_t >= 0:
                                if w_arr[prev_t] >= 0.5:
                                    eligible = True
                                    break
                            elif init < 0:
                                # pre-horizon shutdown |init| periods before t=0
                                shutdown_ago = t + abs(init)
                                if lo <= shutdown_ago <= hi:
                                    eligible = True
                                    break
                    if eligible:
                        c = tiers[k]["cost"]
                        if best_cost is None or c < best_cost:
                            best_cost = c
                if best_cost is not None:
                    total += best_cost

    return total


def main():
    args = parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)

    tol = 1e-5
    eps = 1e-5

    generators = data["generators"]
    demand = data["demand"]
    reserve = data.get("reserve_requirement", [0.0] * len(demand))
    T = len(demand)
    G = len(generators)
    schedule = sol["schedule"]
    gen_ids = [gd["id"] for gd in generators]

    violated_set = set()
    violation_messages = []  # (constraint_idx, message)
    violation_magnitudes = []

    def record(constraint_idx, msg, lhs, rhs, operator):
        """Check a constraint and record violation if any."""
        if operator == "<=":
            violation_amount = lhs - rhs
        elif operator == ">=":
            violation_amount = rhs - lhs
        elif operator == "=":
            violation_amount = abs(lhs - rhs)
        else:
            return

        if violation_amount > tol:
            violated_set.add(constraint_idx)
            violation_messages.append((constraint_idx, msg))
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Initial condition helpers
    def u0(g_idx):
        return 1 if generators[g_idx]["initial_status"] > 0 else 0

    def p0(g_idx):
        gd = generators[g_idx]
        return gd["initial_power"] if gd["initial_status"] > 0 else 0.0

    # ==========================================================================
    # Constraint 1: Power balance  sum_g p[g,t] = demand[t]
    # ==========================================================================
    for t in range(T):
        total_p = sum(schedule[gen_ids[g]]["p"][t] for g in range(G))
        record(1,
               f"Power balance violated at t={t+1}: sum_p={total_p:.4f}, demand={demand[t]:.4f}",
               total_p, demand[t], "=")

    # ==========================================================================
    # Per-generator constraints (2-10, 11, 16-18)
    # ==========================================================================
    for g_idx in range(G):
        gd = generators[g_idx]
        g_id = gd["id"]
        s = schedule[g_id]
        u_arr = s["u"]
        p_arr = s["p"]
        pb_arr = s["pb"]

        # Derive v, w from u (1-bin projection): v(t)=1 iff u(t)=1 and u(t-1)=0
        # (startup event), w(t)=1 iff u(t-1)=1 and u(t)=0 (shutdown event).
        # If the solver chose to export v, w explicitly, we still use the
        # derived values since only u is in the original formulation.
        u_prev_arr = [1 if generators[g_idx]["initial_status"] > 0 else 0] + list(u_arr[:-1])
        v_arr = [max(0, u_arr[t] - u_prev_arr[t]) for t in range(T)]
        w_arr = [max(0, u_prev_arr[t] - u_arr[t]) for t in range(T)]

        p_min = gd["p_min"]
        p_max = gd["p_max"]
        RU = gd["ramp_up_limit"]
        RD = gd["ramp_down_limit"]
        SU = gd["startup_ramp_limit"]
        SD = gd["shutdown_ramp_limit"]
        UT = int(gd["min_up_time"])
        DT = int(gd["min_down_time"])
        init = int(gd["initial_status"])

        # Compute TRU, TRD for VUB
        TRU = max(0, math.ceil((p_max - SU) / RU)) if RU > 0 else 0
        TRD_val = max(0, math.ceil((p_max - SD) / RD)) if RD > 0 else 0
        apply_vub = (UT >= TRU + TRD_val + 2)

        for t in range(T):
            ut = u_arr[t]
            vt = v_arr[t]
            wt = w_arr[t]
            pt = p_arr[t]
            pbt = pb_arr[t]
            u_prev = u_arr[t - 1] if t > 0 else u0(g_idx)
            p_prev = p_arr[t - 1] if t > 0 else p0(g_idx)

            # ------------------------------------------------------------------
            # Constraint 2 INTENTIONALLY SKIPPED: the 3-bin definitional
            # identity u(t)-u(t-1) = v(t)-w(t) is not an original 1-bin
            # constraint. With v, w derived from u above, it is trivially
            # satisfied by construction.
            # ------------------------------------------------------------------
            # Constraint 3: Minimum up-time
            # sum_{s=max(1,t-UT+1)}^{t} v[g,s] <= u[g,t]
            # ------------------------------------------------------------------
            start_ut = max(0, t - UT + 1)
            sum_v = sum(v_arr[ss] for ss in range(start_ut, t + 1))
            record(3,
                   f"Min up-time violated for {g_id} at t={t+1}: sum_v={sum_v:.4f}, u={ut:.4f}",
                   sum_v, ut, "<=")

            # ------------------------------------------------------------------
            # Constraint 4: Minimum down-time
            # sum_{s=max(1,t-DT+1)}^{t} w[g,s] <= 1 - u[g,t]
            # ------------------------------------------------------------------
            start_dt = max(0, t - DT + 1)
            sum_w = sum(w_arr[ss] for ss in range(start_dt, t + 1))
            record(4,
                   f"Min down-time violated for {g_id} at t={t+1}: sum_w={sum_w:.4f}, 1-u={1-ut:.4f}",
                   sum_w, 1 - ut, "<=")

            # ------------------------------------------------------------------
            # Constraint 5: Generation lower bound  p[g,t] >= p_min * u[g,t]
            # ------------------------------------------------------------------
            record(5,
                   f"Gen lower limit violated for {g_id} at t={t+1}: "
                   f"p={pt:.4f} < p_min*u={p_min * ut:.4f}",
                   p_min * ut, pt, "<=")

            # ------------------------------------------------------------------
            # Constraint 6: Generation upper bound  pb[g,t] <= p_max * u[g,t]
            # ------------------------------------------------------------------
            record(6,
                   f"Gen upper limit violated for {g_id} at t={t+1}: "
                   f"pb={pbt:.4f} > p_max*u={p_max * ut:.4f}",
                   pbt, p_max * ut, "<=")

            # ------------------------------------------------------------------
            # Constraint 7: Power <= available  p[g,t] <= pb[g,t]
            # ------------------------------------------------------------------
            record(7,
                   f"p > pb for {g_id} at t={t+1}: p={pt:.4f}, pb={pbt:.4f}",
                   pt, pbt, "<=")

            # ------------------------------------------------------------------
            # Constraint 8: Ramp-up
            # p[g,t] - p[g,t-1] <= RU * u[g,t-1] + SU * v[g,t]
            # ------------------------------------------------------------------
            ramp_up_lhs = pt - p_prev
            ramp_up_rhs = RU * u_prev + SU * vt
            record(8,
                   f"Ramp-up violated for {g_id} at t={t+1}: "
                   f"delta_p={ramp_up_lhs:.4f}, limit={ramp_up_rhs:.4f}",
                   ramp_up_lhs, ramp_up_rhs, "<=")

            # ------------------------------------------------------------------
            # Constraint 9: Ramp-down
            # p[g,t-1] - p[g,t] <= RD * u[g,t] + SD * w[g,t]
            # ------------------------------------------------------------------
            ramp_dn_lhs = p_prev - pt
            ramp_dn_rhs = RD * ut + SD * wt
            record(9,
                   f"Ramp-down violated for {g_id} at t={t+1}: "
                   f"delta_p={ramp_dn_lhs:.4f}, limit={ramp_dn_rhs:.4f}",
                   ramp_dn_lhs, ramp_dn_rhs, "<=")

            # ------------------------------------------------------------------
            # Constraint 10: VUB inequality (Slide 10)
            # p(t) <= p_max*u(t) - sum_i coeff_i*v(t-i) - sum_i coeff_i*w(t+1+i)
            # Only when UT >= TRU + TRD + 2
            # ------------------------------------------------------------------
            if apply_vub:
                startup_sum = 0.0
                for i in range(TRU + 1):
                    coeff = max(0.0, p_max - (SU + i * RU))
                    ti = t - i
                    if 0 <= ti < T:
                        startup_sum += coeff * v_arr[ti]

                shutdown_sum = 0.0
                for i in range(TRD_val + 1):
                    coeff = max(0.0, p_max - (SD + i * RD))
                    ti = t + 1 + i
                    if 0 <= ti < T:
                        shutdown_sum += coeff * w_arr[ti]

                vub_rhs = p_max * ut - startup_sum - shutdown_sum
                record(10,
                       f"VUB violated for {g_id} at t={t+1}: "
                       f"p={pt:.4f}, VUB_limit={vub_rhs:.4f}",
                       pt, vub_rhs, "<=")

            # ------------------------------------------------------------------
            # Constraint 11: Piecewise sum feasibility (implied)
            # p_above_min = p[g,t] - p_min*u[g,t] must be in [0, (p_max-p_min)*u[g,t]]
            # Lower part (>= 0) is constraint 5; upper part checked here.
            # ------------------------------------------------------------------
            p_above_min = pt - p_min * ut
            max_above_min = (p_max - p_min) * ut
            record(11,
                   f"Piecewise sum feasibility violated for {g_id} at t={t+1}: "
                   f"p_above_min={p_above_min:.4f}, max={max_above_min:.4f}",
                   p_above_min, max_above_min, "<=")

            # ------------------------------------------------------------------
            # Constraint 16: Binary constraint u in {0, 1} (original variable).
            # v, w are derived from u, so their domain is automatic.
            # ------------------------------------------------------------------
            rounded = round(ut)
            if rounded not in (0, 1):
                dev = min(abs(ut), abs(ut - 1))
                record(16,
                       f"Binary violated for {g_id}.u at t={t+1}: val={ut:.6f}",
                       dev, 0.0, "=")
            else:
                dev = abs(ut - rounded)
                record(16,
                       f"Binary violated for {g_id}.u at t={t+1}: val={ut:.6f}",
                       dev, 0.0, "=")

            # ------------------------------------------------------------------
            # Constraint 17: Non-negativity  p >= 0, pb >= 0
            # ------------------------------------------------------------------
            record(17,
                   f"Negative p for {g_id} at t={t+1}: p={pt:.6f}",
                   0.0, pt, "<=")
            record(17,
                   f"Negative pb for {g_id} at t={t+1}: pb={pbt:.6f}",
                   0.0, pbt, "<=")

        # ----------------------------------------------------------------------
        # Constraint 18: Initial up/down time obligations
        # If init > 0: u[g,t] must be 1 for t=1..max(0, UT-init)
        # If init < 0: u[g,t] must be 0 for t=1..max(0, DT-|init|)
        # ----------------------------------------------------------------------
        if init > 0:
            must_on = max(0, UT - init)
            for t in range(min(must_on, T)):
                if u_arr[t] < 1 - tol:
                    record(18,
                           f"Initial up-time obligation violated for {g_id} at t={t+1}: "
                           f"u={u_arr[t]:.4f}, must be 1",
                           1.0, u_arr[t], "<=")
        elif init < 0:
            must_off = max(0, DT - abs(init))
            for t in range(min(must_off, T)):
                if u_arr[t] > tol:
                    record(18,
                           f"Initial down-time obligation violated for {g_id} at t={t+1}: "
                           f"u={u_arr[t]:.4f}, must be 0",
                           u_arr[t], 0.0, "<=")

    # ==========================================================================
    # Constraint 15: Reserve requirement  sum_g pb[g,t] >= demand[t] + reserve[t]
    # ==========================================================================
    for t in range(T):
        total_pb = sum(schedule[gen_ids[g]]["pb"][t] for g in range(G))
        rhs_val = demand[t] + reserve[t]
        record(15,
               f"Reserve violated at t={t+1}: sum_pb={total_pb:.4f}, "
               f"demand+reserve={rhs_val:.4f}",
               rhs_val, total_pb, "<=")

    # ==========================================================================
    # Constraint 19: Objective consistency (Tier C anti-score-gaming check)
    #
    # Recompute the objective exactly from (u, p) and reject when the
    # self-reported objective_value disagrees. The recompute is exact
    # because the cost-only auxiliaries pl (piecewise segments) and delta
    # (startup tiers) are fully determined by (u, p) under minimisation
    # (see recompute_objective docstring). Tolerance: 1e-6 relative with a
    # 1e-3 absolute floor -- loose enough to absorb float summation noise
    # (observed |rel diff| <= 1e-13 on Gurobi references), tight enough to
    # catch any fabricated objective_value.
    # ==========================================================================
    reported_raw = sol.get("objective_value")
    reported = None
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None

    if reported is not None:
        true_obj = recompute_objective(data, sol)
        # Map a non-finite reported objective onto an effective value that
        # always trips the tolerance test (still a fabricated objective).
        if math.isnan(reported):
            eff = true_obj + max(1.0, abs(true_obj))
        elif math.isinf(reported):
            eff = math.copysign(sys.float_info.max, reported)
        else:
            eff = reported
        abs_diff = abs(eff - true_obj)
        obj_tol = max(1e-3, 1e-6 * abs(true_obj))
        if abs_diff > obj_tol:
            violated_set.add(19)
            msg = (f"Objective consistency violated: reported objective_value="
                   f"{reported} differs from recomputed cost "
                   f"(no-load + piecewise production + startup) ="
                   f"{true_obj:.6f} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})")
            violation_messages.append((19, msg))
            normalizer = max(abs(true_obj), eps)
            violation_magnitudes.append({
                "constraint": 19,
                "lhs": float(eff),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer),
            })

    # ==========================================================================
    # Compile results
    # ==========================================================================
    violated_constraints = sorted(violated_set)

    # Aggregate messages: collect unique messages per constraint
    aggregated_violations = []
    for c_idx in violated_constraints:
        msgs = [msg for ci, msg in violation_messages if ci == c_idx]
        if len(msgs) <= 3:
            aggregated_violations.extend(msgs)
        else:
            aggregated_violations.extend(msgs[:2])
            aggregated_violations.append(
                f"... and {len(msgs) - 2} more constraint {c_idx} violations"
            )

    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": violated_constraints,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {violated_constraints}")
        print(f"Total violation instances: {len(violation_magnitudes)}")

    print(f"Result written to: {args.result_path}")


if __name__ == "__main__":
    main()
