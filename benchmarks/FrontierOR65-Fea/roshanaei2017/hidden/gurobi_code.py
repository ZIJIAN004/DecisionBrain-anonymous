"""
CORPS (Collaborative Operating Room Planning and Scheduling) model
from Roshanaei et al. (2017), implemented as a monolithic MIP solved by Gurobi.

This implements the FULL original problem:
  - Allocation Master Problem (MP), constraints (1)-(8)
  - Sequencing Subproblem (SP), constraints (9)-(22)
both merged into a single MIP. Sequencing decisions (finish times f_p, OR-order
binaries eta, surgeon-order binaries pi, OR completion c_{h,d,r}, surgeon
start/end i_s/e_s) are added per (hospital, day) so that the assignment
produced by MP is guaranteed to admit a feasible no-overlap schedule respecting
turnover/cleaning, surgeon availability, and OR overtime caps.
"""

import argparse
import json
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


def solve(instance_path: str, solution_path: str, time_limit: float) -> dict:
    with open(instance_path, "r") as f:
        data = json.load(f)

    num_patients = data["num_patients"]
    num_surgeons = data["num_surgeons"]
    num_hospitals = data["num_hospitals"]
    num_days = data["num_days"]
    alpha = data["alpha"]

    patients = data["patients"]
    surgeons = data["surgeons"]
    hospitals = data["hospitals"]

    # ----------------------------------------------------------------
    # Pre-compute derived data
    # ----------------------------------------------------------------

    # Surgeon operating days (set of 0-indexed days)
    surgeon_days = {}  # surgeon_id -> set of day indices
    surgeon_avail = {}  # (s, d) -> A_sd
    for surg in surgeons:
        s = surg["surgeon_id"]
        surgeon_days[s] = set(surg["operating_days"])
        for day_str, avail in surg["availability_by_day"].items():
            surgeon_avail[(s, int(day_str))] = avail

    # Hospital / OR data
    or_data = {}  # (h, r) -> OR dict
    or_ids_by_hospital = {}  # h -> list of or_id
    for hosp in hospitals:
        h = hosp["hospital_id"]
        or_ids_by_hospital[h] = []
        for or_info in hosp["ORs"]:
            r = or_info["or_id"]
            or_ids_by_hospital[h].append(r)
            or_data[(h, r)] = or_info

    # Surgeon fixed cost per day at hospital h: L_{s,h,d} (same for all days)
    surgeon_hosp_cost = {}  # (s, h) -> cost
    for hosp in hospitals:
        h = hosp["hospital_id"]
        for s_str, cost in hosp["surgeon_fixed_costs_per_day"].items():
            surgeon_hosp_cost[(int(s_str), h)] = cost

    # Patient data helpers
    patient_data = {}
    for pat in patients:
        p = pat["patient_id"]
        patient_data[p] = pat

    # ----------------------------------------------------------------
    # Cost / parameter accessors
    # ----------------------------------------------------------------

    def get_K(h, d, r):
        info = or_data[(h, r)]
        B = info["daily"][str(d)]["regular_time"]  # minutes
        return info["fixed_cost_per_hour"] * (B / 60.0)

    def get_C(h, d, r):
        info = or_data[(h, r)]
        return info["overtime_cost_per_hour"] / 60.0

    def get_B(h, d, r):
        return or_data[(h, r)]["daily"][str(d)]["regular_time"]

    def get_V(h, d, r):
        return or_data[(h, r)]["daily"][str(d)]["max_overtime"]

    def get_L(s, h, d):
        return surgeon_hosp_cost.get((s, h), 0.0)

    def get_F(p):
        return patient_data[p]["preparation_time"]

    def get_G(p):
        return patient_data[p]["cleaning_time"]

    def get_T(p, s):
        pat = patient_data[p]
        E_ps = pat["surgeon_specific_times"][str(s)]
        return pat["preparation_time"] + E_ps + pat["cleaning_time"]

    def get_E(p, s):
        return patient_data[p]["surgeon_specific_times"][str(s)]

    # ----------------------------------------------------------------
    # Build feasible (p, s, h, d, r) tuples (allocation domain)
    # ----------------------------------------------------------------
    feasible_tuples = []
    tuples_by_patient = {}
    tuples_by_or_day = {}        # (h,d,r) -> list of (p,s,h,d,r)
    tuples_by_surgeon_day = {}   # (s,d) -> list of (p,s,h,d,r)
    tuples_by_pshd = {}          # (p,s,h,d) -> list of r
    tuples_by_shd = {}           # (s,h,d) -> list of (p,s,h,d,r)
    tuples_by_phd = {}           # (p,h,d) -> list of (p,s,h,d,r)
    tuples_by_phdr = {}          # (p,h,d,r) -> list of (p,s,h,d,r)

    for pat in patients:
        p = pat["patient_id"]
        due_date = pat["due_date"]
        tuples_by_patient[p] = []

        for s in pat["eligible_surgeons"]:
            if s not in surgeon_days:
                continue
            for h_str, eligible_ors in pat["eligible_ORs_by_hospital"].items():
                h = int(h_str)
                for r in eligible_ors:
                    for d in surgeon_days[s]:
                        # due_date 是允许完成的最晚日(inclusive),所以仅当 d > due_date 时跳过
                        # 原代码 `due_date <= d: continue` off-by-one,把 d == due_date 也错误排除
                        if d > due_date:
                            continue
                        tup = (p, s, h, d, r)
                        feasible_tuples.append(tup)
                        tuples_by_patient[p].append(tup)
                        tuples_by_or_day.setdefault((h, d, r), []).append(tup)
                        tuples_by_surgeon_day.setdefault((s, d), []).append(tup)
                        tuples_by_pshd.setdefault((p, s, h, d), []).append(r)
                        tuples_by_shd.setdefault((s, h, d), []).append(tup)
                        tuples_by_phd.setdefault((p, h, d), []).append(tup)
                        tuples_by_phdr.setdefault((p, h, d, r), []).append(tup)

    # Active sets
    active_hdr = set()
    active_shd = set()
    active_hd = set()
    for (p, s, h, d, r) in feasible_tuples:
        active_hdr.add((h, d, r))
        active_shd.add((s, h, d))
        active_hd.add((h, d))

    # Patients eligible at (h,d): those with at least one tuple at (h,d,*)
    patients_at_hd = {}
    for (h, d) in active_hd:
        plist = sorted({p for (p, s, hh, dd, r) in feasible_tuples
                        if hh == h and dd == d})
        patients_at_hd[(h, d)] = plist

    # Surgeons eligible at (h,d)
    surgeons_at_hd = {}
    for (h, d) in active_hd:
        slist = sorted({s for (s, hh, dd) in active_shd if hh == h and dd == d})
        surgeons_at_hd[(h, d)] = slist

    # ORs eligible at (h,d)
    ors_at_hd = {}
    for (h, d) in active_hd:
        rlist = sorted({r for (hh, dd, r) in active_hdr if hh == h and dd == d})
        ors_at_hd[(h, d)] = rlist

    # For each (h,d,r): list of patients that can be scheduled there
    patients_at_hdr = {}
    for (h, d, r) in active_hdr:
        plist = sorted({p for (p, s, hh, dd, rr) in feasible_tuples
                        if hh == h and dd == d and rr == r})
        patients_at_hdr[(h, d, r)] = plist

    # For each (h,d,s): patients that can be scheduled with that surgeon there
    patients_at_hds = {}
    for (s, h, d) in active_shd:
        plist = sorted({p for (p, ss, hh, dd, r) in feasible_tuples
                        if hh == h and dd == d and ss == s})
        patients_at_hds[(h, d, s)] = plist

    # ----------------------------------------------------------------
    # Big-M values per (h, d): max total OR time on that day + 1
    # (per math_model.txt comment 4)
    # ----------------------------------------------------------------
    bigM_hd = {}
    for (h, d) in active_hd:
        rlist = ors_at_hd[(h, d)]
        max_or_time = max(get_B(h, d, r) + get_V(h, d, r) for r in rlist)
        bigM_hd[(h, d)] = max_or_time + 1.0

    # ----------------------------------------------------------------
    # Create Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("CORPS")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # ----------------- Allocation variables (MP) -----------------
    x = model.addVars(feasible_tuples, vtype=GRB.BINARY, name="x")
    y = model.addVars(active_hdr, vtype=GRB.BINARY, name="y")
    z = model.addVars(active_shd, vtype=GRB.BINARY, name="z")
    v = model.addVars(active_hdr, lb=0.0, name="v")  # overtime
    for (h, d, r) in active_hdr:
        v[h, d, r].ub = get_V(h, d, r)

    # ----------------- Sequencing variables (SP) -----------------
    # f[p, h, d]  finishing time of patient p if scheduled at hospital h on day d
    f_keys = list(tuples_by_phd.keys())
    f = model.addVars(f_keys, lb=0.0, name="f")

    # Per-OR upper bound on f_p when scheduled there
    for (p, h, d) in f_keys:
        f[p, h, d].ub = bigM_hd[(h, d)]

    # eta[p, k, h, d, r]  : 1 if p is operated AFTER k in OR r at (h,d)
    # Defined for p < k where p, k could share OR r (both have a tuple (.,.,h,d,r))
    eta_keys = []
    for (h, d, r) in active_hdr:
        plist = patients_at_hdr[(h, d, r)]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                eta_keys.append((p, k, h, d, r))
    eta = model.addVars(eta_keys, vtype=GRB.BINARY, name="eta")

    # pi[p, k, h, d, s]  : 1 if p is operated AFTER k on surgeon s's list at (h,d)
    pi_keys = []
    for (s, h, d) in active_shd:
        plist = patients_at_hds[(h, d, s)]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                pi_keys.append((p, k, h, d, s))
    pi = model.addVars(pi_keys, vtype=GRB.BINARY, name="pi")

    # c[h, d, r]  : completion time of OR r at (h,d)
    c = model.addVars(active_hdr, lb=0.0, name="c")
    for (h, d, r) in active_hdr:
        c[h, d, r].ub = bigM_hd[(h, d)]

    # i_s[s, h, d]  : starting time of surgeon s's day at (h,d)
    # e_s[s, h, d]  : ending time of surgeon s's day at (h,d)
    i_var = model.addVars(active_shd, lb=0.0, name="iS")
    e_var = model.addVars(active_shd, lb=0.0, name="eS")
    for (s, h, d) in active_shd:
        i_var[s, h, d].ub = bigM_hd[(h, d)]
        e_var[s, h, d].ub = bigM_hd[(h, d)]

    # ----------------------------------------------------------------
    # Objective: identical to MP
    # ----------------------------------------------------------------
    obj = gp.LinExpr()
    for (h, d, r) in active_hdr:
        obj += get_K(h, d, r) * y[h, d, r]
    for (s, h, d) in active_shd:
        obj += get_L(s, h, d) * z[s, h, d]
    for (h, d, r) in active_hdr:
        obj += get_C(h, d, r) * v[h, d, r]
    for pat in patients:
        p = pat["patient_id"]
        if not pat["is_mandatory"] and pat["reward"] > 0:
            for tup in tuples_by_patient.get(p, []):
                obj -= pat["reward"] * x[tup]
    model.setObjective(obj, GRB.MINIMIZE)

    # ================================================================
    # ALLOCATION (MP) constraints (1)-(8)
    # ================================================================

    # (1) Mandatory patients scheduled exactly once
    for pat in patients:
        p = pat["patient_id"]
        if pat["is_mandatory"]:
            model.addConstr(
                gp.quicksum(x[tup] for tup in tuples_by_patient.get(p, [])) == 1,
                name=f"c1_mand_{p}"
            )

    # (2) Optional patients at most once
    for pat in patients:
        p = pat["patient_id"]
        if not pat["is_mandatory"]:
            model.addConstr(
                gp.quicksum(x[tup] for tup in tuples_by_patient.get(p, [])) <= 1,
                name=f"c2_opt_{p}"
            )

    # (3) Surgeon at most one hospital per day
    surgeon_day_hospitals = {}
    for (s, h, d) in active_shd:
        surgeon_day_hospitals.setdefault((s, d), []).append((s, h, d))
    for (s, d), shd_list in surgeon_day_hospitals.items():
        if len(shd_list) >= 1:
            model.addConstr(
                gp.quicksum(z[s, h, d] for (s, h, d) in shd_list) <= 1,
                name=f"c3_surg_one_hosp_{s}_{d}"
            )

    # (4) x_pshdr <= z_shd (aggregated per (p,s,h,d))
    for (p, s, h, d), r_list in tuples_by_pshd.items():
        model.addConstr(
            gp.quicksum(x[p, s, h, d, r] for r in r_list) <= z[s, h, d],
            name=f"c4_xz_{p}_{s}_{h}_{d}"
        )

    # (5) x_pshdr <= y_hdr
    for tup in feasible_tuples:
        p, s, h, d, r = tup
        model.addConstr(x[tup] <= y[h, d, r], name=f"c5_xy_{p}_{s}_{h}_{d}_{r}")

    # (6) OR capacity (MP-level): sum T*x <= B*y + v
    for (h, d, r), tup_list in tuples_by_or_day.items():
        model.addConstr(
            gp.quicksum(get_T(p, s) * x[p, s, h, d, r]
                        for (p, s, _, _, _) in tup_list)
            <= get_B(h, d, r) * y[h, d, r] + v[h, d, r],
            name=f"c6_or_cap_{h}_{d}_{r}"
        )

    # (7) Surgeon weighted-time availability (MP-level)
    for (s, h, d), tup_list in tuples_by_shd.items():
        A_sd = surgeon_avail.get((s, d), 0)
        model.addConstr(
            gp.quicksum(
                (alpha * get_E(p, s) + (1 - alpha) * get_T(p, s))
                * x[p, s, h, d, r]
                for (p, s_val, h_val, d_val, r) in tup_list
            ) <= A_sd * z[s, h, d],
            name=f"c7_surg_avail_{s}_{h}_{d}"
        )

    # (8) Overtime bounds 0 <= v <= V_hdr (already set via lb/ub)
    # Add explicit upper bound constraint for documentation / dual visibility
    for (h, d, r) in active_hdr:
        model.addConstr(v[h, d, r] <= get_V(h, d, r),
                        name=f"c8_overtime_max_{h}_{d}_{r}")

    # ================================================================
    # SEQUENCING (SP) constraints (9)-(22)
    # ================================================================

    # (9) For every patient assigned at (h,d), exactly one (s,r) pair
    # This is implied by combination of (1)/(2) plus (4)/(5), but we add
    # the consistency form: sum_{s,r} x_pshdr at fixed (h,d) <= 1.
    # The "= 1" form only applies if patient is at (h,d), which is captured
    # by linking sequencing variables only to assigned tuples below.
    # No explicit constraint needed beyond (1)/(2).

    # (10) f_p >= F_p + sum_s,r E_ps * x_pshdr   [for any (p, h, d) where p is assigned]
    #      Linearize:  f[p,h,d] >= F_p * (sum x_phdrs) + sum E_ps x_pshdr
    # We tie f[p,h,d] to assignment: if patient not scheduled at (h,d), f=0 by lb.
    # Actually f[p,h,d] >= F_p*assigned + sum_s,r E_ps x_pshdr is fine.
    for (p, h, d), tup_list in tuples_by_phd.items():
        Fp = get_F(p)
        # assigned indicator = sum_{s,r} x_pshdr  (0 or 1)
        assigned_expr = gp.quicksum(x[t] for t in tup_list)
        E_expr = gp.quicksum(get_E(t[0], t[1]) * x[t] for t in tup_list)
        model.addConstr(
            f[p, h, d] >= Fp * assigned_expr + E_expr,
            name=f"c10_f_min_{p}_{h}_{d}"
        )
        # Also: if not assigned at (h,d), force f=0 (tightening, optional)
        model.addConstr(
            f[p, h, d] <= bigM_hd[(h, d)] * assigned_expr,
            name=f"c10b_f_zero_{p}_{h}_{d}"
        )

    # (11)/(12) OR no-overlap: for p<k sharing OR r at (h,d)
    # If both assigned to OR r:
    #   if eta_pkr=1 (p after k): f_p >= f_k + G_k + F_p + E_ps
    #   else (k after p):         f_k >= f_p + G_p + F_k + E_ks
    # Linearization with M*(slack) per math_model.txt.
    # x_psr (in math_model) corresponds here to sum_{s in Omega_p} x_pshdr (binary).
    # We use the unique (p,h,d,r)-row sum since at most one s.
    def x_at_phdr(p, h, d, r):
        """Sum x_pshdr over s for fixed (p,h,d,r)."""
        tlist = tuples_by_phdr.get((p, h, d, r), [])
        if not tlist:
            return None
        return gp.quicksum(x[t] for t in tlist)

    for (h, d, r) in active_hdr:
        plist = patients_at_hdr[(h, d, r)]
        M = bigM_hd[(h, d)]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                xp = x_at_phdr(p, h, d, r)
                xk = x_at_phdr(k, h, d, r)
                Fp, Fk = get_F(p), get_F(k)
                Gp, Gk = get_G(p), get_G(k)
                # E_ps depends on s — but since at most one s wins, we can
                # use sum_s E_ps * x_pshdr (this is exactly E_ps if assigned).
                E_p_expr = gp.quicksum(get_E(t[0], t[1]) * x[t]
                                       for t in tuples_by_phdr[(p, h, d, r)])
                E_k_expr = gp.quicksum(get_E(t[0], t[1]) * x[t]
                                       for t in tuples_by_phdr[(k, h, d, r)])
                # (11) p after k
                model.addConstr(
                    f[p, h, d] >= f[k, h, d] + Gk * xk + Fp * xp + E_p_expr
                                 - M * (3 - eta[p, k, h, d, r] - xp - xk),
                    name=f"c11_ord_pk_{p}_{k}_{h}_{d}_{r}"
                )
                # (12) k after p
                model.addConstr(
                    f[k, h, d] >= f[p, h, d] + Gp * xp + Fk * xk + E_k_expr
                                 - M * (2 + eta[p, k, h, d, r] - xp - xk),
                    name=f"c12_ord_kp_{p}_{k}_{h}_{d}_{r}"
                )

    # (13)/(14) Surgeon no-overlap: for p<k sharing surgeon s at (h,d)
    # If both assigned to surgeon s (xpsr summed over r=1):
    #   if pi_pks=1 (p after k): f_p >= f_k + E_ps
    #   else                   : f_k >= f_p + E_ks
    # NOTE: for surgeon ordering, paper writes E_ps without prep time -- this
    # is because the surgeon only needs to be present DURING the surgery proper.
    def x_at_phds(p, h, d, s):
        """Sum x_pshdr over r for fixed (p,s,h,d)."""
        rlist = tuples_by_pshd.get((p, s, h, d), [])
        if not rlist:
            return None
        return gp.quicksum(x[(p, s, h, d, r)] for r in rlist)

    for (s, h, d) in active_shd:
        plist = patients_at_hds[(h, d, s)]
        M = bigM_hd[(h, d)]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                xp = x_at_phds(p, h, d, s)
                xk = x_at_phds(k, h, d, s)
                E_ps = get_E(p, s)
                E_ks = get_E(k, s)
                # (13) p after k
                model.addConstr(
                    f[p, h, d] >= f[k, h, d] + E_ps * xp
                                 - M * (3 - pi[p, k, h, d, s] - xp - xk),
                    name=f"c13_surg_pk_{p}_{k}_{h}_{d}_{s}"
                )
                # (14) k after p
                model.addConstr(
                    f[k, h, d] >= f[p, h, d] + E_ks * xk
                                 - M * (2 + pi[p, k, h, d, s] - xp - xk),
                    name=f"c14_surg_kp_{p}_{k}_{h}_{d}_{s}"
                )

    # (15) f_p + G_p <= B_r + v_r  whenever p uses OR r at (h,d)
    # Linearized: f_p + G_p - M*(1 - x_at_phdr) <= B_r*y_hdr + v_hdr
    for (h, d, r) in active_hdr:
        plist = patients_at_hdr[(h, d, r)]
        M = bigM_hd[(h, d)]
        Br = get_B(h, d, r)
        for p in plist:
            xpr = x_at_phdr(p, h, d, r)
            Gp = get_G(p)
            model.addConstr(
                f[p, h, d] + Gp * xpr - M * (1 - xpr)
                <= Br * y[h, d, r] + v[h, d, r],
                name=f"c15_f_le_Bv_{p}_{h}_{d}_{r}"
            )

    # (16) e_s >= f_p whenever p assigned to s at (h,d)
    # (17) i_s <= f_p - E_ps whenever p assigned to s at (h,d)
    for (s, h, d) in active_shd:
        plist = patients_at_hds[(h, d, s)]
        M = bigM_hd[(h, d)]
        for p in plist:
            xps = x_at_phds(p, h, d, s)
            E_ps = get_E(p, s)
            # (16)
            model.addConstr(
                e_var[s, h, d] >= f[p, h, d] - M * (1 - xps),
                name=f"c16_es_ge_fp_{s}_{p}_{h}_{d}"
            )
            # (17)
            model.addConstr(
                i_var[s, h, d] <= f[p, h, d] - E_ps * xps + M * (1 - xps),
                name=f"c17_is_le_fp_{s}_{p}_{h}_{d}"
            )

    # (18) e_s - i_s <= A_s   (use A_sd; the SP uses surgeon-day availability)
    for (s, h, d) in active_shd:
        A_sd = surgeon_avail.get((s, d), 0)
        model.addConstr(
            e_var[s, h, d] - i_var[s, h, d] <= A_sd,
            name=f"c18_surg_window_{s}_{h}_{d}"
        )

    # (19) c_r >= f_p + G_p whenever p uses OR r
    for (h, d, r) in active_hdr:
        plist = patients_at_hdr[(h, d, r)]
        M = bigM_hd[(h, d)]
        for p in plist:
            xpr = x_at_phdr(p, h, d, r)
            Gp = get_G(p)
            model.addConstr(
                c[h, d, r] >= f[p, h, d] + Gp * xpr - M * (1 - xpr),
                name=f"c19_cr_ge_fp_{p}_{h}_{d}_{r}"
            )

    # (20) 0 <= v_r <= V_r  (already enforced via bounds + (8))
    # (21) v_r >= c_r - B_r
    for (h, d, r) in active_hdr:
        Br = get_B(h, d, r)
        model.addConstr(
            v[h, d, r] >= c[h, d, r] - Br * y[h, d, r],
            name=f"c21_overtime_def_{h}_{d}_{r}"
        )

    # (22) integrality (already declared via vtype=BINARY)

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    wall_start = time.time()
    model.optimize()
    wall_time = time.time() - wall_start

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    result = {
        "instance_path": instance_path,
        "solver": "gurobi",
        "time_limit": time_limit,
        "wall_time": round(wall_time, 2),
        "status": model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status)),
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap

        # --- Patient assignments + finish times ---
        assignments = []
        for tup in feasible_tuples:
            if x[tup].X > 0.5:
                p, s, h, d, r = tup
                pat = patient_data[p]
                fp_val = f[p, h, d].X if (p, h, d) in f_keys else 0.0
                E_ps = get_E(p, s)
                Fp = get_F(p)
                Gp = get_G(p)
                start_time = max(0.0, fp_val - E_ps)
                room_in = max(0.0, fp_val - E_ps - Fp)
                room_out = fp_val + Gp
                assignments.append({
                    "patient_id": p,
                    "surgeon_id": s,
                    "hospital_id": h,
                    "day": d,
                    "or_id": r,
                    "is_mandatory": pat["is_mandatory"],
                    "T_ps": get_T(p, s),
                    "finish_time": round(fp_val, 4),
                    "surgery_start_time": round(start_time, 4),
                    "room_entry_time": round(room_in, 4),
                    "room_exit_time": round(room_out, 4),
                })

        # --- Opened ORs with completion + overtime ---
        opened_ors = []
        for (h, d, r) in active_hdr:
            if y[h, d, r].X > 0.5:
                opened_ors.append({
                    "hospital_id": h,
                    "day": d,
                    "or_id": r,
                    "regular_time": get_B(h, d, r),
                    "completion_time": round(c[h, d, r].X, 4),
                    "overtime": round(v[h, d, r].X, 4),
                })

        # --- Surgeon assignments with start/end ---
        surgeon_assignments = []
        for (s, h, d) in active_shd:
            if z[s, h, d].X > 0.5:
                surgeon_assignments.append({
                    "surgeon_id": s,
                    "hospital_id": h,
                    "day": d,
                    "start_time": round(i_var[s, h, d].X, 4),
                    "end_time": round(e_var[s, h, d].X, 4),
                })

        # --- Sequencing (order) decisions ---
        or_sequences = []
        for (p, k, h, d, r) in eta_keys:
            xp_v = sum(x[t].X for t in tuples_by_phdr.get((p, h, d, r), []))
            xk_v = sum(x[t].X for t in tuples_by_phdr.get((k, h, d, r), []))
            if xp_v > 0.5 and xk_v > 0.5:
                or_sequences.append({
                    "hospital_id": h,
                    "day": d,
                    "or_id": r,
                    "patient_p": p,
                    "patient_k": k,
                    "p_after_k": int(eta[p, k, h, d, r].X > 0.5),
                })

        surgeon_sequences = []
        for (p, k, h, d, s) in pi_keys:
            xp_v = sum(x[(p, s, h, d, r)].X
                       for r in tuples_by_pshd.get((p, s, h, d), []))
            xk_v = sum(x[(k, s, h, d, r)].X
                       for r in tuples_by_pshd.get((k, s, h, d), []))
            if xp_v > 0.5 and xk_v > 0.5:
                surgeon_sequences.append({
                    "hospital_id": h,
                    "day": d,
                    "surgeon_id": s,
                    "patient_p": p,
                    "patient_k": k,
                    "p_after_k": int(pi[p, k, h, d, s].X > 0.5),
                })

        result["assignments"] = assignments
        result["opened_ors"] = opened_ors
        result["surgeon_assignments"] = surgeon_assignments
        result["or_sequences"] = or_sequences
        result["surgeon_sequences"] = surgeon_sequences
        result["num_patients_scheduled"] = len(assignments)
        result["num_mandatory_scheduled"] = sum(
            1 for a in assignments if a["is_mandatory"])
        result["num_optional_scheduled"] = sum(
            1 for a in assignments if not a["is_mandatory"])
    else:
        result["objective_value"] = None
        result["best_bound"] = None
        result["mip_gap"] = None
        result["assignments"] = []
        result["opened_ors"] = []
        result["surgeon_assignments"] = []
        result["or_sequences"] = []
        result["surgeon_sequences"] = []
        result["num_patients_scheduled"] = 0
        result["num_mandatory_scheduled"] = 0
        result["num_optional_scheduled"] = 0

    with open(solution_path, "w") as f_out:
        json.dump(result, f_out, indent=2)

    print(f"Status: {result['status_name']}")
    print(f"Objective: {result['objective_value']}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Solution written to {solution_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the CORPS model (Roshanaei et al. 2017) with Gurobi"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Gurobi time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
