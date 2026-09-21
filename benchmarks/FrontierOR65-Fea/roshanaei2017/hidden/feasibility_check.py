#!/usr/bin/env python3
"""
Feasibility checker for CORPS (Collaborative Operating Room Planning and
Scheduling) solutions from Roshanaei et al. (2017).

Verifies the FULL original problem (constraints 1-22) plus a Tier C
objective-consistency check (constraint 23) that recomputes the MP
objective from the solution variables and rejects values that disagree
with the reported objective_value.

Constraint enumeration:
  0  : domain (continuous time vars >= 0, eta/pi in {0,1})
  1  : mandatory patients scheduled exactly once
  2  : optional patients scheduled at most once
  3  : surgeon at most one hospital per day
  4  : x_pshdr <= z_shd
  5  : x_pshdr <= y_hdr
  6  : OR capacity   sum_p T_ps x_pshdr <= B_hdr y_hdr + v_hdr
  7  : surgeon weighted-time availability
  8  : 0 <= v_hdr <= V_hdr
  9  : exactly one (s, r) per scheduled patient at (h, d)
  10 : f_p >= F_p + E_ps
  11 : per-OR no-overlap (also subsumes 12)
  13 : per-surgeon no-overlap (also subsumes 14)
  15 : f_p + G_p <= B_r + v_r
  16 : e_s >= f_p
  17 : i_s <= f_p - E_ps
  18 : e_s - i_s <= A_sd
  19 : c_r >= f_p + G_p
  21 : v_r >= c_r - B_r
  23 : objective consistency (Tier C: recomputed objective must match
       reported objective_value within tolerance)
"""

import argparse
import json
from collections import defaultdict


# ----------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------

def load_instance(path):
    with open(path) as f:
        data = json.load(f)

    patients = {p["patient_id"]: p for p in data["patients"]}
    surgeons = {s["surgeon_id"]: s for s in data["surgeons"]}
    hospitals = {h["hospital_id"]: h for h in data["hospitals"]}

    or_lookup = {}  # (h, r) -> or_info
    for h in data["hospitals"]:
        hid = h["hospital_id"]
        for or_info in h["ORs"]:
            or_lookup[(hid, or_info["or_id"])] = or_info

    return data, patients, surgeons, hospitals, or_lookup


def load_solution(path):
    with open(path) as f:
        sol = json.load(f)

    # Allocation
    x = {}
    finish_times = {}            # (p, h, d) -> f_p
    surgery_start = {}           # (p, h, d) -> start of surgery proper
    for a in sol.get("assignments", []):
        key = (a["patient_id"], a["surgeon_id"], a["hospital_id"],
               a["day"], a["or_id"])
        x[key] = 1
        ph = (a["patient_id"], a["hospital_id"], a["day"])
        if "finish_time" in a:
            finish_times[ph] = a["finish_time"]
        if "surgery_start_time" in a:
            surgery_start[ph] = a["surgery_start_time"]

    y = {}
    completion = {}              # (h, d, r) -> c_r
    if "opened_ors" in sol:
        for o in sol["opened_ors"]:
            y[(o["hospital_id"], o["day"], o["or_id"])] = 1
            if "completion_time" in o:
                completion[(o["hospital_id"], o["day"], o["or_id"])] = \
                    o["completion_time"]
    if "or_details" in sol:
        for o in sol["or_details"]:
            y[(o["hospital_id"], o["day"], o["or_id"])] = 1
    for (p, s, h, d, r) in x:
        y[(h, d, r)] = 1

    z = {}
    surg_start = {}              # (s, h, d) -> i_s
    surg_end = {}                # (s, h, d) -> e_s
    if "surgeon_assignments" in sol:
        for sa in sol["surgeon_assignments"]:
            key = (sa["surgeon_id"], sa["hospital_id"], sa["day"])
            z[key] = 1
            if "start_time" in sa:
                surg_start[key] = sa["start_time"]
            if "end_time" in sa:
                surg_end[key] = sa["end_time"]
    for (p, s, h, d, r) in x:
        z[(s, h, d)] = 1

    v = {}
    if "opened_ors" in sol:
        for o in sol["opened_ors"]:
            v[(o["hospital_id"], o["day"], o["or_id"])] = o.get("overtime", 0.0)
    elif "or_details" in sol:
        for o in sol["or_details"]:
            v[(o["hospital_id"], o["day"], o["or_id"])] = o.get(
                "mp_overtime_minutes", 0.0)
    for key in y:
        if key not in v:
            v[key] = 0.0

    return (x, y, z, v, finish_times, surgery_start, completion,
            surg_start, surg_end)


# ----------------------------------------------------------------
# Parameter accessors
# ----------------------------------------------------------------

def get_E(patients, p, s):
    return patients[p]["surgeon_specific_times"][str(s)]


def get_T(patients, p, s):
    pat = patients[p]
    return pat["preparation_time"] + get_E(patients, p, s) + pat["cleaning_time"]


def get_F(patients, p):
    return patients[p]["preparation_time"]


def get_G(patients, p):
    return patients[p]["cleaning_time"]


def get_B(or_lookup, h, d, r):
    info = or_lookup.get((h, r))
    if info and str(d) in info["daily"]:
        return info["daily"][str(d)]["regular_time"]
    return 480


def get_V_max(or_lookup, h, d, r):
    info = or_lookup.get((h, r))
    if info and str(d) in info["daily"]:
        return info["daily"][str(d)]["max_overtime"]
    return 120


def get_K(or_lookup, h, d, r):
    """Fixed cost of opening OR r at hospital h on day d.

    K_{hdr} = fixed_cost_per_hour * (regular_time_minutes / 60).
    """
    info = or_lookup.get((h, r))
    if info is None:
        return 0.0
    B = get_B(or_lookup, h, d, r)
    return float(info.get("fixed_cost_per_hour", 0.0)) * (B / 60.0)


def get_C(or_lookup, h, d, r):
    """Overtime cost per minute for OR r at hospital h on day d.

    C_{hdr} = overtime_cost_per_hour / 60  (since v is in minutes).
    """
    info = or_lookup.get((h, r))
    if info is None:
        return 0.0
    return float(info.get("overtime_cost_per_hour", 0.0)) / 60.0


def get_L(hospitals, s, h, d):
    """Surgeon-hospital-day fixed cost. Stored per hospital, per surgeon."""
    hosp = hospitals.get(h)
    if hosp is None:
        return 0.0
    return float(hosp.get("surgeon_fixed_costs_per_day", {}).get(str(s), 0.0))


# ----------------------------------------------------------------
# Main check
# ----------------------------------------------------------------

def check_feasibility(instance_path, solution_path, result_path):
    tol = 1e-3   # SP constraints often need slightly looser tolerance
    eps = 1e-5

    data, patients, surgeons, hospitals, or_lookup = load_instance(instance_path)

    with open(solution_path) as f:
        raw_sol = json.load(f)
    sol_status = raw_sol.get("status")
    sol_assignments = raw_sol.get("assignments", [])
    if sol_status in (3, "3", "INFEASIBLE", "NO_SOLUTION_FOUND") or \
       (raw_sol.get("objective_value") is None and len(sol_assignments) == 0):
        status_desc = raw_sol.get("status_name", raw_sol.get("status_description",
                                  raw_sol.get("status", "unknown")))
        result = {
            "feasible": False,
            "violated_constraints": [1],
            "violations": [
                f"No solution provided (solver status={status_desc}). "
                f"Empty assignments cannot satisfy constraint (1)."
            ],
            "violation_magnitudes": [{
                "constraint": 1, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            }],
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    (x, y, z, v, finish_times, surgery_start, completion,
     surg_start_d, surg_end_d) = load_solution(solution_path)

    num_days = data["num_days"]
    alpha = data["alpha"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def add_violation(constraint_idx, msg, lhs_val, rhs_val, raw_excess):
        violated_constraints.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs_val), eps)
        ratio = raw_excess / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs_val, "rhs": rhs_val,
            "raw_excess": raw_excess,
            "normalizer": normalizer, "ratio": ratio,
        })

    # ================================================================
    # VARIABLE-DOMAIN CHECKS
    # ----------------------------------------------------------------
    # Math model declares the continuous time variables non-negative
    # (f_p, c_r, i_s, e_s >= 0) and the order indicators (eta_{pk,hdr},
    # pi_{pk,hd}) binary in {0, 1}. The binary {0,1} domain of
    # x_{pshdr}, y_{hdr}, z_{shd} is enforced implicitly by how
    # load_solution reads them (set membership -> value 1; absence -> 0),
    # so they need no explicit numeric check here.
    # Use constraint index 0 to flag any domain violation distinctly
    # from the numbered model constraints (1)-(22).
    # ================================================================

    # f_p >= 0 (finish_time per assignment)
    for (p, h, d), fp in finish_times.items():
        if fp < -tol:
            add_violation(
                0,
                f"Domain: finish_time f_p < 0 for p={p},h={h},d={d}: "
                f"f_p={fp:.6f}",
                fp, 0.0, abs(fp),
            )

    # c_r >= 0 (completion_time per opened OR)
    for (h, d, r), cr in completion.items():
        if cr < -tol:
            add_violation(
                0,
                f"Domain: completion_time c_r < 0 for h={h},d={d},r={r}: "
                f"c_r={cr:.6f}",
                cr, 0.0, abs(cr),
            )

    # i_s >= 0 (surgeon start_time)
    for (s, h, d), is_ in surg_start_d.items():
        if is_ < -tol:
            add_violation(
                0,
                f"Domain: surgeon start_time i_s < 0 for s={s},h={h},d={d}: "
                f"i_s={is_:.6f}",
                is_, 0.0, abs(is_),
            )

    # e_s >= 0 (surgeon end_time)
    for (s, h, d), es in surg_end_d.items():
        if es < -tol:
            add_violation(
                0,
                f"Domain: surgeon end_time e_s < 0 for s={s},h={h},d={d}: "
                f"e_s={es:.6f}",
                es, 0.0, abs(es),
            )

    # eta in {0,1}: or_sequences[*].p_after_k
    for entry in raw_sol.get("or_sequences", []) or []:
        eta = entry.get("p_after_k")
        if eta is None:
            continue
        if not (abs(eta - 0.0) <= tol or abs(eta - 1.0) <= tol):
            add_violation(
                0,
                f"Domain: or_sequences p_after_k not binary for "
                f"h={entry.get('hospital_id')},d={entry.get('day')},"
                f"r={entry.get('or_id')},p={entry.get('patient_p')},"
                f"k={entry.get('patient_k')}: eta={eta}",
                float(eta), 1.0, min(abs(eta - 0.0), abs(eta - 1.0)),
            )

    # pi in {0,1}: surgeon_sequences[*].p_after_k
    for entry in raw_sol.get("surgeon_sequences", []) or []:
        pi_v = entry.get("p_after_k")
        if pi_v is None:
            continue
        if not (abs(pi_v - 0.0) <= tol or abs(pi_v - 1.0) <= tol):
            add_violation(
                0,
                f"Domain: surgeon_sequences p_after_k not binary for "
                f"h={entry.get('hospital_id')},d={entry.get('day')},"
                f"s={entry.get('surgeon_id')},p={entry.get('patient_p')},"
                f"k={entry.get('patient_k')}: pi={pi_v}",
                float(pi_v), 1.0, min(abs(pi_v - 0.0), abs(pi_v - 1.0)),
            )

    # ================================================================
    # (1) Mandatory patients scheduled exactly once
    # ================================================================
    patient_count = defaultdict(int)
    for (p, s, h, d, r) in x:
        patient_count[p] += 1

    for pid, pat in patients.items():
        if pat["due_date"] <= num_days:
            count = patient_count.get(pid, 0)
            if abs(count - 1.0) > tol:
                add_violation(
                    1,
                    f"Mandatory patient {pid} scheduled {count} time(s)",
                    float(count), 1.0, abs(count - 1.0),
                )

    # ================================================================
    # (2) Optional patients scheduled at most once
    # ================================================================
    for pid, pat in patients.items():
        if pat["due_date"] > num_days:
            count = patient_count.get(pid, 0)
            if count - 1.0 > tol:
                add_violation(
                    2,
                    f"Optional patient {pid} scheduled {count} times",
                    float(count), 1.0, count - 1.0,
                )

    # ================================================================
    # (3) Surgeon at most one hospital per day
    # ================================================================
    surgeon_hosp_per_day = defaultdict(set)
    for (s, h, d) in z:
        surgeon_hosp_per_day[(s, d)].add(h)

    for (s, d), hosp_set in surgeon_hosp_per_day.items():
        lhs_val = float(len(hosp_set))
        if lhs_val - 1.0 > tol:
            add_violation(
                3,
                f"Surgeon {s} at {len(hosp_set)} hospitals on day {d}",
                lhs_val, 1.0, lhs_val - 1.0,
            )

    # ================================================================
    # (4) x_pshdr <= z_shd
    # ================================================================
    for (p, s, h, d, r) in x:
        z_val = 1.0 if (s, h, d) in z else 0.0
        if 1.0 - z_val > tol:
            add_violation(
                4,
                f"Patient {p}/surgeon {s} at hospital {h} day {d} room {r} "
                f"but surgeon not present",
                1.0, z_val, 1.0 - z_val,
            )

    # ================================================================
    # (5) x_pshdr <= y_hdr
    # ================================================================
    for (p, s, h, d, r) in x:
        y_val = 1.0 if (h, d, r) in y else 0.0
        if 1.0 - y_val > tol:
            add_violation(
                5,
                f"Patient {p} at hospital {h} day {d} room {r} but OR closed",
                1.0, y_val, 1.0 - y_val,
            )

    # ================================================================
    # (6) OR capacity: sum T*x <= B*y + v
    # ================================================================
    or_load = defaultdict(float)
    for (p, s, h, d, r) in x:
        or_load[(h, d, r)] += get_T(patients, p, s)

    for (h, d, r), total_load in or_load.items():
        B_hdr = get_B(or_lookup, h, d, r)
        y_val = 1.0 if (h, d, r) in y else 0.0
        v_val = v.get((h, d, r), 0.0)
        rhs_val = B_hdr * y_val + v_val
        if total_load - rhs_val > tol:
            add_violation(
                6,
                f"OR cap exceeded h={h} d={d} r={r}: "
                f"load={total_load:.2f} > B*y+v={rhs_val:.2f}",
                total_load, rhs_val, total_load - rhs_val,
            )

    # ================================================================
    # (7) Surgeon weighted-time availability
    # ================================================================
    surgeon_load = defaultdict(float)
    for (p, s, h, d, r) in x:
        E_ps = get_E(patients, p, s)
        T_ps = get_T(patients, p, s)
        surgeon_load[(s, h, d)] += alpha * E_ps + (1.0 - alpha) * T_ps

    for (s, h, d), weighted in surgeon_load.items():
        A_sd = surgeons[s]["availability_by_day"].get(str(d), 0)
        z_val = 1.0 if (s, h, d) in z else 0.0
        rhs_val = float(A_sd) * z_val
        if weighted - rhs_val > tol:
            add_violation(
                7,
                f"Surgeon {s} availability exceeded h={h} d={d}: "
                f"load={weighted:.2f} > A*z={rhs_val:.2f}",
                weighted, rhs_val, weighted - rhs_val,
            )

    # ================================================================
    # (8) Overtime bounds: 0 <= v <= V_hdr
    # ================================================================
    for (h, d, r), v_val in v.items():
        V_hdr = get_V_max(or_lookup, h, d, r)
        if v_val < -tol:
            add_violation(8, f"Negative overtime h={h} d={d} r={r}: v={v_val}",
                          v_val, 0.0, abs(v_val))
        if v_val - V_hdr > tol:
            add_violation(8, f"Overtime > V_max h={h} d={d} r={r}: "
                          f"v={v_val:.2f} > {V_hdr}",
                          v_val, float(V_hdr), v_val - V_hdr)

    # ================================================================
    # SEQUENCING constraints (9)-(22)
    # ----------------------------------------------------------------
    patients_at_hdr = defaultdict(list)
    patients_at_hds = defaultdict(list)
    surgeon_for_pat_at_hd = {}   # (p,h,d) -> s
    or_for_pat_at_hd = {}        # (p,h,d) -> r
    for (p, s, h, d, r) in x:
        patients_at_hdr[(h, d, r)].append(p)
        patients_at_hds[(h, d, s)].append(p)
        surgeon_for_pat_at_hd[(p, h, d)] = s
        or_for_pat_at_hd[(p, h, d)] = r

    # ================================================================
    # (9) Each scheduled patient has exactly one (s,r) at (h,d)
    # ================================================================
    pat_phd = defaultdict(list)
    for (p, s, h, d, r) in x:
        pat_phd[(p, h, d)].append((s, r))
    for (p, h, d), srs in pat_phd.items():
        if len(srs) != 1:
            add_violation(
                9,
                f"Patient {p} at h={h} d={d} has {len(srs)} (s,r) pairs",
                float(len(srs)), 1.0, abs(len(srs) - 1.0),
            )

    # ================================================================
    # (10) f_p >= F_p + E_ps
    # ================================================================
    for (p, s, h, d, r), _ in x.items():
        if (p, h, d) not in finish_times:
            add_violation(
                10,
                f"Missing finish_time for patient {p} at h={h} d={d}",
                0.0, 1.0, 1.0,
            )
            continue
        fp = finish_times[(p, h, d)]
        Fp = get_F(patients, p)
        Eps = get_E(patients, p, s)
        if fp + tol < Fp + Eps:
            add_violation(
                10,
                f"f_p < F_p + E_ps for p={p}: f={fp:.2f} < {Fp + Eps:.2f}",
                fp, Fp + Eps, (Fp + Eps) - fp,
            )

    # ================================================================
    # (11)/(12) Per-OR no-overlap
    # ================================================================
    for (h, d, r), plist in patients_at_hdr.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                if (p, h, d) not in finish_times or (k, h, d) not in finish_times:
                    continue
                fp = finish_times[(p, h, d)]
                fk = finish_times[(k, h, d)]
                sp_ = surgeon_for_pat_at_hd[(p, h, d)]
                sk_ = surgeon_for_pat_at_hd[(k, h, d)]
                Fp, Fk = get_F(patients, p), get_F(patients, k)
                Gp, Gk = get_G(patients, p), get_G(patients, k)
                Eps = get_E(patients, p, sp_)
                Eks = get_E(patients, k, sk_)
                p_after_k_ok = (fp + tol >= fk + Gk + Fp + Eps)
                k_after_p_ok = (fk + tol >= fp + Gp + Fk + Eks)
                if not (p_after_k_ok or k_after_p_ok):
                    short_p = (fk + Gk + Fp + Eps) - fp
                    short_k = (fp + Gp + Fk + Eks) - fk
                    add_violation(
                        11,
                        f"OR overlap p={p}/k={k} at (h={h},d={d},r={r}): "
                        f"neither order feasible (p-after-k short {short_p:.2f}, "
                        f"k-after-p short {short_k:.2f})",
                        max(fp, fk),
                        max(fk + Gk + Fp + Eps, fp + Gp + Fk + Eks),
                        min(short_p, short_k),
                    )

    # ================================================================
    # (13)/(14) Per-surgeon no-overlap
    # ================================================================
    for (h, d, s), plist in patients_at_hds.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                p, k = plist[i], plist[j]
                if (p, h, d) not in finish_times or (k, h, d) not in finish_times:
                    continue
                fp = finish_times[(p, h, d)]
                fk = finish_times[(k, h, d)]
                Eps = get_E(patients, p, s)
                Eks = get_E(patients, k, s)
                p_after_k_ok = (fp + tol >= fk + Eps)
                k_after_p_ok = (fk + tol >= fp + Eks)
                if not (p_after_k_ok or k_after_p_ok):
                    short_p = (fk + Eps) - fp
                    short_k = (fp + Eks) - fk
                    add_violation(
                        13,
                        f"Surgeon overlap p={p}/k={k} surgeon={s} at "
                        f"(h={h},d={d}): neither order feasible "
                        f"(p-after-k short {short_p:.2f}, k-after-p short "
                        f"{short_k:.2f})",
                        max(fp, fk), max(fk + Eps, fp + Eks),
                        min(short_p, short_k),
                    )

    # ================================================================
    # (15) f_p + G_p <= B_r + v_r
    # ================================================================
    for (p, s, h, d, r), _ in x.items():
        if (p, h, d) not in finish_times:
            continue
        fp = finish_times[(p, h, d)]
        Gp = get_G(patients, p)
        Br = get_B(or_lookup, h, d, r)
        vr = v.get((h, d, r), 0.0)
        if fp + Gp - (Br + vr) > tol:
            add_violation(
                15,
                f"f_p+G_p > B_r+v_r for p={p}, r={r}, h={h}, d={d}: "
                f"{fp + Gp:.2f} > {Br + vr:.2f}",
                fp + Gp, Br + vr, (fp + Gp) - (Br + vr),
            )

    # ================================================================
    # (16) e_s >= f_p   /   (17) i_s <= f_p - E_ps
    # ================================================================
    for (p, s, h, d, r), _ in x.items():
        if (p, h, d) not in finish_times:
            continue
        fp = finish_times[(p, h, d)]
        Eps = get_E(patients, p, s)
        es = surg_end_d.get((s, h, d))
        if es is not None and es + tol < fp:
            add_violation(
                16,
                f"e_s < f_p for s={s},p={p},h={h},d={d}: e_s={es:.2f} < "
                f"f_p={fp:.2f}",
                es, fp, fp - es,
            )
        is_ = surg_start_d.get((s, h, d))
        if is_ is not None and is_ - (fp - Eps) > tol:
            add_violation(
                17,
                f"i_s > f_p - E_ps for s={s},p={p},h={h},d={d}: "
                f"i_s={is_:.2f} > {fp - Eps:.2f}",
                is_, fp - Eps, is_ - (fp - Eps),
            )

    # ================================================================
    # (18) e_s - i_s <= A_s
    # ================================================================
    for (s, h, d), is_ in surg_start_d.items():
        es = surg_end_d.get((s, h, d))
        if es is None:
            continue
        A_sd = surgeons[s]["availability_by_day"].get(str(d), 0)
        if (es - is_) - A_sd > tol:
            add_violation(
                18,
                f"surgeon-window > A_sd for s={s},h={h},d={d}: "
                f"e-i={es - is_:.2f} > A={A_sd}",
                es - is_, float(A_sd), (es - is_) - A_sd,
            )

    # ================================================================
    # (19) c_r >= f_p + G_p
    # ================================================================
    for (p, s, h, d, r), _ in x.items():
        if (p, h, d) not in finish_times:
            continue
        fp = finish_times[(p, h, d)]
        Gp = get_G(patients, p)
        cr = completion.get((h, d, r))
        if cr is None:
            add_violation(
                19,
                f"Missing completion_time for OR (h={h},d={d},r={r})",
                0.0, fp + Gp, fp + Gp,
            )
            continue
        if cr + tol < fp + Gp:
            add_violation(
                19,
                f"c_r < f_p+G_p for p={p}, OR (h={h},d={d},r={r}): "
                f"c={cr:.2f} < {fp + Gp:.2f}",
                cr, fp + Gp, (fp + Gp) - cr,
            )

    # ================================================================
    # (20) 0 <= v_r <= V_r   (covered by (8))
    # ================================================================

    # ================================================================
    # (21) v_r >= c_r - B_r
    # ================================================================
    for (h, d, r), cr in completion.items():
        Br = get_B(or_lookup, h, d, r)
        vr = v.get((h, d, r), 0.0)
        if (cr - Br) - vr > tol:
            add_violation(
                21,
                f"v_r < c_r - B_r for (h={h},d={d},r={r}): "
                f"v={vr:.2f} < c-B={cr - Br:.2f}",
                vr, cr - Br, (cr - Br) - vr,
            )

    # ================================================================
    # (22) Integrality of x  (handled by x being read as 0/1 from solution)
    # ================================================================

    # ================================================================
    # (23) Objective consistency (Tier C defense against score-gaming)
    # ----------------------------------------------------------------
    # Recompute the MP objective directly from the solution variables
    # (y, z, v, x for optional patients) and reject when the reported
    # objective_value disagrees beyond tolerance. This catches LLM
    # exploits that return fabricated obj values (e.g. 0 or float-max)
    # while submitting otherwise-feasible routes/assignments.
    #
    # Formula (Roshanaei et al. 2017, MP):
    #   obj = sum_{h,d,r} K_{hdr} y_{hdr}
    #       + sum_{s,h,d} L_{shd} z_{shd}
    #       + sum_{h,d,r} C_{hdr} v_{hdr}
    #       - sum_{p optional} U_p * (sum x_pshdr)
    # ================================================================
    reported_obj_raw = raw_sol.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None:
            opened_fixed_cost = sum(
                get_K(or_lookup, h, d, r) for (h, d, r) in y
            )
            surgeon_fixed_cost = sum(
                get_L(hospitals, s, h, d) for (s, h, d) in z
            )
            overtime_cost = sum(
                get_C(or_lookup, h, d, r) * v_val
                for (h, d, r), v_val in v.items()
            )
            optional_reward = 0.0
            for (p, s, h, d, r) in x:
                pat = patients.get(p)
                if pat is None:
                    continue
                if pat["due_date"] > num_days:
                    optional_reward += float(pat.get("reward", 0.0))

            true_obj = (opened_fixed_cost + surgeon_fixed_cost
                        + overtime_cost - optional_reward)
            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative tolerance with 1e-2 absolute floor (costs are
            # small floats per-minute, summed across many ORs).
            obj_tol = max(1e-2, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                add_violation(
                    23,
                    f"Objective consistency violated: reported "
                    f"objective_value={reported_obj} differs from recomputed "
                    f"MP objective sum(K*y)+sum(L*z)+sum(C*v)-sum(U_p*x)="
                    f"{true_obj} (K*y={opened_fixed_cost:.4f}, "
                    f"L*z={surgeon_fixed_cost:.4f}, "
                    f"C*v={overtime_cost:.4f}, "
                    f"-U*x=-{optional_reward:.4f}; |diff|={abs_diff:.3g}, "
                    f"tol={obj_tol:.3g})",
                    reported_obj, true_obj, abs_diff,
                )

    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CORPS (Roshanaei et al. 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    result = check_feasibility(args.instance_path, args.solution_path,
                               args.result_path)

    if result["feasible"]:
        print("FEASIBLE: All constraints (1)-(23) satisfied.")
    else:
        print(f"INFEASIBLE: violated constraints {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
