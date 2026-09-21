#!/usr/bin/env python3
"""
Feasibility checker for Congestion-Aware System Optimal SAV Routing LP.

Based on: Levin (2017) "Congestion-aware system optimal route choice for shared
autonomous vehicles", Transportation Research Part C, 82, 229-247.

Checks all 22 hard constraints (Eqs. 34-55) from the combined LP formulation,
plus an objective-consistency check (constraint 23) that recomputes TSTT
(Eq. 30/33) from the solution's N_U/N_D/omega variables and rejects reported
objective values that disagree.

Expected solution JSON format:
{
  "objective_value": float,
  "y": {"i|j|k|s|t": float},         // turning flow y_{ijk}^s(t)
  "y_centroid": {"i|j|s|t": float},   // centroid departure flow y_{ij}^s(t)
  "N_U": {"i|j|s|t": float},          // upstream cumulative count
  "N_D": {"i|j|s|t": float},          // downstream cumulative count
  "p": {"j|t": float},                // parking at centroid j at time t
  "e": {"r|s|t": float},              // departing travelers
  "omega": {"r|s|t": float}           // waiting demand
}
"""

import argparse
import json
from collections import defaultdict

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def gv(d, key):
    """Get value from dict, default 0.0."""
    return d.get(key, 0.0)


def compute_violation(lhs, rhs, op):
    """Compute violation_amount for a constraint."""
    if op == "eq":
        return abs(lhs - rhs)
    elif op == "leq":
        return max(lhs - rhs, 0.0)
    elif op == "geq":
        return max(rhs - lhs, 0.0)
    return 0.0


def record_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs, op):
    """Check and record a violation if it exceeds tolerance."""
    viol_amt = compute_violation(lhs, rhs, op)
    if viol_amt > TOL:
        violations_list.append((constraint_idx, msg))
        normalizer = max(abs(rhs), EPS)
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(viol_amt),
            "normalizer": float(normalizer),
            "ratio": float(viol_amt / normalizer),
        })


# --- Key construction helpers ---

def yk(i, j, k, s, t):
    """Key for turning flow y_{ijk}^s(t)."""
    return f"{i}|{j}|{k}|{s}|{t}"


def yck(i, j, s, t):
    """Key for centroid departure flow y_{ij}^s(t)."""
    return f"{i}|{j}|{s}|{t}"


def nk(i, j, s, t):
    """Key for cumulative counts N_{ij}^{Us}(t) or N_{ij}^{Ds}(t)."""
    return f"{i}|{j}|{s}|{t}"


def pk(j, t):
    """Key for parking p_j(t)."""
    return f"{j}|{t}"


def ek(r, s, t):
    """Key for e_r^s(t) or omega_r^s(t)."""
    return f"{r}|{s}|{t}"


def parse_instance(instance):
    """Extract network structure and parameters from instance JSON."""
    net = instance["network"]
    T = instance["time_parameters"]["time_horizon_T"]

    centroids = set()
    junctions = set()
    for node in net["nodes"]:
        if node["type"] == "centroid":
            centroids.add(node["id"])
        else:
            junctions.add(node["id"])

    links = {}
    A_o = []
    A_z_plus = []
    A_z_minus = []

    for link in net["links"]:
        i, j = link["from"], link["to"]
        links[(i, j)] = link
        if link["type"] == "centroid_connector":
            if i in centroids:
                A_z_plus.append((i, j))
            if j in centroids:
                A_z_minus.append((i, j))
        else:
            A_o.append((i, j))

    all_links = list(links.keys())

    # Gamma_j^+ = outgoing links from j; Gamma_j^- = incoming links to j
    gamma_plus = defaultdict(list)
    gamma_minus = defaultdict(list)
    for (i, j) in all_links:
        gamma_plus[i].append((i, j))
        gamma_minus[j].append((i, j))

    # Demand d[r][s][t]
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for od in instance["demand"]["od_pairs"]:
        r, s = od["origin"], od["destination"]
        for t_dep in od["departure_times"]:
            d[r][s][t_dep] += 1.0

    # Initial parking
    p0 = {}
    for c, count in instance["fleet"]["initial_distribution"].items():
        p0[c] = float(count)

    return {
        "T": T,
        "centroids": centroids,
        "links": links,
        "all_links": all_links,
        "A_o": A_o,
        "A_z_plus": A_z_plus,
        "A_z_minus": A_z_minus,
        "gamma_plus": gamma_plus,
        "gamma_minus": gamma_minus,
        "d": d,
        "p0": p0,
    }


def check_feasibility(instance, solution):
    # If the solver reported infeasible/unbounded (no solution found), return null
    status = solution.get("status")
    if status in ("INF_OR_UNBD", "INFEASIBLE", "UNBOUNDED"):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": [f"No solution to check: solver status is {status}"],
            "violation_magnitudes": [],
        }
    obj = solution.get("objective_value")
    if obj is None and not solution.get("y"):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution to check: objective_value is null and no variable data present"],
            "violation_magnitudes": [],
        }

    inst = parse_instance(instance)

    # Solution variables
    y = solution.get("y", {})
    yc = solution.get("y_centroid", {})
    N_U = solution.get("N_U", {})
    N_D = solution.get("N_D", {})
    p_sol = solution.get("p", {})
    e_sol = solution.get("e", {})
    omega_sol = solution.get("omega", {})

    T = inst["T"]
    centroids = inst["centroids"]
    links_data = inst["links"]
    A_o = inst["A_o"]
    A_z_plus = inst["A_z_plus"]
    A_z_minus = inst["A_z_minus"]
    gamma_plus = inst["gamma_plus"]
    gamma_minus = inst["gamma_minus"]
    d = inst["d"]

    A_o_Az_plus = A_o + A_z_plus
    A_o_Az_minus = A_o + A_z_minus

    def fftt(i, j):
        return links_data[(i, j)]["free_flow_travel_time_steps"]

    def cwtt(i, j):
        return links_data[(i, j)]["congested_travel_time_steps"]

    dt_sec = instance["time_parameters"]["time_step_duration_seconds"]

    def cap(i, j):
        return links_data[(i, j)]["capacity_vph"] * (dt_sec / 3600.0)

    def jam_veh(i, j):
        return links_data[(i, j)]["jam_density_vehicles"]

    violations_list = []
    magnitudes_list = []

    od_pairs = [(r, s) for r in centroids for s in centroids]

    # =========================================================================
    # Constraint 1 (Eq. 34): Upstream cumulative count evolution
    # N_{ij}^{Us}(t+1) = N_{ij}^{Us}(t) + sum_{(j,k) in Gamma_j^+} y_{ijk}^s(t)
    # for (i,j) in A_o, s in Z, t in [0, T-1]
    # (A_z^+ links are handled by constraint 13/Eq.46)
    # =========================================================================
    for (i, j) in A_o:
        for s in centroids:
            for t in range(T):
                lhs = gv(N_U, nk(i, j, s, t + 1))
                flow_sum = sum(gv(y, yk(i, j, k, s, t))
                               for (_, k) in gamma_plus.get(j, []))
                rhs = gv(N_U, nk(i, j, s, t)) + flow_sum
                record_violation(violations_list, magnitudes_list, 1,
                    f"Constraint 1 (Eq.34) violated: link ({i},{j}), dest {s}, t={t}: "
                    f"N_U(t+1)={lhs:.6g} != {rhs:.6g}",
                    lhs, rhs, "eq")

    # =========================================================================
    # Constraint 2 (Eq. 35): Downstream cumulative count evolution
    # N_{jk}^{Ds}(t+1) = N_{jk}^{Ds}(t) + sum_{(i,j) in Gamma_j^-} y_{ijk}^s(t)
    # for (j,k) in A_o, s in Z, t in [0, T-1]
    # (A_z^- links are handled by constraint 14/Eq.47)
    # =========================================================================
    for (j, k) in A_o:
        for s in centroids:
            for t in range(T):
                lhs = gv(N_D, nk(j, k, s, t + 1))
                flow_sum = sum(gv(y, yk(i, j, k, s, t))
                               for (i, _) in gamma_minus.get(j, []))
                rhs = gv(N_D, nk(j, k, s, t)) + flow_sum
                record_violation(violations_list, magnitudes_list, 2,
                    f"Constraint 2 (Eq.35) violated: link ({j},{k}), dest {s}, t={t}: "
                    f"N_D(t+1)={lhs:.6g} != {rhs:.6g}",
                    lhs, rhs, "eq")

    # =========================================================================
    # Constraint 3 (Eq. 36): Sending flow constraint
    # sum_{(j,k)} y_{ijk}^s(t) <= N_{ij}^{Us}(t - L/v + 1) - N_{ij}^{Ds}(t)
    # for (i,j) in A_o, s in Z, t in [L/v - 1, T]
    # (A_z^+ centroid departure bounded by constraint 12/Eq.45)
    # =========================================================================
    for (i, j) in A_o:
        lv = fftt(i, j)
        for s in centroids:
            for t in range(lv - 1, T + 1):
                lhs = sum(gv(y, yk(i, j, k, s, t))
                          for (_, k) in gamma_plus.get(j, []))
                t_shift = t - lv + 1
                rhs = gv(N_U, nk(i, j, s, t_shift)) - gv(N_D, nk(i, j, s, t))
                record_violation(violations_list, magnitudes_list, 3,
                    f"Constraint 3 (Eq.36) violated: link ({i},{j}), dest {s}, t={t}: "
                    f"sum_y={lhs:.6g} > {rhs:.6g}",
                    lhs, rhs, "leq")

    # =========================================================================
    # Constraint 4 (Eq. 37): Zero flow before free-flow travel time elapses
    # y_{ijk}^s(t) = 0 for t in [0, L/v - 1)
    # for (i,j) in A_o ∪ A_z^+, (j,k) in Gamma_j^+, s in Z
    # =========================================================================
    for (i, j) in A_o_Az_plus:
        lv = fftt(i, j)
        if lv <= 1:
            continue
        for (_, k) in gamma_plus.get(j, []):
            for s in centroids:
                for t in range(lv - 1):
                    val = gv(y, yk(i, j, k, s, t))
                    record_violation(violations_list, magnitudes_list, 4,
                        f"Constraint 4 (Eq.37) violated: y({i},{j},{k},{s},{t})={val:.6g} != 0",
                        val, 0.0, "eq")

    # =========================================================================
    # Constraint 5 (Eq. 38): Sending flow capacity
    # sum_s sum_k y_{ijk}^s(t) <= Q_{ij}
    # for (i,j) in A_o, t in [0, T]
    # =========================================================================
    for (i, j) in A_o:
        Q = cap(i, j)
        for t in range(T + 1):
            lhs = sum(gv(y, yk(i, j, k, s, t))
                      for s in centroids
                      for (_, k) in gamma_plus.get(j, []))
            record_violation(violations_list, magnitudes_list, 5,
                f"Constraint 5 (Eq.38) violated: link ({i},{j}), t={t}: "
                f"sum_y={lhs:.6g} > Q={Q}",
                lhs, Q, "leq")

    # =========================================================================
    # Constraint 6 (Eq. 39): Receiving flow capacity
    # sum_i sum_s y_{ijk}^s(t) <= Q_{jk}
    # for (j,k) in A_o, t in [0, T]
    # =========================================================================
    for (j, k) in A_o:
        Q = cap(j, k)
        for t in range(T + 1):
            lhs = sum(gv(y, yk(i, j, k, s, t))
                      for (i, _) in gamma_minus.get(j, [])
                      for s in centroids)
            record_violation(violations_list, magnitudes_list, 6,
                f"Constraint 6 (Eq.39) violated: link ({j},{k}), t={t}: "
                f"sum_y={lhs:.6g} > Q={Q}",
                lhs, Q, "leq")

    # =========================================================================
    # Constraint 7 (Eq. 40): Receiving flow congested wave constraint
    # sum_i sum_s y_{ijk}^s(t) <= sum_s(N_{jk}^{Us}(t-L/w+1) - N_{jk}^{Ds}(t)) + KL
    # for (j,k) in A_o, t in [L/w - 1, T]
    # =========================================================================
    for (j, k) in A_o:
        lw = cwtt(j, k)
        KL = jam_veh(j, k)
        for t in range(lw - 1, T + 1):
            lhs = sum(gv(y, yk(i, j, k, s, t))
                      for (i, _) in gamma_minus.get(j, [])
                      for s in centroids)
            t_shift = t - lw + 1
            rhs_flow = sum(gv(N_U, nk(j, k, s, t_shift)) - gv(N_D, nk(j, k, s, t))
                           for s in centroids)
            rhs = rhs_flow + KL
            record_violation(violations_list, magnitudes_list, 7,
                f"Constraint 7 (Eq.40) violated: link ({j},{k}), t={t}: "
                f"sum_y={lhs:.6g} > {rhs:.6g}",
                lhs, rhs, "leq")

    # =========================================================================
    # Constraint 8 (Eq. 41): Initial upstream counts zero
    # N_{ij}^{Us}(0) = 0 for all (i,j) in A, s in Z
    # =========================================================================
    for (i, j) in inst["all_links"]:
        for s in centroids:
            val = gv(N_U, nk(i, j, s, 0))
            record_violation(violations_list, magnitudes_list, 8,
                f"Constraint 8 (Eq.41) violated: N_U({i},{j},{s},0)={val:.6g} != 0",
                val, 0.0, "eq")

    # =========================================================================
    # Constraint 9 (Eq. 42): Initial downstream counts zero
    # N_{ij}^{Ds}(0) = 0 for all (i,j) in A, s in Z
    # =========================================================================
    for (i, j) in inst["all_links"]:
        for s in centroids:
            val = gv(N_D, nk(i, j, s, 0))
            record_violation(violations_list, magnitudes_list, 9,
                f"Constraint 9 (Eq.42) violated: N_D({i},{j},{s},0)={val:.6g} != 0",
                val, 0.0, "eq")

    # =========================================================================
    # Constraint 10 (Eq. 43): Parking evolution at centroids
    # p_j(t+1) = p_j(t) + sum_{(i,j) in Gamma_j^-}(N_{ij}^{Uj}(t) - N_{ij}^{Dj}(t))
    #            - sum_{(j,k) in Gamma_j^+} sum_s y_{jk}^s(t)
    # for j in Z, t in [0, T-1]
    # Note: superscript j on N means only vehicles destined for centroid j.
    #       y_{jk}^s is the centroid departure flow (from y_centroid dict).
    # =========================================================================
    for j in centroids:
        for t in range(T):
            p_next = gv(p_sol, pk(j, t + 1))
            p_curr = gv(p_sol, pk(j, t))

            # Arriving: vehicles on incoming links destined for j (occupancy)
            arriving = sum(gv(N_U, nk(i, j, j, t)) - gv(N_D, nk(i, j, j, t))
                           for (i, _) in gamma_minus.get(j, []))

            # Departing: centroid departure flow from j to all destinations
            departing = sum(gv(yc, yck(j, k, s, t))
                            for (_, k) in gamma_plus.get(j, [])
                            for s in centroids)

            rhs = p_curr + arriving - departing
            record_violation(violations_list, magnitudes_list, 10,
                f"Constraint 10 (Eq.43) violated: centroid {j}, t={t}: "
                f"p(t+1)={p_next:.6g} != {rhs:.6g}",
                p_next, rhs, "eq")

    # =========================================================================
    # Constraint 11 (Eq. 44): No through-flow on centroid connectors to centroid
    # y_{ijk}^s(t) = 0 for (j,k) in A_z^-, (i,j) in Gamma_j^-, s != k, t in [0,T]
    # =========================================================================
    for (j, k) in A_z_minus:
        for (i, _) in gamma_minus.get(j, []):
            for s in centroids:
                if s != k:
                    for t in range(T + 1):
                        val = gv(y, yk(i, j, k, s, t))
                        if abs(val) > TOL:
                            record_violation(violations_list, magnitudes_list, 11,
                                f"Constraint 11 (Eq.44) violated: "
                                f"y({i},{j},{k},{s},{t})={val:.6g} but s={s} != k={k}",
                                val, 0.0, "eq")

    # =========================================================================
    # Constraint 12 (Eq. 45): Outgoing flow bounded by parked vehicles
    # sum_{(i,j) in Gamma_i^+} sum_s y_{ij}^s(t) <= p_i(t)
    # for i in Z, t in [0, T]
    # =========================================================================
    for i in centroids:
        for t in range(T + 1):
            lhs = sum(gv(yc, yck(i, j, s, t))
                      for (_, j) in gamma_plus.get(i, [])
                      for s in centroids)
            rhs = gv(p_sol, pk(i, t))
            record_violation(violations_list, magnitudes_list, 12,
                f"Constraint 12 (Eq.45) violated: centroid {i}, t={t}: "
                f"sum_y={lhs:.6g} > p={rhs:.6g}",
                lhs, rhs, "leq")

    # =========================================================================
    # Constraint 13 (Eq. 46): Upstream count on outgoing centroid connectors
    # N_{ij}^{Us}(t+1) = N_{ij}^{Us}(t) + y_{ij}^s(t)
    # for (i,j) in A_z^+, s in Z, t in [0, T-1]
    # =========================================================================
    for (i, j) in A_z_plus:
        for s in centroids:
            for t in range(T):
                lhs = gv(N_U, nk(i, j, s, t + 1))
                rhs = gv(N_U, nk(i, j, s, t)) + gv(yc, yck(i, j, s, t))
                record_violation(violations_list, magnitudes_list, 13,
                    f"Constraint 13 (Eq.46) violated: link ({i},{j}), dest {s}, t={t}: "
                    f"N_U(t+1)={lhs:.6g} != {rhs:.6g}",
                    lhs, rhs, "eq")

    # =========================================================================
    # Constraint 14 (Eq. 47): Downstream count on incoming centroid connectors
    # N_{ij}^{Ds}(t+1) = N_{ij}^{Us}(t)
    # for (i,j) in A_z^-, s in Z, t in [0, T-1]
    # =========================================================================
    for (i, j) in A_z_minus:
        for s in centroids:
            for t in range(T):
                lhs = gv(N_D, nk(i, j, s, t + 1))
                rhs = gv(N_U, nk(i, j, s, t))
                record_violation(violations_list, magnitudes_list, 14,
                    f"Constraint 14 (Eq.47) violated: link ({i},{j}), dest {s}, t={t}: "
                    f"N_D(t+1)={lhs:.6g} != N_U(t)={rhs:.6g}",
                    lhs, rhs, "eq")

    # =========================================================================
    # Constraint 15 (Eq. 48): Fleet conservation
    # sum_{i in Z} p_i(0) = sum_{i in Z} p_i(T)
    # =========================================================================
    sum_p0 = sum(gv(p_sol, pk(j, 0)) for j in centroids)
    sum_pT = sum(gv(p_sol, pk(j, T)) for j in centroids)
    record_violation(violations_list, magnitudes_list, 15,
        f"Constraint 15 (Eq.48) violated: sum p(0)={sum_p0:.6g} != sum p(T)={sum_pT:.6g}",
        sum_p0, sum_pT, "eq")

    # =========================================================================
    # Constraint 16 (Eq. 49): Departing travelers bounded by waiting demand
    # e_r^s(t) <= omega_r^s(t)
    # for (r,s) in Z^2, t in [0, T]
    # =========================================================================
    for (r, s) in od_pairs:
        for t in range(T + 1):
            e_val = gv(e_sol, ek(r, s, t))
            omega_val = gv(omega_sol, ek(r, s, t))
            record_violation(violations_list, magnitudes_list, 16,
                f"Constraint 16 (Eq.49) violated: ({r},{s}), t={t}: "
                f"e={e_val:.6g} > omega={omega_val:.6g}",
                e_val, omega_val, "leq")

    # =========================================================================
    # Constraint 17 (Eq. 50): Departing travelers bounded by departing vehicles
    # e_r^s(t) <= sum_{(r,j) in Gamma_r^+} y_{rj}^s(t)
    # for (r,s) in Z^2, t in [0, T]
    # =========================================================================
    for (r, s) in od_pairs:
        for t in range(T + 1):
            e_val = gv(e_sol, ek(r, s, t))
            dep_flow = sum(gv(yc, yck(r, j, s, t))
                           for (_, j) in gamma_plus.get(r, []))
            record_violation(violations_list, magnitudes_list, 17,
                f"Constraint 17 (Eq.50) violated: ({r},{s}), t={t}: "
                f"e={e_val:.6g} > sum_y={dep_flow:.6g}",
                e_val, dep_flow, "leq")

    # =========================================================================
    # Constraint 18 (Eq. 51): Waiting demand evolution
    # omega_r^s(t+1) = omega_r^s(t) + d_r^s(t) - e_r^s(t)
    # for (r,s) in Z^2, t in [0, T-1]
    # =========================================================================
    for (r, s) in od_pairs:
        for t in range(T):
            lhs = gv(omega_sol, ek(r, s, t + 1))
            rhs = gv(omega_sol, ek(r, s, t)) + d[r][s][t] - gv(e_sol, ek(r, s, t))
            record_violation(violations_list, magnitudes_list, 18,
                f"Constraint 18 (Eq.51) violated: ({r},{s}), t={t}: "
                f"omega(t+1)={lhs:.6g} != {rhs:.6g}",
                lhs, rhs, "eq")

    # =========================================================================
    # Constraint 19 (Eq. 52): All demand served by end of horizon
    # omega_r^s(T) = 0 for all (r,s) in Z^2
    # =========================================================================
    for (r, s) in od_pairs:
        val = gv(omega_sol, ek(r, s, T))
        record_violation(violations_list, magnitudes_list, 19,
            f"Constraint 19 (Eq.52) violated: omega({r},{s},T)={val:.6g} != 0",
            val, 0.0, "eq")

    # =========================================================================
    # Constraint 20 (Eq. 53): Non-negativity of turning flows
    # y_{ijk}^s(t) >= 0
    # for (i,j) in A_o ∪ A_z^+, (j,k) in Gamma_j^+, s in Z, t in [0,T]
    # =========================================================================
    for (i, j) in A_o_Az_plus:
        for (_, k) in gamma_plus.get(j, []):
            for s in centroids:
                for t in range(T + 1):
                    val = gv(y, yk(i, j, k, s, t))
                    if val < -TOL:
                        record_violation(violations_list, magnitudes_list, 20,
                            f"Constraint 20 (Eq.53) violated: "
                            f"y({i},{j},{k},{s},{t})={val:.6g} < 0",
                            val, 0.0, "geq")

    # =========================================================================
    # Constraint 21 (Eq. 54): Non-negativity of centroid departure flow
    # y_{ij}^s(t) >= 0
    # for (i,j) in A_z^+, s in Z, t in [0, T]
    # =========================================================================
    for (i, j) in A_z_plus:
        for s in centroids:
            for t in range(T + 1):
                val = gv(yc, yck(i, j, s, t))
                if val < -TOL:
                    record_violation(violations_list, magnitudes_list, 21,
                        f"Constraint 21 (Eq.54) violated: "
                        f"y_c({i},{j},{s},{t})={val:.6g} < 0",
                        val, 0.0, "geq")

    # =========================================================================
    # Constraint 22 (Eq. 55): Non-negativity of departing travelers
    # e_r^s(t) >= 0 for (r,s) in Z^2, t in [0, T]
    # =========================================================================
    for (r, s) in od_pairs:
        for t in range(T + 1):
            val = gv(e_sol, ek(r, s, t))
            if val < -TOL:
                record_violation(violations_list, magnitudes_list, 22,
                    f"Constraint 22 (Eq.55) violated: "
                    f"e({r},{s},{t})={val:.6g} < 0",
                    val, 0.0, "geq")

    _domain_check_vars_binary = []
    _domain_check_vars_integer = []

    # =====================================================================
    # Non-negativity checks for levin2017
    # All continuous flow/count variables must be >= 0
    for field in ("y", "y_centroid", "N_U", "N_D", "p", "e", "omega"):
        var_dict = solution.get(field, {})
        if not isinstance(var_dict, dict):
            continue
        for key, val in var_dict.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v < -TOL:
                violations_list.append((3, f"{field}[{key}] = {v} < 0 (non-negativity)"))
                magnitudes_list.append({
                    "constraint": 3, "lhs": v, "rhs": 0.0,
                    "raw_excess": -v, "normalizer": max(abs(v), EPS),
                    "ratio": -v / max(abs(v), EPS),
                })

    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # Adapted: writes (idx, msg) tuples to violations_list so aggregation
    # below picks them up. Constraint indices reused: 1 (binary), 2 (integer).
    # =====================================================================
    # Constraint 1: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > TOL or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > TOL:
                        violations_list.append((1,
                            f"Constraint 1 (binary domain): {var_name}[{key}] = {v} not in {0, 1}"))
                        magnitudes_list.append({
                            "constraint": 1,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 2: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > TOL:
                    violations_list.append((2,
                        f"Constraint 2 (integer domain): {var_name}[{key}] = {v} is not integer"))
                    magnitudes_list.append({
                        "constraint": 2,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), EPS),
                        "ratio": float(frac / max(abs(round(v)), EPS)),
                    })

    # =========================================================================
    # Constraint 23 (Eq. 30/33): Objective consistency (Tier C defense)
    # Recompute TSTT from solution variables and compare to reported obj.
    # Z = sum_{(i,j) in A} sum_s sum_{t=0..T} (N_U(t) - N_D(t))
    #   + sum_{(r,s) in Z^2} sum_{t=0..T} omega(t)
    # All variables required by this formula (N_U, N_D, omega) are written
    # to the solution by every program, so a full recompute is exact.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            link_term = 0.0
            for (i, j) in inst["all_links"]:
                for s in centroids:
                    for t in range(T + 1):
                        link_term += (gv(N_U, nk(i, j, s, t))
                                      - gv(N_D, nk(i, j, s, t)))
            wait_term = 0.0
            for (r, s) in od_pairs:
                for t in range(T + 1):
                    wait_term += gv(omega_sol, ek(r, s, t))
            true_obj = float(link_term + wait_term)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                record_violation(violations_list, magnitudes_list, 23,
                    f"Constraint 23 (Eq.30/33) violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"TSTT={true_obj} (link_term={link_term:.6g}, "
                    f"wait_term={wait_term:.6g}, |diff|={abs_diff:.3g}, "
                    f"tol={tol:.3g})",
                    reported, true_obj, "eq")

    # =========================================================================
    # Aggregate results (moved here so it picks up domain-check writes)
    # =========================================================================
    violated_indices = sorted(set(idx for idx, _ in violations_list))
    msg_by_idx = defaultdict(list)
    for idx, msg in violations_list:
        msg_by_idx[idx].append(msg)

    aggregated_msgs = []
    for idx in violated_indices:
        msgs = msg_by_idx[idx]
        if len(msgs) <= 3:
            aggregated_msgs.extend(msgs)
        else:
            aggregated_msgs.append(
                f"{msgs[0]} (and {len(msgs)-1} more violations of constraint {idx})")

    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": aggregated_msgs,
        "violation_magnitudes": magnitudes_list if not feasible else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SAV routing LP (Levin 2017)")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the feasibility result JSON file")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    result = check_feasibility(instance, solution)
    save_json(args.result_path, result)

    if result["feasible"] is None:
        print("NO SOLUTION: Nothing to check.")
        for msg in result["violations"]:
            print(f"  - {msg}")
    elif result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. "
              f"Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"][:10]:
            print(f"  - {msg}")
        if len(result["violations"]) > 10:
            print(f"  ... and {len(result['violations']) - 10} more")


if __name__ == "__main__":
    main()
