#!/usr/bin/env python3
"""
Feasibility Checker for MD-VCSP (Huisman, Freling, Wagelmans 2005).

Checks all hard constraints from MD-VCSP1 formulation (constraints 2-9),
plus an objective-consistency check (constraint 10).

When vehicle_schedule is absent or empty, the solution is rejected as
infeasible because constraints (2)-(4) and (9) on y variables cannot be
verified (skipping them made those checks vacuous).

Constraint (10) [obj-recompute, Tier C anti-gaming defense]: the reported
objective_value must equal the objective recomputed directly from the
solution variables. Per math_model.txt (reproduction comments 1 & 4) the
MD-VCSP objective under the paper's computational setup counts each vehicle
and each crew duty as cost 1, and solution_schema.json defines
objective_value as "Total number of vehicles plus crew duties used to cover
all trips". Both obj-determining quantities -- the vehicle chains and the
crew duties -- are fully present in the solution, so this is an EXACT full
recompute (true_obj = num_vehicles + num_drivers), not a lower bound. A
reported objective_value that disagrees with the recomputed count is
rejected so that fabricated objective values cannot pass the checker.
"""

import argparse
import json
from collections import defaultdict

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def tt(data, a, b):
    """Travel time from location a to location b."""
    return data["travel_times"][a][b]


def build_network(data):
    """Build the vehicle-scheduling network for each depot."""
    trips = sorted(data["trips"], key=lambda t: t["start_time"])
    n = len(trips)
    depots = [d["name"] for d in data["depots"]]
    tid2t = {t["trip_id"]: t for t in trips}

    arc_exists = set()
    arc_is_short = set()
    arc_is_long = set()
    arc_is_source = set()
    arc_is_sink = set()

    for dep in depots:
        for t in trips:
            arc_exists.add((dep, 'src', t["trip_id"]))
            arc_is_source.add((dep, 'src', t["trip_id"]))
            arc_exists.add((dep, t["trip_id"], 'snk'))
            arc_is_sink.add((dep, t["trip_id"], 'snk'))

        for i in range(n):
            ti = trips[i]
            for j in range(i + 1, n):
                tj = trips[j]
                dh_ij = tt(data, ti["end_location"], tj["start_location"])
                if tj["start_time"] >= ti["end_time"] + dh_ij:
                    k = (dep, ti["trip_id"], tj["trip_id"])
                    arc_exists.add(k)
                    idle = tj["start_time"] - ti["end_time"]
                    rt = tt(data, ti["end_location"], dep) + tt(data, dep, tj["start_location"])
                    if idle >= rt:
                        arc_is_long.add(k)
                    else:
                        arc_is_short.add(k)

    return {
        'trips': trips, 'n': n, 'depots': depots, 'tid2t': tid2t,
        'arc_exists': arc_exists, 'arc_is_short': arc_is_short,
        'arc_is_long': arc_is_long, 'arc_is_source': arc_is_source,
        'arc_is_sink': arc_is_sink,
    }


def build_y_from_vehicle_schedule(vehicle_schedule, depots):
    """Reconstruct binary y variables from the vehicle schedule (chains)."""
    y = {}
    for dep in depots:
        chains = vehicle_schedule.get(dep, [])
        for chain in chains:
            if not chain:
                continue
            y[(dep, 'src', chain[0])] = 1
            for k in range(len(chain) - 1):
                y[(dep, chain[k], chain[k + 1])] = 1
            y[(dep, chain[-1], 'snk')] = 1
    return y


def has_vehicle_schedule(solution, depots):
    """Check if the solution has a non-empty vehicle schedule."""
    vs = solution.get("vehicle_schedule", {})
    if not vs:
        return False
    for dep in depots:
        if vs.get(dep) and any(len(chain) > 0 for chain in vs[dep]):
            return True
    return False


def build_short_adj(net):
    """Build short arc adjacency for piece generation."""
    short_adj = defaultdict(set)
    for dep, i, j in net['arc_is_short']:
        short_adj[(dep, i)].add(j)
    return short_adj


def gen_pieces_for_duty(duty_trips, dep, net, short_adj):
    """Split a duty's trip list into pieces of work.
    A piece is a maximal contiguous sequence connected by short arcs."""
    if not duty_trips:
        return []
    tid2t = net['tid2t']
    sorted_tids = sorted(duty_trips, key=lambda t: tid2t[t]["start_time"])

    pieces = []
    current_piece = [sorted_tids[0]]
    for k in range(1, len(sorted_tids)):
        prev_tid = current_piece[-1]
        cur_tid = sorted_tids[k]
        if cur_tid in short_adj.get((dep, prev_tid), set()):
            current_piece.append(cur_tid)
        else:
            pieces.append(current_piece)
            current_piece = [cur_tid]
    pieces.append(current_piece)
    return pieces


def add_violation(violated_set, violations, violation_magnitudes,
                  constraint_idx, msg, lhs, rhs, violation_amount):
    """Record a constraint violation."""
    violated_set.add(constraint_idx)
    normalizer = max(abs(rhs), EPS)
    violations.append(msg)
    violation_magnitudes.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(violation_amount / normalizer),
    })


def check_feasibility(data, solution):
    """Check all hard constraints from MD-VCSP1 formulation."""
    net = build_network(data)
    trips = net['trips']
    depots = net['depots']
    tid2t = net['tid2t']
    arc_exists = net['arc_exists']
    arc_is_short = net['arc_is_short']
    arc_is_long = net['arc_is_long']
    short_adj = build_short_adj(net)

    violations = []
    violation_magnitudes = []
    violated_set = set()

    vehicle_schedule = solution.get("vehicle_schedule", {})
    crew_schedule = solution.get("crew_schedule", [])

    have_vs = has_vehicle_schedule(solution, depots)

    if not have_vs:
        # Skipping y-variable checks (2)-(4) and (9) when vehicle_schedule
        # is absent/empty makes those constraints vacuous. Reject outright.
        for cidx in (2, 3, 4, 9):
            add_violation(violated_set, violations, violation_magnitudes,
                          cidx,
                          f"Constraint ({cidx}): Missing required field "
                          f"'vehicle_schedule'; cannot evaluate y-variable "
                          f"constraint.",
                          0.0, 1.0, 1.0)
        return {
            "feasible": False,
            "violated_constraints": sorted(violated_set),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }

    # Reconstruct y variables
    if have_vs:
        y = build_y_from_vehicle_schedule(vehicle_schedule, depots)
    else:
        y = {}

    # Pre-index y for efficient lookups
    y_out = defaultdict(int)   # (dep, src) -> sum of y values
    y_in = defaultdict(int)    # (dep, dst) -> sum of y values
    y_out_any = defaultdict(int)  # src -> sum across all depots
    y_in_any = defaultdict(int)   # dst -> sum across all depots
    for (dep, src, dst), val in y.items():
        y_out[(dep, src)] += val
        y_in[(dep, dst)] += val
        y_out_any[src] += val
        y_in_any[dst] += val

    # Pre-index long arcs in y
    y_long_out = defaultdict(int)  # (dep, src) -> sum of y on long arcs from src
    y_long_in = defaultdict(int)   # (dep, dst) -> sum of y on long arcs into dst
    for arc_key in arc_is_long:
        dep, ti, tj = arc_key
        val = y.get(arc_key, 0)
        if val > 0:
            y_long_out[(dep, ti)] += val
            y_long_in[(dep, tj)] += val

    # Pre-compute duty piece information
    duty_pieces_cache = []
    for d in crew_schedule:
        dep = d["depot"]
        pieces = gen_pieces_for_duty(d["trips"], dep, net, short_adj)
        duty_pieces_cache.append(pieces)

    # Pre-index: for each (dep, trip), count duties covering that trip
    duty_trip_count = defaultdict(int)
    for d in crew_schedule:
        dep = d["depot"]
        for tid in d["trips"]:
            duty_trip_count[(dep, tid)] += 1

    # Pre-index: for each (dep, ti, tj) short arc, count duties covering it
    duty_short_count = defaultdict(int)
    for idx, d in enumerate(crew_schedule):
        dep = d["depot"]
        pieces = duty_pieces_cache[idx]
        for piece in pieces:
            for k in range(len(piece) - 1):
                arc = (dep, piece[k], piece[k + 1])
                if arc in arc_is_short:
                    duty_short_count[arc] += 1

    # Pre-index: for each (dep, tid), count duties where tid is last in a piece
    duty_end_piece = defaultdict(int)
    # Pre-index: for each (dep, tid), count duties where tid is first in a piece
    duty_start_piece = defaultdict(int)
    for idx, d in enumerate(crew_schedule):
        dep = d["depot"]
        pieces = duty_pieces_cache[idx]
        for piece in pieces:
            duty_end_piece[(dep, piece[-1])] += 1
            duty_start_piece[(dep, piece[0])] += 1

    # =========================================================================
    # Constraint (2): Each trip has exactly one successor arc
    # sum_{d in D} sum_{j: (i,j) in A^d} y^d_{ij} = 1, for all i in N
    # =========================================================================
    if have_vs:
        for t in trips:
            tid = t["trip_id"]
            lhs = y_out_any.get(tid, 0)
            rhs = 1.0
            va = abs(lhs - rhs)
            if va > TOL:
                add_violation(violated_set, violations, violation_magnitudes,
                              2, f"Constraint (2): Trip {tid} has {lhs} successor arc(s) instead of 1",
                              lhs, rhs, va)

    # =========================================================================
    # Constraint (3): Each trip has exactly one predecessor arc
    # sum_{d in D} sum_{i: (i,j) in A^d} y^d_{ij} = 1, for all j in N
    # =========================================================================
    if have_vs:
        for t in trips:
            tid = t["trip_id"]
            lhs = y_in_any.get(tid, 0)
            rhs = 1.0
            va = abs(lhs - rhs)
            if va > TOL:
                add_violation(violated_set, violations, violation_magnitudes,
                              3, f"Constraint (3): Trip {tid} has {lhs} predecessor arc(s) instead of 1",
                              lhs, rhs, va)

    # =========================================================================
    # Constraint (4): Flow conservation per depot per trip node
    # sum_{i:(i,j) in A^d} y^d_{ij} - sum_{i:(j,i) in A^d} y^d_{ji} = 0
    # =========================================================================
    if have_vs:
        for dep in depots:
            for t in trips:
                tid = t["trip_id"]
                inf = y_in.get((dep, tid), 0)
                outf = y_out.get((dep, tid), 0)
                lhs = inf - outf
                rhs = 0.0
                va = abs(lhs - rhs)
                if va > TOL:
                    add_violation(violated_set, violations, violation_magnitudes,
                                  4, f"Constraint (4): Flow not conserved at trip {tid}, "
                                  f"depot {dep} (in={inf}, out={outf})",
                                  lhs, rhs, va)

    # =========================================================================
    # Constraint (5): Trip task linking
    # sum_{k in K^d(i)} x^d_k - sum_{j:(i,j) in A^d} y^d_{ij} = 0
    # =========================================================================
    if have_vs:
        for dep in depots:
            for t in trips:
                tid = t["trip_id"]
                dc = duty_trip_count.get((dep, tid), 0)
                yo = y_out.get((dep, tid), 0)
                lhs = dc - yo
                rhs = 0.0
                va = abs(lhs - rhs)
                if va > TOL:
                    add_violation(violated_set, violations, violation_magnitudes,
                                  5, f"Constraint (5): Trip-task mismatch at trip {tid}, "
                                  f"depot {dep} (duties={dc}, y_out={yo})",
                                  lhs, rhs, va)
    else:
        # Without vehicle schedule, check that each trip is covered by
        # exactly one duty across all depots (implied by constraints 2+5)
        for t in trips:
            tid = t["trip_id"]
            total_cover = sum(
                duty_trip_count.get((dep, tid), 0) for dep in depots
            )
            lhs = total_cover
            rhs = 1.0
            va = abs(lhs - rhs)
            if va > TOL:
                add_violation(violated_set, violations, violation_magnitudes,
                              5, f"Constraint (5): Trip {tid} covered by {total_cover} "
                              f"duty(ies) instead of 1",
                              lhs, rhs, va)

    # =========================================================================
    # Constraint (6): Short-arc dh-task linking
    # sum_{k in K^d(i,j)} x^d_k - y^d_{ij} = 0
    # =========================================================================
    if have_vs:
        for arc_key in arc_is_short:
            dep, ti, tj = arc_key
            dc = duty_short_count.get(arc_key, 0)
            yv = y.get(arc_key, 0)
            lhs = dc - yv
            rhs = 0.0
            va = abs(lhs - rhs)
            if va > TOL:
                add_violation(violated_set, violations, violation_magnitudes,
                              6, f"Constraint (6): Short-arc dh-task mismatch at ({ti},{tj}), "
                              f"depot {dep} (duties={dc}, y={yv})",
                              lhs, rhs, va)

    # =========================================================================
    # Constraint (7): Long-arc dh-task linking (end to depot)
    # sum_{k in K^d(i,t^d)} x^d_k - y^d_{i,t^d} - sum_{j:(i,j) in A^{ld}} y^d_{ij} = 0
    # =========================================================================
    if have_vs:
        for dep in depots:
            for t in trips:
                tid = t["trip_id"]
                dc = duty_end_piece.get((dep, tid), 0)
                ys = y.get((dep, tid, 'snk'), 0)
                ylo = y_long_out.get((dep, tid), 0)
                lhs = dc - ys - ylo
                rhs = 0.0
                va = abs(lhs - rhs)
                if va > TOL:
                    add_violation(violated_set, violations, violation_magnitudes,
                                  7, f"Constraint (7): End-to-depot dh-task mismatch at trip {tid}, "
                                  f"depot {dep} (duties_end={dc}, y_sink={ys}, y_long_out={ylo})",
                                  lhs, rhs, va)

    # =========================================================================
    # Constraint (8): Long-arc dh-task linking (depot to start)
    # sum_{k in K^d(s^d,j)} x^d_k - y^d_{s^d,j} - sum_{i:(i,j) in A^{ld}} y^d_{ij} = 0
    # =========================================================================
    if have_vs:
        for dep in depots:
            for t in trips:
                tid = t["trip_id"]
                dc = duty_start_piece.get((dep, tid), 0)
                ys = y.get((dep, 'src', tid), 0)
                yli = y_long_in.get((dep, tid), 0)
                lhs = dc - ys - yli
                rhs = 0.0
                va = abs(lhs - rhs)
                if va > TOL:
                    add_violation(violated_set, violations, violation_magnitudes,
                                  8, f"Constraint (8): Depot-to-start dh-task mismatch at trip {tid}, "
                                  f"depot {dep} (duties_start={dc}, y_source={ys}, y_long_in={yli})",
                                  lhs, rhs, va)

    # =========================================================================
    # Constraint (9): Variable domains
    # y^d_{ij} in {0,1}, x^d_k in {0,1}
    # Check that all used y arcs are valid arcs in the network.
    # =========================================================================
    if have_vs:
        for k_arc, val in y.items():
            if val > 0 and k_arc not in arc_exists:
                dep, src, dst = k_arc
                add_violation(violated_set, violations, violation_magnitudes,
                              9, f"Constraint (9): Arc ({src},{dst}) in depot {dep} "
                              f"is not a valid arc in A^d",
                              float(val), 0.0, float(val))

    # =========================================================================
    # Constraint (10): Objective consistency (Tier C anti-gaming defense)
    # objective_value must equal the objective recomputed directly from the
    # solution. Per math_model.txt (reproduction comments 1 & 4) the MD-VCSP
    # objective under the paper's computational setup counts each vehicle and
    # each crew duty as cost 1, and solution_schema.json defines
    # objective_value as "Total number of vehicles plus crew duties used to
    # cover all trips". Both obj-determining quantities are fully present in
    # the solution (vehicle chains -> num_vehicles, crew duties ->
    # num_drivers), so this is an EXACT full recompute:
    #     true_obj = num_vehicles + num_drivers
    # A reported objective_value that disagrees is rejected so that
    # fabricated objective values cannot pass the checker.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # num_vehicles: each non-empty vehicle chain is one vehicle, counted
            # over the same depot set used to reconstruct the y variables (see
            # build_y_from_vehicle_schedule above).
            num_vehicles = 0
            for dep in depots:
                for chain in vehicle_schedule.get(dep, []):
                    if chain:
                        num_vehicles += 1
            # num_drivers: each crew duty is one driver.
            num_drivers = len(crew_schedule)
            true_obj = float(num_vehicles + num_drivers)
            abs_diff = abs(reported - true_obj)
            # The objective is an integer count (vehicles + duties), so any
            # mismatch of magnitude >= 1 must fire; tol = 0.5.
            tol = 0.5
            if abs_diff > tol:
                add_violation(violated_set, violations, violation_magnitudes,
                              10,
                              f"Constraint (10): Objective consistency violated: "
                              f"reported objective_value={reported} differs from "
                              f"recomputed num_vehicles({num_vehicles})+"
                              f"num_drivers({num_drivers})={true_obj} "
                              f"(|diff|={abs_diff:.3g}, tol={tol:.3g})",
                              reported, true_obj, abs_diff)

    # Build result
    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MD-VCSP (Huisman et al. 2005)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write feasibility result JSON file")
    args = parser.parse_args()

    data = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(data, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE - No constraints violated.")
    else:
        print(f"INFEASIBLE - Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
