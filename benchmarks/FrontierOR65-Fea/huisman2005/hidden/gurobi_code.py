#!/usr/bin/env python3
"""
Gurobi implementation of MD-VCSP1 from Huisman, Freling, and Wagelmans (2005).
Multiple-Depot Integrated Vehicle and Crew Scheduling.

Implements the MD-VCSP1 formulation (constraints 1-9) from the paper.
Since K^d (the set of all feasible duties) is exponentially large,
we enumerate feasible duties for the given instance and solve the full MIP.
"""

import argparse
import json
import math
import time
from collections import defaultdict

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
    with open(path, 'r') as f:
        return json.load(f)


def tt(data, a, b):
    """Travel time between locations a and b in minutes."""
    return data["travel_times"][a][b]


def build_data(data):
    """Pre-process instance data into efficient structures."""
    trips = sorted(data["trips"], key=lambda t: t["start_time"])
    n = len(trips)
    depots = [d["name"] for d in data["depots"]]
    IDLE_COST = data["vehicle_parameters"]["fictitious_cost_per_minute_empty_outside_depot"]
    FIXED_VEH = 1.0  # each vehicle costs 1 in objective
    relief_locs = set(data["relief_locations"])
    depot_set = set(depots)

    # Pre-compute all arcs per depot
    # arc = (type, src_id, dst_id, cost)
    # type: 'source', 'sink', 'short', 'long'
    arcs = {}  # (dep, src, dst) -> cost
    arc_type = {}  # (dep, src, dst) -> type
    out_arcs = defaultdict(list)  # (dep, src) -> [(dst, cost)]
    in_arcs = defaultdict(list)   # (dep, dst) -> [(src, cost)]

    short_arcs = defaultdict(set)  # dep -> set of (tid_i, tid_j)
    long_arcs = defaultdict(set)   # dep -> set of (tid_i, tid_j)

    for dep in depots:
        for t in trips:
            dh = tt(data, dep, t["start_location"])
            c = FIXED_VEH + dh * IDLE_COST
            key = (dep, 'src', t["trip_id"])
            arcs[key] = c
            arc_type[key] = 'source'
            out_arcs[(dep, 'src')].append((t["trip_id"], c))
            in_arcs[(dep, t["trip_id"])].append(('src', c))

        for t in trips:
            dh = tt(data, t["end_location"], dep)
            c = dh * IDLE_COST
            key = (dep, t["trip_id"], 'snk')
            arcs[key] = c
            arc_type[key] = 'sink'
            out_arcs[(dep, t["trip_id"])].append(('snk', c))
            in_arcs[(dep, 'snk')].append((t["trip_id"], c))

        for i in range(n):
            ti = trips[i]
            for j in range(i + 1, n):
                tj = trips[j]
                dh_ij = tt(data, ti["end_location"], tj["start_location"])
                if tj["start_time"] >= ti["end_time"] + dh_ij:
                    idle = tj["start_time"] - ti["end_time"]
                    rt = tt(data, ti["end_location"], dep) + tt(data, dep, tj["start_location"])
                    # INFERRED ASSUMPTION: long arc if idle >= round trip to depot
                    # **NOT EXPLICITLY SPECIFIED IN PAPER beyond "long enough to return"**
                    if idle >= rt:
                        c = rt * IDLE_COST
                        atype = 'long'
                        long_arcs[dep].add((ti["trip_id"], tj["trip_id"]))
                    else:
                        c = idle * IDLE_COST
                        atype = 'short'
                        short_arcs[dep].add((ti["trip_id"], tj["trip_id"]))
                    key = (dep, ti["trip_id"], tj["trip_id"])
                    arcs[key] = c
                    arc_type[key] = atype
                    out_arcs[(dep, ti["trip_id"])].append((tj["trip_id"], c))
                    in_arcs[(dep, tj["trip_id"])].append((ti["trip_id"], c))

    return trips, depots, arcs, arc_type, out_arcs, in_arcs, short_arcs, long_arcs, relief_locs, depot_set


def pre_post_time(loc, dep, data, relief_locs, depot_set, is_start):
    """Compute sign-on/off time for a duty starting/ending at loc."""
    cp = data["crew_parameters"]
    if loc == dep or loc in depot_set:
        return cp["sign_on_time_depot_minutes"] if is_start else cp["sign_off_time_depot_minutes"]
    else:
        return cp["extra_time_non_depot_relief_minutes"] + tt(data, loc, dep)


def enumerate_duties(data, trips, depots, short_arcs, long_arcs, relief_locs, depot_set):
    """
    Enumerate feasible duties for all depots.
    Returns list of duty dicts.
    """
    n = len(trips)
    tid2t = {t["trip_id"]: t for t in trips}
    duty_types = data["duty_types"]
    duties = []

    for dep in depots:
        # Build short-arc adjacency
        s_adj = defaultdict(list)
        for (ti, tj) in short_arcs[dep]:
            s_adj[ti].append(tj)

        # Generate pieces of work via DFS through short arcs
        pieces = []
        for si in range(n):
            st = trips[si]
            # DFS stack: (current_tid, path)
            stack = [(st["trip_id"], [st["trip_id"]])]
            while stack:
                cur, path = stack.pop()
                ft = tid2t[path[0]]
                lt = tid2t[cur]
                wt = lt["end_time"] - ft["start_time"]
                if wt > 300:
                    continue
                if wt >= 30:
                    pieces.append({
                        'tids': tuple(path),
                        'st': ft["start_time"], 'et': lt["end_time"],
                        'sl': ft["start_location"], 'el': lt["end_location"],
                        'wt': wt,
                        'sa': [(path[k], path[k+1]) for k in range(len(path)-1)],
                    })
                for nxt in s_adj.get(cur, []):
                    if nxt not in path:
                        nt = tid2t[nxt]
                        if nt["end_time"] - ft["start_time"] <= 300:
                            stack.append((nxt, path + [nxt]))

        print(f"  Depot {dep}: {len(pieces)} pieces")

        # Generate duties from pieces
        for dname, dtype in duty_types.items():
            np_ = dtype["num_pieces"]
            if np_ == 1:
                for p in pieces:
                    d = _try_single(p, dtype, dname, dep, data, relief_locs, depot_set)
                    if d is not None:
                        duties.append(d)
            else:
                # 2-piece duties - use sorted pieces for efficiency
                sorted_p = sorted(pieces, key=lambda x: x['et'])
                for i, p1 in enumerate(sorted_p):
                    # Break must end at relief location
                    if p1['el'] not in relief_locs and p1['el'] not in depot_set:
                        continue
                    min_break = dtype["break_length_min"] or 0
                    for j, p2 in enumerate(sorted_p):
                        if p2['st'] <= p1['et']:
                            continue
                        if p2['st'] - p1['et'] < min_break:
                            continue
                        if p2['sl'] not in relief_locs and p2['sl'] not in depot_set:
                            continue
                        # Check no trip overlap
                        if set(p1['tids']) & set(p2['tids']):
                            continue
                        d = _try_two(p1, p2, dtype, dname, dep, data, relief_locs, depot_set)
                        if d is not None:
                            duties.append(d)

    return duties


def _try_single(p, dtype, dname, dep, data, relief_locs, depot_set):
    if dtype["piece_length_min"] and p['wt'] < dtype["piece_length_min"]:
        return None
    if dtype["piece_length_max"] and p['wt'] > dtype["piece_length_max"]:
        return None
    pre = pre_post_time(p['sl'], dep, data, relief_locs, depot_set, True)
    post = pre_post_time(p['el'], dep, data, relief_locs, depot_set, False)
    ds = p['st'] - pre
    de = p['et'] + post
    dl = de - ds
    if dtype["duty_length_max"] and dl > dtype["duty_length_max"]:
        return None
    if dtype["work_time_max"] and p['wt'] > dtype["work_time_max"]:
        return None
    if dtype["start_time_min"] and ds < dtype["start_time_min"]:
        return None
    if dtype["start_time_max"] and ds > dtype["start_time_max"]:
        return None
    if dtype["end_time_max"] and de > dtype["end_time_max"]:
        return None
    return {
        'dep': dep, 'type': dname, 'tids': list(p['tids']),
        'sa': list(p['sa']),
        'le': [p['tids'][-1]], 'ls': [p['tids'][0]],
        'cost': 1.0,
    }


def _try_two(p1, p2, dtype, dname, dep, data, relief_locs, depot_set):
    for p in [p1, p2]:
        if dtype["piece_length_min"] and p['wt'] < dtype["piece_length_min"]:
            return None
        if dtype["piece_length_max"] and p['wt'] > dtype["piece_length_max"]:
            return None
    pre = pre_post_time(p1['sl'], dep, data, relief_locs, depot_set, True)
    post = pre_post_time(p2['el'], dep, data, relief_locs, depot_set, False)
    ds = p1['st'] - pre
    de = p2['et'] + post
    dl = de - ds
    wt = p1['wt'] + p2['wt']
    if dtype["duty_length_max"] and dl > dtype["duty_length_max"]:
        return None
    if dtype["work_time_max"] and wt > dtype["work_time_max"]:
        return None
    if dtype["start_time_min"] and ds < dtype["start_time_min"]:
        return None
    if dtype["start_time_max"] and ds > dtype["start_time_max"]:
        return None
    if dtype["end_time_max"] and de > dtype["end_time_max"]:
        return None
    return {
        'dep': dep, 'type': dname,
        'tids': list(p1['tids']) + list(p2['tids']),
        'sa': list(p1['sa']) + list(p2['sa']),
        'le': [p1['tids'][-1], p2['tids'][-1]],
        'ls': [p1['tids'][0], p2['tids'][0]],
        'cost': 1.0,
    }


def solve_model(data, time_limit):
    """Build and solve the MD-VCSP1 model with Gurobi."""
    print("Pre-processing instance data...")
    trips, depots, arcs, arc_type, out_arcs, in_arcs, short_arcs, long_arcs, relief_locs, depot_set = build_data(data)
    n = len(trips)
    tids = [t["trip_id"] for t in trips]
    print(f"Trips: {n}, Depots: {len(depots)}, Arcs: {len(arcs)}")

    print("Enumerating feasible duties...")
    duties = enumerate_duties(data, trips, depots, short_arcs, long_arcs, relief_locs, depot_set)
    print(f"Total duties: {len(duties)}")

    if not duties:
        print("WARNING: No feasible duties. Solving MDVSP only.")
        return _solve_mdvsp_only(data, trips, depots, arcs, arc_type, time_limit)

    print("Building Gurobi model...")
    m = gp.Model("MD-VCSP1")
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 1)
    m.setParam("Threads", 1)

    # y variables
    y = {}
    for key, cost in arcs.items():
        y[key] = m.addVar(vtype=GRB.BINARY, obj=cost, name=f"y_{key[0]}_{key[1]}_{key[2]}")

    # x variables (duty selection)
    x = {}
    for k, duty in enumerate(duties):
        x[k] = m.addVar(vtype=GRB.BINARY, obj=duty['cost'], name=f"x_{k}")

    m.update()

    # Pre-index duties for constraint building
    duty_by_dep_trip = defaultdict(list)    # (dep, tid) -> [k]
    duty_by_dep_sa = defaultdict(list)      # (dep, ti, tj) -> [k]
    duty_by_dep_le = defaultdict(list)      # (dep, tid) -> [k] for long_end
    duty_by_dep_ls = defaultdict(list)      # (dep, tid) -> [k] for long_start

    for k, duty in enumerate(duties):
        dep = duty['dep']
        for tid in duty['tids']:
            duty_by_dep_trip[(dep, tid)].append(k)
        for (ti, tj) in duty['sa']:
            duty_by_dep_sa[(dep, ti, tj)].append(k)
        for tid in duty['le']:
            duty_by_dep_le[(dep, tid)].append(k)
        for tid in duty['ls']:
            duty_by_dep_ls[(dep, tid)].append(k)

    print("Adding constraints...")

    # (2) Each trip has exactly one successor
    for tid in tids:
        m.addConstr(
            gp.quicksum(y[key] for key in arcs if key[1] == tid) == 1,
            name=f"succ_{tid}"
        )

    # (3) Each trip has exactly one predecessor
    for tid in tids:
        m.addConstr(
            gp.quicksum(y[key] for key in arcs if key[2] == tid) == 1,
            name=f"pred_{tid}"
        )

    # (4) Flow conservation
    for dep in depots:
        for tid in tids:
            in_keys = [(dep, s, tid) for s, _ in in_arcs.get((dep, tid), []) if (dep, s, tid) in arcs]
            out_keys = [(dep, tid, d) for d, _ in out_arcs.get((dep, tid), []) if (dep, tid, d) in arcs]
            if in_keys or out_keys:
                m.addConstr(
                    gp.quicksum(y[k] for k in in_keys) - gp.quicksum(y[k] for k in out_keys) == 0,
                    name=f"flow_{dep}_{tid}"
                )

    # (5) Trip task linking
    for dep in depots:
        for tid in tids:
            dk = duty_by_dep_trip.get((dep, tid), [])
            out_keys = [(dep, tid, d) for d, _ in out_arcs.get((dep, tid), []) if (dep, tid, d) in arcs]
            if dk or out_keys:
                m.addConstr(
                    gp.quicksum(x[k] for k in dk) - gp.quicksum(y[k] for k in out_keys) == 0,
                    name=f"tlink_{dep}_{tid}"
                )

    # (6) Short-arc dh-task linking
    for dep in depots:
        for (ti, tj) in short_arcs[dep]:
            dk = duty_by_dep_sa.get((dep, ti, tj), [])
            key = (dep, ti, tj)
            if key in arcs:
                m.addConstr(
                    gp.quicksum(x[k] for k in dk) - y[key] == 0,
                    name=f"slink_{dep}_{ti}_{tj}"
                )

    # (7) Long-arc end-to-depot linking
    for dep in depots:
        for tid in tids:
            dk = duty_by_dep_le.get((dep, tid), [])
            rhs_keys = []
            sink_key = (dep, tid, 'snk')
            if sink_key in arcs:
                rhs_keys.append(sink_key)
            for (ti, tj) in long_arcs[dep]:
                if ti == tid and (dep, ti, tj) in arcs:
                    rhs_keys.append((dep, ti, tj))
            if dk or rhs_keys:
                m.addConstr(
                    gp.quicksum(x[k] for k in dk) - gp.quicksum(y[k] for k in rhs_keys) == 0,
                    name=f"lend_{dep}_{tid}"
                )

    # (8) Long-arc depot-to-start linking
    for dep in depots:
        for tid in tids:
            dk = duty_by_dep_ls.get((dep, tid), [])
            rhs_keys = []
            src_key = (dep, 'src', tid)
            if src_key in arcs:
                rhs_keys.append(src_key)
            for (ti, tj) in long_arcs[dep]:
                if tj == tid and (dep, ti, tj) in arcs:
                    rhs_keys.append((dep, ti, tj))
            if dk or rhs_keys:
                m.addConstr(
                    gp.quicksum(x[k] for k in dk) - gp.quicksum(y[k] for k in rhs_keys) == 0,
                    name=f"lstr_{dep}_{tid}"
                )

    print("Optimizing...")
    m.optimize()

    return _extract(m, y, x, duties, depots, trips, arcs)


def _solve_mdvsp_only(data, trips, depots, arcs, arc_type, time_limit):
    """Fallback MDVSP-only solver."""
    m = gp.Model("MDVSP")
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 0)
    tids = [t["trip_id"] for t in trips]

    y = {}
    for key, cost in arcs.items():
        y[key] = m.addVar(vtype=GRB.BINARY, obj=cost)
    m.update()

    for tid in tids:
        m.addConstr(gp.quicksum(y[k] for k in arcs if k[1] == tid) == 1)
        m.addConstr(gp.quicksum(y[k] for k in arcs if k[2] == tid) == 1)
    for dep in depots:
        for tid in tids:
            ink = [k for k in arcs if k[0] == dep and k[2] == tid]
            outk = [k for k in arcs if k[0] == dep and k[1] == tid]
            if ink or outk:
                m.addConstr(gp.quicksum(y[k] for k in ink) - gp.quicksum(y[k] for k in outk) == 0)
    m.optimize()

    nv = 0
    if m.SolCount > 0:
        for k in arcs:
            if k[1] == 'src' and y[k].X > 0.5:
                nv += 1
    return {
        "objective_value": nv * 2 if m.SolCount > 0 else float('inf'),
        "num_vehicles": nv, "num_drivers": nv,
        "gurobi_obj": m.ObjVal if m.SolCount > 0 else None,
        "status": m.Status,
        "vehicle_schedule": (_build_vehicle_schedule(y, arcs, depots)
                              if m.SolCount > 0 else {}),
        "crew_schedule": [],
    }


def _build_vehicle_schedule(y, arcs, depots):
    """Reconstruct per-depot list of trip chains from solved y arcs."""
    succ = defaultdict(dict)
    starts = defaultdict(list)
    for k in arcs:
        if y[k].X > 0.5:
            dep, src, dst = k
            if src == 'src':
                starts[dep].append(dst)
            else:
                succ[dep][src] = dst
    schedule = {dep: [] for dep in depots}
    for dep in depots:
        for first in starts[dep]:
            chain = [first]
            cur = first
            while True:
                nxt = succ[dep].get(cur)
                if nxt is None or nxt == 'snk':
                    break
                chain.append(nxt)
                cur = nxt
            schedule[dep].append(chain)
    return schedule


def _extract(m, y, x, duties, depots, trips, arcs):
    """Extract solution from Gurobi model."""
    if m.SolCount == 0:
        return {
            "objective_value": float('inf'), "num_vehicles": 0, "num_drivers": 0,
            "gurobi_obj": None, "status": m.Status,
            "vehicle_schedule": {}, "crew_schedule": [],
        }

    nv = sum(1 for k in arcs if k[1] == 'src' and y[k].X > 0.5)
    nd = sum(1 for k in range(len(duties)) if x[k].X > 0.5)
    crew = [{"depot": duties[k]['dep'], "type": duties[k]['type'], "trips": duties[k]['tids']}
            for k in range(len(duties)) if x[k].X > 0.5]

    return {
        "objective_value": nv + nd,
        "num_vehicles": nv, "num_drivers": nd,
        "gurobi_obj": m.ObjVal, "status": m.Status,
        "vehicle_schedule": _build_vehicle_schedule(y, arcs, depots),
        "crew_schedule": crew,
    }


def main():
    parser = argparse.ArgumentParser(description="MD-VCSP1 Gurobi Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from {args.instance_path}")
    data = load_instance(args.instance_path)

    t0 = time.time()
    result = solve_model(data, args.time_limit)
    elapsed = time.time() - t0

    print(f"\nSolution found in {elapsed:.1f}s")
    print(f"Objective (vehicles + drivers): {result['objective_value']}")
    print(f"Vehicles: {result['num_vehicles']}, Drivers: {result['num_drivers']}")

    solution = {
        "objective_value": result["objective_value"],
        "num_vehicles": result["num_vehicles"],
        "num_drivers": result["num_drivers"],
        "gurobi_objective": result.get("gurobi_obj"),
        "solver_status": result["status"],
        "solve_time_seconds": elapsed,
        "vehicle_schedule": result["vehicle_schedule"],
        "crew_schedule": result["crew_schedule"],
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2, default=str)
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
