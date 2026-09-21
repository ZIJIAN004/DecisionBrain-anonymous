"""
Feasibility checker for the Capped-Fleet Pickup and Delivery Problem with Time
Windows. Instances are Li & Lim PDPTW benchmarks, restricted to a hard fleet size.

All arithmetic is integral: the distance and travel time between two nodes are
round(sqrt(dx^2 + dy^2)) computed as int(sqrt(...) + 0.5), and every time in the
instance is on that same scale, so feasibility never turns on a tolerance. A
vehicle leaves the depot at time 0, waits when it arrives before a node's window
opens, and may not arrive after the window closes.

Constraint indices follow hidden/mathematical_formulation.md:

    (1)  objective consistency: reported objective_value equals recomputed distance
    (2)  every request is served exactly once -- both its pickup and its delivery
    (3)  at most max_vehicles routes are used
    (4)  the load carried never exceeds capacity and never drops below zero
    (5)  service at each node begins within its time window
    (6)  each route returns to the depot before the depot's window closes
    (7)  a request's pickup and delivery are served by the same vehicle, with the
         pickup strictly earlier in that route

Index 0 is reserved for a solution that cannot be parsed as a set of routes.

CLI contract:
    python feasibility_check.py --instance_path I --solution_path S --result_path R
Writes to R: {"feasible": bool|None, "violated_constraints": [...],
              "violations": [...], "violation_magnitudes": [...]}
"""

import argparse
import json
import math


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dist(coords, a, b):
    dx = coords[a][0] - coords[b][0]
    dy = coords[a][1] - coords[b][1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def write_result(path, result):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def normalise_routes(raw):
    if isinstance(raw, dict):
        return [(str(k), v) for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))]
    if isinstance(raw, list):
        return [(str(i), v) for i, v in enumerate(raw)]
    return None


def check_feasibility(instance_path, solution_path, result_path):
    inst = load_json(instance_path)
    sol = load_json(solution_path)

    coords = inst["coordinates"]          # index 0 is the depot
    demand = inst["demand"]
    tw = inst["time_window"]
    service = inst["service_time"]
    capacity = inst["capacity"]
    max_vehicles = inst["max_vehicles"]
    pairs = inst["requests"]              # [[pickup, delivery], ...]
    n = len(coords)

    if "routes" not in sol or sol["routes"] is None:
        result = write_result(result_path, {
            "feasible": None, "violated_constraints": [],
            "violations": ["No solution data in solution file"],
            "violation_magnitudes": []})
        print("No solution data in solution file")
        print(f"Result written to {result_path}")
        return result

    routes = normalise_routes(sol["routes"])
    if routes is None or any(not isinstance(seq, list) for _, seq in routes):
        result = write_result(result_path, {
            "feasible": None, "violated_constraints": [],
            "violations": ["solution.routes must map each vehicle to a list of node ids"],
            "violation_magnitudes": []})
        print("solution.routes is not a mapping of vehicles to node lists")
        print(f"Result written to {result_path}")
        return result

    violations, violated, magnitudes = [], set(), []

    def add(idx, message, lhs, rhs, excess):
        violated.add(idx)
        violations.append(message)
        norm = max(abs(rhs), 1e-9) if isinstance(rhs, (int, float)) else 1.0
        magnitudes.append({"constraint": idx, "lhs": lhs, "rhs": rhs,
                           "raw_excess": excess, "normalizer": norm,
                           "ratio": (excess / norm) if isinstance(excess, (int, float)) else None})

    for vid, seq in routes:
        for node in seq:
            if isinstance(node, bool) or not isinstance(node, int) or node <= 0 or node >= n:
                add(0, f"route {vid} contains {node!r}, which is not a node index in "
                       f"1..{n - 1} (the depot is implicit and must not be listed)",
                    node, None, None)

    routes = [(vid, [c for c in seq if isinstance(c, int) and not isinstance(c, bool)
                     and 0 < c < n]) for vid, seq in routes]
    nonempty = [(vid, seq) for vid, seq in routes if seq]

    # (3) fleet cap
    if len(nonempty) > max_vehicles:
        add(3, f"{len(nonempty)} routes are used but at most {max_vehicles} are allowed",
            len(nonempty), max_vehicles, len(nonempty) - max_vehicles)

    delivery_of = {p: d for p, d in pairs}
    pickup_of = {d: p for p, d in pairs}

    seen = {}
    total = 0
    for vid, seq in nonempty:
        load = 0
        t = 0
        prev = 0
        picked_here = set()
        for node in seq:
            if node in seen:
                add(2, f"node {node} is visited twice (routes {seen[node]} and {vid})",
                    2, 1, 1)
            seen[node] = vid

            t += dist(coords, prev, node)
            total += dist(coords, prev, node)
            start = max(t, tw[node][0])
            # (5) time window
            if start > tw[node][1]:
                add(5, f"route {vid} starts service at node {node} at {start} but its "
                       f"window closes at {tw[node][1]} (arrival {t})",
                    start, tw[node][1], start - tw[node][1])
            t = start + service[node]

            # (7) pairing and precedence
            if node in delivery_of:
                picked_here.add(node)
            elif node in pickup_of:
                if pickup_of[node] not in picked_here:
                    add(7, f"route {vid} serves delivery {node} before its pickup "
                           f"{pickup_of[node]} on the same route", 0, 1, 1)
            load += demand[node]
            # (4) capacity
            if load < 0:
                add(4, f"route {vid} carries a negative load {load} after node {node}",
                    load, 0, -load)
            if load > capacity:
                add(4, f"route {vid} carries {load} after node {node}, exceeding "
                       f"capacity {capacity}", load, capacity, load - capacity)
            prev = node
        t += dist(coords, prev, 0)
        total += dist(coords, prev, 0)
        # (6) depot return
        if t > tw[0][1]:
            add(6, f"route {vid} returns to the depot at {t}, after it closes at "
                   f"{tw[0][1]}", t, tw[0][1], t - tw[0][1])

    # (2) coverage
    missing = [f"{p}->{d}" for p, d in pairs if p not in seen or d not in seen]
    if missing:
        add(2, f"{len(missing)} requests are not fully served, e.g. {missing[:10]}",
            len(missing), 0, len(missing))
    stray = [x for x in seen if x not in delivery_of and x not in pickup_of]
    if stray:
        add(2, f"{len(stray)} visited nodes belong to no request, e.g. {stray[:10]}",
            len(stray), 0, len(stray))

    # (1) objective consistency
    reported = sol.get("objective_value")
    if reported is None:
        add(1, "solution.objective_value is missing; the reported objective is part "
               "of the submission contract", None, total, None)
    else:
        try:
            val = float(reported)
        except (TypeError, ValueError):
            add(1, f"objective_value {reported!r} is not a number", reported, total, None)
        else:
            if round(val) != total:
                add(1, f"objective_value {reported} does not match the recomputed "
                       f"distance {total}", val, total, abs(val - total))

    feasible = not violations
    result = write_result(result_path, {
        "feasible": feasible,
        "violated_constraints": sorted(violated),
        "violations": violations,
        "violation_magnitudes": magnitudes if not feasible else [],
    })
    print("Feasibility: {}".format("FEASIBLE" if feasible else "INFEASIBLE"))
    print(f"Routes used: {len(nonempty)} / {max_vehicles}   objective: {total}")
    for v in violations[:20]:
        print("  - " + v)
    print(f"Result written to {result_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="capped-fleet PDPTW feasibility checker")
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--result_path", required=True)
    a = ap.parse_args()
    check_feasibility(a.instance_path, a.solution_path, a.result_path)


if __name__ == "__main__":
    main()
