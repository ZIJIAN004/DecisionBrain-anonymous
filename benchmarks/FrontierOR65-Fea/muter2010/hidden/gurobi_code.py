#!/usr/bin/env python3
"""
VRPTW as a Set Covering Problem solved with Gurobi.

Based on: Muter, Birbil, Sahin (2010) - reformulation of VRPTW as SCP.

The SCP formulation:
  min  sum_{p in P} c_p * y_p
  s.t. sum_{p in P} a_{ip} * y_p >= 1   for all i in C
       y_p in {0, 1}

where:
  C   = set of customers
  P   = set of feasible routes (columns)
  c_p = total travel distance of route p
  a_{ip} = 1 if customer i is served by route p, 0 otherwise

A route is feasible if:
  1. Total demand does not exceed vehicle capacity
  2. Vehicle arrives before end of each customer's time window
  3. If vehicle arrives early, it waits until ready_time
  4. Route starts and ends at depot (node 0)

Column generation strategy (since full enumeration is intractable for 25 customers):
  1. Enumerate all feasible short routes (1-3 customers)
  2. Solomon I1 insertion heuristic for initial feasible solution
  3. Nearest-neighbor heuristic variants (different starting customers)
  4. Savings algorithm (Clarke-Wright)
  5. Random insertion heuristics (multiple runs)
  6. 2-opt and Or-opt local search improvements on discovered routes
  7. Solve the SCP IP over all generated columns with Gurobi
"""

import argparse
import json
import math
import random
import time
import itertools
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
        pass# ---------------------------------------------------------------------------
# Data structures and helpers
# ---------------------------------------------------------------------------

class VRPTWInstance:
    """Parsed VRPTW instance."""

    def __init__(self, data: dict):
        depot = data["depot"]
        self.depot_id = depot["id"]
        self.depot_x = depot["x"]
        self.depot_y = depot["y"]
        self.depot_ready = depot["ready_time"]
        self.depot_due = depot["due_date"]
        self.depot_service = depot["service_time"]

        self.capacity = data["vehicle_capacity"]
        self.num_vehicles = data["num_vehicles"]
        self.num_customers = data["num_customers"]

        # All nodes: depot (index 0) + customers (index 1..n)
        self.x = [depot["x"]]
        self.y = [depot["y"]]
        self.demand = [depot["demand"]]
        self.ready = [depot["ready_time"]]
        self.due = [depot["due_date"]]
        self.service = [depot["service_time"]]
        self.customer_ids = []  # original customer ids (1-based)

        for c in data["customers"]:
            self.x.append(c["x"])
            self.y.append(c["y"])
            self.demand.append(c["demand"])
            self.ready.append(c["ready_time"])
            self.due.append(c["due_date"])
            self.service.append(c["service_time"])
            self.customer_ids.append(c["id"])

        self.n = len(self.customer_ids)  # number of customers
        self.N = self.n + 1  # total nodes (depot + customers)

        # Precompute Euclidean distance matrix (double precision, NOT rounded)
        self.dist = [[0.0] * self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                self.dist[i][j] = math.sqrt(dx * dx + dy * dy)


def route_cost(inst: VRPTWInstance, route: list[int]) -> float:
    """Compute total travel distance for a route (depot -> customers -> depot).
    route is a list of customer node indices (1-based internal indices)."""
    if not route:
        return 0.0
    cost = inst.dist[0][route[0]]
    for i in range(len(route) - 1):
        cost += inst.dist[route[i]][route[i + 1]]
    cost += inst.dist[route[-1]][0]
    return cost


def check_route_feasibility(inst: VRPTWInstance, route: list[int]) -> bool:
    """Check if a route (list of customer node indices) is feasible.
    Returns True if feasible, False otherwise."""
    if not route:
        return True

    # Check capacity
    total_demand = sum(inst.demand[c] for c in route)
    if total_demand > inst.capacity:
        return False

    # Check time windows: simulate the route from depot
    current_time = 0.0
    prev = 0  # depot
    for c in route:
        travel = inst.dist[prev][c]
        arrival = current_time + travel
        # Must arrive before due_date
        if arrival > inst.due[c]:
            return False
        # Wait if arriving early
        start_service = max(arrival, inst.ready[c])
        current_time = start_service + inst.service[c]
        prev = c

    # Check return to depot is feasible
    arrival_depot = current_time + inst.dist[prev][0]
    if arrival_depot > inst.due[0]:
        return False

    return True


def get_earliest_completion(inst: VRPTWInstance, route: list[int]) -> float:
    """Return the time the vehicle finishes the last customer's service,
    or -1 if the route is infeasible."""
    current_time = 0.0
    prev = 0
    for c in route:
        travel = inst.dist[prev][c]
        arrival = current_time + travel
        if arrival > inst.due[c]:
            return -1.0
        start_service = max(arrival, inst.ready[c])
        current_time = start_service + inst.service[c]
        prev = c
    return current_time


# ---------------------------------------------------------------------------
# Column generation: enumerate short routes
# ---------------------------------------------------------------------------

def enumerate_short_routes(inst: VRPTWInstance, max_len: int = 3) -> list[tuple[int, ...]]:
    """Enumerate all feasible routes of length 1 to max_len."""
    routes = []
    customers = list(range(1, inst.N))

    for length in range(1, max_len + 1):
        for combo in itertools.permutations(customers, length):
            route = list(combo)
            if check_route_feasibility(inst, route):
                routes.append(tuple(route))
    return routes


# ---------------------------------------------------------------------------
# Column generation: Solomon I1 insertion heuristic
# ---------------------------------------------------------------------------

def solomon_i1(inst: VRPTWInstance, seed_order: list[int] | None = None,
               alpha1: float = 1.0, alpha2: float = 0.0,
               mu: float = 1.0) -> list[list[int]]:
    """Solomon's I1 sequential insertion heuristic.

    Parameters:
        seed_order: order in which seed customers are chosen; if None, use
                    farthest-from-depot ordering.
        alpha1, alpha2: weights for the two insertion cost criteria.
        mu: weight balancing distance savings vs time urgency.

    Returns a list of routes (each route is a list of customer node indices).
    """
    unrouted = set(range(1, inst.N))
    routes = []

    if seed_order is None:
        # Seed selection: farthest unrouted customer from depot
        seed_order = sorted(range(1, inst.N),
                            key=lambda c: -inst.dist[0][c])

    while unrouted:
        # Pick seed
        seed = None
        for s in seed_order:
            if s in unrouted:
                seed = s
                break
        if seed is None:
            break

        route = [seed]
        unrouted.discard(seed)

        # Try to insert remaining customers
        improved = True
        while improved:
            improved = False
            best_customer = None
            best_pos = None
            best_c2 = -float("inf")

            for u in list(unrouted):
                # Find best insertion position for customer u
                best_c1_u = float("inf")
                best_pos_u = None

                for pos in range(len(route) + 1):
                    trial = route[:pos] + [u] + route[pos:]
                    if not check_route_feasibility(inst, trial):
                        continue

                    # c11: extra distance
                    if pos == 0:
                        prev_node = 0
                    else:
                        prev_node = route[pos - 1]
                    if pos == len(route):
                        next_node = 0
                    else:
                        next_node = route[pos]

                    c11 = (inst.dist[prev_node][u] + inst.dist[u][next_node]
                           - mu * inst.dist[prev_node][next_node])

                    # c12: time push-forward (simplified)
                    # Assumption: using distance-based c12 approximation since
                    # the exact push-forward computation is complex and this is
                    # for column generation, not the final solve.
                    c12 = 0.0  # simplified

                    c1 = alpha1 * c11 + alpha2 * c12
                    if c1 < best_c1_u:
                        best_c1_u = c1
                        best_pos_u = pos

                if best_pos_u is not None:
                    # c2 criterion: distance savings from direct service
                    c2 = inst.dist[0][u] + inst.dist[u][0] - best_c1_u
                    if c2 > best_c2:
                        best_c2 = c2
                        best_customer = u
                        best_pos = best_pos_u

            if best_customer is not None:
                route.insert(best_pos, best_customer)
                unrouted.discard(best_customer)
                improved = True

        routes.append(route)

    return routes


# ---------------------------------------------------------------------------
# Column generation: nearest neighbor heuristic
# ---------------------------------------------------------------------------

def nearest_neighbor(inst: VRPTWInstance, start_customer: int) -> list[int]:
    """Build a single route starting with start_customer using nearest-neighbor."""
    route = [start_customer]
    unvisited = set(range(1, inst.N)) - {start_customer}

    while unvisited:
        last = route[-1]
        best_next = None
        best_dist = float("inf")
        for c in unvisited:
            trial = route + [c]
            if check_route_feasibility(inst, trial):
                d = inst.dist[last][c]
                if d < best_dist:
                    best_dist = d
                    best_next = c
        if best_next is None:
            break
        route.append(best_next)
        unvisited.discard(best_next)

    return route


def nearest_neighbor_all(inst: VRPTWInstance) -> list[list[int]]:
    """Generate routes by nearest-neighbor starting from each customer."""
    routes = []
    for c in range(1, inst.N):
        route = nearest_neighbor(inst, c)
        if route and check_route_feasibility(inst, route):
            routes.append(route)
    return routes


# ---------------------------------------------------------------------------
# Column generation: Clarke-Wright savings algorithm
# ---------------------------------------------------------------------------

def savings_algorithm(inst: VRPTWInstance) -> list[list[int]]:
    """Clarke-Wright savings algorithm adapted for VRPTW."""
    # Start with each customer in its own route
    routes = {c: [c] for c in range(1, inst.N)}
    route_of = {c: c for c in range(1, inst.N)}  # customer -> route key

    # Compute savings
    savings = []
    for i in range(1, inst.N):
        for j in range(i + 1, inst.N):
            s = inst.dist[0][i] + inst.dist[0][j] - inst.dist[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if ri not in routes or rj not in routes:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Try merging: i must be last in route_i, j must be first in route_j
        # or vice versa
        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j

        if merged is not None and check_route_feasibility(inst, merged):
            new_key = ri
            routes[new_key] = merged
            if rj in routes:
                del routes[rj]
            for c in merged:
                route_of[c] = new_key

    return list(routes.values())


# ---------------------------------------------------------------------------
# Column generation: random insertion
# ---------------------------------------------------------------------------

def random_insertion(inst: VRPTWInstance, rng: random.Random) -> list[list[int]]:
    """Build a full solution by randomly inserting customers into routes."""
    customers = list(range(1, inst.N))
    rng.shuffle(customers)

    routes = []
    unrouted = set(customers)

    while unrouted:
        # Start a new route with a random unrouted customer
        seed = rng.choice(list(unrouted))
        route = [seed]
        unrouted.discard(seed)

        # Try inserting remaining customers in random order
        candidates = list(unrouted)
        rng.shuffle(candidates)
        for u in candidates:
            # Find best feasible insertion position
            best_pos = None
            best_cost = float("inf")
            for pos in range(len(route) + 1):
                trial = route[:pos] + [u] + route[pos:]
                if check_route_feasibility(inst, trial):
                    cost = route_cost(inst, trial)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
            if best_pos is not None:
                route.insert(best_pos, u)
                unrouted.discard(u)

        routes.append(route)

    return routes


# ---------------------------------------------------------------------------
# Local search: 2-opt within a route
# ---------------------------------------------------------------------------

def two_opt_improve(inst: VRPTWInstance, route: list[int]) -> list[int]:
    """Apply 2-opt local search to improve a single route."""
    if len(route) <= 2:
        return route

    improved = True
    best = list(route)
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if check_route_feasibility(inst, new_route):
                    if route_cost(inst, new_route) < route_cost(inst, best):
                        best = new_route
                        improved = True
    return best


# ---------------------------------------------------------------------------
# Local search: Or-opt (move a segment of 1-3 customers)
# ---------------------------------------------------------------------------

def or_opt_improve(inst: VRPTWInstance, route: list[int]) -> list[int]:
    """Apply Or-opt local search: relocate segments of 1, 2, or 3 customers."""
    if len(route) <= 1:
        return route

    best = list(route)
    improved = True
    while improved:
        improved = False
        for seg_len in range(1, min(4, len(best) + 1)):
            for i in range(len(best) - seg_len + 1):
                segment = best[i:i + seg_len]
                remainder = best[:i] + best[i + seg_len:]
                for j in range(len(remainder) + 1):
                    new_route = remainder[:j] + segment + remainder[j:]
                    if new_route == best:
                        continue
                    if check_route_feasibility(inst, new_route):
                        if route_cost(inst, new_route) < route_cost(inst, best):
                            best = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return best


# ---------------------------------------------------------------------------
# Generate all sub-routes from a given route
# ---------------------------------------------------------------------------

def extract_subroutes(inst: VRPTWInstance, route: list[int],
                      max_subroute_len: int | None = None) -> list[tuple[int, ...]]:
    """Extract all feasible contiguous sub-routes from a given route."""
    if max_subroute_len is None:
        max_subroute_len = len(route)
    subroutes = []
    for length in range(1, min(max_subroute_len, len(route)) + 1):
        for start in range(len(route) - length + 1):
            sub = route[start:start + length]
            if check_route_feasibility(inst, sub):
                subroutes.append(tuple(sub))
    return subroutes


# ---------------------------------------------------------------------------
# Master column pool builder
# ---------------------------------------------------------------------------

def build_column_pool(inst: VRPTWInstance, time_budget: float) -> list[tuple[int, ...]]:
    """Generate a diverse set of feasible routes (columns) for the SCP.

    Uses multiple heuristics and local search to create a rich column pool.
    time_budget is in seconds.
    """
    start_time = time.time()
    pool_set: set[tuple[int, ...]] = set()

    def add_routes(routes: list[list[int]]):
        for r in routes:
            if r and check_route_feasibility(inst, r):
                pool_set.add(tuple(r))

    def elapsed():
        return time.time() - start_time

    # --- Phase 1: Enumerate short routes (1-3 customers) ---
    print(f"  [ColGen] Phase 1: Enumerating short routes (1-3 customers)...")
    short_routes = enumerate_short_routes(inst, max_len=3)
    for r in short_routes:
        pool_set.add(r)
    print(f"  [ColGen]   Found {len(short_routes)} short routes, pool size = {len(pool_set)}")

    if elapsed() > time_budget * 0.8:
        return list(pool_set)

    # --- Phase 2: Solomon I1 heuristic with different parameters ---
    print(f"  [ColGen] Phase 2: Solomon I1 insertion heuristic...")
    for alpha1, alpha2, mu in [(1.0, 0.0, 1.0), (0.5, 0.5, 1.0),
                                (1.0, 0.0, 0.5), (0.0, 1.0, 1.0)]:
        routes = solomon_i1(inst, alpha1=alpha1, alpha2=alpha2, mu=mu)
        add_routes(routes)
        # Also extract sub-routes and apply local search
        for r in routes:
            for sub in extract_subroutes(inst, r):
                pool_set.add(sub)
            improved = two_opt_improve(inst, r)
            if improved and check_route_feasibility(inst, improved):
                pool_set.add(tuple(improved))
                for sub in extract_subroutes(inst, improved):
                    pool_set.add(sub)

    # Also try with different seed orderings
    # Seed by earliest due date
    seed_by_due = sorted(range(1, inst.N), key=lambda c: inst.due[c])
    routes = solomon_i1(inst, seed_order=seed_by_due)
    add_routes(routes)
    for r in routes:
        for sub in extract_subroutes(inst, r):
            pool_set.add(sub)

    # Seed by earliest ready time
    seed_by_ready = sorted(range(1, inst.N), key=lambda c: inst.ready[c])
    routes = solomon_i1(inst, seed_order=seed_by_ready)
    add_routes(routes)
    for r in routes:
        for sub in extract_subroutes(inst, r):
            pool_set.add(sub)

    print(f"  [ColGen]   Pool size = {len(pool_set)}")

    if elapsed() > time_budget * 0.8:
        return list(pool_set)

    # --- Phase 3: Nearest-neighbor heuristic ---
    print(f"  [ColGen] Phase 3: Nearest-neighbor heuristic variants...")
    nn_routes = nearest_neighbor_all(inst)
    add_routes(nn_routes)
    for r in nn_routes:
        for sub in extract_subroutes(inst, r):
            pool_set.add(sub)
        improved = two_opt_improve(inst, r)
        if improved and check_route_feasibility(inst, improved):
            pool_set.add(tuple(improved))
            for sub in extract_subroutes(inst, improved):
                pool_set.add(sub)
    print(f"  [ColGen]   Pool size = {len(pool_set)}")

    if elapsed() > time_budget * 0.8:
        return list(pool_set)

    # --- Phase 4: Clarke-Wright savings ---
    print(f"  [ColGen] Phase 4: Clarke-Wright savings algorithm...")
    cw_routes = savings_algorithm(inst)
    add_routes(cw_routes)
    for r in cw_routes:
        for sub in extract_subroutes(inst, r):
            pool_set.add(sub)
        improved = two_opt_improve(inst, r)
        if improved and check_route_feasibility(inst, improved):
            pool_set.add(tuple(improved))
            for sub in extract_subroutes(inst, improved):
                pool_set.add(sub)
        improved = or_opt_improve(inst, r)
        if improved and check_route_feasibility(inst, improved):
            pool_set.add(tuple(improved))
            for sub in extract_subroutes(inst, improved):
                pool_set.add(sub)
    print(f"  [ColGen]   Pool size = {len(pool_set)}")

    if elapsed() > time_budget * 0.8:
        return list(pool_set)

    # --- Phase 5: Random insertion (multiple runs) ---
    print(f"  [ColGen] Phase 5: Random insertion heuristics...")
    num_random_runs = 200
    for run in range(num_random_runs):
        if elapsed() > time_budget * 0.6:
            break
        rng = random.Random(42 + run)
        ri_routes = random_insertion(inst, rng)
        add_routes(ri_routes)
        for r in ri_routes:
            for sub in extract_subroutes(inst, r):
                pool_set.add(sub)
            # Apply local search to some runs
            if run < 50:
                improved = two_opt_improve(inst, r)
                if improved and check_route_feasibility(inst, improved):
                    pool_set.add(tuple(improved))
                    for sub in extract_subroutes(inst, improved):
                        pool_set.add(sub)
    print(f"  [ColGen]   Pool size = {len(pool_set)}")

    if elapsed() > time_budget * 0.8:
        return list(pool_set)

    # --- Phase 6: Enumerate routes of length 4-5 if time permits ---
    # Assumption: For 25 customers, length-4 permutations = 25*24*23*22 = 303,600
    # which is feasible to check. Length-5 may be too large, but we try with
    # time bounds.
    print(f"  [ColGen] Phase 6: Enumerating medium-length routes...")
    customers = list(range(1, inst.N))
    for length in range(4, 7):
        count = 0
        for combo in itertools.permutations(customers, length):
            if elapsed() > time_budget * 0.8:
                break
            route = list(combo)
            # Quick demand check before full feasibility
            total_demand = sum(inst.demand[c] for c in route)
            if total_demand > inst.capacity:
                continue
            if check_route_feasibility(inst, route):
                pool_set.add(tuple(route))
                count += 1
        print(f"  [ColGen]   Length {length}: found {count} routes, "
              f"pool size = {len(pool_set)}")
        if elapsed() > time_budget * 0.8:
            break

    print(f"  [ColGen] Final pool size = {len(pool_set)}")
    return list(pool_set)


# ---------------------------------------------------------------------------
# Solve the Set Covering Problem with Gurobi
# ---------------------------------------------------------------------------

def solve_scp(inst: VRPTWInstance, columns: list[tuple[int, ...]],
              time_limit: float) -> dict:
    """Solve the SCP formulation of VRPTW with Gurobi.

    Returns a dict with solution details.
    """
    customers = list(range(1, inst.N))  # internal node indices
    num_columns = len(columns)

    # Precompute column costs and coverage
    col_costs = []
    col_coverage = []  # col_coverage[p] = set of customers covered
    for p, route in enumerate(columns):
        col_costs.append(route_cost(inst, list(route)))
        col_coverage.append(set(route))

    # Build coverage map: customer -> list of column indices
    coverage_map = defaultdict(list)
    for p, route in enumerate(columns):
        for c in route:
            coverage_map[c].append(p)

    # Check that every customer is covered by at least one column
    uncovered = [c for c in customers if c not in coverage_map]
    if uncovered:
        print(f"  [WARNING] {len(uncovered)} customers have no covering column!")
        print(f"  [WARNING] Uncovered customer nodes: {uncovered}")
        # The model will be infeasible; we still build it so Gurobi reports this.

    print(f"  [SCP] Building Gurobi model with {num_columns} columns "
          f"and {len(customers)} covering constraints...")

    model = gp.Model("VRPTW_SCP")
    model.setParam("Threads", 1)
    # Pass shorter TimeLimit so optimize() returns before outer wrapper SIGKILL.
    # Plan F 1h run found incumbent obj=2972.6 (jsonl) but solution.json missing
    # — gurobi was mid-simplex at 2407s and got SIGKILLed at 3631s before
    # json.dump ran. The 120s buffer lets the final LP wind down + writeback.
    _internal_tl = max(60, time_limit - 120) if time_limit > 240 else time_limit
    model.setParam("TimeLimit", _internal_tl)
    model.setParam("OutputFlag", 1)
    # Solver hints: l11 (33 customers, 11 vehicles VRPTW) was 1h TLE no
    # incumbent in prior runs while l21-51 (same dims, different seeds)
    # solved OPT in 800-1070s. MIPFocus=1 prioritizes finding feasible
    # over closing gap; NoRelHeurTime budgets root-node heuristics.
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))

    # Decision variables: y_p in {0, 1} for each column p
    y = model.addVars(num_columns, vtype=GRB.BINARY, name="y",
                      obj=col_costs)

    # Covering constraints: each customer must be served by at least one route
    for c in customers:
        covering_cols = coverage_map.get(c, [])
        if covering_cols:
            model.addConstr(
                gp.quicksum(y[p] for p in covering_cols) >= 1,
                name=f"cover_{c}"
            )
        else:
            # Add infeasible constraint so Gurobi reports infeasibility
            model.addConstr(0 >= 1, name=f"cover_{c}_infeasible")

    model.setAttr("ModelSense", GRB.MINIMIZE)
    model.update()

    # Solve
    model.optimize()

    # Extract solution
    result = {
        "status": model.Status,
        "status_name": _gurobi_status_name(model.Status),
        "objective_value": None,
        "routes": [],
        "num_columns_generated": num_columns,
        "gap": None,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["gap"] = model.MIPGap if hasattr(model, "MIPGap") else None

        selected_routes = []
        for p in range(num_columns):
            if y[p].X > 0.5:
                route = list(columns[p])
                cost = col_costs[p]
                # Map internal node indices back to original customer IDs
                customer_ids = [inst.customer_ids[c - 1] for c in route]
                selected_routes.append({
                    "route_nodes": route,
                    "customer_ids": customer_ids,
                    "cost": cost,
                    "demand": sum(inst.demand[c] for c in route),
                })
        result["routes"] = selected_routes
    else:
        print("  [SCP] No feasible solution found.")

    return result


def _gurobi_status_name(status: int) -> str:
    """Convert Gurobi status code to human-readable name."""
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"UNKNOWN({status})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve VRPTW as Set Covering Problem using Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the VRPTW instance JSON file.")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Time limit in seconds for the Gurobi solver.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    print(f"Loading instance from {args.instance_path}...")
    with open(args.instance_path, "r") as f:
        data = json.load(f)
    inst = VRPTWInstance(data)
    print(f"  Customers: {inst.n}, Capacity: {inst.capacity}, "
          f"Vehicles: {inst.num_vehicles}")

    overall_start = time.time()

    # Allocate time: ~60% for column generation, ~40% for Gurobi solving
    # (with a minimum of 10 seconds for Gurobi)
    colgen_budget = max(args.time_limit * 0.6, args.time_limit - 60)
    colgen_budget = min(colgen_budget, args.time_limit - 10)
    colgen_budget = max(colgen_budget, 5)  # at least 5 seconds for colgen

    # Generate columns
    print(f"Generating columns (budget: {colgen_budget:.1f}s)...")
    columns = build_column_pool(inst, colgen_budget)

    colgen_elapsed = time.time() - overall_start
    solver_time_limit = max(args.time_limit - colgen_elapsed - 1, 5)

    # Solve SCP
    print(f"Solving SCP with Gurobi (time limit: {solver_time_limit:.1f}s)...")
    result = solve_scp(inst, columns, solver_time_limit)

    total_elapsed = time.time() - overall_start

    # Build output JSON
    output = {
        "objective_value": result["objective_value"],
        "solver_status": result["status_name"],
        "mip_gap": result["gap"],
        "num_routes": len(result["routes"]),
        "num_columns_generated": result["num_columns_generated"],
        "total_time_seconds": round(total_elapsed, 2),
        "routes": [],
    }

    for i, r in enumerate(result["routes"]):
        route_entry = {
            "route_id": i + 1,
            "customer_ids": r["customer_ids"],
            "cost": round(r["cost"], 6),
            "demand": r["demand"],
            # Full path: depot -> customers -> depot
            "path": [0] + r["customer_ids"] + [0],
        }
        output["routes"].append(route_entry)

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSolution written to {args.solution_path}")
    print(f"  Objective value: {result['objective_value']}")
    print(f"  Number of routes: {len(result['routes'])}")
    print(f"  Total time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
