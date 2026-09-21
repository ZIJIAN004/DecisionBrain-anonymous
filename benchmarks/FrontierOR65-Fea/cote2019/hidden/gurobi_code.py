#!/usr/bin/env python3
"""
S2L-CVRP (Stochastic 2D Loading CVRP) solver using Gurobi.

Implements the mathematical model from Côté, Gendreau, and Potvin (2020).
Branch-and-cut with lazy constraint callbacks for:
  - Subtour elimination / rounded capacity inequalities (Eq 4)
  - Infeasible path inequalities (Eq 5)
  - Aggregated optimality cuts for recourse (Eq 43)

Usage:
    python gurobi_code.py --instance_path instance_1.json --solution_path sol.json --time_limit 3600
"""

import argparse
import json
import math
import time
from itertools import product as cartesian_product
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
# Instance loading
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Precomputation helpers
# ---------------------------------------------------------------------------

def compute_expected_area_weight(customers):
    """
    Compute expected area ã_j and expected weight q̃_j for each customer j.
    ã_j = Σ_{i∈I_j} Σ_r p_i^r * h_i^r * w_i^r
    q̃_j = Σ_{i∈I_j} Σ_r p_i^r * q_i^r
    """
    expected = {}
    for cust in customers:
        cid = cust["id"]
        a_tilde = 0.0
        q_tilde = 0.0
        for item in cust["items"]:
            for r in item["realizations"]:
                p = r["probability"]
                a_tilde += p * r["height"] * r["width"]
                q_tilde += p * r["weight"]
        expected[cid] = (a_tilde, q_tilde)
    return expected


def rhs_capacity_bound(subset_ids, expected, H, W, Q):
    """
    Compute right-hand side of RCI (Eq 4):
    |S| - max{ceil(Σ ã_j / (H*W)), ceil(Σ q̃_j / Q)}
    """
    total_area = sum(expected[j][0] for j in subset_ids)
    total_weight = sum(expected[j][1] for j in subset_ids)
    area_vehicles = math.ceil(total_area / (H * W))
    weight_vehicles = math.ceil(total_weight / Q)
    return len(subset_ids) - max(area_vehicles, weight_vehicles)


# ---------------------------------------------------------------------------
# Route extraction from edge solution
# ---------------------------------------------------------------------------

def extract_routes(edge_vals, n):
    """
    Given a dict {(j,k): value} of edge variables with value ~1,
    extract routes as lists of customer IDs (not including depot 0).
    Depot = 0, customers = 1..n.
    """
    # Build adjacency list (only edges with value close to 1)
    adj = defaultdict(list)
    for (j, k), val in edge_vals.items():
        if val > 0.5:
            adj[j].append(k)
            adj[k].append(j)

    # Each route starts and ends at depot 0.
    # Depot degree = 2K, so there are K routes.
    visited_edges = set()
    routes = []

    # Find all neighbors of depot 0
    depot_neighbors = sorted(adj[0])

    for start in depot_neighbors:
        edge = (min(0, start), max(0, start))
        if edge in visited_edges:
            continue
        visited_edges.add(edge)

        route = []
        prev = 0
        cur = start
        while cur != 0:
            route.append(cur)
            neighbors = adj[cur]
            next_node = None
            for nb in neighbors:
                e = (min(cur, nb), max(cur, nb))
                if e not in visited_edges:
                    next_node = nb
                    break
            if next_node is None:
                # **INFERRED ASSUMPTION**: if stuck, route is malformed; break.
                break
            visited_edges.add((min(cur, next_node), max(cur, next_node)))
            prev = cur
            cur = next_node

        if route:
            routes.append(route)

    return routes


# ---------------------------------------------------------------------------
# 2D Packing feasibility: Bottom-Left heuristic with unloading constraints
# ---------------------------------------------------------------------------

def bottom_left_heuristic_with_unloading(items_with_delivery_order, H, W):
    """
    Bottom-left heuristic for 2OPP with unloading constraints.

    items_with_delivery_order: list of (item_h, item_w, delivery_position)
        sorted by delivery_position descending (last delivered packed first).
    H: vehicle height (length along which items are unloaded from the top/rear).
    W: vehicle width.

    Unloading constraint: items delivered earlier (higher delivery_position
    numerically = later in delivery) must not block items delivered later
    from being pulled out from the top (y-direction = H direction).
    **INFERRED ASSUMPTION**: "top" means the open end at y=H; items are
    unloaded by sliding out in the +y direction. An item i (delivered at
    position a) is blocked by item j (delivered at position b, b>a meaning
    j is delivered after i) if j overlaps i in x and j.y > i.y. Since j is
    still on the truck when i is delivered, this is a conflict. We pack in
    reverse delivery order (last-delivered = first packed, placed deepest).

    Returns True if all items packed feasibly, False otherwise.
    """
    # Sort: last-delivered customers first (they go deepest in truck).
    sorted_items = sorted(items_with_delivery_order, key=lambda t: -t[2])

    placements = []  # (x, y, w, h, delivery_pos)

    for (ih, iw, dpos) in sorted_items:
        placed = False
        # Try all candidate positions in bottom-left order
        # Candidate y positions: 0, and top edges of existing placements
        # Candidate x positions: 0, and right edges of existing placements
        y_candidates = sorted(set([0] + [p[1] + p[3] for p in placements]))
        x_candidates = sorted(set([0] + [p[0] + p[2] for p in placements]))

        best_pos = None
        for y in y_candidates:
            if y + ih > H:
                continue
            for x in x_candidates:
                if x + iw > W:
                    continue
                # Check no overlap with existing placements
                overlap = False
                for (px, py, pw, ph, pd) in placements:
                    if x < px + pw and x + iw > px and y < py + ph and y + ih > py:
                        overlap = True
                        break
                if overlap:
                    continue

                # Check unloading constraints:
                # This item has delivery_pos = dpos.
                # For every already-placed item with delivery_pos pd < dpos
                # (delivered earlier, so still on truck when this item should
                # be unloaded), that item must not be above this item
                # in a way that blocks pulling this item out.
                # **INFERRED ASSUMPTION**: unloading from y=H end.
                # Item (x,y,iw,ih) is blocked by (px,py,pw,ph) if they
                # overlap in x range AND py >= y+ih (the other item is
                # further from the open end). Actually, the blocker is
                # between this item and the exit. So item j blocks item i
                # if j overlaps i in x and j is between i and exit (y=H).
                # i.e., px < x+iw and px+pw > x and py >= y (some part of j
                # is at or above y) but we need j to actually be between i
                # and the exit. The precise constraint: j's y range overlaps
                # (y+ih, H) in x-projection overlapping (x, x+iw).
                # Simpler conservative check: if j has delivery_pos < dpos
                # (j is delivered BEFORE this item, so j must be unloaded
                # first), then j must be closer to exit (higher y) or
                # non-overlapping in x.
                # Actually let's be precise: when delivering customer at
                # position dpos, all items of customers with position > dpos
                # have already been delivered and removed. Items with position
                # < dpos are still on the truck. So we need: no item with
                # position < dpos blocks this item's removal from y=H.
                # An item blocks if it overlaps in x and has y >= this item's
                # y + ih (between this item and exit at y=H).
                # Wait—items with smaller delivery position are delivered LATER.
                # Delivery position 1 = first delivered, N = last delivered.
                # When unloading customer at position p, customers p+1..N
                # have not yet been delivered? No—delivery order is 1,2,3...
                # Customer 1 is delivered first. When delivering customer 1,
                # customers 2..N are still on truck. So items of customers
                # 2..N must not block customer 1's items.
                # So for this item (dpos), items with delivery_pos > dpos are
                # still on truck during this item's delivery and could block.
                unloading_ok = True
                for (px, py, pw, ph, pd) in placements:
                    if pd > dpos:
                        # pd is delivered AFTER dpos, so pd's items are still
                        # on the truck when dpos is being delivered.
                        # Check if pd's item blocks this item from exit (y=H).
                        if (px < x + iw and px + pw > x and
                                py + ph > y + ih and py < H):
                            # pd's item occupies space between this item's top
                            # and the exit in overlapping x range → blocks.
                            # **INFERRED ASSUMPTION**: blocking means any part
                            # of the other item is above (closer to exit) this
                            # item in the same x column.
                            unloading_ok = False
                            break
                if not unloading_ok:
                    continue

                # Valid position found (bottom-left preference)
                if best_pos is None or (y, x) < (best_pos[1], best_pos[0]):
                    best_pos = (x, y)

            if best_pos is not None and best_pos[1] == y:
                # Already found the bottom-most valid position at this y
                break

        if best_pos is None:
            return False

        placements.append((best_pos[0], best_pos[1], iw, ih, dpos))

    return True


# ---------------------------------------------------------------------------
# 2OPP / 2OPPUL exact MIP solver
# ---------------------------------------------------------------------------

def solve_2opp_exact(items_with_delivery_order, H, W, time_limit=30.0):
    """
    Exact MIP for 2D Orthogonal Packing with Unloading constraints (2OPPUL).

    items_with_delivery_order: list of (item_h, item_w, delivery_position)
    H, W: container dimensions
    time_limit: seconds for Gurobi

    Returns True if feasible, False otherwise.
    """
    n_items = len(items_with_delivery_order)
    if n_items == 0:
        return True

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("2OPPUL", env=env)
    m.setParam("Threads", 1)
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 0)

    items = items_with_delivery_order  # alias

    # Position variables
    x = {}
    y = {}
    for i in range(n_items):
        x[i] = m.addVar(lb=0, ub=W - items[i][1], vtype=GRB.CONTINUOUS,
                         name=f"x_{i}")
        y[i] = m.addVar(lb=0, ub=H - items[i][0], vtype=GRB.CONTINUOUS,
                         name=f"y_{i}")

    # Non-overlap disjunction: for each pair (i,j), i<j, at least one of:
    #   x_i + w_i <= x_j  (i left of j)   -> delta1
    #   x_j + w_j <= x_i  (j left of i)   -> delta2
    #   y_i + h_i <= y_j  (i below j)      -> delta3
    #   y_j + h_j <= y_i  (j below i)      -> delta4
    M_x = W
    M_y = H

    delta = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            d = {}
            for k in range(4):
                d[k] = m.addVar(vtype=GRB.BINARY, name=f"d_{i}_{j}_{k}")
            delta[(i, j)] = d

            hi, wi, di_pos = items[i]
            hj, wj, dj_pos = items[j]

            # x_i + w_i <= x_j + M*(1 - delta1)
            m.addConstr(x[i] + wi <= x[j] + M_x * (1 - d[0]))
            # x_j + w_j <= x_i + M*(1 - delta2)
            m.addConstr(x[j] + wj <= x[i] + M_x * (1 - d[1]))
            # y_i + h_i <= y_j + M*(1 - delta3)
            m.addConstr(y[i] + hi <= y[j] + M_y * (1 - d[2]))
            # y_j + h_j <= y_i + M*(1 - delta4)
            m.addConstr(y[j] + hj <= y[i] + M_y * (1 - d[3]))

            # At least one must hold
            m.addConstr(d[0] + d[1] + d[2] + d[3] >= 1)

            # Unloading constraints:
            # If di_pos < dj_pos: i is delivered before j. When delivering i,
            # j is still on truck. j must not block i from exit (y=H).
            # "j above i" = y_j + h_j > y_i + h_i with x-overlap.
            # The "j below i" option (delta4: y_j + h_j <= y_i) means j is
            # deeper than i. This is fine—j doesn't block i from exit.
            # The "i below j" option (delta3: y_i + h_i <= y_j) means i is
            # deeper and j is above i between i and exit. This blocks i.
            # So we must forbid delta3 when di_pos < dj_pos.
            # **INFERRED ASSUMPTION**: "above" in the unloading direction
            # means higher y, and items are pulled out from y=H.
            # When i is delivered before j (di_pos < dj_pos), j is still on
            # truck. j must not be between i and exit.
            # Forbid: delta3 (y_i + h_i <= y_j) because this puts j above i.
            if di_pos < dj_pos:
                m.addConstr(d[2] == 0)  # forbid "i below j"
            elif dj_pos < di_pos:
                m.addConstr(d[3] == 0)  # forbid "j below i"
            # If same delivery position (same customer), no unloading constraint.

    m.setObjective(0, GRB.MINIMIZE)  # feasibility problem
    m.optimize()

    feasible = (m.Status == GRB.OPTIMAL or
                (m.Status == GRB.TIME_LIMIT and m.SolCount > 0))
    m.dispose()
    env.dispose()
    return feasible


# ---------------------------------------------------------------------------
# Packing feasibility check (combined heuristic + exact)
# ---------------------------------------------------------------------------

def check_packing_feasibility(items_with_delivery_order, H, W, Q):
    """
    Check if items can be packed in the vehicle with unloading constraints.

    items_with_delivery_order: list of (height, width, weight, delivery_position)
    H, W: vehicle dimensions
    Q: vehicle weight capacity

    Returns True if feasible.
    """
    if not items_with_delivery_order:
        return True

    # Area bound
    total_area = sum(h * w for h, w, wt, dp in items_with_delivery_order)
    if total_area > H * W:
        return False

    # Weight bound
    total_weight = sum(wt for h, w, wt, dp in items_with_delivery_order)
    if total_weight > Q:
        return False

    # Individual item dimension check
    for h, w, wt, dp in items_with_delivery_order:
        # **NOT SPECIFIED IN PAPER**: Items cannot be rotated (stated in problem).
        if h > H or w > W:
            return False

    # Build list for packing (h, w, delivery_position)
    pack_items = [(h, w, dp) for h, w, wt, dp in items_with_delivery_order]

    # Try bottom-left heuristic first (fast)
    if bottom_left_heuristic_with_unloading(pack_items, H, W):
        return True

    # Fall back to exact MIP
    return solve_2opp_exact(pack_items, H, W, time_limit=10.0)


# ---------------------------------------------------------------------------
# Recourse computation for a single route
# ---------------------------------------------------------------------------

def compute_route_recourse(route, customers_by_id, H, W, Q, cf):
    """
    Compute the recourse cost F(R) for a route.

    route: list of customer IDs in delivery order.
    Returns the minimum recourse over both delivery orderings (forward/reverse).

    F(R) = cf * Σ_{ω} p_ω * F(ω)
    where F(ω) = number of unserved customers under scenario ω.
    """
    if not route:
        return 0.0

    # **INFERRED ASSUMPTION**: Both orderings are checked since two-index
    # formulation doesn't determine direction; use the better one.
    best_recourse = float("inf")

    for delivery_order in [route, list(reversed(route))]:
        recourse = _compute_recourse_for_ordering(
            delivery_order, customers_by_id, H, W, Q, cf
        )
        best_recourse = min(best_recourse, recourse)

    return best_recourse


def _compute_recourse_for_ordering(delivery_order, customers_by_id, H, W, Q, cf):
    """
    Compute recourse for a specific delivery ordering.
    Enumerate scenarios (Cartesian product of item realizations).
    """
    # Collect all items on this route with their realizations.
    # For each item, we have a list of (realization_data, probability).
    # A scenario is one realization per item.
    item_realizations = []  # list of lists of (h, w, weight, prob, customer_id, delivery_pos)
    for pos_idx, cid in enumerate(delivery_order):
        cust = customers_by_id[cid]
        delivery_pos = pos_idx + 1  # 1-indexed delivery position
        for item in cust["items"]:
            reals = []
            for r in item["realizations"]:
                reals.append((
                    r["height"], r["width"], r["weight"],
                    r["probability"], cid, delivery_pos
                ))
            item_realizations.append(reals)

    # Check if all items are deterministic (1 realization each)
    all_deterministic = all(len(reals) == 1 for reals in item_realizations)

    if all_deterministic:
        # Single scenario with probability 1
        items = [(r[0], r[1], r[2], r[5]) for reals in item_realizations
                 for r in reals]
        feasible = check_packing_feasibility(items, H, W, Q)
        if feasible:
            return 0.0
        else:
            # Infeasible deterministic route: F(R) = infinity conceptually.
            # In practice, all customers are unserved.
            # **NOT SPECIFIED IN PAPER**: For a deterministic infeasible route,
            # we return cf * n_customers (all unserved). The outer logic will
            # add an infeasible path inequality instead.
            return cf * len(delivery_order)

    # Enumerate all scenarios
    # **NOT SPECIFIED IN PAPER**: For routes with very many scenarios, this
    # could be exponential. We cap at a reasonable number.
    n_scenarios = 1
    for reals in item_realizations:
        n_scenarios *= len(reals)

    # If too many scenarios, use sampling
    # **NOT SPECIFIED IN PAPER**: sampling threshold
    MAX_SCENARIOS = 10000
    if n_scenarios > MAX_SCENARIOS:
        return _compute_recourse_sampled(
            delivery_order, item_realizations, customers_by_id, H, W, Q, cf,
            MAX_SCENARIOS
        )

    total_recourse = 0.0

    for scenario in cartesian_product(*item_realizations):
        # scenario is a tuple of (h, w, weight, prob, cid, delivery_pos) per item
        prob = 1.0
        for item_real in scenario:
            prob *= item_real[3]

        if prob < 1e-15:
            continue

        # Build item list: (h, w, weight, delivery_pos)
        items_scenario = [(s[0], s[1], s[2], s[5]) for s in scenario]

        # Check full route feasibility
        feasible = check_packing_feasibility(items_scenario, H, W, Q)
        if feasible:
            # F(ω) = 0
            continue

        # Not feasible: find max customers that can be served.
        unserved = _count_unserved(delivery_order, scenario, H, W, Q)
        total_recourse += prob * cf * unserved

    return total_recourse


def _compute_recourse_sampled(delivery_order, item_realizations, customers_by_id,
                              H, W, Q, cf, n_samples):
    """Estimate recourse by sampling scenarios."""
    import random
    total_recourse = 0.0

    for _ in range(n_samples):
        scenario = []
        for reals in item_realizations:
            # Sample according to probabilities
            r_val = random.random()
            cumul = 0.0
            chosen = reals[0]
            for real in reals:
                cumul += real[3]
                if r_val <= cumul:
                    chosen = real
                    break
            scenario.append(chosen)

        items_scenario = [(s[0], s[1], s[2], s[5]) for s in scenario]
        feasible = check_packing_feasibility(items_scenario, H, W, Q)
        if not feasible:
            unserved = _count_unserved(delivery_order, scenario, H, W, Q)
            total_recourse += cf * unserved

    return total_recourse / n_samples


def _count_unserved(delivery_order, scenario, H, W, Q):
    """
    Count unserved customers in a scenario.
    Try dropping customers one at a time (greedy) to find the maximum number
    that can be served.

    **NOT SPECIFIED IN PAPER**: Exact approach would try all subsets; we use
    greedy removal of the customer whose items contribute most area, then
    re-check feasibility. This is a heuristic for the recourse sub-problem.
    """
    n_cust = len(delivery_order)

    # Group items by customer
    cust_items = defaultdict(list)
    for s in scenario:
        cust_items[s[4]].append(s)  # keyed by customer id

    # Try serving all customers first (already failed), then drop one at a time.
    served = list(delivery_order)

    for drop_count in range(1, n_cust):
        # Try dropping each remaining customer, pick the one that makes it feasible
        # **INFERRED ASSUMPTION**: greedy drop by largest area contribution
        best_to_drop = None
        best_area = -1
        for cid in served:
            area = sum(s[0] * s[1] for s in cust_items[cid])
            if area > best_area:
                best_area = area
                best_to_drop = cid

        served.remove(best_to_drop)

        # Rebuild item list with remaining customers
        remaining_items = []
        for cid in served:
            for s in cust_items[cid]:
                # Recompute delivery_pos based on position in served list
                dpos = served.index(cid) + 1
                remaining_items.append((s[0], s[1], s[2], dpos))

        if check_packing_feasibility(remaining_items, H, W, Q):
            return drop_count  # number unserved

    # If even single customer can't be served (shouldn't happen normally)
    return n_cust


# ---------------------------------------------------------------------------
# Check if route is infeasible under ALL scenarios
# ---------------------------------------------------------------------------

def is_route_always_infeasible(route, customers_by_id, H, W, Q):
    """
    Check if a route is infeasible under every possible scenario.
    Used for infeasible path inequalities (Eq 5).
    Both delivery orderings are checked.
    """
    for delivery_order in [route, list(reversed(route))]:
        item_realizations = []
        for pos_idx, cid in enumerate(delivery_order):
            cust = customers_by_id[cid]
            delivery_pos = pos_idx + 1
            for item in cust["items"]:
                reals = []
                for r in item["realizations"]:
                    reals.append((r["height"], r["width"], r["weight"],
                                  r["probability"], cid, delivery_pos))
                item_realizations.append(reals)

        # Check each scenario
        n_scenarios = 1
        for reals in item_realizations:
            n_scenarios *= len(reals)

        if n_scenarios > 10000:
            # **NOT SPECIFIED IN PAPER**: too many scenarios to check all;
            # conservatively say not always infeasible.
            return False

        any_feasible = False
        for scenario in cartesian_product(*item_realizations):
            items_scenario = [(s[0], s[1], s[2], s[5]) for s in scenario]
            if check_packing_feasibility(items_scenario, H, W, Q):
                any_feasible = True
                break

        if any_feasible:
            return False

    return True


# ---------------------------------------------------------------------------
# Connected components from edges
# ---------------------------------------------------------------------------

def find_connected_components(edge_vals, n):
    """
    Find connected components from edges with value > 0.5.
    Returns list of sets of node IDs.
    """
    adj = defaultdict(set)
    for (j, k), val in edge_vals.items():
        if val > 0.5:
            adj[j].add(k)
            adj[k].add(j)

    visited = set()
    components = []

    for node in range(n + 1):  # 0..n
        if node in visited or node not in adj:
            continue
        comp = set()
        stack = [node]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.add(v)
            for nb in adj[v]:
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)

    return components


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(args):
    data = load_instance(args.instance_path)

    params = data["parameters"]
    vehicle = data["vehicle"]
    depot = data["depot"]
    customers = data["customers"]
    dist_matrix = data["distance_matrix"]

    n = params["n_customers"]
    H = vehicle["H"]
    W = vehicle["W"]
    Q = vehicle["Q"]
    K = vehicle["K"]
    cf = params["recourse_cost_cf"]

    customers_by_id = {c["id"]: c for c in customers}
    expected = compute_expected_area_weight(customers)

    # Node set: 0 = depot, 1..n = customers
    C = list(range(1, n + 1))  # customer set
    V = list(range(0, n + 1))  # all nodes

    # Edge set: (j, k) with j < k
    edges = []
    for j in V:
        for k in V:
            if j < k:
                edges.append((j, k))

    # Distance / cost
    c = {}
    for j, k in edges:
        c[(j, k)] = dist_matrix[j][k]

    # ----- Build Gurobi model -----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model("S2L_CVRP", env=env)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("LazyConstraints", 1)
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, args.time_limit * 0.05))

    # Decision variables
    x = {}
    for j, k in edges:
        x[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}")

    # Recourse variable theta >= 0
    theta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")

    # Objective: min Σ c_jk * x_jk + theta
    model.setObjective(
        gp.quicksum(c[(j, k)] * x[(j, k)] for j, k in edges) + theta,
        GRB.MINIMIZE
    )

    # Constraint (2): Σ_{j∈C} x_{0,j} = 2K
    model.addConstr(
        gp.quicksum(x[(0, j)] for j in C) == 2 * K,
        name="depot_degree"
    )

    # Constraint (3): degree constraint for each customer j
    for j in C:
        incident = []
        for h in V:
            if h < j:
                incident.append(x[(h, j)])
            elif h > j:
                incident.append(x[(j, h)])
        model.addConstr(gp.quicksum(incident) == 2, name=f"degree_{j}")

    # Constraint (6): x binary - already set in var definition

    model.update()

    # ----- Callback -----
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution
            x_vals = {}
            for j, k in edges:
                val = model.cbGetSolution(x[(j, k)])
                x_vals[(j, k)] = val

            theta_val = model.cbGetSolution(theta)

            # 1. Find connected components and add SECs / RCIs
            components = find_connected_components(x_vals, n)
            for comp in components:
                if 0 in comp:
                    continue
                # Subtour: component without depot
                S = comp
                if len(S) < 2:
                    continue

                # RCI right-hand side (Eq 4)
                rhs = rhs_capacity_bound(S, expected, H, W, Q)

                # Σ_{j,k∈S, j<k} x_{jk} <= rhs
                lhs = gp.quicksum(
                    x[(min(j, k), max(j, k))]
                    for j in S for k in S if j < k
                )
                model.cbLazy(lhs <= rhs)

            # 1b. RCI cuts for customer subsets within depot-connected routes
            #     (Eq 4 applies to ALL S ⊆ C, not just disconnected subtours)
            routes = extract_routes(x_vals, n)
            for route in routes:
                S = set(route)
                if len(S) < 2:
                    continue
                rhs = rhs_capacity_bound(S, expected, H, W, Q)
                # Count internal edges (edges between customers in S)
                lhs_val = sum(
                    1 for j in S for k in S if j < k
                    and x_vals.get((j, k), 0) > 0.5
                )
                if lhs_val > rhs + 0.5:
                    lhs_expr = gp.quicksum(
                        x[(min(j, k), max(j, k))]
                        for j in S for k in S if j < k
                    )
                    model.cbLazy(lhs_expr <= rhs)

            # 2. Extract routes and compute recourse
            # (routes already extracted above)

            total_recourse = 0.0
            active_edges = set()
            for j, k in edges:
                if x_vals[(j, k)] > 0.5:
                    active_edges.add((j, k))

            for route in routes:
                # Check if always infeasible → add infeasible path inequality (Eq 5)
                if is_route_always_infeasible(route, customers_by_id, H, W, Q):
                    # Add infeasible path inequality:
                    # Σ_{(j,k)∈R} x_{jk} <= |R| - 1
                    # R includes depot-customer edges and inter-customer edges
                    route_edges = set()
                    full_path = [0] + route + [0]
                    for i in range(len(full_path) - 1):
                        j, k = full_path[i], full_path[i + 1]
                        route_edges.add((min(j, k), max(j, k)))
                    model.cbLazy(
                        gp.quicksum(x[e] for e in route_edges)
                        <= len(route_edges) - 1
                    )
                else:
                    # Compute recourse for this route
                    route_recourse = compute_route_recourse(
                        route, customers_by_id, H, W, Q, cf
                    )
                    total_recourse += route_recourse

            # 3. Optimality cut for recourse (Eq 43)
            if total_recourse > theta_val + 1e-6:
                # θ >= F(x^v) * (Σ_{active edges} x_{jk} - (n + K - 1))
                # **INFERRED ASSUMPTION**: The cut uses the total number of
                # edges in the solution which is n + K (K depot edges + n
                # customer-to-customer edges for K routes of total n customers).
                # Actually: each route with m_i customers has m_i + 1 edges.
                # Total edges = Σ(m_i + 1) = n + K.
                n_active = len(active_edges)
                model.cbLazy(
                    theta >= total_recourse * (
                        gp.quicksum(x[e] for e in active_edges) -
                        (n_active - 1)
                    )
                )

    model.optimize(callback)

    # ----- Extract solution -----
    result = {
        "instance_id": data.get("instance_id", None),
        "problem": "S2L-CVRP",
        "solver": "Gurobi",
        "status": None,
        "objective_value": None,
        "routing_cost": None,
        "recourse_cost": None,
        "routes": [],
        "time_limit": args.time_limit,
        "solve_time": model.Runtime,
    }

    if model.SolCount > 0:
        # Extract edge values
        x_sol = {}
        for j, k in edges:
            x_sol[(j, k)] = x[(j, k)].X

        routing_cost = sum(c[e] * x_sol[e] for e in edges if x_sol[e] > 0.5)
        theta_sol = theta.X

        routes = extract_routes(x_sol, n)

        # Recompute exact recourse for final solution
        total_recourse = 0.0
        for route in routes:
            route_recourse = compute_route_recourse(
                route, customers_by_id, H, W, Q, cf
            )
            total_recourse += route_recourse

        result["status"] = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible"
        result["objective_value"] = routing_cost + total_recourse
        result["routes"] = [list(r) for r in routes]
    else:
        result["status"] = "NoSolution"
        result["objective_value"] = None

    # Save solution
    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {result['status']}")
    print(f"Objective: {result['objective_value']}")
    print(f"Routing cost: {result['routing_cost']}")
    print(f"Recourse cost: {result['recourse_cost']}")
    print(f"Routes: {result['routes']}")
    print(f"Solve time: {model.Runtime:.2f}s")

    model.dispose()
    env.dispose()

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="S2L-CVRP solver using Gurobi (Côté, Gendreau, Potvin 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=float, default=3600.0,
                        help="Time limit in seconds (default: 3600)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args)


if __name__ == "__main__":
    main()
