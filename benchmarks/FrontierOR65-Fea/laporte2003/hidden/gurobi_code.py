"""
Gurobi MIP implementation for the Undirected Traveling Purchaser Problem (TPP).

Based on: Laporte, Riera-Ledesma, and Salazar-Gonzalez (2003)
"A Branch-and-Cut Algorithm for the Undirected Traveling Purchaser Problem"
Operations Research 51(6):940-951

Implements the ILP formulation (1)-(9) with:
  - Lazy connectivity constraints (3) via callback
  - Valid inequalities (10), (14) added upfront for strengthening
"""

import argparse
import json
import time
import sys
from itertools import combinations

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


def load_instance(instance_path):
    """Load a TPP instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n_vertices = data["num_vertices"]
    n_markets = data["num_markets"]
    n_products = data["num_products"]
    depot = data["depot"]
    supply_type = data["supply_type"]

    # Build vertex set V = {0, 1, ..., n_vertices-1}
    V = list(range(n_vertices))
    # Markets M = V \ {depot}
    M = [v for v in V if v != depot]

    # Travel costs: symmetric, stored as "i,j" with i < j
    travel_costs = {}
    for key, cost in data["travel_costs"].items():
        i, j = map(int, key.split(","))
        travel_costs[(min(i, j), max(i, j))] = cost

    # Edge set E = {(i,j) : i < j, i,j in V}
    E = [(i, j) for i in V for j in V if i < j]

    # Edge costs
    c = {e: travel_costs.get(e, 0) for e in E}

    # Product demands
    demands = data["product_demands"]  # list of length n_products

    # Product markets: product k -> list of markets where it can be purchased
    product_markets = {}
    for k_str, markets in data["product_markets"].items():
        product_markets[int(k_str)] = markets

    # Purchase costs: b[k][i] = cost of product k at market i
    purchase_costs = {}
    for key, cost in data["purchase_costs"].items():
        parts = key.split(",")
        market = int(parts[0])
        product = int(parts[1])
        if product not in purchase_costs:
            purchase_costs[product] = {}
        purchase_costs[product][market] = cost

    # Supplies: q[k][i] = supply of product k at market i
    supplies = {}
    if supply_type == "unlimited":
        # Unlimited supply: q_{ki} = d_k for all k, i in M_k
        for k in range(n_products):
            supplies[k] = {}
            for i in product_markets[k]:
                supplies[k][i] = demands[k]
    else:
        for key, val in data["supplies"].items():
            parts = key.split(",")
            product = int(parts[0])
            market = int(parts[1])
            if product not in supplies:
                supplies[product] = {}
            supplies[product][market] = val

    return {
        "n_vertices": n_vertices,
        "n_markets": n_markets,
        "n_products": n_products,
        "depot": depot,
        "supply_type": supply_type,
        "V": V,
        "M": M,
        "E": E,
        "c": c,
        "demands": demands,
        "product_markets": product_markets,
        "purchase_costs": purchase_costs,
        "supplies": supplies,
    }


def compute_mandatory_sets(inst):
    """
    Compute M* (mandatory markets) and K* (products without market choice).
    M* = {v_0} union {v_i in M : exists p_k in K such that sum_{v_j in M_k \\ {v_i}} q_{kj} < d_k}
    K* = {p_k in K : sum_{v_i in M_k} q_{ki} = d_k}
    """
    depot = inst["depot"]
    n_products = inst["n_products"]
    demands = inst["demands"]
    product_markets = inst["product_markets"]
    supplies = inst["supplies"]

    M_star = {depot}
    for i in inst["M"]:
        for k in range(n_products):
            if i in product_markets[k]:
                total_without_i = sum(
                    supplies[k].get(j, 0)
                    for j in product_markets[k]
                    if j != i
                )
                if total_without_i < demands[k]:
                    M_star.add(i)
                    break

    K_star = set()
    for k in range(n_products):
        total = sum(supplies[k].get(i, 0) for i in product_markets[k])
        if total == demands[k]:
            K_star.add(k)

    return M_star, K_star


def find_connected_components(vertices, edges_used):
    """Find connected components given a set of vertices and edges."""
    adj = {v: [] for v in vertices}
    for (i, j) in edges_used:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    visited = set()
    components = []
    for v in vertices:
        if v not in visited:
            comp = []
            stack = [v]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for w in adj[u]:
                    if w not in visited:
                        stack.append(w)
            components.append(set(comp))
    return components


def solve_tpp(inst, time_limit):
    """Solve the TPP using Gurobi with branch-and-cut."""
    V = inst["V"]
    M = inst["M"]
    E = inst["E"]
    c = inst["c"]
    n_products = inst["n_products"]
    demands = inst["demands"]
    product_markets = inst["product_markets"]
    purchase_costs = inst["purchase_costs"]
    supplies = inst["supplies"]
    depot = inst["depot"]

    M_star, K_star = compute_mandatory_sets(inst)

    # Create Gurobi model
    model = gp.Model("TPP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)

    # Decision variables
    # x[e] = 1 if edge e is in the solution
    x = {}
    for e in E:
        x[e] = model.addVar(vtype=GRB.BINARY, name=f"x_{e[0]}_{e[1]}")

    # y[i] = 1 if market i is visited
    y = {}
    for i in V:
        if i in M_star:
            y[i] = model.addVar(vtype=GRB.BINARY, lb=1.0, ub=1.0, name=f"y_{i}")
        elif i == depot:
            y[i] = model.addVar(vtype=GRB.BINARY, lb=1.0, ub=1.0, name=f"y_{i}")
        else:
            y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # z[k][i] = amount of product k purchased at market i
    z = {}
    for k in range(n_products):
        z[k] = {}
        for i in product_markets[k]:
            if k in K_star:
                # Fixed: z_{ki} = q_{ki}
                z[k][i] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=supplies[k][i],
                    ub=supplies[k][i],
                    name=f"z_{k}_{i}",
                )
            else:
                z[k][i] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=supplies[k].get(i, 0),
                    name=f"z_{k}_{i}",
                )

    model.update()

    # Objective (1): min sum c_e * x_e + sum b_{ki} * z_{ki}
    obj = gp.LinExpr()
    for e in E:
        obj += c[e] * x[e]
    for k in range(n_products):
        for i in product_markets[k]:
            b_ki = purchase_costs.get(k, {}).get(i, 0)
            obj += b_ki * z[k][i]
    model.setObjective(obj, GRB.MINIMIZE)

    # Helper: edges incident to vertex i -> delta({v_i})
    def delta_vertex(i):
        return [e for e in E if i in e]

    # Helper: edges in delta(S) for a subset S
    def delta_set(S):
        S_set = set(S)
        return [e for e in E if (e[0] in S_set) != (e[1] in S_set)]

    # Constraint (2): degree constraints
    for i in V:
        inc_edges = delta_vertex(i)
        model.addConstr(
            gp.quicksum(x[e] for e in inc_edges) == 2 * y[i],
            name=f"degree_{i}",
        )

    # Constraint (4): demand satisfaction
    for k in range(n_products):
        model.addConstr(
            gp.quicksum(z[k][i] for i in product_markets[k]) == demands[k],
            name=f"demand_{k}",
        )

    # Constraint (5): products can only be purchased at visited markets
    for k in range(n_products):
        for i in product_markets[k]:
            q_ki = supplies[k].get(i, 0)
            model.addConstr(z[k][i] <= q_ki * y[i], name=f"supply_{k}_{i}")

    # Valid inequality (10): x_{v0,vj} <= y_j
    for j in M:
        e = (min(depot, j), max(depot, j))
        if e in x:
            model.addConstr(x[e] <= y[j], name=f"trivial_{j}")

    # Valid inequality (14) with S = M_k for each product k
    # sum_{e in delta(S)} x_e >= 2 for S = M_k
    # Condition: sum_{v_i in M_k \ S} q_{ki} < d_k, which with S = M_k gives
    # sum over empty set = 0 < d_k, so the condition always holds.
    # This ensures the tour must enter/leave the set of markets selling product k.
    for k in range(n_products):
        Mk = set(product_markets[k])
        S = Mk
        if S:
            delta_S = delta_set(S)
            if delta_S:
                model.addConstr(
                    gp.quicksum(x[e] for e in delta_S) >= 2,
                    name=f"cover_{k}",
                )

    # Callback for lazy connectivity constraints (3)
    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(x)
            y_vals = model.cbGetSolution(y)

            # Find edges used in the solution
            edges_used = [e for e in E if x_vals[e] > 0.5]
            # Find visited vertices
            visited = [v for v in V if y_vals[v] > 0.5]

            if len(visited) < 2:
                return

            # Find connected components among visited vertices
            components = find_connected_components(visited, edges_used)

            # If all visited vertices are connected (single component with depot), OK
            if len(components) == 1 and depot in components[0]:
                return

            # For each component not containing the depot, add subtour elimination
            for comp in components:
                if depot not in comp:
                    # Constraint (3): sum_{e in delta(S)} x_e >= 2 y_i for all v_i in S
                    S = comp
                    delta_S = delta_set(S)
                    for i in S:
                        if i in y:
                            model.cbLazy(
                                gp.quicksum(x[e] for e in delta_S) >= 2 * y[i]
                            )

    # Solve
    model.optimize(subtour_callback)

    # Extract solution
    result = {"objective_value": None, "status": None}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract visited markets
        visited_markets = [i for i in M if y[i].X > 0.5]
        result["visited_markets"] = visited_markets

        # Extract tour edges
        tour_edges = [(e[0], e[1]) for e in E if x[e].X > 0.5]
        result["tour_edges"] = tour_edges

        # Extract purchase assignments
        purchases = {}
        for k in range(n_products):
            purchases[k] = {}
            for i in product_markets[k]:
                val = z[k][i].X
                if val > 1e-6:
                    purchases[k][i] = val
        result["purchases"] = {
            str(k): {str(i): v for i, v in mk.items()}
            for k, mk in purchases.items()
        }

        # Compute routing cost and purchase cost breakdown
        routing_cost = sum(c[e] * x[e].X for e in E)
        purchase_cost = sum(
            purchase_costs.get(k, {}).get(i, 0) * z[k][i].X
            for k in range(n_products)
            for i in product_markets[k]
        )
        result["routing_cost"] = routing_cost
        result["purchase_cost"] = purchase_cost

        # Extract tour order
        if tour_edges:
            adj = {}
            for (i, j) in tour_edges:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
            tour = [depot]
            prev = -1
            cur = depot
            while True:
                neighbors = adj.get(cur, [])
                next_v = None
                for n in neighbors:
                    if n != prev:
                        next_v = n
                        break
                if next_v is None or next_v == depot:
                    break
                tour.append(next_v)
                prev = cur
                cur = next_v
            result["tour"] = tour
    else:
        result["status"] = "infeasible"
        result["objective_value"] = None

    result["solve_time"] = model.Runtime
    result["mip_gap"] = model.MIPGap if model.SolCount > 0 else None
    result["num_nodes"] = int(model.NodeCount)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for the Undirected Traveling Purchaser Problem"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to the JSON instance file"
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path for the output solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum solver runtime in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    inst = load_instance(args.instance_path)
    print(
        f"Instance: {inst['n_vertices']} vertices, {inst['n_markets']} markets, "
        f"{inst['n_products']} products, supply_type={inst['supply_type']}"
    )

    M_star, K_star = compute_mandatory_sets(inst)
    print(f"Mandatory markets |M*| = {len(M_star)}, Fixed products |K*| = {len(K_star)}")

    print(f"Solving with time limit = {args.time_limit} seconds...")
    result = solve_tpp(inst, args.time_limit)

    print(f"\n--- Results ---")
    print(f"Status: {result['status']}")
    print(f"Objective value: {result['objective_value']}")
    if result["objective_value"] is not None:
        print(f"Routing cost: {result.get('routing_cost', 'N/A')}")
        print(f"Purchase cost: {result.get('purchase_cost', 'N/A')}")
        print(f"Visited markets: {result.get('visited_markets', [])}")
        print(f"Tour: {result.get('tour', [])}")
    print(f"Solve time: {result['solve_time']:.2f}s")
    print(f"MIP gap: {result.get('mip_gap', 'N/A')}")
    print(f"Nodes explored: {result.get('num_nodes', 'N/A')}")

    # Save solution
    output = {
        "objective_value": result["objective_value"],
        "status": result["status"],
        "solve_time": result["solve_time"],
        "mip_gap": result.get("mip_gap"),
        "num_nodes": result.get("num_nodes"),
        "visited_markets": result.get("visited_markets", []),
        "tour": result.get("tour", []),
        "tour_edges": result.get("tour_edges", []),
        "purchases": result.get("purchases", {}),
        "routing_cost": result.get("routing_cost"),
        "purchase_cost": result.get("purchase_cost"),
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSolution saved to: {args.solution_path}")


if __name__ == "__main__":
    main()
