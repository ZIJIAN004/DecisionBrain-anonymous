#!/usr/bin/env python3
"""
Gurobi implementation of the Multi-Commodity (MC) formulation for the
Two-Stage Stochastic Three-Level Lot Sizing and Replenishment Problem
with a Distribution Structure (2S-3LSPD).

Reference: Gruson, Cordeau, and Jans (2021), European Journal of Operational Research.

This implements the deterministic equivalent (scenario-based) MC formulation
described in Equations 11-19 of the paper, which can be solved directly by
a general-purpose MIP solver.
"""

import argparse
import json
import sys

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
    """Load problem instance from JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build and solve the MC formulation (Eqs. 11-19) using Gurobi.

    Sets and indices (from paper):
      F = {plant} ∪ W ∪ R  : set of all facilities
      T : set of time periods (1-indexed in the model)
      Ω : set of demand scenarios

    Decision variables:
      y[i,t] ∈ {0,1} : 1 iff there is production/order at facility i in period t
      x0[r,k,t,ω] : quantity produced at plant in period k to satisfy d_{r,t,ω}
      x1[r,k,t,ω] : quantity ordered at warehouse level in period k to satisfy d_{r,t,ω}
      x2[r,k,t,ω] : quantity ordered at retailer level in period k to satisfy d_{r,t,ω}
      σ0[r,k,t,ω] : stock at plant at end of period k for commodity d_{r,t,ω}
      σ1[r,k,t,ω] : stock at warehouse at end of period k for commodity d_{r,t,ω}
      σ2[r,k,t,ω] : stock at retailer at end of period k for commodity d_{r,t,ω}
    """
    # ------------------------------------------------------------------
    # Extract instance data
    # ------------------------------------------------------------------
    T = data["num_periods"]          # number of time periods
    nW = data["num_warehouses"]      # number of warehouses
    nR = data["num_retailers"]       # number of retailers
    nS = data["num_scenarios"]       # number of scenarios (|Ω|)

    # retailer_to_warehouse[r] gives the warehouse index (0-based) for retailer r
    r2w = data["retailer_to_warehouse"]

    # Scenario probabilities
    p_omega = data["scenario_probabilities"]  # length nS

    # Holding costs. Per math model (Eq. 11) plant cost is hc_{pk} and warehouse
    # cost is hc_{W(r),k}; normalize possibly scalar instance data to the fully
    # indexed form so the objective can use hc_plant[k] and hc_warehouse[w][k].
    hc_plant = data["holding_cost_plant"]
    if not isinstance(hc_plant, (list, tuple)):
        hc_plant = [hc_plant] * T
    hc_warehouse = data["holding_cost_warehouse"]
    if not isinstance(hc_warehouse, (list, tuple)):
        hc_warehouse = [[hc_warehouse] * T for _ in range(nW)]
    elif hc_warehouse and not isinstance(hc_warehouse[0], (list, tuple)):
        hc_warehouse = [[v] * T for v in hc_warehouse]
    hc_retailers = data["holding_costs_retailers"]  # list of length nR

    # Setup costs: indexed [period] for plant, [warehouse][period] for warehouses,
    #              [retailer][period] for retailers
    sc_plant = data["setup_costs_plant"]          # list of length T
    sc_warehouses = data["setup_costs_warehouses"]  # list of nW lists, each length T
    sc_retailers = data["setup_costs_retailers"]    # list of nR lists, each length T

    # Demands: demands[scenario][retailer][period]
    demands = data["demands"]  # nS x nR x T

    # ------------------------------------------------------------------
    # Create Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("2S-3LSPD_MC")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 1e-6)
    # INFERRED ASSUMPTION: Use single thread to match paper's setting
    # (paper uses CPLEX with parallel mode OFF)
    model.setParam("Threads", 1)

    # Periods: 1..T (0-indexed internally as 0..T-1)
    periods = range(T)
    retailers = range(nR)
    warehouses = range(nW)
    scenarios = range(nS)

    # ------------------------------------------------------------------
    # Facility indexing for y variables
    # We use a single set of y variables indexed by (facility_id, period).
    # facility_id 0 = plant
    # facility_id 1..nW = warehouses (warehouse w has id w+1)
    # facility_id nW+1..nW+nR = retailers (retailer r has id nW+1+r)
    # ------------------------------------------------------------------
    PLANT_ID = 0

    def warehouse_id(w):
        return 1 + w

    def retailer_id(r):
        return 1 + nW + r

    nF = 1 + nW + nR  # total number of facilities

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # y[i,t] ∈ {0,1}: setup variable for facility i in period t
    y = {}
    for i in range(nF):
        for t in periods:
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")

    # For the MC formulation, flow/stock variables are indexed by (r, k, t, omega)
    # where k <= t (period k, demand period t)
    # x0[r,k,t,omega]: production at plant in period k for d_{r,t,omega}
    # x1[r,k,t,omega]: order at warehouse in period k for d_{r,t,omega}
    # x2[r,k,t,omega]: order at retailer in period k for d_{r,t,omega}
    # s0[r,k,t,omega]: stock at plant at end of period k for d_{r,t,omega}
    # s1[r,k,t,omega]: stock at warehouse at end of period k for d_{r,t,omega}
    # s2[r,k,t,omega]: stock at retailer at end of period k for d_{r,t,omega}

    x0 = {}
    x1 = {}
    x2 = {}
    s0 = {}
    s1 = {}
    s2 = {}

    for omega in scenarios:
        for r in retailers:
            for t in periods:
                for k in range(t + 1):  # k = 0, 1, ..., t (0-indexed: k <= t)
                    x0[r, k, t, omega] = model.addVar(lb=0.0, name=f"x0_{r}_{k}_{t}_{omega}")
                    x1[r, k, t, omega] = model.addVar(lb=0.0, name=f"x1_{r}_{k}_{t}_{omega}")
                    x2[r, k, t, omega] = model.addVar(lb=0.0, name=f"x2_{r}_{k}_{t}_{omega}")
                    s0[r, k, t, omega] = model.addVar(lb=0.0, name=f"s0_{r}_{k}_{t}_{omega}")
                    s1[r, k, t, omega] = model.addVar(lb=0.0, name=f"s1_{r}_{k}_{t}_{omega}")
                    s2[r, k, t, omega] = model.addVar(lb=0.0, name=f"s2_{r}_{k}_{t}_{omega}")

    model.update()

    # ------------------------------------------------------------------
    # Objective function (Eq. 11)
    # Min Σ_t ( Σ_{i∈F} sc_{it} y_{it}
    #          + Σ_ω p_ω Σ_r Σ_{k≤t} (hc_p * s0 + hc_{W(r)} * s1 + hc_r * s2) )
    # ------------------------------------------------------------------
    obj = gp.LinExpr()

    # Setup costs
    for t in periods:
        # Plant setup cost
        obj += sc_plant[t] * y[PLANT_ID, t]
        # Warehouse setup costs
        for w in warehouses:
            obj += sc_warehouses[w][t] * y[warehouse_id(w), t]
        # Retailer setup costs
        for r in retailers:
            obj += sc_retailers[r][t] * y[retailer_id(r), t]

    # Expected holding costs
    for omega in scenarios:
        pw = p_omega[omega]
        for r in retailers:
            w_r = r2w[r]  # warehouse index for retailer r
            hc_r = hc_retailers[r]
            for t in periods:
                for k in range(t + 1):
                    # Holding cost at plant in period k
                    obj += pw * hc_plant[k] * s0[r, k, t, omega]
                    # Holding cost at warehouse in period k
                    obj += pw * hc_warehouse[w_r][k] * s1[r, k, t, omega]
                    # Holding cost at retailer in period k
                    obj += pw * hc_r * s2[r, k, t, omega]

    model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    for omega in scenarios:
        for r in retailers:
            w_r = r2w[r]          # warehouse index for this retailer
            wid = warehouse_id(w_r)
            rid = retailer_id(r)
            d_rt_omega = demands[omega][r]  # list of length T

            for t in periods:
                d_val = d_rt_omega[t]

                for k in range(t + 1):
                    # Initial stock = 0 (Section 5.1: no initial inventory)
                    # σ^{lr}_{-1,t,ω} = 0 (conceptually: k-1 < 0 means no prior stock)
                    s0_prev = s0[r, k - 1, t, omega] if k > 0 else 0.0
                    s1_prev = s1[r, k - 1, t, omega] if k > 0 else 0.0
                    s2_prev = s2[r, k - 1, t, omega] if k > 0 else 0.0

                    # Eq. 12: Plant inventory balance
                    # x1_{kt} + s0_{kt} = s0_{k-1,t} + x0_{kt}
                    model.addConstr(
                        x1[r, k, t, omega] + s0[r, k, t, omega]
                        == s0_prev + x0[r, k, t, omega],
                        name=f"eq12_r{r}_k{k}_t{t}_w{omega}"
                    )

                    # Eq. 13: Warehouse inventory balance
                    # x2_{kt} + s1_{kt} = s1_{k-1,t} + x1_{kt}
                    model.addConstr(
                        x2[r, k, t, omega] + s1[r, k, t, omega]
                        == s1_prev + x1[r, k, t, omega],
                        name=f"eq13_r{r}_k{k}_t{t}_w{omega}"
                    )

                    # Eq. 14: Retailer inventory balance
                    # δ_{kt} * d_{rtω} + (1-δ_{kt}) * s2_{kt} = s2_{k-1,t} + x2_{kt}
                    delta_kt = 1.0 if k == t else 0.0
                    if k == t:
                        # When k=t: demand is consumed
                        model.addConstr(
                            d_val + 0 == s2_prev + x2[r, k, t, omega],
                            name=f"eq14_r{r}_k{k}_t{t}_w{omega}"
                        )
                    else:
                        # When k<t: stock is carried forward
                        model.addConstr(
                            s2[r, k, t, omega] == s2_prev + x2[r, k, t, omega],
                            name=f"eq14_r{r}_k{k}_t{t}_w{omega}"
                        )

                    # Eq. 15: Setup forcing at plant
                    # x0_{kt} ≤ d_{rtω} * y_{p,k}
                    model.addConstr(
                        x0[r, k, t, omega] <= d_val * y[PLANT_ID, k],
                        name=f"eq15_r{r}_k{k}_t{t}_w{omega}"
                    )

                    # Eq. 16: Setup forcing at warehouse
                    # x1_{kt} ≤ d_{rtω} * y_{W(r),k}
                    model.addConstr(
                        x1[r, k, t, omega] <= d_val * y[wid, k],
                        name=f"eq16_r{r}_k{k}_t{t}_w{omega}"
                    )

                    # Eq. 17: Setup forcing at retailer
                    # x2_{kt} ≤ d_{rtω} * y_{r,k}
                    model.addConstr(
                        x2[r, k, t, omega] <= d_val * y[rid, k],
                        name=f"eq17_r{r}_k{k}_t{t}_w{omega}"
                    )

    # ------------------------------------------------------------------
    # Initial setups imposed (Section 5.1, math_model assumption #5):
    # "Initial setups are imposed: there must be production and an order
    # placed by each warehouse and retailer to satisfy the demand of the
    # first period for each retailer." This is a hard model assumption
    # in the paper, not a per-instance switch — enforce unconditionally.
    # ------------------------------------------------------------------
    for i in range(nF):
        model.addConstr(y[i, 0] == 1, name=f"init_setup_{i}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    solution = {}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal

        # ---- First-stage decisions (binary setup) ----
        y_sol = {}
        for i in range(nF):
            for t in periods:
                val = y[i, t].X
                y_sol[f"y_{i}_{t}"] = 1 if val > 0.5 else 0
        solution["setup_variables"] = y_sol

        # ---- Aggregate per-(period, scenario) recourse quantities ----
        # The MC formulation uses disaggregated (r, k, t, omega)-indexed
        # variables tied to specific retailer-period demands. We project
        # them onto the original aggregated solution structure (per-period
        # / per-scenario quantities at plant / each warehouse / each
        # retailer) so feasibility_check verifies the original problem
        # without depending on the MC disaggregation.
        production_plant = {}      # (t, omega) -> float
        delivery_warehouse = {}    # (w, t, omega) -> float
        delivery_retailer = {}     # (r, t, omega) -> float
        inventory_plant = {}       # (t, omega) -> float
        inventory_warehouse = {}   # (w, t, omega) -> float
        inventory_retailer = {}    # (r, t, omega) -> float

        for omega in scenarios:
            for k in periods:
                production_plant[(k, omega)] = sum(
                    x0[r, k, t, omega].X
                    for r in retailers for t in periods if t >= k
                )
                inventory_plant[(k, omega)] = sum(
                    s0[r, k, t, omega].X
                    for r in retailers for t in periods if t >= k
                )
                for w in warehouses:
                    rs_under_w = [r for r in retailers if r2w[r] == w]
                    delivery_warehouse[(w, k, omega)] = sum(
                        x1[r, k, t, omega].X
                        for r in rs_under_w for t in periods if t >= k
                    )
                    inventory_warehouse[(w, k, omega)] = sum(
                        s1[r, k, t, omega].X
                        for r in rs_under_w for t in periods if t >= k
                    )
                for r in retailers:
                    delivery_retailer[(r, k, omega)] = sum(
                        x2[r, k, t, omega].X
                        for t in periods if t >= k
                    )
                    inventory_retailer[(r, k, omega)] = sum(
                        s2[r, k, t, omega].X
                        for t in periods if t >= k
                    )

        solution["production_plant"] = {
            f"{t}_{omega}": float(v)
            for (t, omega), v in production_plant.items()
        }
        solution["delivery_warehouse"] = {
            f"{w}_{t}_{omega}": float(v)
            for (w, t, omega), v in delivery_warehouse.items()
        }
        solution["delivery_retailer"] = {
            f"{r}_{t}_{omega}": float(v)
            for (r, t, omega), v in delivery_retailer.items()
        }
        solution["inventory_plant"] = {
            f"{t}_{omega}": float(v)
            for (t, omega), v in inventory_plant.items()
        }
        solution["inventory_warehouse"] = {
            f"{w}_{t}_{omega}": float(v)
            for (w, t, omega), v in inventory_warehouse.items()
        }
        solution["inventory_retailer"] = {
            f"{r}_{t}_{omega}": float(v)
            for (r, t, omega), v in inventory_retailer.items()
        }

        # Extract solver statistics
        solution["solver_status"] = model.Status
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        solution["best_bound"] = model.ObjBound if model.SolCount > 0 else None
        solution["num_variables"] = model.NumVars
        solution["num_constraints"] = model.NumConstrs
        solution["solve_time"] = model.Runtime
    else:
        solution["objective_value"] = None
        solution["solver_status"] = model.Status
        solution["error"] = "No feasible solution found within time limit."

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve the 2S-3LSPD using the MC formulation with Gurobi."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the problem instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path where the final solution JSON file must be written."
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Maximum solver runtime in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    # Build and solve
    solution = build_and_solve(data, args.time_limit)

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
