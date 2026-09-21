"""
Gurobi implementation of DEF_V (Deterministic Equivalent Formulation with Variable Allocation)
using the flow-based mixed-integer linear reformulation of Ernst & Krishnamoorthy (1996).

Source: Rostami, Kämmerling, Naoum-Sawaya, Buchheim, Clausen (2021)
        "Stochastic single-allocation hub location"
        European Journal of Operational Research

This program solves the deterministic equivalent of SP_V (stochastic SAHLP with variable
allocation) by linearizing the quadratic inter-hub transport terms using flow variables
y^s_{ikl}, following the approach described in Section 2 (SAHLP-flow) extended per scenario.

Supports three problem variants:
  - SAHLP: Single Allocation Hub Location Problem (with fixed hub setup costs)
  - SApHMP: Single Allocation p-Hub Median Problem (exactly p hubs, no setup costs)
  - CSAHLP: Capacitated SAHLP (with hub capacity constraints)
"""

import json
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Solve the stochastic SAHLP (DEF_V) using flow-based linearization with Gurobi."
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path for the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # =========================================================================
    # Load instance data
    # =========================================================================
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    num_scenarios = instance['num_scenarios']
    chi = instance['cost_parameters']['chi']
    alpha = instance['cost_parameters']['alpha']
    delta = instance['cost_parameters']['delta']
    d = instance['distances']
    f_costs = instance['fixed_costs']
    p_hubs = instance.get('p_hubs')
    hub_capacities = instance.get('hub_capacities')
    variant = instance['problem_info']['variant']
    scenarios = instance['scenarios']

    N = range(n)
    S = range(num_scenarios)

    # =========================================================================
    # Precompute scenario-dependent parameters
    # =========================================================================
    w_s = []   # w_s[s][i][j]: flow from i to j under scenario s
    O_s = []   # O_s[s][i]: total outgoing flow from node i under scenario s
    D_s = []   # D_s[s][i]: total incoming flow to node i under scenario s
    p_s = []   # p_s[s]: probability of scenario s
    c_s = []   # c_s[s][i][k]: collection/distribution cost coefficient

    for s in S:
        ws = scenarios[s]['demands']
        w_s.append(ws)
        p_s.append(scenarios[s]['probability'])

        Os = [sum(ws[i][j] for j in N) for i in N]
        Ds = [sum(ws[j][i] for j in N) for i in N]
        O_s.append(Os)
        D_s.append(Ds)

        # c^s_{ik} = d_{ik} * (chi * O^s_i + delta * D^s_i)
        cs = [[d[i][k] * (chi * Os[i] + delta * Ds[i]) for k in N] for i in N]
        c_s.append(cs)

    # =========================================================================
    # Build Gurobi model: DEF_V with flow-based linearization
    # =========================================================================
    model = gp.Model("DEF_V_flow")
    model.setParam("TimeLimit", args.time_limit)
    # Paper uses single thread for experiments
    model.setParam("Threads", 1)

    # ----- First-stage decision variables -----
    # z_k in {0,1}: 1 if hub is opened at node k
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # ----- Second-stage decision variables (per scenario) -----
    # x^s_{ik} in {0,1}: allocation of node i to hub k (for i != k)
    x = {}
    for s in S:
        for i in N:
            for k in N:
                if i != k:
                    x[s, i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}_{k}")

    # ----- Flow variables for linearization -----
    # y^s_{ikl} >= 0: total flow originating at node i, routed via hub k then hub l
    y = {}
    for s in S:
        for i in N:
            for k in N:
                for l in N:
                    y[s, i, k, l] = model.addVar(
                        lb=0.0, vtype=GRB.CONTINUOUS,
                        name=f"y_{s}_{i}_{k}_{l}"
                    )

    model.update()

    # ----- Helper: X^s_{ik} = z_i if i==k, x^s_{ik} if i!=k -----
    # This captures the convention that x_{kk} = z_k (a hub is allocated to itself)
    def X(s, i, k):
        if i == k:
            return z[i]
        else:
            return x[s, i, k]

    # =========================================================================
    # Objective function
    # =========================================================================
    # Hub setup costs (not used for SApHMP)
    if variant == "SApHMP":
        hub_cost = 0
    else:
        hub_cost = gp.quicksum(f_costs[k] * z[k] for k in N)

    # Collection/distribution costs: sum_s p_s sum_{i,k: i!=k} c^s_{ik} x^s_{ik}
    cd_cost = gp.quicksum(
        p_s[s] * c_s[s][i][k] * x[s, i, k]
        for s in S for i in N for k in N if i != k
    )

    # Inter-hub transfer costs (linearized): sum_s p_s sum_{i,k,l} alpha d_{kl} y^s_{ikl}
    transfer_cost = gp.quicksum(
        p_s[s] * alpha * d[k][l] * y[s, i, k, l]
        for s in S for i in N for k in N for l in N
    )

    model.setObjective(hub_cost + cd_cost + transfer_cost, GRB.MINIMIZE)

    # =========================================================================
    # Constraints
    # =========================================================================

    # (25) Single allocation: sum_{k!=i} x^s_{ik} = 1 - z_i, for all i, s
    for s in S:
        for i in N:
            model.addConstr(
                gp.quicksum(x[s, i, k] for k in N if k != i) == 1 - z[i],
                name=f"alloc_{s}_{i}"
            )

    # (26) Linking: x^s_{ik} <= z_k, for all i, k (i!=k), s
    for s in S:
        for i in N:
            for k in N:
                if i != k:
                    model.addConstr(
                        x[s, i, k] <= z[k],
                        name=f"link_{s}_{i}_{k}"
                    )

    # Hub count constraint depends on variant
    if variant == "SApHMP" and p_hubs is not None:
        # Exactly p hubs
        model.addConstr(
            gp.quicksum(z[k] for k in N) == p_hubs,
            name="p_hubs"
        )
    else:
        # (20) At least one hub
        model.addConstr(
            gp.quicksum(z[k] for k in N) >= 1,
            name="at_least_one_hub"
        )

    # CSAHLP capacity constraints (per scenario)
    if variant == "CSAHLP" and hub_capacities is not None:
        for s in S:
            for k in N:
                # sum_i O^s_i * X^s_{ik} <= Gamma_k * z_k
                model.addConstr(
                    gp.quicksum(O_s[s][i] * X(s, i, k) for i in N)
                    <= hub_capacities[k] * z[k],
                    name=f"cap_{s}_{k}"
                )

    # ----- Flow-based linearization constraints (per scenario) -----
    # These follow the pattern of SAHLP-flow (eqs 9-11) applied per scenario
    # with X^s_{ik} replacing x_{ik}.

    # Flow balance (eq 9 per scenario):
    # sum_l y^s_{ikl} - sum_l y^s_{ilk} = O^s_i * X^s_{ik} - sum_j w^s_{ij} * X^s_{jk}
    # for all i, k, s
    for s in S:
        for i in N:
            for k in N:
                lhs = (
                    gp.quicksum(y[s, i, k, l] for l in N)
                    - gp.quicksum(y[s, i, l, k] for l in N)
                )
                rhs = (
                    O_s[s][i] * X(s, i, k)
                    - gp.quicksum(w_s[s][i][j] * X(s, j, k) for j in N)
                )
                model.addConstr(lhs == rhs, name=f"flow_bal_{s}_{i}_{k}")

    # Flow bound (eq 10 per scenario):
    # sum_l y^s_{ikl} <= O^s_i * X^s_{ik}, for all i, k, s
    for s in S:
        for i in N:
            for k in N:
                model.addConstr(
                    gp.quicksum(y[s, i, k, l] for l in N)
                    <= O_s[s][i] * X(s, i, k),
                    name=f"flow_bnd_{s}_{i}_{k}"
                )

    # (y >= 0 is enforced by lb=0 in variable definition)

    # =========================================================================
    # Solve
    # =========================================================================
    model.optimize()

    # =========================================================================
    # Extract and save solution
    # =========================================================================
    solution = {}

    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['hubs'] = [k for k in N if z[k].X > 0.5]

        # Extract allocations per scenario
        allocations = {}
        for s in S:
            alloc_s = {}
            for i in N:
                if z[i].X > 0.5:
                    # Node i is a hub, allocated to itself
                    alloc_s[str(i)] = i
                else:
                    for k in N:
                        if k != i and x[s, i, k].X > 0.5:
                            alloc_s[str(i)] = k
                            break
            allocations[str(s)] = alloc_s
        solution['allocations'] = allocations

        # Raw solver values for the ORIGINAL DEF_V decision variables (z, x^s).
        # Flow variables y^s_{ikl} are an internal linearization auxiliary
        # (Ernst & Krishnamoorthy) and are NOT part of the original
        # formulation, so they are not exported.
        solution['z_values'] = [float(z[k].X) for k in N]
        solution['x_values'] = {
            str(s): {
                str(i): {
                    str(k): float(x[s, i, k].X)
                    for k in N if k != i
                }
                for i in N
            }
            for s in S
        }

        solution['status'] = model.Status
        solution['mip_gap'] = model.MIPGap if hasattr(model, 'MIPGap') else None
        solution['runtime'] = model.Runtime
        solution['node_count'] = int(model.NodeCount)
    else:
        solution['objective_value'] = None
        solution['status'] = model.Status
        solution['runtime'] = model.Runtime

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution['objective_value'] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Hubs: {solution.get('hubs', [])}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
