# Original Formulation: Split-Delivery Vehicle Routing Problem with Time Windows (SDVRPTW) — Arc-Flow Formulation

*Source: Branch-and-Price-and-Cut for the Split-Delivery Vehicle Routing Problem with Time Windows, Guy Desaulniers, 2010.*

## Sets and Parameters

- $\mathcal{N} = \{1,\dots,n\}$: set of $n$ customers.

- $\mathcal{V} = \mathcal{N} \cup \{0, n+1\}$: node set, where $0$ and $n+1$ represent the depot at the start and end of the planning horizon.

- $\mathcal{A} \subset \mathcal{V}\times\mathcal{V}$: set of arcs; $(i,j) \in \mathcal{A}$ if $e_i + t_{ij} \le l_j$ (with the depot arcs $(0,j)$ and $(j,n+1)$ defined analogously, but not $(n+1,0)$).

- $\mathcal{F}$: set of available (identical) vehicles, each with capacity $Q$.

- $\mathcal{V}^{+}(i) = \{j \in \mathcal{V} : (i,j) \in \mathcal{A}\}$: successor set of $i$.

- $\mathcal{V}^{-}(i) = \{j \in \mathcal{V} : (j,i) \in \mathcal{A}\}$: predecessor set of $i$.

- $c_{ij} \ge 0$: cost of arc $(i,j)$.

- $t_{ij} \ge 0$: travel time of arc $(i,j)$ (includes service time at $i$ if any).

- $d_i$: demand of customer $i$; $\bar{d}_i = \min\{d_i, Q\}$.

- $[e_i, l_i]$: time window at node $i$.

- $k^{C}_{i}$: minimum number of vehicles needed to service customer $i$ (with $\mathcal{U} = \{i\}$) respecting only the vehicle-capacity constraints.

- $k^{C}(\mathcal{N}) = \lceil \sum_{i\in\mathcal{N}} d_i / Q \rceil$: minimum number of vehicles needed to service all customers under vehicle-capacity constraints.

- $\mathcal{P}(\mathcal{N})$: collection of subsets $\mathcal{U} \subseteq \mathcal{N}$ with $|\mathcal{U}| \ge 2$ and $k_{\mathcal{U}} > 1$ (here $k_{\mathcal{U}} = \max\{k^{C}_{\mathcal{U}}, k^{T}_{\mathcal{U}}\}$).

- $\mathcal{A}^{-}(\mathcal{U}) = \{(i,j)\in\mathcal{A} : i \in \mathcal{V}\setminus\mathcal{U},\; j \in \mathcal{U}\}$: arcs entering $\mathcal{U}$.

- $\mathcal{A}^{*}(\mathcal{N}) \subseteq \mathcal{A}$: a chosen subset containing, for each pair of reverse arcs $(i,j),(j,i) \in \mathcal{A}(\mathcal{N})$, exactly one of them.

## Decision Variables

- $x^{f}_{ij} \in \{0,1\}$, $\forall f \in \mathcal{F}, (i,j) \in \mathcal{A}$: 1 if vehicle $f$ uses arc $(i,j)$, 0 otherwise.

- $\delta^{f}_{i} \ge 0$, $\forall f \in \mathcal{F}, i \in \mathcal{N}$: quantity delivered by vehicle $f$ to customer $i$.

- $s^{f}_{i} \in \mathbb{R}$, $\forall f \in \mathcal{F}, i \in \mathcal{V}$: visit start time of vehicle $f$ at node $i$ (value is irrelevant if $f$ does not visit $i$).

- $H \ge 0$: total number of vehicles used (integer).

## Objective

$$\begin{align}
\min \quad & \sum_{f \in \mathcal{F}} \sum_{(i,j) \in \mathcal{A}} c_{ij}\, x^{f}_{ij} \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{f \in \mathcal{F}} \delta^{f}_{i} &\ge d_i,
  & \forall i \in \mathcal{N}, \tag{2}\\
\sum_{f \in \mathcal{F}} \sum_{j \in \mathcal{V}^{+}(i)} x^{f}_{ij} &\ge k^{C}_{i},
  & \forall i \in \mathcal{N}, \tag{3}\\
\sum_{f \in \mathcal{F}} \sum_{j \in \mathcal{V}^{+}(0)} x^{f}_{0j} &= H, \tag{4}\\
H &\in \bigl[k^{C}(\mathcal{N}),\; |\mathcal{F}|\bigr],\; H\ \text{integer}, \tag{5}\\
\sum_{f \in \mathcal{F}} \sum_{(i,j) \in \mathcal{A}^{-}(\mathcal{U})} x^{f}_{ij} &\ge k_{\mathcal{U}},
  & \forall \mathcal{U} \in \mathcal{P}(\mathcal{N}), \tag{6}\\
\sum_{f \in \mathcal{F}} \sum_{(i,j) \in \mathcal{A}^{*}_{i'j'}} x^{f}_{ij} &\le 1,
  & \forall (i',j') \in \mathcal{A}^{*}(\mathcal{N}), \tag{7}\\
\sum_{j \in \mathcal{V}^{+}(0)} x^{f}_{0j} &= 1,
  & \forall f \in \mathcal{F}, \tag{8}\\
\sum_{j \in \mathcal{V}^{+}(i)} x^{f}_{ij} - \sum_{j \in \mathcal{V}^{-}(i)} x^{f}_{ji} &= 0,
  & \forall f \in \mathcal{F},\; i \in \mathcal{N}, \tag{9}\\
\sum_{i \in \mathcal{V}^{-}(n+1)} x^{f}_{i,n+1} &= 1,
  & \forall f \in \mathcal{F}, \tag{10}\\
x^{f}_{ij}\bigl(s^{f}_{i} + t_{ij} - s^{f}_{j}\bigr) &\le 0,
  & \forall f \in \mathcal{F},\; (i,j) \in \mathcal{A}, \tag{11}\\
e_i \le s^{f}_{i} &\le l_i,
  & \forall f \in \mathcal{F},\; i \in \mathcal{V}, \tag{12}\\
\sum_{i \in \mathcal{N}} \delta^{f}_{i} &\le Q,
  & \forall f \in \mathcal{F}, \tag{13}\\
0 \le \delta^{f}_{i} &\le \bar{d}_i \sum_{j \in \mathcal{V}^{+}(i)} x^{f}_{ij},
  & \forall f \in \mathcal{F},\; i \in \mathcal{N}, \tag{14}\\
x^{f}_{ij} &\in \{0,1\},
  & \forall f \in \mathcal{F},\; (i,j) \in \mathcal{A}. \tag{15}
\end{align}$$

The objective (1) minimizes the total travel cost. Constraints (2) ensure that the demand of each customer is fulfilled. Constraints (3)–(6) are redundant inequalities used to strengthen the LP relaxation: (3) imposes a minimum number of visits at each customer, (4)–(5) compute and bound the number of vehicles used, and (6) are the $k$-path inequalities. Constraints (7) restrict the feasible space while keeping at least one optimal integer solution. Constraints (8)–(10) define a path for each vehicle from $0$ to $n+1$. Constraints (11) ensure that the customer time windows are respected whenever an arc is used; **they are the bilinear coupling $x^{f}_{ij}(s^{f}_{i} + t_{ij} - s^{f}_{j}) \le 0$ in its original nonlinear form**. Constraints (12) enforce time windows on the $s$ variables, (13) are vehicle-capacity constraints, and (14) limit the quantity delivered to a customer and force it to zero if the route of the vehicle does not visit that customer. Finally, binary requirements on $x^{f}_{ij}$ are given in (15).
