# Original Formulation: Stochastic Multi-commodity Capacitated Fixed-charge Network Design (MCFND)

*Source: A Stochastic Benders Decomposition Scheme for Large-Scale Stochastic Network Design, Dimitris Bertsimas, Ryan Cory-Wright, Jean Pauphilet, Periklis Petridis, 2024.*

## Sets and Indices

- $\mathcal{N}$: set of nodes of the capacitated directed network; index $n \in \mathcal{N}$.

- $\mathcal{E}$: set of edges (arcs) of the directed network $(\mathcal{N},\mathcal{E})$; index $(i,j) \in \mathcal{E}$.

- $\mathcal{K}$: index set of commodities to be shipped; index $k \in \mathcal{K}$.

- $\mathcal{R}$: set of historical demand observations (scenarios); index $r \in \mathcal{R}$.

## Parameters

- $\boldsymbol{A}$: flow conservation matrix (node–arc incidence matrix) of the network $(\mathcal{N},\mathcal{E})$.

- $u_{i,j}$: capacity of arc $(i,j) \in \mathcal{E}$ (total flow of all commodities combined).

- $d_n^{k,r}$: amount of commodity $k$ supplied (positive) or demanded (negative) at node $n$ in scenario $r$; $\boldsymbol{d}^{k,r}$ is the corresponding vector over nodes.

- $c_{i,j}$: fixed cost of activating (constructing) edge $(i,j) \in \mathcal{E}$.

- $f_{i,j}^{k}$: marginal transportation cost, i.e. per-unit cost of transporting commodity $k$ through edge $(i,j)$.

- $c_0$: fixed upper limit on the number of edges that may be activated.

- $\gamma > 0$: regularization parameter controlling the strongly quadratic penalty term in the objective.

## Decision Variables

- $z_{i,j} \in \{0,1\}$: binary design variable; $1$ if edge $(i,j)$ is activated, $0$ otherwise, for all $(i,j) \in \mathcal{E}$.

- $x_{i,j}^{k,r} \ge 0$: continuous flow variable; quantity of commodity $k$ routed on edge $(i,j)$ in scenario $r$, for all $(i,j) \in \mathcal{E},\, k \in \mathcal{K},\, r \in \mathcal{R}$; $\boldsymbol{x}^{k,r}$ is the corresponding vector over edges.

## Objective and Constraints

The complete optimization formulation for MCFND is Problem (1), page 4: $$\begin{align}
\min \quad
  & \sum_{(i,j) \in \mathcal{E}} c_{i,j}\, z_{i,j}
    + \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \sum_{(i,j) \in \mathcal{E}}
      \left(
        \sum_{k \in \mathcal{K}} f_{i,j}^{k}\, x_{i,j}^{k,r}
        + \frac{1}{2\gamma} \Big( \sum_{k \in \mathcal{K}} x_{i,j}^{k,r} \Big)^{2}
      \right) \notag \\
\text{s.t.} \quad
  & \boldsymbol{A}\,\boldsymbol{x}^{k,r} = \boldsymbol{d}^{k,r},
    && \forall k \in \mathcal{K},\, r \in \mathcal{R}, \notag \\
  & \sum_{k \in \mathcal{K}} x_{i,j}^{k,r} \le u_{i,j},
    && \forall (i,j) \in \mathcal{E},\, r \in \mathcal{R}, \notag \\
  & \boldsymbol{x}^{k,r} \ge 0,\ \ x_{i,j}^{k,r} = 0 \text{ if } z_{i,j} = 0,
    && \forall (i,j) \in \mathcal{E}, \notag \\
  & \sum_{(i,j) \in \mathcal{E}} z_{i,j} \le c_0,\ \ z_{i,j} \in \{0,1\}
    && \forall (i,j) \in \mathcal{E}.
\tag{1}
\end{align}$$
