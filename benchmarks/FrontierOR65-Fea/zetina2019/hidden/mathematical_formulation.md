# Sets and Indices

- $G = (N, A)$: directed graph with node set $N$ and arc set $A$; $(i,j) \in A$.

- $K$: set of commodities. Each commodity $k \in K$ is characterized by the tuple $(o_k, d_k, W^k)$, where $o_k$ is its origin, $d_k$ its destination, and $W^k$ its demand quantity.

# Parameters

- $f_{ij}$: fixed cost for installing arc $(i,j) \in A$.

- $c^k_{ij}$: per-unit transportation cost of commodity $k$ on arc $(i,j)$.

- $W^k$: demand quantity of commodity $k \in K$.

# Decision Variables

- $y_{ij} \in \{0,1\}$: equals $1$ if arc $(i,j) \in A$ is installed, $0$ otherwise.

- $x^k_{ij} \ge 0$: fraction of commodity $k$’s demand routed on arc $(i,j)$.

# Formulation (P)

$$\begin{align}
  \min \quad & \sum_{(i,j) \in A} f_{ij}\, y_{ij} + \sum_{k \in K} \sum_{(i,j) \in A} W^k c^k_{ij}\, x^k_{ij} \tag{1} \\
  \text{s.t.}\quad & \sum_{j \in N} x^k_{ji} - \sum_{j \in N} x^k_{ij} =
    \begin{cases}
      -1 & \text{if } i = o_k \\
      \phantom{-}1 & \text{if } i = d_k \\
      \phantom{-}0 & \text{otherwise}
    \end{cases}, && \forall\, i \in N,\ k \in K \tag{2} \\
  & x^k_{ij} \le y_{ij}, && \forall\, (i,j) \in A,\ k \in K \tag{3} \\
  & x^k_{ij} \ge 0, && \forall\, (i,j) \in A,\ k \in K \tag{4} \\
  & y_{ij} \in \{0,1\}, && \forall\, (i,j) \in A \tag{5}
\end{align}$$

Objective (1) sums fixed installation costs and variable routing costs. Constraints (2) are flow conservation per commodity. Constraints (3) are disaggregated linking constraints forbidding flow on uninstalled arcs. Constraints (4)–(5) are domain restrictions.
