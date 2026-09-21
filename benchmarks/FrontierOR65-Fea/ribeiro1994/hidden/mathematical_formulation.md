# Original Formulation: Multiple-Depot Vehicle Scheduling Problem (MDVSP)

*Source: A Column Generation Approach to the Multiple-Depot Vehicle Scheduling Problem, Celso C. Ribeiro and François Soumis, 1994 (Operations Research 42(1):41–52).*

The formulation below is the integer multicommodity flow program “Problem MDVSP” given first in Section 2 (“Model Formulation and Lower Bounds”) as the definition of the problem the paper studies; its notation ($x^k_{ij}$, $A^k$, $V^k$, $r_k$) is carried into every subsequent section.

## Sets and Indices

- $N = \{1,\dots,n\}$: set of trips, with index $j$ (or $i$).

- $K = \{1,\dots,m\}$: set of depots, with index $k$.

- For each depot $k \in K$ a graph $G^k = (V^k, A^k)$ is associated, where:

  - $V^k = N \cup \{n+k\}$: node set (trips of $N$ plus the depot node $n+k$).

  - $A^k = N \times N \;\cup\; \{n+k\} \times N \;\cup\; N \times \{n+k\}$: arc set (trip-to-trip arcs, depot-to-trip arcs, and trip-to-depot arcs).

## Parameters

- $n$: number of trips; $m$: number of depots.

- Trip $T_j$ starts at time $s_j$ and ends at time $e_j$, $j = 1,\dots,n$.

- $\tau_{ij}$: travel time from the ending point of trip $T_i$ to the starting point of trip $T_j$.

- Compatibility: an ordered pair $(T_i, T_j)$ is *compatible* iff $e_i + \tau_{ij} \le s_j$; only compatible trip-to-trip arcs are included in $A^k$.

- $r_k$: number of vehicles stationed at depot $D_k$, for each $k \in K$.

- $c_{ij}$: finite cost incurred if a vehicle performs trip $T_j$ immediately after trip $T_i$.

- $c_{n+k,j}$ (resp. $c_{j,n+k}$): finite cost incurred if a vehicle stationed at depot $D_k$ starts (resp. ends) with trip $T_j$.

## Decision Variables

- $x^k_{ij}$: flow of type $k$ (i.e., number of vehicles from depot $k$) through arc $(i,j) \in A^k$, for all $k \in K$ and all $(i,j) \in A^k$. Nonnegative and integer.

## Objective

$$\text{Minimize} \quad \sum_{k=1}^{m} \sum_{(i,j) \in A^k} c_{ij}\, x^k_{ij}$$

## Constraints

$$\begin{align}
& \sum_{k=1}^{m} \sum_{i \in V^k} x^k_{ij} = 1 && \text{for all } j \in N \tag{1}\\[4pt]
& \sum_{i \in V^k} x^k_{ij} - \sum_{i \in V^k} x^k_{ji} = 0 && \text{for all } k \in K,\ \text{for all } j \in V^k \tag{2}\\[4pt]
& \sum_{j \in N} x^k_{n+k,j} \le r_k && \text{for all } k \in K \notag\\[4pt]
& x^k_{ij} \ge 0 && \text{for all } k \in K,\ \text{for all } (i,j) \in A^k \tag{3}\\[4pt]
& x^k_{ij}\ \text{integer} && \text{for all } k \in K,\ \text{for all } (i,j) \in A^k \notag
\end{align}$$

Constraint (1) ensures that each trip is performed exactly once. Constraints (2) are the flow-conservation constraints, and the capacity constraints $\sum_{j\in N} x^k_{n+k,j}\le r_k$ limit the number of vehicles leaving each depot. As remarked by the paper, $x^k_{ij}$ takes binary values for all $i,j \in N$ once (1) is imposed; the integrality requirement is stated on all arcs.
