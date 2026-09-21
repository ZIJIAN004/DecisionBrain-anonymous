# Original Formulation: Capacitated Vehicle Routing Problem with Stochastic Two-Dimensional Items (S2L-CVRP)

*Source: The Vehicle Routing Problem with Stochastic Two-Dimensional Items, Jean-François Côté, Michel Gendreau, Jean-Yves Potvin, Transportation Science (2020).*

## Sets and Parameters

- $G = (V, E)$: complete undirected graph.

- $V = \{0, 1, 2, \ldots, n\}$: set of vertices ($n+1$ vertices); vertex $0$ is the depot.

- $C = V \setminus \{0\}$: set of customers.

- $E = \{(j,k) : j,k \in V,\; j < k\}$: set of edges.

- $c_{jk}$: cost associated with edge $(j,k) \in E$.

- $K$: number of identical vehicles available (all must be used).

- $H$: height (length) of the loading area of each vehicle.

- $W$: width of the loading area of each vehicle.

- $Q$: maximum weight capacity of each vehicle.

- For each customer $j \in C$: a set $I_j$ of two-dimensional items, of cardinality $m_j$. $\bigcup_{j \in C} I_j = I$ is the set of all items, $\sum_{j \in C} m_j = m$.

- For each item $i \in I$: there are $d_i$ possible sizes in height, width, and weight with an associated probability distribution ($d_i = 1$ for a deterministic item), $\sum_{r=1}^{d_i} p_i^r = 1$.

- $p_i^r$: probability that item $i$ has realization $r$.

- $w_i^r \le W$, $h_i^r \le H$, $q_i^r \le Q$: width, height, and weight of item $i$ under realization $r$.

- Expected area covered by items of customer $j$: $\displaystyle \tilde{a}_j = \sum_{i \in I_j} \sum_{r=1}^{d_i} p_i^r h_i^r w_i^r$.

- Expected weight of items of customer $j$: $\displaystyle \tilde{q}_j = \sum_{i \in I_j} \sum_{r=1}^{d_i} p_i^r q_i^r$.

- $c_f$: cost (penalty) associated with each unserved customer (recourse cost).

- $\Omega_R$: set of all possible realizations/scenarios for route $R$.

- $p_{\omega_R}$: probability of scenario $\omega_R \in \Omega_R$.

- $F(\omega_R)$: number of unserved customers under scenario $\omega_R$.

- $\mathcal{R}^{\mathit{inf}}$: set of infeasible routes (routes that do not satisfy the two-dimensional packing and unloading requirements, including deterministic routes and stochastic routes infeasible under all scenarios).

- $\mathcal{R}_{x^v}$: set of routes in integer solution $x^v$.

## Decision Variables

- $x_{jk} \in \{0,1\}$, $0 \le j < k \le n$: equals $1$ if edge $(j,k)$ is used, $0$ otherwise.

- $F(x)$: expected cost of the recourse of solution $x = (x_{jk})$ (defined below via <a href="#eq:F7" data-reference-type="eqref" data-reference="eq:F7">[eq:F7]</a>–<a href="#eq:F8" data-reference-type="eqref" data-reference="eq:F8">[eq:F8]</a>).

## Objective

$$\begin{align}
\min \quad & \sum_{j < k} c_{jk}\, x_{jk} + F(x) \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{j \in C} x_{0j} = 2K, \tag{2} \\[4pt]
& \sum_{h < j} x_{hj} + \sum_{k > j} x_{jk} = 2,
  && j \in C, \tag{3} \\[4pt]
& \sum_{\substack{j,k \in S \\ j < k}} x_{jk} \leq |S|
   - \left\lceil \max\!\left\{
     \frac{\sum_{j \in S} \tilde{a}_j}{HW},\;
     \frac{\sum_{j \in S} \tilde{q}_j}{Q}
   \right\} \right\rceil,
  && S \subseteq C,\; 2 \leq |S| \leq n, \tag{4} \\[4pt]
& \sum_{(j,k) \in R} x_{jk} \leq |R| - 1,
  && R \in \mathcal{R}^{\mathit{inf}}, \tag{5} \\[4pt]
& x_{jk} \in \{0,1\},
  && 0 \leq j < k \leq n. \tag{6}
\end{align}$$

The recourse cost. For an integer solution $x^v$, the recourse cost $F(x^v)$ is $$\begin{align}
F(x^v) = \sum_{R \in \mathcal{R}_{x^v}} F(R), \tag{7} \label{eq:F7}
\end{align}$$ where $F(R)$ is the recourse cost of route $R$. When route $R$ has only deterministic items, $F(R) = 0$ (feasible) or $F(R) = \infty$ (infeasible). When stochastic items are present, $F(R) = \infty$ if $F(\omega_R) > 0$ for every $\omega_R \in \Omega_R$; otherwise route $R$ is feasible and $$\begin{align}
F(R) = c_f \cdot \sum_{\omega_R \in \Omega_R} p_{\omega_R}\, F(\omega_R). \tag{8} \label{eq:F8}
\end{align}$$ The recourse cost is set to $\infty$ in the case of a fractional solution.
