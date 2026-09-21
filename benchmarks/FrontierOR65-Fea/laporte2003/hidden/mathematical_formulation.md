# Original Formulation: Undirected Traveling Purchaser Problem

*Source*: Laporte, Riera-Ledesma, Salazar-González (2003), “A Branch-and-Cut Algorithm for the Undirected Traveling Purchaser Problem,” *Operations Research* 51(6):940–951. Equations (1)–(9) in the paper.

## Sets and Indices

- $v_0$: depot vertex.

- $M = \{v_1, \ldots, v_n\}$: set of markets ($n \ge 4$).

- $V = \{v_0\} \cup M$: vertex set.

- $E = \{[v_i,v_j] : v_i,v_j \in V,\ i<j\}$: edge set of the undirected complete graph $G=(V,E)$.

- $K = \{p_1,\ldots,p_m\}$: set of products ($m \ge 1$).

- $M_k \subseteq M$: subset of markets where product $p_k$ is available.

- $\delta(S) = \{[v_i,v_j]\in E : v_i \in S,\, v_j \in V\setminus S\}$ for $S\subset V$.

- $M^{*} = \{v_0\} \cup \bigl\{ v_i \in M : \exists\,p_k \in K \text{ s.t. }
          \sum_{v_j \in M_k\setminus\{v_i\}} q_{kj} < d_k\bigr\}$: mandatory vertices.

## Parameters

- $c_e$: travel cost of edge $e \in E$.

- $b_{ki}$: price of product $p_k$ at market $v_i \in M_k$.

- $d_k$: demand for product $p_k$.

- $q_{ki}$: available supply of product $p_k$ at market $v_i \in M_k$ ($0 < q_{ki} \le d_k$ and $\sum_{v_j \in M_k} q_{kj} \ge d_k$).

## Decision Variables

- $x_e \in \{0,1\}$ for $e \in E$: 1 iff edge $e$ is used.

- $y_i \in \{0,1\}$ for $v_i \in M\setminus M^{*}$: 1 iff vertex $v_i$ is visited. (For $v_i \in M^{*}$, $y_i$ is fixed to $1$.)

- $z_{ki} \ge 0$ for $p_k \in K,\ v_i \in M_k$: amount of product $p_k$ purchased at market $v_i$.

## Objective

$$\begin{equation}
  w^{\mathrm{OPT}} \;=\; \min \ \sum_{e \in E} c_e\, x_e
                       + \sum_{p_k \in K} \sum_{v_i \in M_k} b_{ki}\, z_{ki}.
  \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  \sum_{e \in \delta(\{v_i\})} x_e &\;=\; 2\, y_i,
    && \forall\, v_i \in V, \tag{2}\\[2pt]
  \sum_{e \in \delta(S)} x_e &\;\ge\; 2\, y_i,
    && \forall\, S \subseteq M,\ \forall\, v_i \in S, \tag{3}\\[2pt]
  \sum_{v_i \in M_k} z_{ki} &\;=\; d_k,
    && \forall\, p_k \in K, \tag{4}\\[2pt]
  z_{ki} &\;\le\; q_{ki}\, y_i,
    && \forall\, p_k \in K,\ \forall\, v_i \in M_k, \tag{5}\\[2pt]
  x_e &\;\in\; \{0,1\},
    && \forall\, e \in E, \tag{6}\\[2pt]
  y_i &\;\in\; \{0,1\},
    && \forall\, v_i \in M\setminus M^{*}, \tag{7}\\[2pt]
  y_i &\;=\; 1,
    && \forall\, v_i \in M^{*}, \tag{8}\\[2pt]
  z_{ki} &\;\ge\; 0,
    && \forall\, p_k \in K,\ \forall\, v_i \in M_k. \tag{9}
\end{align}$$
