# Original Formulation: Minimum Graph Coloring Problem (Min–GCP)

*Source: Exact Solution of Graph Coloring Problems via Constraint Programming and Column Generation, Stefano Gualandi and Federico Malucelli, INFORMS Journal on Computing, 24(1), 81–100, 2012.*

The paper states this as the constraint programming (CP) model of Min–GCP in Section 2 (“Graph Coloring via Constraint Programming”), Eqs. (1)–(4). It is the first formulation defining the problem the paper studies, and its notation ($x_i$, $x_0$, $K$, $\mathscr{C}$) is carried into every subsequent section. The variables are finite-domain integer variables and the `alldifferent` / $\max$ constraints are kept literally as written (no linearization, no big-M reformulation), per the paper.

## Sets and Indices

- $G=(V,E)$: undirected graph; $V$ set of vertices, $E$ set of edges (unordered pairs $\{i,j\}$).

- $K=\{1,\dots,\bar{\chi}\}$: set of available colors (colors map to natural numbers).

- $\mathscr{C}$: a collection of cliques of $G$ on which the redundant `alldifferent` constraints are posted (defined in the preprocessing of Section 2.1).

## Parameters

- $\underline{\chi}$: lower bound on $\chi(G)$ (e.g. size of a maximal clique).

- $\bar{\chi}$: upper bound on $\chi(G)$ (e.g. number of colors used by a heuristic).

- $x_0^{*}$: cost (number of colors) of the last solution found during the CP search; used for the cost-bounding constraint (4).

## Decision Variables

$$\begin{align*}
  & x_i \in K, && \text{domain}(x_i)=K, \quad \forall\, i \in V
        \qquad \text{(color assigned to vertex $i$)},\\
  & x_0, && \text{domain}(x_0)=\{\underline{\chi},\dots,\bar{\chi}\}
        \qquad \text{(number of used colors; $x_0=\chi(G)$ at optimum)}.
\end{align*}$$

## Objective

$$\begin{align}
  \min \quad & x_0. \notag
\end{align}$$

## Constraints

$$\begin{align}
  & x_i \neq x_j
        && \forall\, \{i,j\} \in E,                                 \tag{1}\\
  & \texttt{alldifferent}(\{x_i \mid i \in C\})
        && \forall\, C \in \mathscr{C},                             \tag{2}\\
  & x_0 = \max(\{x_i \mid i \in V\}),
        &&                                                          \tag{3}\\
  & x_0 \leq x_0^{*}.
        &&                                                          \tag{4}
\end{align}$$
