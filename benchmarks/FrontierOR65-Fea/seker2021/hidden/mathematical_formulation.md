# Sets and Indices

- $G=(V,E)$: undirected (perfect) graph with vertex set $V=\{1,\ldots,n\}$ and edge set $E$.

- $\{V_1, V_2, \ldots, V_P\}$: partition of $V$ into $P$ clusters; $p \in \{1,\ldots,P\}$ indexes clusters.

- $k \in \{1,\ldots,P\}$: color index (at most $P$ colors are needed).

# Decision Variables

- $y_k \in \{0,1\}$: equals $1$ if color $k$ is used, $0$ otherwise, for $k \in \{1,\ldots,P\}$.

- $w_{ik} \in \{0,1\}$: equals $1$ if vertex $i \in V$ is selected and assigned color $k$, $0$ otherwise, for $i \in V,\, k \in \{1,\ldots,P\}$.

# Model 1

$$\begin{align}
  \min \quad & \sum_{k=1}^{P} y_k \tag{1a} \\
  \text{s.t.}\quad & w_{ik} \le y_k, && \forall\, i \in V,\ k \in \{1,\ldots,P\} \tag{1b} \\
  & w_{ik} + w_{jk} \le 1, && \forall\, \{i,j\} \in E,\ k \in \{1,\ldots,P\} \tag{1c} \\
  & \sum_{i \in V_p} \sum_{k=1}^{P} w_{ik} = 1, && \forall\, p \in \{1,\ldots,P\} \tag{1d} \\
  & y_k \in \{0,1\}, && \forall\, k \in \{1,\ldots,P\} \tag{1e} \\
  & w_{ik} \in \{0,1\}, && \forall\, i \in V,\ k \in \{1,\ldots,P\} \tag{1f}
\end{align}$$

# Symmetry-Breaking Constraints

The following constraints are added to Model 1 to break color-index symmetry: $$\begin{align}
  y_k \le y_{k-1}, && \forall\, k \in \{2,\ldots,P\} \tag{2}
\end{align}$$
