# Original Formulation: Bin Packing with Conflicts

Sadykov & Vanderbeck. “Bin Packing with Conflicts: a Generic Branch-and-Price Algorithm.” Compact formulation (1a)–(1f).

## Sets, Indices and Parameters

- $V = \{1,\dots,n\}$: set of items.

- $G = (V,E)$: conflict graph where $(i,j)\in E$ indicates items $i$ and $j$ cannot share a bin.

- $K\le n$: upper bound on the number of identical bins.

- $W\in\mathbb{Z}_+$: capacity of each bin.

- $w_i \in \mathbb{Z}_+$: weight of item $i$, with $w_i\le W$.

## Decision Variables

$$\begin{align*}
  x_{ik} &\in \{0,1\}, \quad i=1,\dots,n,\ k=1,\dots,K \quad
    \text{($=1$ iff item $i$ is placed in bin $k$)} \\
  y_k   &\in \{0,1\}, \quad k=1,\dots,K \quad
    \text{($=1$ iff bin $k$ is used).}
\end{align*}$$

## Compact Formulation (1)

$$\begin{align}
\min\ & \sum_{k=1}^{K} y_k \tag{1a} \\
\text{s.t.}\ & \sum_{k=1}^{K} x_{ik} \ge 1,
  && i = 1,\dots,n, \tag{1b} \\
& \sum_{i=1}^{n} w_i\, x_{ik} \le W\, y_k,
  && k = 1,\dots,K, \tag{1c} \\
& x_{ik} + x_{jk} \le y_k,
  && (i,j)\in E,\ k = 1,\dots,K, \tag{1d} \\
& y_k \in \{0,1\},
  && k = 1,\dots,K, \tag{1e} \\
& x_{ik} \in \{0,1\},
  && i = 1,\dots,n,\ k = 1,\dots,K. \tag{1f}
\end{align}$$
