# Original Formulation: Quadratic Multiknapsack Problem (QMKP)

*Source: An Exact Algorithm for the Quadratic Multiknapsack Problem with an Application to Event Seating, David Bergman, 2019 (INFORMS Journal on Computing).*

## Sets and Indices

- $n \in \mathbb{Z}^+$: number of items; $[n] := \{1,\dots,n\}$.

- $m \in \mathbb{Z}^+$: number of knapsacks; $[m] := \{1,\dots,m\}$.

- $i,j \in [n]$: item indices.

- $k \in [m]$: knapsack index.

## Parameters

- $p_i \in \mathbb{Z}$: individual profit of item $i \in [n]$ (not required to be nonnegative).

- $p_{i,j} \in \mathbb{Z}$: pairwise profit of distinct items $i,j \in [n]$, with $p_{i,j} = p_{j,i}$ (not required to be nonnegative).

- $w_i \geq 0$: weight of item $i \in [n]$.

- $C_k \in \mathbb{Z}^+$: capacity of knapsack $k \in [m]$.

## Decision Variables

- $x_{i,k} \in \{0,1\}$ for $i \in [n],\, k \in [m]$: equals $1$ iff item $i$ is placed in knapsack $k$.

## Objective

$$\begin{align}
\text{maximize} \quad
  & \sum_{i=1}^{n}\sum_{k=1}^{m} p_i\, x_{i,k}
    + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\sum_{k=1}^{m} x_{i,k}\, x_{j,k}\, p_{i,j}
    \notag
\end{align}$$

## Constraints

$$\begin{align}
\text{subject to} \quad
  & \sum_{i=1}^{n} w_i\, x_{i,k} \leq C_k,
    && k \in [m], \\
  & \sum_{k=1}^{m} x_{i,k} \leq 1,
    && i \in [n], \\
  & x_{i,k} \in \{0,1\},
    && i \in [n],\; k \in [m].
    \tag{QMKP-QP}
\end{align}$$
