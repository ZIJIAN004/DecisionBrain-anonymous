# Original Formulation: Quadratic Knapsack Problem (QKP)

*Source: Exact Solution of the Quadratic Knapsack Problem, Alberto Caprara, David Pisinger, Paolo Toth, 1999 (INFORMS Journal on Computing 11(2):125–137).*

## Sets and Parameters

- $N := \{1, \dots, n\}$ — set of items; $i, j \in N$ are item indices.

- $n$ — number of items.

- $w_j$ — positive integer weight of item $j$, for $j \in N$.

- $c$ — positive integer knapsack capacity.

- $P = (p_{ij})$ — $n \times n$ nonnegative integer profit matrix, assumed symmetric, i.e. $p_{ij} = p_{ji}$ for all $i, j \in N,\ j > i$. The diagonal element $q_j := p_{jj}$ is the profit achieved if item $j$ is selected; for $j > i$, the quantity $p_{ij} + p_{ji}$ is the profit achieved if both items $i$ and $j$ are selected.

It is assumed without loss of generality that $\max_{j \in N} w_j \leq c < \sum_{j \in N} w_j$.

## Decision Variables

- $x_j \in \{0, 1\}$ — equal to $1$ if item $j$ is selected, $0$ otherwise, for $j \in N$.

## Objective

$$\begin{align}
\text{maximize} \quad & z(\text{QKP}) = \sum_{i \in N} \sum_{j \in N} p_{ij}\, x_i x_j \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{subject to} \quad & \sum_{j \in N} w_j x_j \leq c \tag{1} \\
                        & x_j \in \{0, 1\}, \quad j \in N. \tag{1}
\end{align}$$
