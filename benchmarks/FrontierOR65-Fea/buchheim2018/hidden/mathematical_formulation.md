# Original Formulation: Quadratic Shortest Path Problem (QSPP)

*Source: Quadratic Combinatorial Optimization Using Separable Underestimators, Christoph Buchheim and Emiliano Traversi, 2018 (INFORMS Journal on Computing 30(3):424–437).*

## Sets and Indices

- $G = (N, A)$: directed graph with node set $N$ and arc set $A$.

- $s \in N$: origin (source) node of the path.

- $t \in N$: destination (sink) node of the path.

- $\delta^+(i)$: set of outgoing arcs of node $i$.

- $\delta^-(i)$: set of ingoing arcs of node $i$.

## Parameters

- $Q_{ab} \in \mathbb{R}$: quadratic cost coefficient incurred when arcs $a$ and $b$ are used together, $\forall\, a, b \in A$. The matrix $Q$ is symmetric.

- $L_a \in \mathbb{R}$: linear traversal cost of arc $a$, $\forall\, a \in A$.

## Decision Variables

- $x_a \in \{0,1\}$: equals $1$ if arc $a$ is used in the path, $0$ otherwise, $\forall\, a \in A$.

## Objective

$$\begin{align}
\min \quad & \sum_{a,b \in A} Q_{ab}\, x_a x_b + \sum_{a \in A} L_a\, x_a \tag{19}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{a \in \delta^+(i)} x_a - \sum_{a \in \delta^-(i)} x_a = 0 && \forall\, i \in N \setminus \{s, t\} \tag{19}\\
& \sum_{a \in \delta^+(s)} x_a = 1 \tag{19}\\
& \sum_{a \in \delta^-(t)} x_a = 1 \tag{19}\\
& x_a \in \{0,1\} && \forall\, a \in A \tag{19}
\end{align}$$
