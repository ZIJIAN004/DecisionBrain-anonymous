# Original Formulation: Minimum Convex Cost Flows in Bipartite Networks (MCCFBN)

*Source: A specialized interior-point algorithm for huge minimum convex cost flows in bipartite networks, Jordi Castro and Stefano Nasini, 2018/2021.*

## Sets and Indices

- $I$ : set of supply nodes (operating suppliers or machines), with $n = |I|$.

- $J$ : set of demand nodes (customers or tasks), with $m = |J|$.

- $(i,j)$ : arc from $i \in I$ to $j \in J$ in the complete bipartite network $I \times J$.

## Parameters

- $f_{ij} : \mathbb{R} \to \mathbb{R}$, convex cost function of the flow from $i \in I$ to $j \in J$.

- $d_j \in \mathbb{R}_+$, demand of node $j \in J$.

- $s_i \in \mathbb{R}_+$, supply (or supply capacity) of node $i \in I$.

- $u_{ij} \in \mathbb{R}_+$, capacity of the arc $(i,j) \in I \times J$.

where $\mathbb{R}$ and $\mathbb{R}_+$ are the sets of real and nonnegative real numbers respectively.

## Decision Variables

- $x_{ij} \in \mathbb{R}$ : flow from node $i \in I$ to node $j \in J$.

## Objective

$$\begin{align}
\min \quad & \sum_{i \in I} \sum_{j \in J} f_{ij}(x_{ij}), \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{subject to} \quad
& \sum_{i \in I} x_{ij} = d_j, && j \in J, \tag{2} \\
& \sum_{j \in J} x_{ij} \le s_i, && i \in I, \tag{3} \\
& 0 \le x_{ij} \le u_{ij}, && i \in I,\ j \in J. \tag{4}
\end{align}$$
