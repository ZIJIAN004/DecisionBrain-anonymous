# Original Formulation: Set Partitioning Problem (SPP)

*Source: A Unified Column Generation and Elimination Method for Solving Large-Scale Set Partitioning Problems, Yasuyuki Ihara (NEC Solution Innovators, Ltd.), 2025 (arXiv:2503.16652).*

## Sets and Parameters

$$\begin{align*}
V = \{v_i\}_{i \in I} \quad & \text{ground set of elements;} \\
\{U_j\}_{j \in J}     \quad & \text{family of subsets of } V; \\
I                     \quad & \text{index set for the elements of } V; \\
J = \{1,\ldots,N\}    \quad & \text{index set for the subsets;} \\
M                     \quad & \text{number of elements in } V \ (=|I|); \\
N                     \quad & \text{number of indices in } J \ (\text{number of subsets}); \\
c_j > 0               \quad & \text{cost of selecting subset } U_j,\ j \in J; \\
\mathbf{c} = (c_1,\ldots,c_N)^\top \quad & \text{cost vector;} \\
\mathbf{1}_M          \quad & M\text{-dimensional vector with all components equal to } 1.
\end{align*}$$

The $M \times N$ binary incidence matrix $A = (a_{i,j})_{1 \le i \le M,\, 1 \le j \le N}$ has entries $$\begin{equation}
a_{i,j} =
\begin{cases}
1, & \text{if } v_i \in U_j, \\
0, & \text{otherwise.}
\end{cases}
\tag{3}
\end{equation}$$

## Decision Variables

For each $j \in J$, a binary variable $x_j$ indicates whether subset $U_j$ is included in the selected partition $K \subseteq J$: $$\begin{equation}
x_j =
\begin{cases}
1, & \text{if } j \in K, \\
0, & \text{otherwise,}
\end{cases}
\tag{1}
\end{equation}$$ with $\mathbf{x} = (x_1,\ldots,x_N)^\top$.

## Objective

$$\begin{equation}
\min_{\mathbf{x}} \quad \mathbf{c}^\top \mathbf{x}
\tag{2}
\end{equation}$$

## Constraints

$$\begin{align}
& A\mathbf{x} = \mathbf{1}_M, \tag{2}\\
& x_j \in \{0,1\}, \quad \forall\, j \in J. \tag{1}
\end{align}$$
