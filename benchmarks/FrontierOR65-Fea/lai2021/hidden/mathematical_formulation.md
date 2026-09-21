# Original Formulation: Maximally Diverse Grouping Problem (MDGP)

*Source: Neighborhood Decomposition Based Variable Neighborhood Search and Tabu Search for Maximally Diverse Grouping, Xiangjing Lai, Jin-Kao Hao, Zhang-Hua Fu, and Dong Yue, 2021 (quadratic binary program of Gallego, Laguna, Martı́, and Duarte, 2013; Rodrı́guez, Lozano, Garcı́a-Martı́nez, and González-Barrera, 2013).*

## Sets and Parameters

- $V$: set of $N$ elements to be partitioned.

- $m$: number of groups (positive integer), indexed by $g$.

- $D = [d_{ij}]_{N \times N}$: symmetric distance (dissimilarity) matrix between elements.

- $L_g$, $U_g$ with $1 \le g \le m$, $L_g \le U_g$: lower and upper capacity limits of group $g$.

## Decision Variables

- $X_{ig} \in \{0,1\}$, $\forall i \in \{1,\dots,N\},\; \forall g \in \{1,\dots,m\}$: 1 if element (vertex) $i$ is assigned to group $g$, 0 otherwise.

## Objective (Quadratic Binary Program)

$$\begin{align}
\max \quad & \sum_{g=1}^{m} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} d_{ij}\, X_{ig}\, X_{jg} \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{g=1}^{m} X_{ig} &= 1,
  & \forall i \in \{1,\dots,N\}, \tag{2}\\
L_g \le \sum_{i=1}^{N} X_{ig} &\le U_g,
  & \forall g \in \{1,\dots,m\}, \tag{3}\\
X_{ig} &\in \{0,1\},
  & \forall i \in \{1,\dots,N\},\; \forall g \in \{1,\dots,m\}. \tag{4}
\end{align}$$

The objective (1) is the quadratic sum of the pairwise distances within each group, which must be maximized. Constraints (2) guarantee that every vertex is placed in exactly one group. Constraints (3) ensure that the size of group $g$ lies within the interval $[L_g, U_g]$. Constraints (4) are the binary restrictions on the assignment variables. **The products $X_{ig}\,X_{jg}$ in (1) make the formulation a genuine 0–1 quadratic program, left in its original nonlinear form here.**
