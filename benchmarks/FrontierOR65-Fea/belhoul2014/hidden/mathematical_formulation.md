# Sets and Indices

- $n$: number of tasks (equal to the number of agents); $i, j \in \{1,\ldots,n\}$.

- $p$: number of objectives, indexed by $k = 1,\ldots,p$.

# Parameters

- $c^k_{ij} \in \mathbb{Z}_+$: cost of assigning task $i$ to agent $j$ with respect to objective $k$.

- $\lambda = (\lambda_1,\ldots,\lambda_p) \in \mathbb{R}^p_{>0}$: strictly positive weighting vector representing the decision-maker’s search direction.

- $\bar{z} = (\bar{z}_1,\ldots,\bar{z}_p) \in \mathbb{R}^p$: reference point (e.g. the ideal point $z^*$).

# Decision Variables

- $x_{ij} \in \{0,1\}$: equals $1$ if task $i$ is assigned to agent $j$, $0$ otherwise, for $i,j = 1,\ldots,n$.

- $\mu$: unrestricted scalar variable (epigraph of the Tchebychev objective).

# Compromise Assignment Problem (CAP)

The Compromise Assignment Problem uses the weighted Tchebychev / achievement scalarizing function: $$\begin{align}
  \min \quad & \max_{k=1,\ldots,p}\ \bigl\{\lambda_k\!\left(\sum_{i=1}^{n}\sum_{j=1}^{n} c^k_{ij}\, x_{ij} - \bar{z}_k\right)\bigr\} \tag{CAP} \\
  \text{s.t.}\quad
    & \sum_{j=1}^{n} x_{ij} = 1, && i = 1,\ldots,n \notag \\
    & \sum_{i=1}^{n} x_{ij} = 1, && j = 1,\ldots,n \notag \\
    & x_{ij} \in \{0,1\}, && i,j = 1,\ldots,n \notag
\end{align}$$

# Linearized Compromise Assignment Problem (LCAP)

The paper linearizes (CAP) by introducing a scalar variable $\mu$, yielding the mixed-integer linear program: $$\begin{align}
  \min \quad & \mu \tag{LCAP-1} \\
  \text{s.t.}\quad
    & \mu \;\ge\; \lambda_k\!\left(\sum_{i=1}^{n}\sum_{j=1}^{n} c^k_{ij}\, x_{ij} - \bar{z}_k\right), && k = 1,\ldots,p \tag{LCAP-2} \\
    & \sum_{j=1}^{n} x_{ij} = 1, && i = 1,\ldots,n \tag{LCAP-3} \\
    & \sum_{i=1}^{n} x_{ij} = 1, && j = 1,\ldots,n \tag{LCAP-4} \\
    & x_{ij} \in \{0,1\}, && i, j = 1,\ldots,n \notag \\
    & \mu \text{ unrestricted}. \notag
\end{align}$$
