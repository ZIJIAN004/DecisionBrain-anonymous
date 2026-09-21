# Sets and Indices

- $m$: number of agents, indexed by $i \in \{1,\ldots,m\}$.

- $n$: number of jobs, indexed by $j \in \{1,\ldots,n\}$.

# Parameters

- $p_{ij} \in \mathbb{Z}_+$: profit associated with assigning job $j$ to agent $i$.

- $w_{ij} \in \mathbb{Z}_+$: capacity consumption on agent $i$ if job $j$ is assigned to agent $i$.

- $c_i \in \mathbb{Z}_+$: capacity of agent $i$.

# Decision Variables

- $x_{ij} \in \{0,1\}$: equals $1$ if job $j$ is assigned to agent $i$, and $0$ otherwise, for $i \in \{1,\ldots,m\}$, $j \in \{1,\ldots,n\}$.

# Objective

Maximize total profit of the assignment: $$\begin{align}
  \max \quad & \sum_{i=1}^{m} \sum_{j=1}^{n} p_{ij}\, x_{ij} \tag{1}
\end{align}$$

# Constraints

$$\begin{align}
  \sum_{i=1}^{m} x_{ij} &= 1, && j \in \{1,\ldots,n\} \tag{2} \\
  \sum_{j=1}^{n} w_{ij}\, x_{ij} &\le c_i, && i \in \{1,\ldots,m\} \tag{3} \\
  x_{ij} &\in \{0,1\}, && i \in \{1,\ldots,m\},\ j \in \{1,\ldots,n\} \tag{4}
\end{align}$$

Constraint (2) is the semi-assignment constraint (each job assigned to exactly one agent); (3) is the knapsack/capacity constraint per agent; (4) gives binary integrality.
