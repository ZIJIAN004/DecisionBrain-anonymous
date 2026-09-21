# Sets and Parameters

- $N = \{1,\ldots,n\}$: set of orders (items); index $i$.

- $M = \{1,\ldots,m\}$: set of slabs (knapsacks); index $j$.

- $M_i \subseteq M$: set of slabs incident to order $i$.

- $N_j \subseteq N$: set of orders incident to slab $j$.

- $C_j$: set of colors incident on slab $j$.

- $w_i$: weight of order $i$.

- $W_j$: weight of slab $j$.

- $c_i$: color of order $i$.

# Decision Variables

$$\begin{align*}
x_{ij} &\in \{0,1\} && \forall\, i \in N,\; j \in M_i \quad (\text{1 if order $i$ is assigned to slab $j$}) \\
y_{cj} &\in \{0,1\} && \forall\, c \in C_j,\; j \in M \quad (\text{1 if orders of color $c$ use slab $j$}) \\
z_{j}  &\in \{0,1\} && \forall\, j \in M \quad (\text{1 if any order is incident to slab $j$})
\end{align*}$$

# Objective

The paper states the natural formulation with the nonlinear objective (1): $$\begin{equation}
\max \; \sum_{i \in N}\sum_{j \in M_i} w_i\, x_{ij} \;-\; \sum_{j \in M} \left( W_j - \sum_{i \in N_j} w_i\, x_{ij} \right) z_j \tag{1}
\end{equation}$$

Because $z_j = 0$ forces $x_{ij} = 0$ for all $i \in N_j$ and $z_j = 1$ implies $x_{ij} z_j = x_{ij}$, for all feasible solutions the objective is equivalent to the linear form: $$\begin{equation}
\max \; \sum_{i \in N}\sum_{j \in M_i} 2 w_i\, x_{ij} \;-\; \sum_{j \in M} W_j\, z_j \tag{1$'$}
\end{equation}$$

# Constraints

$$\begin{align}
\sum_{i \in N_j} w_i\, x_{ij} &\leq W_j\, z_j && \forall\, j \in M \tag{2} \\
\sum_{j \in M_i} x_{ij} &\leq 1 && \forall\, i \in N \tag{3} \\
\sum_{c \in C_j} y_{cj} &\leq 2 && \forall\, j \in M \tag{4}
\end{align}$$ $$\begin{align}
x_{ij} &\leq y_{c_i,\, j} && \forall\, i \in N,\; j \in M_i \tag{5a} \\
x_{ij} &\in \{0,1\} && \forall\, i \in N,\; j \in M_i \tag{5b} \\
y_{cj} &\in \{0,1\} && \forall\, c \in C_j,\; j \in M \tag{5c} \\
z_{j}  &\in \{0,1\} && \forall\, j \in M \tag{5d}
\end{align}$$
