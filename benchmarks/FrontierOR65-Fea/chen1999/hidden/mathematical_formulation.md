# Original Formulation: Parallel Machine Scheduling with an Additive Criterion (PMAC)

*Source: Solving Parallel Machine Scheduling Problems by Column Generation, Zhi-Long Chen and Warren B. Powell, 1999 (INFORMS Journal on Computing 11(1):78–94).*

The paper studies the general PMAC problem of scheduling $n$ jobs on $m$ parallel machines (identical $P$, uniform $Q$, or unrelated $R$) to minimize an additive criterion $\sum_{j\in N} f_j(C_j)$. The first and defining formulation, written in Section 1.1 (“Integer Programming Formulation”), is **IP1** (Eqs. (1)–(6)); its notation $x_{ij}^k$, $C_j$ is carried into every later section. The set-partitioning forms SP1/SP2 (Section 1.2) are a Dantzig–Wolfe decomposition used by the algorithm and are therefore not part of the original formulation.

## Sets and Indices

$$\begin{align*}
N &= \{1,2,\dots,n\} && \text{set of jobs};\quad i,j \in N \\
M &= \{1,2,\dots,m\} && \text{set of machines};\quad k \in M \\
0,\, n+1 && && \text{dummy ``start'' and ``end'' jobs on a machine} \\
A_j^k &= \{i \in N \mid i \text{ can succeed } j \text{ in a feasible partial schedule on machine } k\} \\
B_j^k &= \{i \in N \mid i \text{ can precede } j \text{ in a feasible partial schedule on machine } k\}
\end{align*}$$

## Parameters

$$\begin{align*}
p_{ij} &: \text{processing time of job } i \text{ on machine } j \quad
   (p_{ij}=p_i \text{ for } P;\ p_{ij}=p_i/s_j \text{ for } Q;\ \text{arbitrary for } R) \\
s_j &: \text{speed of machine } j \quad (\text{uniform machines}) \\
w_i &: \text{weight of job } i \\
d_i &: \text{due date of job } i \\
f_j(\cdot) &: \text{real-valued additive cost function for job } j
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_{ij}^k &\in \{0,1\}, \quad i,j\in N,\ k\in M:
   && = 1 \text{ if job } j \text{ is processed immediately after job } i \text{ on machine } k \\
x_{0j}^k &\in \{0,1\}, \quad j\in N,\ k\in M:
   && = 1 \text{ if job } j \text{ is processed first on machine } k \\
x_{j,n+1}^k &\in \{0,1\}, \quad j\in N,\ k\in M:
   && = 1 \text{ if job } j \text{ is processed last on machine } k \\
C_j &\ge 0, \quad j\in N: && \text{completion time of job } j
\end{align*}$$

## Objective

$$\begin{align}
\min \quad \sum_{j \in N} f_j(C_j) \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{k \in M} \sum_{i \in B_j^k \cup \{0\}} x_{ij}^k &= 1,
   && \forall j \in N \tag{2}\\[2pt]
\sum_{j \in N} x_{0j}^k &\le 1,
   && \forall k \in M \tag{3}\\[2pt]
\sum_{i \in B_j^k \cup \{0\}} x_{ij}^k &=
   \sum_{i \in A_j^k \cup \{n+1\}} x_{ji}^k,
   && \forall j \in N,\ k \in M \tag{4}\\[2pt]
C_j &= \sum_{k \in M}\!\left( p_{jk}\, x_{0j}^k
   + \sum_{i \in B_j^k} (C_i + p_{jk})\, x_{ij}^k \right),
   && \forall j \in N \tag{5}\\[2pt]
x_{ij}^k &\in \{0,1\},
   && \forall i,j \in N,\ k \in M \tag{6}
\end{align}$$
