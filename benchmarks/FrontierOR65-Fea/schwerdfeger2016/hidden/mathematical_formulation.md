# Original Formulation: Workload Balancing on Identical Parallel Machines ($P \parallel \mathrm{NSSWD}$)

*Source: A fast and effective subset sum based improvement procedure for workload balancing on identical parallel machines, Stefan Schwerdfeger & Rico Walter, Computers & Operations Research 73 (2016) 84–91.*

## Sets and Indices

$$\begin{align*}
\mathcal{I} &: \text{set of } m \ge 2 \text{ identical parallel machines, indexed by } i = 1, \ldots, m, \\
\mathcal{J} &: \text{set of } n > m \text{ independent jobs, indexed by } j = 1, \ldots, n.
\end{align*}$$

## Parameters

$$\begin{align*}
p_j \in \mathbb{N} &: \text{integer processing time of job } j \ (j = 1, \ldots, n);\\
                   &\quad \text{w.l.o.g. } p_1 \ge p_2 \ge \cdots \ge p_n > 0, \\
\mu = \frac{1}{m}\sum_{j=1}^{n} p_j &: \text{average machine completion time.}
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_{ij} &\in \{0,1\} : \text{$1$ if job $j$ is assigned to machine $i$, $0$ otherwise},\\
C_i    &: \text{completion time of machine $i$ (auxiliary, fixed by constraint (3)).}
\end{align*}$$

## Objective

The normalized sum of squared workload deviations (NSSWD-criterion) is defined as $$\begin{equation}
\mathrm{NSSWD} = \frac{1}{\mu}\left[\sum_{i=1}^{m}\left(C_i - \mu\right)^2\right]^{1/2}.
\tag{1}
\end{equation}$$ $$\begin{equation}
\text{Minimize}\quad z = \frac{1}{\mu}\left[\sum_{i=1}^{m}\left(C_i - \mu\right)^2\right]^{1/2}.
\tag{2}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j=1}^{n} p_j\, x_{ij} &= C_i, & i &= 1, \ldots, m, \tag{3}\\
\sum_{i=1}^{m} x_{ij} &= 1, & j &= 1, \ldots, n, \tag{4}\\
x_{ij} &\in \{0,1\}, & i &= 1, \ldots, m;\ j = 1, \ldots, n. \tag{5}
\end{align}$$
