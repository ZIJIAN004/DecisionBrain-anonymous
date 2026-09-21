# Original Formulation: Two-Stage Stochastic Planning and Scheduling

*Source: Stochastic Planning and Scheduling with Logic-Based Benders Decomposition, Özgün Elçi and J. N. Hooker, 2020/2022.*

## Sets and Parameters

- $J$: set of jobs, indexed by $j$.

- $I$: set of facilities, indexed by $i$.

- $\Omega$: finite set of scenarios, indexed by $\omega$; $\pi_\omega \ge 0$ is the probability of scenario $\omega$, with $\sum_{\omega \in \Omega} \pi_\omega = 1$.

- $p^{\omega}_{ij}$: processing time of job $j$ on facility $i$ in scenario $\omega$.

- $r_j$: release time of job $j$.

- $d_j$: deadline of job $j$; each job must be processed within $[r_j, d_j]$.

- $c_{ij}$: resource consumption of job $j$ on facility $i$ (cumulative scheduling).

- $K_i$: maximum total resource consumption on facility $i$ at any instant.

- $g(\mathbf{x})$: first-stage cost as a function of the assignment.

- $h(\mathbf{s},\mathbf{x},\omega)$: second-stage objective.

## Decision Variables

- **First-stage (binary)**: $x_j \in I$ for all $j \in J$, i.e. $x_j$ denotes the facility to which job $j$ is assigned. Equivalently, $x_{ij} \in \{0,1\}$ with $x_{ij} = 1$ iff job $j$ is assigned to facility $i$. For each $\mathbf{x}$, let $J_i(\mathbf{x}) = \{j \in J : x_j = i\}$ denote the set of jobs assigned to facility $i$.

- **Second-stage (continuous)**: $s_j \ge 0$ for all $j \in J$, the start time of job $j$; $s_j$ is determined after the scenario $\omega$ is revealed.

## Objective

$$\begin{align}
\min_{\mathbf{x}} \quad & g(\mathbf{x}) + \sum_{\omega \in \Omega} \pi_\omega\, Q(\mathbf{x}, \omega)
  \;:\; x_j \in I,\ \forall j \in J, \tag{3}
\end{align}$$ where the second-stage value is $$\begin{align}
Q(\mathbf{x}, \omega) = \min_{\mathbf{s}} \quad & h(\mathbf{s}, \mathbf{x}, \omega) \tag{2}
\end{align}$$ *(With (1): $\min_{\mathbf{x}\in X}\{f(\mathbf{x}) + \mathbb{E}_{\omega}[Q(\mathbf{x},\omega)]\}$.)*

## Constraints of the Second Stage (for each scenario $\omega \in \Omega$)

$$\begin{align}
s_j \in [\, r_j,\; d_j - p^{\omega}_{x_j,j} \,],
  & \quad \forall j \in J, \tag{TW}\\
\sum_{\substack{j \in J_i(\mathbf{x}) \\ 0 \le t \le s_j + p^{\omega}_{x_j,j}}} c_{ij} \le K_i,
  & \quad \forall i \in I,\; \forall t \ge 0. \tag{CUM}
\end{align}$$ Here (TW) are the time-window constraints ensuring each job finishes before its deadline under its (scenario-dependent) processing time. The cumulative resource constraint (CUM) states that, at every point in time $t$, the total resource consumption of jobs being processed on facility $i$ must not exceed $K_i$. Each assignment $\mathbf{x}$ induces, for every $(\omega, i)$, a cumulative scheduling problem over the jobs in $J_i(\mathbf{x})$ with continuous start-time variables $s_j$.

## First-Stage Constraints

$$\begin{align}
\sum_{i \in I} x_{ij} &= 1, & \forall j \in J, \notag\\
x_{ij} &\in \{0,1\}, & \forall i \in I,\; j \in J. \notag
\end{align}$$
