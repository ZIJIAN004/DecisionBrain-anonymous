# Original Formulation: Periodic Vehicle Routing Problem with Time Windows and Flexible Schedule Structures (PVRPTW)

*Source: A Branch-and-Price-and-Cut Algorithm for the Periodic Vehicle Routing Problem with Flexible Schedule Structures, Florian Rothenbächer, Transportation Science, 2019.*

## Sets and Indices

$$\begin{align*}
P &= \{0, \ldots, |P|-1\} && \text{Planning horizon (days); cyclic: day $|P|-1$ is followed by day $0$.}\\
N &&& \text{Set of customers.}\\
d &&& \text{Single depot.}\\
O &= N \cup \{d\} && \text{Set of all locations.}\\
S_n &\subseteq \mathcal{P}(P) && \text{Set of offered schedules for customer $n \in N$ ($\mathcal{P}(P)$ = power set of $P$).}\\
L &= \{1, \ldots, |P|\} && \text{Set of possible schedule-part lengths.}\\
S_n^{p:l} &&& \text{Subset of $S_n$ whose schedules include the schedule part $n^{p:l}$.}\\
R &&& \text{Set of all feasible routes.}\\
R^p &\subset R && \text{Subset of routes feasible for day $p \in P$.}
\end{align*}$$

## Parameters

$$\begin{align*}
m &&& \text{Number of homogeneous vehicles available each day.}\\
Q &&& \text{Vehicle capacity.}\\
D &&& \text{Maximal route duration.}\\
[a_d, b_d] &&& \text{Time window of the depot.}\\
q_n &&& \text{Demand per day for customer $n \in N$.}\\
[a_n, b_n] &&& \text{Time window for customer $n \in N$.}\\
c_r &&& \text{Cost of route $r$ (depends exclusively on traveled distance).}\\
t_{o_1,o_2} &&& \text{Travel time from $o_1$ to $o_2$ (including service time at $o_1$).}\\
c_{o_1,o_2} &&& \text{Travel cost from $o_1$ to $o_2$.}\\
a_{rn}^{p:l} &&& \text{How often route $r$ includes schedule part $n^{p:l}$ (a visit to $n$ on day $p$}\\
&&& \text{delivering $l$ days of demand); $0$ or $1$ for elementary routes.}\\
b_s^{p:l} &&& \text{Binary: whether schedule $s$ induces the schedule part $(p:l)$.}
\end{align*}$$ A schedule part $n^{p:l}$ represents a visit to customer $n$ on day $p$ delivering the demand for $l$ consecutive days; the delivered amount is $q_n \cdot l$.

## Decision Variables

$$\begin{align*}
\lambda_r^p &\in \{0,1\} && \text{$1$ if route $r \in R^p$ is performed on day $p \in P$, $0$ otherwise.}\\
z_n^s &\in \{0,1\} && \text{$1$ if schedule $s \in S_n$ is chosen for customer $n \in N$, $0$ otherwise.}
\end{align*}$$

## Objective

$$\begin{align}
\min \quad & \sum_{p \in P} \sum_{r \in R^p} c_r\, \lambda_r^p \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{s \in S_n} z_n^s = 1
  && (\pi_n) && \forall\, n \in N \tag{1b}\\
& \sum_{r \in R^p} a_{rn}^{p:l}\, \lambda_r^p = \sum_{s \in S_n} b_s^{p:l}\, z_n^s
  && (\rho_n^{p:l}) && \forall\, n \in N,\; p \in P,\; l \in L : S_n^{p:l} \neq \emptyset \tag{1c}\\
& \sum_{r \in R^p} \lambda_r^p \leq m
  && (\mu_p) && \forall\, p \in P \tag{1d}\\
& z_n^s \in \{0,1\}
  && && \forall\, n \in N,\; s \in S_n \tag{1e}\\
& \lambda_r^p \in \{0,1\}
  && && \forall\, p \in P,\; r \in R^p \tag{1f}
\end{align}$$
