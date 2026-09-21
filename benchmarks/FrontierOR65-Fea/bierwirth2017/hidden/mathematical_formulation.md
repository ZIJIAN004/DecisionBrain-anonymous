# Original Formulation: Job Shop Scheduling Problem with Total Weighted Tardiness (JSPTWT)

*Source: “Extended GRASP for the Job Shop Scheduling Problem with Total Weighted Tardiness Objective,” Bierwirth and Kuhpfahl, European Journal of Operational Research, 2017.*

## Sets and Parameters

- $J = \{1,\ldots,n\}$: set of jobs; $M = \{1,\ldots,m\}$: set of machines.

- Each job $j \in J$ has an ordered sequence of $m$ operations; operation $(i,j)$ denotes the processing of job $j$ on machine $i$.

- $p_{ij} \ge 0$: processing time of job $j$ on machine $i$.

- $w_j$: weight of job $j$; $d_j$: due date of job $j$; $r_j \ge 0$: release date of job $j$.

- $\sigma_j(k)$: the machine of the $k$-th operation of job $j$ in its technological sequence.

- $V$: a sufficiently large constant (e.g., $V = \sum_{i,j} p_{ij} + \max_j r_j$).

## Decision Variables

- $s_{ij} \ge 0$: start time of operation $(i,j)$.

- $c_j \ge 0$: completion time of job $j$; $T_j \ge 0$: tardiness of job $j$.

- $y_{(i,j),(i,k)} \in \{0,1\}$ for each pair of jobs $j \ne k$ sharing machine $i$: $1$ if $(i,j)$ precedes $(i,k)$ on machine $i$.

## Objective

$$\begin{equation}
\min \; TWT \;=\; \sum_{j \in J} w_j\, T_j \tag{1}
\end{equation}$$

## Constraints (Reconstruction – paper does not provide a MIP)

$$\begin{align}
T_j &\ge c_j - d_j, \quad T_j \ge 0, & \forall j \in J \tag{2} \\
c_j &= s_{\sigma_j(m),\, j} + p_{\sigma_j(m),\, j}, & \forall j \in J \tag{3} \\
s_{\sigma_j(k+1),\, j} &\ge s_{\sigma_j(k),\, j} + p_{\sigma_j(k),\, j}, & \forall j \in J,\; k = 1,\ldots,m-1 \tag{4} \\
s_{\sigma_j(1),\, j} &\ge r_j, & \forall j \in J \tag{5} \\
s_{i,k} &\ge s_{i,j} + p_{i,j} - V\,(1 - y_{(i,j),(i,k)}), & \forall i \in M,\; j \ne k \text{ on } i \tag{6} \\
s_{i,j} &\ge s_{i,k} + p_{i,k} - V\, y_{(i,j),(i,k)}, & \forall i \in M,\; j \ne k \text{ on } i \tag{7} \\
s_{ij} &\ge 0,\; T_j \ge 0,\; c_j \ge 0, \; y_{(i,j),(i,k)} \in \{0,1\}. & \tag{8}
\end{align}$$
