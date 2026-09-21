# Original Formulation: Workload Smoothing Problem (WSP)

*Source: A Branch and Cut Approach for Workload Smoothing on Assembly Lines, Anulark Pinnoi and Wilbert E. Wilhelm, INFORMS Journal on Computing 9(4):335–350, 1997.*

## Sets and Indices

$$\begin{align*}
&T = \{1, \ldots, \tau\} && \text{set of all tasks; } \tau = |T|.\\
&s = 1, \ldots, S_U && \text{station index.}\\
&i, j, t \in T && \text{task indices.}\\
&\Theta = \{(i,j): i,j \in T,\ i \text{ is an immediate predecessor of } j\} && \text{arc set of the precedence graph.}\\
&H = (T, \Theta) && \text{precedence (directed) graph.}\\
&M(t),\ A(t) && \text{immediate / all predecessors of task } t.\\
&R(t),\ B(t) && \text{immediate / all successors of task } t.\\
&T(s) && \text{set of task candidates that may be processed at station } s.
\end{align*}$$

## Parameters

$$\begin{align*}
&c && \text{cycle time (maximum workload allowed per station).}\\
&p_t && \text{processing time of task } t \text{ (integer).}\\
&E_t && \text{earliest station to which task } t \text{ may be assigned.}\\
&L_t && \text{latest station to which task } t \text{ may be assigned.}\\
&S_U && \text{upper bound on the optimal number of stations.}\\
&S_L = \left\lceil \tfrac{\sum_{t \in T} p_t}{c} \right\rceil && \text{lower bound on the optimal number of stations.}\\
&S^* && \text{optimal (fixed) number of stations, given as input to the WSP.}
\end{align*}$$

## Decision Variables

$$\begin{align*}
&x_{si} =
\begin{cases}
1 & \text{if task } i \text{ is assigned to station } s,\\
0 & \text{otherwise,}
\end{cases}
\qquad \forall\, s \in S(i),\ i \in T.\\[4pt]
&z_{\max} \;=\; \text{the maximum station idle time (a real / continuous variable).}
\end{align*}$$

## Objective

The WSP assigns tasks to the optimal number of stations $S^{*}$ while minimizing the maximum idle time on any station to balance workloads: $$\begin{align}
\text{Minimize } \quad z_{\max} \tag{WSP}
\end{align}$$

## Constraints

Subject to (2)–(5) with $s = 1, \ldots, S^{*}$ in (4), and (6)–(7): $$\begin{align}
&\sum_{s=E_i}^{L_i} x_{si} = 1 && \forall\, i \in T, \tag{2}\\[4pt]
&\sum_{s=E_i}^{L_i} s\, x_{si} \;-\; \sum_{s=E_j}^{L_j} s\, x_{sj} \;\leq\; 0 && \forall\, (i,j) \in \Theta, \tag{3}\\[4pt]
&\sum_{i \in T(s)} p_i\, x_{si} \;\leq\; c && \forall\, s = 1, \ldots, S^{*}, \tag{4}\\[4pt]
&x_{si} \in \{0,1\} && \forall\, s = E_i, \ldots, L_i,\ i \in T, \tag{5}\\[4pt]
&\sum_{t} p_t\, x_{st} \;+\; z_{\max} \;\geq\; c && \forall\, s = 1, \ldots, S^{*}, \tag{6}\\[4pt]
&z_{\max} \;\geq\; 0. \tag{7}
\end{align}$$
