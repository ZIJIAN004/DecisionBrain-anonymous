# Sets and Indices

- $\mathcal{N}$: set of vessels ($|\mathcal{N}| = n$), indexed by $i$.

- $\mathcal{M}$: set of (discrete) berths ($|\mathcal{M}| = m$), indexed by $k$.

- $\mathcal{F}$: set of working shifts, indexed by $f$.

- $\mathcal{S}$: set of time-step indices $\{1, \ldots, |\mathcal{S}|\}$ within a working shift, indexed by $s$.

- $\mathcal{H}$: set of equal-length time steps, indexed by $h$ or $t$; $h = s + |\mathcal{S}|(f-1)$.

- $\mathcal{H}^s$: subset of $\mathcal{H}$ containing all time steps with the same within-shift index $s$.

- $\mathcal{P}_i$: set of feasible QC profiles for vessel $i$.

- $\mathcal{P}_i^s$: subset of $\mathcal{P}_i$ for profiles starting at time steps with index $s$ within a working shift.

# Parameters

- $d_i^{sp}$: handling duration (number of time steps) of profile $p \in \mathcal{P}_i^s$.

- $q_i^{spu}$: number of QCs used by profile $p$ at within-profile step $u \in \{1,\ldots,d_i^{sp}\}$, given start index $s$.

- $(a_i, b_i)$: feasible service time window for vessel $i$.

- $(a^k, b^k)$: available time window for berth $k$.

- $\bar{k}_i$: least-cost berthing position for vessel $i$.

- $\bar{t}_i$: expected time of arrival (ETA) of vessel $i$.

- $c_1, c_2$: unit penalty costs for deviation from $\bar{k}_i$ and $\bar{t}_i$.

- $q^h$: QC capacity at time step $h$.

- $M$: sufficiently large positive constant.

# Decision Variables

- $Y_i^k \in \{0,1\}$: $1$ if berth $k$ is assigned to vessel $i$.

- $\Lambda_i^p \in \{0,1\}$: $1$ if QC profile $p$ is assigned to vessel $i$.

- $\Gamma_i^h \in \{0,1\}$: $1$ if vessel $i$ begins berthing at time step $h$.

- $T_i, E_i \ge 0$, integer: start/end time steps of service of vessel $i$.

- $\Omega_i^{ph} \in \{0,1\}$: $1$ iff profile $p$ AND start step $h$ are both assigned to $i$.

- $\overline{AS}_i^h \in \{0,1\}$: $1$ iff $h \ge T_i$.

- $\overline{BE}_i^h \in \{0,1\}$: $1$ iff $h \le E_i$.

- $\overline{BT}_i^h \in \{0,1\}$: $1$ iff vessel $i$ is being served at time step $h$.

- $X_i^{kh} \in \{0,1\}$: $1$ iff berth $k$ is occupied by vessel $i$ at time $h$.

- $U1_i, V1_i, U2_i, V2_i \ge 0$: linearization variables for the $|\cdot|$ penalties.

# Original Nonlinear Objective (Section 2.2, eq. (1))

$$\begin{align}
  \min \; & \sum_{i \in \mathcal{N}} \Bigl( c_1 \Bigl|\sum_{k \in \mathcal{M}} k\, Y_i^k - \bar{k}_i\Bigr| + c_2\, |T_i - \bar{t}_i| \Bigr) \tag{1}
\end{align}$$

# Constraints (2)–(31)

$$\begin{align}
  & \sum_{k \in \mathcal{M}} Y_i^k = 1, && \forall\, i \in \mathcal{N} \tag{2} \\
  & \sum_{p \in \mathcal{P}_i} \Lambda_i^p = 1, && \forall\, i \in \mathcal{N} \tag{3} \\
  & \sum_{h \in \mathcal{H}} \Gamma_i^h = 1, && \forall\, i \in \mathcal{N} \tag{4} \\
  & \sum_{h \in \mathcal{H}^s} \Gamma_i^h \le \sum_{p \in \mathcal{P}_i^s} \Lambda_i^p, && \forall\, s \in \mathcal{S},\ i \in \mathcal{N} \tag{5} \\
  & T_i = \sum_{h \in \mathcal{H}} h\, \Gamma_i^h, && \forall\, i \in \mathcal{N} \tag{6} \\
  & E_i = \sum_{h \in \mathcal{H}} h\, \Gamma_i^h + \sum_{h \in \mathcal{H}^s}\sum_{p \in \mathcal{P}_i^s} d_i^{sp}\, \Omega_i^{ph} - 1, && \forall\, i \in \mathcal{N},\ s \in \mathcal{S} \tag{7} \\
  & a_i \le T_i, && \forall\, i \in \mathcal{N} \tag{8} \\
  & T_i \le b_i, && \forall\, i \in \mathcal{N} \tag{9} \\
  & \sum_{k \in \mathcal{M}} a^k Y_i^k \le T_i, && \forall\, i \in \mathcal{N} \tag{10} \\
  & T_i \le \sum_{k \in \mathcal{M}} b^k Y_i^k, && \forall\, i \in \mathcal{N} \tag{11} \\
  & h - T_i + 1 \le M\, \overline{AS}_i^h, && \forall\, h \in \mathcal{H},\ i \in \mathcal{N} \tag{12} \\
  & E_i - h + 1 \le M\, \overline{BE}_i^h, && \forall\, h \in \mathcal{H},\ i \in \mathcal{N} \tag{13} \\
  & \overline{BT}_i^h \ge \overline{AS}_i^h + \overline{BE}_i^h - 1, && \forall\, h \in \mathcal{H},\ i \in \mathcal{N} \tag{14} \\
  & T_i - h \le M(1 - \overline{BT}_i^h), && \forall\, h \in \mathcal{H},\ i \in \mathcal{N} \tag{15} \\
  & h - E_i \le M(1 - \overline{BT}_i^h), && \forall\, h \in \mathcal{H},\ i \in \mathcal{N} \tag{16} \\
  & 2 - \Lambda_i^p - \Gamma_i^h \le M(1 - \Omega_i^{ph}), && \forall\, p \in \mathcal{P}_i,\ h \in \mathcal{H},\ i \in \mathcal{N} \tag{17} \\
  & \Omega_i^{ph} \ge \Lambda_i^p + \Gamma_i^h - 1, && \forall\, p \in \mathcal{P}_i,\ h \in \mathcal{H},\ i \in \mathcal{N} \tag{18} \\
  & 2 - \overline{BT}_i^h - Y_i^k \le M(1 - X_i^{kh}), && \forall\, k \in \mathcal{M},\ h \in \mathcal{H},\ i \in \mathcal{N} \tag{19} \\
  & X_i^{kh} \ge \overline{BT}_i^h + Y_i^k - 1, && \forall\, k \in \mathcal{M},\ h \in \mathcal{H},\ i \in \mathcal{N} \tag{20}
\end{align}$$ $$\begin{align}
  & \sum_{i \in \mathcal{N}} \sum_{p \in \mathcal{P}_i} \sum_{s \in \mathcal{S}}
    \sum_{\substack{t \in \mathcal{H}^s \\ t \le h \le t + d_i^{sp} - 1}}
    q_i^{sp(h-t+1)}\, \Omega_i^{pt} \le q^h, && \forall\, h \in \mathcal{H} \tag{21} \\
  & \sum_{i \in \mathcal{N}} X_i^{kh} \le 1, && \forall\, k \in \mathcal{M},\ h \in \mathcal{H} \tag{22} \\
  & X_i^{kh} \in \{0,1\}, && \forall\, k,\ h,\ i \tag{23} \\
  & \overline{AS}_i^h \in \{0,1\}, && \forall\, h,\ i \tag{24} \\
  & \overline{BE}_i^h \in \{0,1\}, && \forall\, h,\ i \tag{25} \\
  & \overline{BT}_i^h \in \{0,1\}, && \forall\, h,\ i \tag{26} \\
  & Y_i^k \in \{0,1\}, && \forall\, k,\ i \tag{27} \\
  & \Gamma_i^h \in \{0,1\}, && \forall\, h,\ i \tag{28} \\
  & \Lambda_i^p \in \{0,1\}, && \forall\, p \in \mathcal{P}_i,\ i \tag{29} \\
  & \Omega_i^{ph} \in \{0,1\}, && \forall\, p \in \mathcal{P}_i,\ h,\ i \tag{30} \\
  & T_i, E_i \ge 0,\ \text{integer}, && \forall\, i \in \mathcal{N} \tag{31}
\end{align}$$

# Paper’s Own Linearization of $|\cdot|$ (Constraints (32)–(35))

The absolute-value terms in (1) are linearized by splitting them into non-negative components: $$\begin{align}
  & \sum_{k \in \mathcal{M}} k\, Y_i^k - \bar{k}_i + U1_i - V1_i = 0, && \forall\, i \in \mathcal{N} \tag{32} \\
  & T_i - \bar{t}_i + U2_i - V2_i = 0, && \forall\, i \in \mathcal{N} \tag{33} \\
  & U1_i,\ V1_i,\ U2_i,\ V2_i \ge 0, && \forall\, i \in \mathcal{N} \tag{34}
\end{align}$$ The equivalent linear objective replacing (1) is $$\begin{align}
  \text{OP:}\quad \min \sum_{i \in \mathcal{N}} \bigl( c_1 (U1_i + V1_i) + c_2 (U2_i + V2_i) \bigr). \tag{35}
\end{align}$$
