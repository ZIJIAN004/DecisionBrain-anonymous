# Sets and Indices

- $\mathcal{N}$: nodes of the scenario tree, indexed by $n$; $|\mathcal{N}| = N$; $p(n)$ is the direct predecessor of $n$, with $p(1) = 0$.

- $\mathscr{P}_n$: scenario path from the root to node $n$.

- $\mathbb{J}$: buses of the transmission network, indexed by $j$.

- $\hat{j} \in \{1,\ldots,\hat{R}\}$: reserve-margin regions; $\hat{J}(j) = \hat{j}$ maps buses to regions.

- $\mathbb{G}_j,\, \mathbb{G}'_j$: existing / potential generators at bus $j$; $\mathbb{G} = \bigcup_j \mathbb{G}_j$, $\mathbb{G}' = \bigcup_j \mathbb{G}'_j$.

- $\mathbb{L},\, \mathbb{L}'$: existing / potential transmission lines.

- $\mathbb{S}_j,\, \mathbb{S}'_j$: existing / potential storage devices at bus $j$; $\mathbb{S},\, \mathbb{S}'$ aggregated.

- $K_n$: representative days at node $n$; $H_k$: hours of day $k$, indexed by $h$.

- $\mathbb{A},\, \mathbb{A}'$: ordered pairs $(i,j)$ with existing / potential transmission lines.

# Parameters

- $\pi_n$: probability of reaching node $n$.

- $c^n_g, c^n_l, c^n_s$: building costs.

- $a_g, a_l, a_s$: capacities of generators, lines, storages (MW).

- $PK^n_{\hat{j}}, RM_{\hat{j}}, DF_g$: peak demand, reserve-margin fraction, derating factor.

- $SC_g$: start-up cost; $GC^h_g(\cdot,\cdot)$: convex quadratic variable-cost function.

- $P^{\min}_g, P^{\max}_g$: generator output bounds.

- $SR^h_j$: spinning-reserve requirement; $RU_g, RD_g$: ramp rates; $L_g$: minimum up time.

- $D^h_j$: real-time demand; $F_l$: existing-line capacity; $B_{ij}$: transmission-loss fraction; $E_s$: withdrawal efficiency.

# Decision Variables

**Upper level (investment, per scenario-tree node $n$):**

- $x^n_g, x^n_l, x^n_s \in \{0,1\}$: build-decision binaries for generator, line, storage.

- $\kappa^n_g, \kappa^n_l, \kappa^n_s \ge 0$: available capacities (MW).

**Lower level (unit commitment, per hour $h$):**

- $\alpha^h_g \in \{0,1\}$: commitment status of generator $g$.

- $\gamma^h_g \in \{0,1\}$: start-up indicator of generator $g$.

- $p^h_g \ge 0$: energy generated (MWh).

- $s^h_g \ge 0$: spinning reserve (MWh).

- $u^h_s, v^h_s \ge 0$: storage withdrawal / injection (MWh).

- $r^h_s \ge 0$: remaining energy in storage (MWh).

- $f^h_{ij} \in \mathbb{R}$: *signed* energy flow on line $(i,j)$ (MWh), not split into directional components.

# Upper-Level Investment Model (1a)–(1g)

The expected total investment cost is $\min\sum_{n \in \mathcal{N}} \pi_n\,\text{TIC}^n(x_n)$, where $$\begin{align}
  \text{TIC}^n(x_n) := \min \quad & \sum_{g \in \mathbb{G}'} c^n_g x^n_g + \sum_{l \in \mathbb{L}'} c^n_l x^n_l + \sum_{s \in \mathbb{S}'} c^n_s x^n_s \tag{1a} \\
  \text{s.t.}\quad & \kappa^n_g = \kappa^{p(n)}_g + a_g x^n_g, && \forall\, g \in \mathbb{G}' \tag{1b} \\
  & \kappa^n_l = \kappa^{p(n)}_l + a_l x^n_l, && \forall\, l \in \mathbb{L}' \tag{1c} \\
  & \kappa^n_s = \kappa^{p(n)}_s + a_s x^n_s, && \forall\, s \in \mathbb{S}' \tag{1d} \\
  & \kappa^n_g \le a_g,\ \kappa^n_l \le a_l,\ \kappa^n_s \le a_s, && \forall\, g \in \mathbb{G}',\, l \in \mathbb{L}',\, s \in \mathbb{S}' \tag{1e} \\
  & \sum_{j:\,\hat{J}(j)=\hat{j}} \!\Bigl(\sum_{g \in \mathbb{G}_j} DF_g\, a_g + \sum_{g \in \mathbb{G}'_j} DF_g\, \kappa^n_g \Bigr) \ge (1 + RM_{\hat{j}})\, PK^n_{\hat{j}}, && \forall\, \hat{j} \tag{1f} \\
  & \kappa^n_g, \kappa^n_l, \kappa^n_s \ge 0,\ \ x^n_g, x^n_l, x^n_s \in \{0,1\}, && \forall\, g \in \mathbb{G}',\, l \in \mathbb{L}',\, s \in \mathbb{S}' \tag{1g}
\end{align}$$ (At the root node $n=1$, $\kappa^0_\cdot = 0$.)

# Lower-Level Unit-Commitment Model (2a)–(2o)

For each node $n$ and representative day $k \in K_n$, the operating cost is $$\begin{align}
  \min \quad & \text{TOC}^k_n = \sum_{h \in H_k} \sum_{g \in \mathbb{G} \cup \mathbb{G}'} \Bigl[ SC_g\, \gamma^h_g + \bigl(a + b\, p^h_g + c\, (p^h_g)^2\bigr) \Bigr] \tag{2a}
\end{align}$$ where the variable cost $GC^h_g(p^h_g, s^h_g) = a + b\, p^h_g + c\, (p^h_g)^2$ is *quadratic convex* in the generation level $p^h_g$ (coefficients $a,b,c \ge 0$ are generator-specific). Subject to $$\begin{align}
  & \alpha^h_g \le \kappa^n_g / a_g, && \forall\, g \in \mathbb{G}',\, h \in H_k \tag{2b} \\
  & \alpha^\tau_g - 1 \le \alpha^h_g - \alpha^{h-1}_g \le \alpha^\tau_g, && \tau = h, \ldots, \min\{h + L_g - 1, |H_k|\},\notag\\
  & && g \in \mathbb{G} \cup \mathbb{G}',\, h \in H_k \tag{2c} \\
  & \gamma^h_g \ge \alpha^h_g - \alpha^{h-1}_g, && \forall\, g \in \mathbb{G} \cup \mathbb{G}',\, h = 2,\ldots,|H_k| \tag{2d} \\
  & p^h_g + s^h_g \le
    \begin{cases}
      P^{\max}_g \alpha^h_g, & g \in \mathbb{G} \\
      \kappa^n_g, & g \in \mathbb{G}'
    \end{cases}, && \forall\, h \in H_k \tag{2e} \\
  & P^{\min}_g \alpha^h_g \le p^h_g, && \forall\, g \in \mathbb{G} \cup \mathbb{G}',\, h \in H_k \tag{2f} \\
  & \sum_{g \in \mathbb{G}_j \cup \mathbb{G}'_j} s^h_g \ge SR^h_j, && \forall\, j \in \mathbb{J},\, h \in H_k \tag{2g} \\
  & p^h_g - p^{h-1}_g \le RU_g,\ \ p^{h-1}_g - p^h_g \le RD_g, && \forall\, g \in \mathbb{G} \cup \mathbb{G}',\, h = 2, \ldots, |H_k| \tag{2h} \\
  & r^1_s = 0, \ \ r^h_s = r^{h-1}_s + E_s\, v^{h-1}_s - u^{h-1}_s, && \forall\, s \in \mathbb{S} \cup \mathbb{S}',\, h = 2,\ldots,|H_k| \tag{2i} \\
  & u^h_s \le r^h_s, && \forall\, s \in \mathbb{S} \cup \mathbb{S}',\, h \in H_k \tag{2j} \\
  & r^h_s \le \kappa^n_s, && \forall\, s \in \mathbb{S}',\, h \in H_k \tag{2k} \\
  & D^h_j + \sum_{i:(j,i) \in \mathbb{A} \cup \mathbb{A}'} f^h_{ji} - \sum_{i:(i,j) \in \mathbb{A} \cup \mathbb{A}'} (1 - B_{ij})\, f^h_{ij} \notag \\
  &\qquad = \sum_{g \in \mathbb{G}_j \cup \mathbb{G}'_j} p^h_g + \sum_{s \in \mathbb{S}_j \cup \mathbb{S}'_j} (u^h_s - v^h_s), && \forall\, j \in \mathbb{J},\, h \in H_k \tag{2l} \\
  & |f^h_{ij}| \le \kappa^n_{ij}, && \forall\, (i,j) \in \mathbb{A} \cup \mathbb{A}',\, h \in H_k \tag{2m} \\
  & \alpha^h_g,\ \gamma^h_g \in \{0,1\}, && \forall\, g \in \mathbb{G} \cup \mathbb{G}',\, h \in H_k \tag{2n} \\
  & p^h_g,\ s^h_g,\ r^h_s,\ u^h_s,\ v^h_s \ge 0,\ \ f^h_{ij} \in \mathbb{R}, && \forall\, g,\, (i,j),\, s,\, h \tag{2o}
\end{align}$$ Flow $f^h_{ij}$ is *signed* (not split into positive/negative directional components); its magnitude is bounded by line capacity $\kappa^n_{ij} \in \{F_l,\, \kappa^n_l\}$ in (2m).

# Complete Compact MM-SMIP Formulation (3a)–(3d)

$$\begin{align}
  [\text{MM-SMIP}]\!:\;\; \min_{x,y} \quad & \sum_{n \in \mathcal{N}} \pi_n \Bigl[ \text{TIC}^n(x_n) + \sum_{k \in K_n} \mathbf{E}_{\zeta_{n_k}} \min_{y^k_n} \text{TOC}\bigl(y^k_n(\zeta_{n_k})\bigr) \Bigr] \tag{3a} \\
  \text{s.t.}\quad & B\, y^k_n(\zeta_{n_k}) \le V(\zeta_{n_k})\Bigl( b + \sum_{m \in \mathscr{P}_n} A_m\, x_m \Bigr), && \forall\, n \in \mathcal{N},\, k \in K_n,\, \zeta_{n_k} \in \Xi^{n_k} \tag{3b} \\
  & x_n \in X_n \cap \{0,1\}^{G' + L' + S'}, && \forall\, n \in \mathcal{N} \tag{3c} \\
  & y^k_n(\zeta_{n_k}) \in Y^k_n(\zeta_{n_k}), && \forall\, n \in \mathcal{N},\, k \in K_n,\, \zeta_{n_k} \in \Xi^{n_k} \tag{3d}
\end{align}$$ Here (3b) bundles the linkage constraints (2b), (2e), (2k), (2m) coupling upper- and lower-level decisions; (3c) collects upper-level constraints (1b)–(1g); (3d) collects the remaining lower-level UC constraints.
