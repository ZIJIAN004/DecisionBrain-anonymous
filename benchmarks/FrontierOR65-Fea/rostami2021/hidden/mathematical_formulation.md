# Original Formulation: Stochastic Single-Allocation Hub Location with Variable Allocation (DEF$_V$)

Rostami, Kämmerling, Naoum-Sawaya, Buchheim, Clausen. “Stochastic single-allocation hub location,” *European Journal of Operational Research*, 2021. Section 3.2.

## Sets, Indices and Parameters

- $N = \{1,\dots,n\}$: set of nodes (possible hub locations, origins, destinations).

- $S_w = \{s_1,\dots,s_m\}$: finite support (scenarios) of the random parameter $\xi$, with probabilities $p_s = \mathbb{P}(\xi=s)$.

- $d_{ij}$: distance between nodes $i$ and $j$.

- $w_{ij}^s$: flow of items from node $i$ to node $j$ under scenario $s$; $O_i^s = \sum_{j\in N} w_{ij}^s$ (outgoing), $D_i^s = \sum_{j\in N} w_{ji}^s$ (incoming).

- $\chi$, $\alpha$, $\delta$: collection, transfer (inter-hub), and distribution costs per unit flow per unit distance.

- $f_k$: fixed set-up cost for locating a hub at node $k$.

- $c_{ik}^s = d_{ik}(\chi\, O_i^s + \delta\, D_i^s)$: scenario-dependent collection/distribution cost coefficient.

## Decision Variables

**First-stage** (location decisions): $$z_k \in \{0,1\}, \quad k\in N,\qquad z_k=1 \iff \text{a hub is opened at node }k.$$ **Second-stage** (scenario-dependent allocation of non-hub nodes): $$x_{ik}^s \in \{0,1\}, \quad i,k\in N,\ i\neq k,\ s\in S_w,$$ with $x_{ik}^s=1$ iff node $i$ is allocated to hub $k$ under scenario $s$.

## Deterministic Equivalent Formulation DEF$_V$ (Section 3.2, eqs. 25–28)

DEF$_V$ is the mixed-integer *quadratic* program: $$\begin{align}
\text{DEF}_V:\quad \min\ &
  \sum_{k\in N} f_k\, z_k
  + \sum_{s\in S_w} p_s \sum_{\substack{i,k\in N \\ i\neq k}} c_{ik}^s\, x_{ik}^s \notag\\
& {}+ \sum_{s\in S_w} p_s \sum_{i,j\in N} \alpha\, w_{ij}^s \Biggl[
  d_{ij}\, z_i\, z_j
  + \sum_{\substack{\ell\in N \\ \ell\neq j}} d_{i\ell}\, z_i\, x_{j\ell}^s \notag\\
& \qquad\qquad\qquad\qquad
  + \sum_{\substack{k\in N \\ k\neq i}} d_{kj}\, x_{ik}^s\, z_j
  + \sum_{\substack{k,\ell\in N \\ k\neq i,\ \ell\neq j}} d_{k\ell}\, x_{ik}^s\, x_{j\ell}^s
  \Biggr] \tag{DEF$_V$} \\
\text{s.t.}\ &
  \sum_{\substack{k\in N \\ k\neq i}} x_{ik}^s = 1 - z_i,
  && i\in N,\ s\in S_w \tag{25}\\
& x_{ik}^s \le z_k,
  && i,k\in N,\ i\neq k,\ s\in S_w \tag{26}\\
& z_i \in \{0,1\},
  && i\in N \tag{27}\\
& x_{ik}^s \in \{0,1\},
  && i,k\in N,\ i\neq k,\ s\in S_w. \tag{28}
\end{align}$$
