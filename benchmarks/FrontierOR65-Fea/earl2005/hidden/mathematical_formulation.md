# Original Formulation: Defensive Drill 1 MILP (Earl & D’Andrea, 2005)

**Source.** M. G. Earl and R. D’Andrea, “Multi-vehicle Cooperative Control Using Mixed Integer Linear Programming,” Sections II–IV. This file transcribes the paper’s formulation for the *one-on-one* defensive drill (one defender, one attacker). For the $N_D$-on-$N_A$ case the paper states that the constraints follow a “similar trend” but are not written out explicitly, so the one-on-one form is used here.

## Sets and Indices

- $i \in \{1,\ldots,N_D\}$ — defenders (here $N_D = 1$).

- $j \in \{1,\ldots,N_A\}$ — attackers (here $N_A = 1$; the index $j$ is dropped in the one-on-one case).

- $k \in \{0,\ldots,N_u-1\}$ — defender control steps.

- $k \in \{0,\ldots,N_a\}$ — attacker discretization steps.

- $k \in \{1,\ldots,N_o\}$ — obstacle (Defense Zone) avoidance checks.

- $m \in \{1,\ldots,M_u\}$ — sides of the control-input polygon.

- $m \in \{1,\ldots,M_I\}$ — sides of the intercept-region polygon.

- $m \in \{1,\ldots,M_{dz}\}$ — sides of the Defense Zone polygon.

- $m \in \{1,\ldots,M_o\}$ — sides of the obstacle polygon.

## Parameters

$T_u[k]>0$, $T_a[k]>0$ (step durations); $R_{dz}$ (Defense Zone radius); $R_{obst}$ (obstacle radius); $R_I$ (inscribed radius of intercept polygon); $H$ (big-M); $\epsilon_c>0$ (small constant for strict inequalities); $\epsilon\ge0$ (control-effort weight); $(v_{pj},v_{qj})$ (attacker constant velocity); $\mathbf{x}_{s,i}=(x_{s,i},y_{s,i},\dot x_{s,i},\dot y_{s,i})$ (defender initial state); $(p_{s,j},q_{s,j})$ (attacker initial position). Matrices $\mathbf{A}[k],\mathbf{B}[k]$ are defined in Eq. (6) of the paper from $T_u[k]$.

## Decision Variables

Continuous: $\mathbf{x}_{u,i}[k] = (x_i[k],y_i[k],\dot x_i[k],\dot y_i[k])$ (defender state); $\mathbf{u}_i[k] = (u_{xi}[k], u_{yi}[k])$ (defender control); $z_{xi}[k],\, z_{yi}[k] \ge 0$ (auxiliaries for $|u_{xi}[k]|,|u_{yi}[k]|$); $p_j[k],\, q_j[k]$ (attacker position).

Binary: $a_j[k]\in\{0,1\}$ (attack mode); $\gamma_j[k]\in\{0,1\}$ ($=1$ iff attacker $j$ is inside the Defense Zone at step $k$); $g_{mj}[k]\in\{0,1\}$ (auxiliary for $\gamma_j[k]$); $\delta_{ij}[k]\in\{0,1\}$ ($=1$ iff attacker $j$ is inside intercept region of defender $i$); $d_{mij}[k]\in\{0,1\}$ (auxiliary for $\delta_{ij}[k]$); $b_{mij}[k]\in\{0,1\}$ (auxiliary for defender obstacle avoidance).

## Objective (Eq. 44 of the paper)

$$\begin{align}
\min\ J \;=\; \sum_{j=1}^{N_A}\sum_{k=1}^{N_a} \gamma_j[k]
\;+\; \epsilon \sum_{i=1}^{N_D}\sum_{k=0}^{N_u-1}
\bigl(z_{xi}[k] + z_{yi}[k]\bigr). \tag{44}
\end{align}$$

## Constraints

#### Defender dynamics and initial condition.

For $i\in\{1,\ldots,N_D\}$, $k\in\{0,\ldots,N_u-1\}$: $$\begin{align}
\mathbf{x}_{u,i}[k+1] \;=\; \mathbf{A}[k]\,\mathbf{x}_{u,i}[k]
  + \mathbf{B}[k]\,\mathbf{u}_i[k], \qquad
\mathbf{x}_{u,i}[0] = \mathbf{x}_{s,i}. \tag{6}
\end{align}$$

#### Defender control-input feasibility (polygon approximation of the unit disk).

For each $i$, $k$, $m\in\{1,\ldots,M_u\}$: $$\begin{align}
u_{xi}[k]\sin\!\tfrac{2\pi m}{M_u}
 + u_{yi}[k]\cos\!\tfrac{2\pi m}{M_u}
 \;\le\; \cos\!\tfrac{\pi}{M_u}. \tag{8}
\end{align}$$

#### Absolute value auxiliaries for control.

For each $i$, $k$: $$\begin{align}
-z_{xi}[k] \le u_{xi}[k] \le z_{xi}[k], \tag{10a}\\
-z_{yi}[k] \le u_{yi}[k] \le z_{yi}[k]. \tag{10b}
\end{align}$$

#### Attacker dynamics.

For $j\in\{1,\ldots,N_A\}$, $k\in\{1,\ldots,N_a\}$: $$\begin{align}
p_j[k+1] &= p_j[k] + v_{pj}\,T_a[k]\,a_j[k], \tag{17a}\\
q_j[k+1] &= q_j[k] + v_{qj}\,T_a[k]\,a_j[k]. \tag{17b}
\end{align}$$

#### Attacker initial conditions.

For each $j$: $$\begin{align}
p_j[0] = p_{s,j},\quad q_j[0] = q_{s,j},\quad a_j[0] = 1. \tag{19}
\end{align}$$

#### Defense Zone indicator for attacker (polygon approximation).

For each $j$, $k\in\{1,\ldots,N_a\}$, $m\in\{1,\ldots,M_{dz}\}$: $$\begin{align}
p_j[k]\sin\!\tfrac{2\pi m}{M_{dz}} + q_j[k]\cos\!\tfrac{2\pi m}{M_{dz}}
  &\le R_{dz} + H(1 - g_{mj}[k]), \tag{23a}\\
p_j[k]\sin\!\tfrac{2\pi m}{M_{dz}} + q_j[k]\cos\!\tfrac{2\pi m}{M_{dz}}
  &\ge R_{dz} + \epsilon_c - (H+\epsilon_c)\,g_{mj}[k]. \tag{23b}
\end{align}$$ For each $j$, $k$ (and each $m$ in the first inequality): $$\begin{align}
g_{mj}[k] - \gamma_j[k] &\ge 0, \tag{25a}\\
\sum_{l=1}^{M_{dz}}\bigl(1 - g_{lj}[k]\bigr) + \gamma_j[k] &\ge 1. \tag{25b}
\end{align}$$

#### Intercept region indicator (defender $i$ vs. attacker $j$).

Let $(x_{a,i}[k],y_{a,i}[k])$ denote defender $i$’s position at attacker step time $t_{a,k}=\sum_{l=0}^{k-1}T_a[l]$, computed from $\mathbf{x}_{u,i}$ via Eq. (7) of the paper. For each $i$, $j$, $k\in\{1,\ldots,N_a\}$, $m\in\{1,\ldots,M_I\}$: $$\begin{align}
(p_j[k]-x_{a,i}[k])\sin\!\tfrac{2\pi m}{M_I}
  + (q_j[k]-y_{a,i}[k])\cos\!\tfrac{2\pi m}{M_I}
  &\le R_I + H(1 - d_{mij}[k]), \tag{29a}\\
(p_j[k]-x_{a,i}[k])\sin\!\tfrac{2\pi m}{M_I}
  + (q_j[k]-y_{a,i}[k])\cos\!\tfrac{2\pi m}{M_I}
  &\ge R_I + \epsilon_c - (H+\epsilon_c)\,d_{mij}[k]. \tag{29b}
\end{align}$$ For each $i$, $j$, $k$ (and each $m$ in the first): $$\begin{align}
d_{mij}[k] - \delta_{ij}[k] &\ge 0, \tag{31a}\\
\sum_{l=1}^{M_I}\bigl(1 - d_{lij}[k]\bigr) + \delta_{ij}[k] &\ge 1. \tag{31b}
\end{align}$$

#### Attacker state machine (one-on-one, Eq. 34).

For $k\in\{1,\ldots,N_a\}$: $$\begin{align}
a[k+1] + \delta[k]                       &\le 1, \tag{34a}\\
a[k+1] - a[k]                            &\le 0, \tag{34b}\\
a[k+1] + \gamma[k]                       &\le 1, \tag{34c}\\
a[k] - \delta[k] - \gamma[k] - a[k+1]    &\le 0. \tag{34d}
\end{align}$$ For the $N_D$-on-$N_A$ case the paper states that (34) is extended “in a similar way,” without writing the generalization explicitly.

#### Defender obstacle (Defense Zone) avoidance.

Let $(x_{o,i}[k],y_{o,i}[k])$ denote defender $i$’s position at obstacle check time $t_o[k]$ (from Eq. (7) of the paper), and $(x_{obst}[k],y_{obst}[k])$ the obstacle centre (taken as $(0,0)$ for the stationary Defense Zone). For each $i$, $k\in\{1,\ldots,N_o\}$, $m\in\{1,\ldots,M_o\}$: $$\begin{align}
(x_{o,i}[k]-x_{obst}[k])\sin\!\tfrac{2\pi m}{M_o}
  + (y_{o,i}[k]-y_{obst}[k])\cos\!\tfrac{2\pi m}{M_o}
  &\;>\; R_{obst} - H\,b_{mij}[k], \tag{14}\\
\sum_{m=1}^{M_o} b_{mij}[k] &\;\le\; M_o - 1. \tag{15}
\end{align}$$

## Variable Domains

$$\begin{align*}
& \mathbf{x}_{u,i}[k]\in\mathbb{R}^4,\ \mathbf{u}_i[k]\in\mathbb{R}^2,\
   z_{xi}[k],z_{yi}[k]\ge 0,\ (p_j[k],q_j[k])\in\mathbb{R}^2,\\
& a_j[k],\gamma_j[k],g_{mj}[k],\delta_{ij}[k],d_{mij}[k],b_{mij}[k]\in\{0,1\}.
\end{align*}$$
