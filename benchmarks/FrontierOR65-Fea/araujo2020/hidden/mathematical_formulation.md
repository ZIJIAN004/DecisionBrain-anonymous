# Original Formulation: Multi-mode Resource-Constrained Multi-Project Scheduling Problem (MMRCMPSP)

*Source: Strong Bounds for Resource Constrained Project Scheduling: Preprocessing and Cutting Planes, Janniele A. S. Araujo, Haroldo G. Santos, Bernard Gendron, Sanjay Dominik Jena, Samuel S. Brito, Danilo S. Souza, 2019.*

This is the time-indexed (pulse discrete-time, PDT) integer programming formulation of Section 3.3, which the paper introduces as the definition of the problem and carries into all subsequent (cutting-plane / experimental) sections. It unifies the three variants studied in the paper (SMRCPSP, MMRCPSP, MMRCMPSP).

## Sets and Indices

$$\begin{align*}
\mathcal{P}        &: \text{set of all projects } p;\\
\mathcal{J}        &: \text{set of all jobs } j;\\
\mathcal{M}_j      &: \text{set of modes available for job } j \in \mathcal{J}, \text{ index } m;\\
\mathcal{J}_p      &: \text{set of jobs belonging to project } p, \quad \mathcal{J}_p \subseteq \mathcal{J}\ \ \forall p \in \mathcal{P};\\
\mathcal{K}        &: \text{set of non-renewable resources, index } k;\\
\mathcal{R}        &: \text{set of renewable resources, index } r;\\
\mathcal{S}        &: \text{set of direct precedence relationships } (j,s) \in \mathcal{J} \times \mathcal{J};\\
\mathcal{S}_j      &: \text{direct successors } s \text{ of job } j \ \text{(used in constraint (11))};\\
\mathcal{T} \subset \mathbb{Z}^+ &: \text{set of time periods in the planning horizon, index } t;\\
\mathcal{T}_{jm} \subseteq \mathcal{T} &: \text{time horizon for job } j \in \mathcal{J} \text{ on mode } m \in \mathcal{M}_j \ \text{(after preprocessing).}
\end{align*}$$

## Parameters

$$\begin{align*}
d_{jm} \in \mathbb{Z}^+        &: \text{duration of job } j \text{ on mode } m \in \mathcal{M}_j;\\
q_{kjm} \in \mathbb{Z}^+       &: \text{amount of non-renewable resource } k \in \mathcal{K} \text{ to execute job } j \text{ on mode } m;\\
q_{rjm} \in \mathbb{Z}^+       &: \text{amount of renewable resource } r \in \mathcal{R} \text{ to execute job } j \text{ on mode } m;\\
\breve{q}_k \in \mathbb{Z}^+   &: \text{available amount of non-renewable resource } k \in \mathcal{K};\\
\breve{q}_r \in \mathbb{Z}^+   &: \text{available amount of renewable resource } r \in \mathcal{R};\\
\sigma_p \in \mathcal{T}       &: \text{release date of project } p;\\
\lambda_p                      &: \text{critical path duration (CPD) lower bound of project } p;\\
a_p \in \mathcal{J}_p          &: \text{artificial job representing the end of project } p;\\
\epsilon > 0                   &: \text{small tie-breaking coefficient (value not specified in paper).}
\end{align*}$$ (The horizons $\mathcal{T}$ and $\mathcal{T}_{jm}$ follow from preprocessing Eqs. (4)–(6) via $\alpha=\sum_{p\in\mathcal{P}}(\beta_p-\sigma_p-\lambda_p)$, $\breve{t}=\max_{p\in P}(\sigma_p+\lambda_p+\alpha)$, $\mathcal{T}=\{0,\dots,\breve{t}\}$, and $\mathcal{T}_{jm}=\{\breve{e}^s_j,\dots,\breve{l}^s_{jm}\}$; these only restrict the index sets above.)

## Decision Variables

$$\begin{align*}
x_{jmt} \in \{0,1\} &: 1 \text{ if job } j \in \mathcal{J} \text{ is allocated on mode } m \in \mathcal{M}_j \text{ at starting time } t \in \mathcal{T}_{jm};\\
z_{jmt} \in \{0,1\} &: 1 \text{ if job } j \in \mathcal{J} \text{ on mode } m \in \mathcal{M}_j \text{ is being processed during time } t \in \mathcal{T}_{jm};\\
h \in \mathbb{Z}^+  &: \text{integer variable used to compute the total makespan (tie-breaker).}
\end{align*}$$

## Objective

$$\begin{equation}
\text{Minimize} \quad
\sum_{p \in \mathcal{P}} \sum_{m \in \mathcal{M}_{a_p}} \sum_{t \in \mathcal{T}_{a_p m}}
\left[ t - (\sigma_p + \lambda_p) \right] x_{a_p m t} \;+\; \epsilon\, h
\tag{7}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{m \in \mathcal{M}_j} \sum_{t \in \mathcal{T}_{jm}} x_{jmt} = 1
  && \forall j \in \mathcal{J} \tag{8}\\[4pt]
& \sum_{j \in \mathcal{J}} \sum_{m \in \mathcal{M}_j} \sum_{t \in \mathcal{T}_{jm}} q_{kjm}\, x_{jmt} \le \breve{q}_k
  && \forall k \in \mathcal{K} \tag{9}\\[4pt]
& \sum_{j \in \mathcal{J}} \sum_{m \in \mathcal{M}_j} q_{rjm}\, z_{jmt} \le \breve{q}_r
  && \forall r \in \mathcal{R},\ \forall t \in \mathcal{T} \tag{10}\\[4pt]
& \sum_{m \in \mathcal{M}_j} \sum_{t \in \mathcal{T}_{jm}} (t + d_{jm})\, x_{jmt}
  \;-\; \sum_{z \in \mathcal{M}_s} \sum_{i \in \mathcal{T}_{sz}} i\, x_{szi} \le 0
  && \forall j \in \mathcal{J},\ \forall s \in \mathcal{S}_j \tag{11}\\[4pt]
& z_{jmt} - \sum_{t' = (t - d_{jm} + 1)}^{t} x_{jmt'} = 0
  && \forall j \in \mathcal{J},\ \forall m \in \mathcal{M}_j,\ \forall t \in \mathcal{T}_{jm} \tag{12}\\[4pt]
& h - \sum_{m \in \mathcal{M}_{a_p}} \sum_{t \in \mathcal{T}_{a_p m}} t\, x_{a_p m t} \ge 0
  && \forall p \in \mathcal{P} \tag{13}\\[4pt]
& x_{jmt} \in \{0,1\}
  && \forall j \in \mathcal{J},\ \forall m \in \mathcal{M}_j,\ \forall t \tag{14}\\[4pt]
& z_{jmt} \in \{0,1\}
  && \forall j \in \mathcal{J},\ \forall m \in \mathcal{M}_j,\ \forall t \in \mathcal{T}_{jm} \tag{15}\\[4pt]
& h \ge 0 \tag{16}
\end{align}$$
