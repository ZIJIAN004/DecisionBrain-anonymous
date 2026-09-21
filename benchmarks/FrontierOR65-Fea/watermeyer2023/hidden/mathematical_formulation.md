# Original Formulation: Project Duration Problem with Partially Renewable Resources and General Temporal Constraints (RCPSP/max-$\pi$)

*Source: A constructive branch-and-bound algorithm for the project duration problem with partially renewable resources and general temporal constraints, Kai Watermeyer & Jürgen Zimmermann, 2023 (Journal of Scheduling 26:95–111).*

**Note on the source formulation.** In Section 2 (“Problem description”) the paper states the problem compactly as the abstract optimization problem (P): minimize $f(S)=S_{n+1}$ over the set $\mathcal{S}=\mathcal{S}_T\cap\mathcal{S}_R$ of all feasible schedules. The defining conditions of the time-feasible set $\mathcal{S}_T$ and the resource-feasible set $\mathcal{S}_R$ are given inline in the same section; they are reproduced below as the explicit constraint block. The paper does *not* provide an explicit MIP/linearization: the resource usage $r^u_{ik}(S_i)$ (hence the resource consumption $r^c_{ik}(S_i)$) is a nonlinear function of the start time $S_i$. The formulation below is therefore the conceptual program (P) that the paper’s branch-and-bound algorithm solves, with constraints stated exactly as in Section 2.

## Sets and Indices

$$\begin{align*}
V &= \{0, 1, \ldots, n, n+1\} && \text{set of all activities (nodes), including}\\
  &&& \text{fictitious project start } 0 \text{ and project end } n+1\\
E &\subseteq V \times V && \text{arc set of the activity-on-node network } N\\
  &&& \text{(start-to-start precedence/temporal relations)}\\
\mathcal{R} &&& \text{set of partially renewable resources}\\
\Pi_k &\subseteq \{1, 2, \ldots, \bar{d}\}, \quad k \in \mathcal{R} && \text{subset of time periods over which resource } k \text{ is defined}\\
V_k &:= \{\, i \in V \mid r^d_{ik} > 0 \,\}, \quad k \in \mathcal{R} && \text{activities with positive demand for resource } k
\end{align*}$$

## Parameters

$$\begin{align*}
n &&& \text{number of real activities}\\
p_i &\in \mathbb{Z}_{\geq 0}, \quad i \in V && \text{processing time of activity } i \;\; (p_0 = p_{n+1} = 0)\\
\delta_{ij} &\in \mathbb{Z}, \quad (i,j) \in E && \text{arc weight / time-lag of arc } (i,j)\\
  &&& (\delta_{ij}\geq 0:\text{ minimum time lag};\;\; \delta_{ji}<0:\text{ maximum time lag from } i \text{ to } j)\\
r^d_{ik} &\in \mathbb{Z}_{\geq 0}, \quad i \in V,\, k \in \mathcal{R} && \text{demand of activity } i \text{ for partially renewable resource } k\\
R_k &\in \mathbb{Z}_{\geq 0}, \quad k \in \mathcal{R} && \text{capacity of partially renewable resource } k\\
\bar{d} &&& \text{prescribed project deadline}
\end{align*}$$

## Decision Variables

$$\begin{align*}
S &= (S_i)_{i \in V}, \quad S_i \in \mathbb{Z}_{\geq 0} && \text{schedule: } S_i \text{ is the (integer) start time of activity } i
\end{align*}$$

*Derived (nonlinear) quantities used in the resource constraints, as defined in Section 2:* $$\begin{align*}
r^u_{ik}(S_i) &:= \bigl|\, (S_i,\, S_i + p_i] \cap \Pi_k \,\bigr|
  && \text{resource usage: number of periods of } \Pi_k \text{ in the}\\
  &&& \text{half-open execution interval } (S_i, S_i+p_i]\\
r^c_{ik}(S_i) &:= r^u_{ik}(S_i)\cdot r^d_{ik}
  && \text{resource consumption of activity } i \text{ for resource } k
\end{align*}$$

## Objective

The paper states the problem (P): determine a feasible schedule $S^\ast$ of shortest project duration. $$\begin{align}
\text{Minimize} \quad & f(S) = S_{n+1}
  \qquad \text{subject to} \quad S \in \mathcal{S} = \mathcal{S}_T \cap \mathcal{S}_R \tag{P}
\end{align}$$

## Constraints

The feasibility conditions defining $\mathcal{S}_T$ (time-feasible) and $\mathcal{S}_R$ (resource-feasible), as given in Section 2: $$\begin{align}
S_j &\geq S_i + \delta_{ij}
  && \forall\, (i,j) \in E \tag{T}\\[2pt]
S_0 &= 0 \tag{Init}\\[2pt]
S_{n+1} &\leq \bar{d} \tag{Dl}\\[2pt]
\sum_{i \in V} r^c_{ik}(S_i) &\leq R_k
  && \forall\, k \in \mathcal{R} \tag{Res}\\[2pt]
S_i &\in \mathbb{Z}_{\geq 0}
  && \forall\, i \in V \tag{Int}
\end{align}$$
