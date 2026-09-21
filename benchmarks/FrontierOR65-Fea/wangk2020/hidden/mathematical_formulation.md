# Original Formulation: Integrated Model of Scheduling and Operations in Airport Networks (IMSOAN)

*Source: “A Stochastic Integer Programming Approach to Air Traffic Scheduling and Operations,” Kai Wang, Alexandre Jacquillat, Operations Research (Articles in Advance), 2020.*

IMSOAN is a biobjective two-stage stochastic integer program. The first stage optimizes strategic scheduling interventions (SI); the second stage is the multiairport ground-holding problem (MAGHO) solved per scenario. The block below reproduces Section 3.2 (“Model Formulation”) with the paper’s original equation tags.

## Sets and Indices

- $\mathcal{T}$: set of time periods, $t \in \{1,\ldots,T\}$.

- $\mathcal{F}$: set of flights, $i,j \in \{1,\ldots,F\}$.

- $\mathcal{H}$: set of airports, $k \in \{1,\ldots,K\}$.

- $\mathcal{S}$: set of operating scenarios, $s \in \{1,\ldots,S\}$.

- $\mathcal{E}^1 \subset \mathcal{F}\times\mathcal{F}$: flight pairs $(i,j)$ with an aircraft or passenger connection.

- $\mathcal{E}^2 \subset \mathcal{F}\times\mathcal{F}$: flight pairs $(i,j)$ with an aircraft connection.

- $\mathcal{T}_k^{\mathrm{dep}} / \mathcal{T}_k^{\mathrm{arr}}$: set of departing / arriving flights at airport $k \in \mathcal{H}$.

- $\mathcal{D}_k$: set of capacity (envelope-segment) constraints at airport $k \in \mathcal{H}$.

## Parameters

- $S_i^{\mathrm{dep}} / S_i^{\mathrm{arr}}$: requested departure / arrival time period of flight $i \in \mathcal{F}$.

- $\delta$: maximum displacement (same limit for all flights).

- $\bar{\mathcal{T}}_i^{\mathrm{dep}} = \{S_i^{\mathrm{dep}}-\delta+1,\ldots,S_i^{\mathrm{dep}}+\delta\}$: possible *scheduled* departure periods of flight $i$.

- $\bar{\mathcal{T}}_i^{\mathrm{arr}} = \{S_i^{\mathrm{arr}}-\delta+1,\ldots,S_i^{\mathrm{arr}}+\delta\}$: possible *scheduled* arrival periods of flight $i$.

- $g_{it} = |t - S_i^{\mathrm{dep}}|$, $\forall i \in \mathcal{F},\, t \in \bar{\mathcal{T}}_i^{\mathrm{dep}}$: displacement cost when flight $i$ departs in period $t$.

- $\tau_{ij}^{(1)}$: minimum connecting time between flights $i$ and $j$, for $(i,j) \in \mathcal{E}^1$.

- $\tau_{ij}^{(2)}$: minimum connecting time between flights $i$ and $j$, for $(i,j) \in \mathcal{E}^2$.

- $\Delta_i^{\min} / \Delta_i^{\max} / \Delta_i^{\mathrm{sch}}$: minimum / maximum / scheduled en-route time of flight $i$, with $\Delta_i^{\min} \le \Delta_i^{\mathrm{sch}} \le \Delta_i^{\max}$ and $\Delta_i^{\mathrm{sch}} = S_i^{\mathrm{arr}} - S_i^{\mathrm{dep}}$.

- $l_i^{\mathrm{dep}} / l_i^{\mathrm{arr}}$: maximum departure / arrival delay allowed for flight $i \in \mathcal{F}$.

- $\mathcal{T}_i^{\mathrm{dep}} = \{S_i^{\mathrm{dep}}-\delta+1,\ldots,S_i^{\mathrm{dep}}+\delta+l_i^{\mathrm{dep}}\}$: possible *operated* departure periods of flight $i$.

- $\mathcal{T}_i^{\mathrm{arr}} = \{S_i^{\mathrm{dep}}-\delta+\Delta_i^{\min}+1,\ldots,S_i^{\mathrm{arr}}+\delta+l_i^{\mathrm{arr}}\}$: possible *operated* arrival periods of flight $i$.

- $c_i^{\mathrm{dep}} / c_i^{\mathrm{arr}}$: unit cost of departure / arrival delay per period for flight $i \in \mathcal{F}$.

- $\Phi_{kt}$: random operating condition at airport $k \in \mathcal{H}$, time $t \in \mathcal{T}$ (VMC or IMC).

- $\phi_{kts}$: realization of the operating condition at airport $k$, time $t$, scenario $s \in \mathcal{S}$.

- $p_s$: probability of scenario $s \in \mathcal{S}$.

- $a_{kq}, b_{kq}, Q_{kq}(\phi)$: parameters of the piecewise-linear capacity envelope $a_{kq}X + b_{kq}Y \le Q_{kq}(\phi)$ at airport $k$ under operating condition $\phi$, for segment $q \in \mathcal{D}_k$ ($X$ = departures, $Y$ = arrivals).

- $\rho \in [0,1]$: weight balancing schedule displacement against expected delay cost.

## Decision Variables

**First-stage (SI) variables:**

- $w_{it}^{\mathrm{dep}} \in \{0,1\}$: $1$ if flight $i$ is scheduled to depart no earlier than period $t$, for $t \in \bar{\mathcal{T}}_i^{\mathrm{dep}}$; the sequence $(1,\ldots,1,0,\ldots,0)$ has its last $1$ at the scheduled departure period. By convention $w_{it}^{\mathrm{dep}}=1$ for $t \le S_i^{\mathrm{dep}}-\delta$ and $w_{it}^{\mathrm{dep}}=0$ for $t > S_i^{\mathrm{dep}}+\delta$.

- $w_{it}^{\mathrm{arr}} \in \{0,1\}$: $1$ if flight $i$ is scheduled to arrive no earlier than period $t$, for $t \in \bar{\mathcal{T}}_i^{\mathrm{arr}}$; analogous boundary convention.

**Second-stage (MAGHO) variables, per scenario $s \in \mathcal{S}$:**

- $x_{its}^{\mathrm{dep}} \in \{0,1\}$: $1$ if flight $i$ is operated (departs) no earlier than period $t$ in scenario $s$, for $t \in \mathcal{T}_i^{\mathrm{dep}}$.

- $x_{its}^{\mathrm{arr}} \in \{0,1\}$: $1$ if flight $i$ arrives no earlier than period $t$ in scenario $s$, for $t \in \mathcal{T}_i^{\mathrm{arr}}$.

- $v_{is}^{\mathrm{dep}} \ge 0$: departure delay of flight $i$ in scenario $s$.

- $v_{is}^{\mathrm{arr}} \ge 0$: arrival delay of flight $i$ in scenario $s$.

## Objective

$$\begin{align}
(\text{IMSOAN})\quad
\min\;\; & \rho \sum_{i \in \mathcal{F}} \sum_{t \in \bar{\mathcal{T}}_i^{\mathrm{dep}}}
   g_{it}\!\left(w_{i,t-1}^{\mathrm{dep}} - w_{it}^{\mathrm{dep}}\right)
   + (1-\rho)\,\mathbb{E}_{\Phi}\!\left[\Psi(w)\right],
   \tag{1a}
\end{align}$$ where the expected second-stage cost is $\mathbb{E}_{\Phi}[\Psi(w)] = \sum_{s \in \mathcal{S}} p_s\, \Psi(w,\phi_s)$.

## Constraints

#### First-stage (SI) constraints.

$$\begin{align}
\text{s.t.}\quad
& w_{it}^{\mathrm{dep}} \le w_{i,t-1}^{\mathrm{dep}}
   && \forall i \in \mathcal{F},\; t \in \bar{\mathcal{T}}_i^{\mathrm{dep}},
   \tag{1b}\\
& w_{it}^{\mathrm{arr}} \le w_{i,t-1}^{\mathrm{arr}}
   && \forall i \in \mathcal{F},\; t \in \bar{\mathcal{T}}_i^{\mathrm{arr}},
   \tag{1c}\\
& \sum_{t \in \mathcal{T}} \!\left(w_{it}^{\mathrm{arr}} - w_{it}^{\mathrm{dep}}\right) = \Delta_i^{\mathrm{sch}}
   && \forall i \in \mathcal{F},
   \tag{1d}\\
& \sum_{t \in \mathcal{T}} \!\left(w_{jt}^{\mathrm{dep}} - w_{it}^{\mathrm{arr}}\right) \ge \tau_{ij}^{(1)}
   && \forall (i,j) \in \mathcal{E}^1,
   \tag{1e}\\
& w_{it}^{\mathrm{dep}},\; w_{it'}^{\mathrm{arr}} \in \{0,1\}
   && \forall i \in \mathcal{F},\; t \in \bar{\mathcal{T}}_i^{\mathrm{dep}},\; t' \in \bar{\mathcal{T}}_i^{\mathrm{arr}}.
   \tag{1f}
\end{align}$$

#### Second-stage (MAGHO) recourse problem $\Psi(w,\phi_s)$ in scenario $s$.

$$\begin{align}
\Psi(w,\phi_s) = \min\;\;
& \sum_{i \in \mathcal{F}} \!\left(c_i^{\mathrm{dep}} v_{is}^{\mathrm{dep}} + c_i^{\mathrm{arr}} v_{is}^{\mathrm{arr}}\right)
   \tag{2a}\\
\text{s.t.}\quad
& x_{its}^{\mathrm{dep}} \le x_{i,t-1,s}^{\mathrm{dep}}
   && \forall i \in \mathcal{F},\; t \in \mathcal{T}_i^{\mathrm{dep}},
   \tag{2b}\\
& x_{its}^{\mathrm{arr}} \le x_{i,t-1,s}^{\mathrm{arr}}
   && \forall i \in \mathcal{F},\; t \in \mathcal{T}_i^{\mathrm{arr}},
   \tag{2c}\\
& \sum_{t \in \mathcal{T}} \!\left(x_{its}^{\mathrm{dep}} - w_{it}^{\mathrm{dep}}\right) = v_{is}^{\mathrm{dep}}
   && \forall i \in \mathcal{F},
   \tag{2d}\\
& \sum_{t \in \mathcal{T}} \!\left(x_{its}^{\mathrm{arr}} - w_{it}^{\mathrm{arr}}\right) \le v_{is}^{\mathrm{arr}}
   && \forall i \in \mathcal{F},
   \tag{2e}\\
& v_{is}^{\mathrm{dep}} \le l_i^{\mathrm{dep}}
   && \forall i \in \mathcal{F},
   \tag{2f}\\
& v_{is}^{\mathrm{arr}} \le l_i^{\mathrm{arr}}
   && \forall i \in \mathcal{F},
   \tag{2g}\\
& \sum_{t \in \mathcal{T}} \!\left(x_{jts}^{\mathrm{dep}} - x_{its}^{\mathrm{arr}}\right) \ge \tau_{ij}^{(2)}
   && \forall (i,j) \in \mathcal{E}^2,
   \tag{2h}\\
& \sum_{t \in \mathcal{T}} \!\left(x_{its}^{\mathrm{arr}} - x_{its}^{\mathrm{dep}}\right) \ge \Delta_i^{\min}
   && \forall i \in \mathcal{F},
   \tag{2i}\\
& \sum_{t \in \mathcal{T}} \!\left(x_{its}^{\mathrm{arr}} - x_{its}^{\mathrm{dep}}\right) \le \Delta_i^{\max}
   && \forall i \in \mathcal{F},
   \tag{2j}\\
& a_{kq} \sum_{i \in \mathcal{T}_k^{\mathrm{dep}}} \!\left(x_{i,t-1,s}^{\mathrm{dep}} - x_{its}^{\mathrm{dep}}\right)
   + b_{kq} \sum_{i \in \mathcal{T}_k^{\mathrm{arr}}} \!\left(x_{i,t-1,s}^{\mathrm{arr}} - x_{its}^{\mathrm{arr}}\right)
   \le Q_{kq}(\phi_{kts})
   \nonumber\\
& \hspace{4.2cm} \forall q \in \mathcal{D}_k,\; t \in \mathcal{T},\; k \in \mathcal{H},
   \tag{2k}\\
% Eqs. (2l) and (2m) are stated by the paper as VALID INEQUALITIES that tighten
% the LP relaxation of IMSOAN (proved in Appendix A); they are written inside the
% same model-formulation block (2a)-(2o) in Section 3.2 and are retained here as
% the paper presents them. They do not change the integer feasible region.
& x_{its}^{\mathrm{dep}} - w_{it}^{\mathrm{dep}} \ge 0
   && \forall i \in \mathcal{F},\; t \in \mathcal{T}_i^{\mathrm{dep}},
   \tag{2l}\\
& x_{i,\,t-(\Delta_i^{\mathrm{sch}}-\Delta_i^{\min}),\,s}^{\mathrm{arr}} - w_{it}^{\mathrm{arr}} \ge 0
   && \forall i \in \mathcal{F},\; t \in \mathcal{T}_i^{\mathrm{arr}},
   \tag{2m}\\
& x_{its}^{\mathrm{dep}},\; x_{it's}^{\mathrm{arr}} \in \{0,1\}
   && \forall i \in \mathcal{F},\; t \in \mathcal{T}_i^{\mathrm{dep}},\; t' \in \mathcal{T}_i^{\mathrm{arr}},
   \tag{2n}\\
& v_{is}^{\mathrm{dep}},\; v_{is}^{\mathrm{arr}} \ge 0
   && \forall i \in \mathcal{F}.
   \tag{2o}
\end{align}$$
