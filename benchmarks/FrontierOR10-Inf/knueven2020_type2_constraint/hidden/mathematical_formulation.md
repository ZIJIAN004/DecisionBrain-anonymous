# Original Formulation: Unit Commitment (1-binary-variable model)

*Source*: Knueven, Ostrowski, Watson (2020), “On Mixed Integer Programming Formulations for the Unit Commitment Problem”. The first concrete formulation presented in the paper (Slide 4) is the *1-bin* (one-binary-variable) unit commitment formulation.

## Sets and Indices

- $\mathcal{G}$: set of thermal generators, index $g \in \mathcal{G}$.

- $\mathcal{T}$: set of time periods, index $t \in \mathcal{T}$.

- $\mathcal{L}$: set of piecewise-linear production-cost segments, index $l \in \mathcal{L}$.

## Parameters (per generator $g$, indices suppressed)

- $\overline{P},\,\underline{P}$: maximum / minimum power output.

- $SU,\,SD$: start-up / shut-down ramp rates.

- $RU,\,RD$: ramp-up / ramp-down rates.

- $UT,\,DT$: minimum up-time / down-time.

- $L(t)$: system load (demand) at time $t$.

- $N(s)$: net injection from slack / transmission (possibly zero).

- $f^l$: marginal cost (slope) of piecewise segment $l$; $\overline{P}^{\,l}$ and $\overline{P}^{\,l-1}$: upper and lower breakpoints of segment $l$.

## Decision Variables (1-bin model)

- $p_g(t) \ge 0$: power output of generator $g$ at time $t$ (continuous).

- $\overline{p}_g(t) \ge 0$: power available / reserve-capable output of generator $g$ at time $t$ (continuous).

- $u_g(t) \in \{0,1\}$: on/off commitment status of generator $g$ at time $t$.

- $c_g(t) \ge 0$: production cost of generator $g$ at time $t$ (continuous).

## Objective

$$\begin{equation}
  \min \ \sum_{g \in \mathcal{G}} \sum_{t \in \mathcal{T}} c_g(t)
  \tag{1}
\end{equation}$$

## Constraints

#### Power balance (system level).

$$\begin{equation}
  \sum_{g \in \mathcal{G}} A_g\bigl(p_g,\,\overline{p}_g,\,u_g\bigr) + N(s) \;=\; L,
  \tag{2}
\end{equation}$$ where $A_g(\cdot)$ is the mapping from generator output variables to net power injection (reduces to $A_g(p_g,\overline{p}_g,u_g)(t) = p_g(t)$ in the copper-plate case).

#### Generator technical constraints.

For each generator $g \in \mathcal{G}$, $$\begin{equation}
  (p_g,\;\overline{p}_g,\;u_g,\;c_g) \;\in\; \Pi_g,
  \tag{3}
\end{equation}$$ where $\Pi_g$ encodes all generator-level technical restrictions on $(p_g(\cdot),\overline{p}_g(\cdot),u_g(\cdot),c_g(\cdot))$, namely: $$\begin{align}
  \underline{P}\, u_g(t) \;\le\; p_g(t) \;\le\; \overline{p}_g(t)
    &\;\le\; \overline{P}\, u_g(t),
    &&\forall t \in \mathcal{T}, \tag{4}\\[2pt]
  p_g(t) - p_g(t-1) &\;\le\; RU\, u_g(t-1) + SU\bigl(u_g(t) - u_g(t-1)\bigr),
    &&\forall t \in \mathcal{T}, \tag{5}\\[2pt]
  p_g(t-1) - p_g(t) &\;\le\; RD\, u_g(t) + SD\bigl(u_g(t-1) - u_g(t)\bigr),
    &&\forall t \in \mathcal{T}, \tag{6}\\[2pt]
  \sum_{\tau = t-UT+1}^{t} u_g(\tau) &\;\ge\; UT\bigl[u_g(t) - u_g(t-1)\bigr]^{+},
    &&\forall t \in \mathcal{T}, \tag{7}\\[2pt]
  \sum_{\tau = t-DT+1}^{t} \bigl(1-u_g(\tau)\bigr)
     &\;\ge\; DT\bigl[u_g(t-1) - u_g(t)\bigr]^{+},
    &&\forall t \in \mathcal{T}, \tag{8}\\[2pt]
  c_g(t) &\;=\; \sum_{l \in \mathcal{L}} f^l\, p_g^{\,l}(t) + (\text{no-load and start-up costs}),
    &&\forall t \in \mathcal{T}, \tag{9}\\[2pt]
  \sum_{l \in \mathcal{L}} p_g^{\,l}(t) &\;=\; p_g(t) - \underline{P}\, u_g(t),
    &&\forall t \in \mathcal{T}, \tag{10}\\[2pt]
  0 \;\le\; p_g^{\,l}(t)
     &\;\le\; \overline{P}^{\,l} - \overline{P}^{\,l-1},
    &&\forall l \in \mathcal{L},\, t \in \mathcal{T}. \tag{11}
\end{align}$$

## Variable Domains

$$\begin{equation}
  u_g(t) \in \{0,1\}, \qquad
  p_g(t),\,\overline{p}_g(t),\,p_g^{\,l}(t),\,c_g(t) \ge 0,
  \quad \forall g,t,l.
  \tag{12}
\end{equation}$$

*Note.* This is the 1-bin original formulation as presented on Slide 4 of Knueven, Ostrowski, Watson (2020). Start-up / shut-down indicators $v_g(t),\,w_g(t)$ and shortest-path / extended formulations are *reformulations* of this model and are not included here.
