# Original Formulation: Stochastic Three-Level Lot Sizing and Replenishment Problem with a Distribution Structure (2S-3LSPD)

*Source: Benders decomposition for a stochastic three-level lot sizing and replenishment problem with a distribution structure, M. Gruson, J.-F. Cordeau and R. Jans, European Journal of Operational Research, 2021.*

The paper’s Section 3 first states a two-stage stochastic program on a random demand $\tilde d$ (Eqs. 1–10), which is intractable because of the expectation operator. To obtain a solvable model the authors discretize the uncertainty into a finite set of scenarios $\Omega$ and write the *multi-commodity (MC)* deterministic-equivalent formulation (Eqs. 11–19) below. The MC notation (scenario index $\omega$, probabilities $p_\omega$) is the one carried into the Benders reformulation and into every reported experiment, so it is the canonical original formulation of the problem. The two-stage origin (1–10) is reproduced in the Remarks; the lost-sales form (20) appears under Variants.

## Sets and Indices

$$\begin{align*}
&G=(F,A) && \text{distribution graph: nodes (facilities) $F$, arcs $A$}\\
&P=\{p\}\subset F && \text{singleton set containing the unique production plant $p$}\\
&W\subset F && \text{set of warehouses}\\
&R\subset F && \text{set of retailers}\\
&S(i) && \text{set of all direct successors of facility $i$}\\
&T && \text{set of time periods, indexed by $t$ (and by $k$ for production/order periods)}\\
&\Omega && \text{set of demand scenarios, indexed by $\omega$}
\end{align*}$$ Levels: level $0$ = production plant, level $1$ = warehouses, level $2$ = retailers.

## Parameters

$$\begin{align*}
&W(r) && \text{warehouse linked to retailer $r\in R$}\\
&d_{rt\omega} && \text{demand of retailer $r$ in period $t$ under scenario $\omega$}\\
&p_\omega && \text{probability of realization of scenario $\omega$}\\
&\delta_{kt} && \text{Kronecker delta: $1$ if $k=t$, $0$ otherwise}\\
&sc_{it} && \text{setup cost at facility $i$ in period $t$}\\
&hc_{pk} && \text{unit holding cost at the plant $p$ in period $k$}\\
&hc_{W(r),k} && \text{unit holding cost at the warehouse linked to $r$ in period $k$}\\
&hc_{rk} && \text{unit holding cost at retailer $r$ in period $k$}
\end{align*}$$ Holding costs are nondecreasing downstream: $hc_{pk}\le hc_{W(r),k}\le hc_{rk}$.

## Decision Variables

$$\begin{align*}
&y_{it}\in\{0,1\} && \text{$1$ iff there is production or an order placed by facility $i$ in period $t$}\\
&x^{0r}_{kt\omega}\ge 0 && \text{quantity produced at the plant in period $k$ to satisfy $d_{rt\omega}$}\\
&x^{1r}_{kt\omega}\ge 0 && \text{quantity ordered at the warehouse in period $k$ to satisfy $d_{rt\omega}$}\\
&x^{2r}_{kt\omega}\ge 0 && \text{quantity ordered at the retailer in period $k$ to satisfy $d_{rt\omega}$}\\
&\sigma^{0r}_{kt\omega}\ge 0 && \text{plant stock at the end of period $k$ for demand $d_{rt\omega}$}\\
&\sigma^{1r}_{kt\omega}\ge 0 && \text{warehouse stock at the end of period $k$ for demand $d_{rt\omega}$}\\
&\sigma^{2r}_{kt\omega}\ge 0 && \text{retailer stock at the end of period $k$ for demand $d_{rt\omega}$}
\end{align*}$$

## Objective

$$\begin{align}
\min\ \sum_{t\in T}\left(\sum_{i\in F} sc_{it}\,y_{it}
  + \sum_{\omega\in\Omega} p_\omega \sum_{r\in R}\sum_{k\le t}
  \Big( hc_{pk}\,\sigma^{0r}_{kt\omega} + hc_{W(r),k}\,\sigma^{1r}_{kt\omega}
        + hc_{rk}\,\sigma^{2r}_{kt\omega}\Big)\right)
\tag{11}
\end{align}$$

## Constraints

$$\begin{align}
& x^{1r}_{kt\omega} + \sigma^{0r}_{kt\omega} = \sigma^{0r}_{k-1,t,\omega} + x^{0r}_{kt\omega}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{12}\\
& x^{2r}_{kt\omega} + \sigma^{1r}_{kt\omega} = \sigma^{1r}_{k-1,t,\omega} + x^{1r}_{kt\omega}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{13}\\
& \delta_{kt}\,d_{rt\omega} + (1-\delta_{kt})\,\sigma^{2r}_{kt\omega} = \sigma^{2r}_{k-1,t,\omega} + x^{2r}_{kt\omega}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{14}\\
& x^{0r}_{kt\omega} \le d_{rt\omega}\,y_{pk}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{15}\\
& x^{1r}_{kt\omega} \le d_{rt\omega}\,y_{W(r),k}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{16}\\
& x^{2r}_{kt\omega} \le d_{rt\omega}\,y_{rk}
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{17}\\
& x^{0r}_{kt\omega},\, x^{1r}_{kt\omega},\, x^{2r}_{kt\omega},\,
  \sigma^{0r}_{kt\omega},\, \sigma^{1r}_{kt\omega},\, \sigma^{2r}_{kt\omega} \ge 0
  && \forall\, t\in T,\ k\le t,\ r\in R,\ \omega\in\Omega \tag{18}\\
& y_{it} \in \{0,1\}
  && \forall\, t\in T,\ i\in F \tag{19}
\end{align}$$
