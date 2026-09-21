# Original Formulation: Inventory-Routing Problem (IRP)

*Source: A Branch-Price-and-Cut Algorithm for the Inventory-Routing Problem, G. Desaulniers, J.G. Rakke, L.C. Coelho, Les Cahiers du GERAD G-2014-19, 2014.*

## Sets and Indices

$$\begin{align*}
& P = \{1, 2, \ldots, \rho\} && \text{set of periods in the planning horizon}\\
& \rho + 1 && \text{fictitious period to handle end inventories}\\
& N && \text{set of customers}\\
& 0 && \text{the supplier (depot)}\\
& R && \text{set of feasible routes}\\
& N_r && \text{set of customers visited in route } r \in R\\
& W^p_r && \text{set of extreme RDPs (route delivery patterns) when route } r\\
&&& \text{is used in period } p \in P\\
& P^+_{ip} && \text{periods of the sub-deliveries of a delivery to customer } i\\
&&& \text{in period } p \ (\text{defined below})\\
& P^-_{is} = \{ p \in P \mid s \in P^+_{ip} \} && \text{periods at which a sub-delivery can fulfill the demand of}\\
&&& \text{customer } i \text{ in period } s
\end{align*}$$ The set of sub-delivery periods is $$P^+_{ip} = \Big\{ s \in \{p{+}1, \ldots, \rho{+}1\} \;\Big|\;
\big(s \in P,\ \bar{d}^s_i > 0,\ \textstyle\sum_{\ell=p}^{s} d^\ell_i \le C_i\big)
\ \text{or}\ \big(s = \rho{+}1,\ \textstyle\sum_{\ell=p}^{s} d^\ell_i < C_i\big) \Big\}.$$

## Parameters

$$\begin{align*}
& d^p_0 && \text{quantity produced by the supplier in period } p \in P\\
& d^p_i && \text{demand (consumption) of customer } i \in N \text{ in period } p \in P\\
& C_i && \text{inventory (holding) capacity of customer } i \in N\\
& C_0 && \text{inventory capacity of the supplier}\\
& I^0_i && \text{initial inventory at customer } i \in N,\ I^0_i \le C_i\\
& I^0_0 && \text{initial inventory at the supplier,}\ I^0_0 \le C_0\\
& h_i && \text{unit holding cost at customer } i \in N\\
& h_0 && \text{unit holding cost at the supplier}\\
& Q && \text{vehicle capacity (homogeneous fleet)}\\
& K && \text{number of available vehicles}\\
& c_{ij} && \text{travel cost between locations } i \text{ and } j,\ i,j \in N \cup \{0\}\\
& a_{ri} && \text{1 if route } r \in R \text{ visits customer } i \in N,\ 0 \text{ otherwise}
\end{align*}$$

Residual inventory from the initial stock at customer $i$ at the end of period $s$: $$I^{0,s}_i = \max\Big\{0,\ I^0_i - \textstyle\sum_{\ell=1}^{s} d^\ell_i\Big\}, \qquad \forall i \in N,\ s \in P.$$ Residual demands: $$\bar{d}^s_i =
\begin{cases}
\max\{0,\ d^1_i - I^0_i\} & \text{if } s = 1\\[2pt]
\max\{0,\ d^s_i - I^{0,s-1}_i\} & \text{otherwise,}
\end{cases}
\qquad \forall i \in N,\ s \in P.$$ Upper bound on the quantity dedicated to each sub-delivery period $s \in P^+_{ip}$: $$u^s_{ip} =
\begin{cases}
\min\{\bar{d}^s_i,\ C_i - I^{0,s-1}_i\} & \text{if } s = p\\[2pt]
C_i - \sum_{\ell=p}^{s-1} d^\ell_i - I^{0,s-1}_i & \text{if } s = \rho+1\\[2pt]
\min\{\bar{d}^s_i,\ C_i - \sum_{\ell=p}^{s-1} d^\ell_i - I^{0,s-1}_i\} & \text{otherwise.}
\end{cases}$$ Route delivery pattern (RDP) parameters. An extreme RDP $w \in W^p_r$ specifies $q^s_{wi} \in [0, u^s_{ip}]$, the quantity delivered to customer $i \in N_r$ dedicated to period $s \in P^+_{ip}$ (an extreme RDP contains at most one partial sub-delivery, i.e. at most one $q^s_{wi} \in (0, u^s_{ip})$): $$\begin{align*}
& q_w = \textstyle\sum_{i \in N_r} \sum_{s \in P^+_{ip}} q^s_{wi} && \text{total quantity delivered (loaded at the supplier) in RDP } w\\
& b^s_{wi} && \text{quantity delivered to customer } i \in N_r \text{ in inventory at end of } s \in P^+_{ip}\\
& c_{rw} && \text{sum of travel costs and holding costs for route } r \text{ with RDP } w
\end{align*}$$

## Decision Variables

$$\begin{align*}
& y^p_{rw} \in [0,1] && \text{proportion of route } r \in R \text{ operated with extreme RDP } w \in W^p_r\\
&&& \text{in period } p \in P \ (\text{continuous})\\
& I^p_0 \ge 0 && \text{inventory at the supplier at the end of period } p \in P
\end{align*}$$

## Objective

$$\begin{equation}
\min \quad \sum_{p \in P} \sum_{r \in R} \sum_{w \in W^p_r} c_{rw}\, y^p_{rw} \;+\; \sum_{p \in P} h_0\, I^p_0 \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& I^{p-1}_0 + d^p_0 - \sum_{r \in R} \sum_{w \in W^p_r} q_w\, y^p_{rw} = I^p_0,
&& \forall p \in P, \tag{2}\\
& \sum_{p \in P^-_{is}} \sum_{r \in R} \sum_{w \in W^p_r} q^s_{wi}\, y^p_{rw} = \bar{d}^s_i,
&& \forall i \in N,\ s \in P \text{ such that } \bar{d}^s_i > 0, \tag{3}\\
& I^{0,s}_i + \sum_{p \in P^-_{is}} \sum_{r \in R} \sum_{w \in W^p_r} b^s_{wi}\, y^p_{rw} \le C_i - d^s_i,
&& \forall i \in N,\ s \in P, \tag{4}\\
& \sum_{r \in R} \sum_{w \in W^p_r} a_{ri}\, y^p_{rw} \le 1,
&& \forall i \in N,\ p \in P, \tag{5}\\
& \sum_{r \in R} \sum_{w \in W^p_r} y^p_{rw} \le K,
&& \forall p \in P, \tag{6}\\
& 0 \le I^p_0 \le C_0,
&& \forall p \in P, \tag{7}\\
& y^p_{rw} \ge 0,
&& \forall p \in P,\ r \in R,\ w \in W^p_r, \tag{8}\\
& \sum_{w \in W^p_r} y^p_{rw} \in \{0,1\},
&& \forall p \in P,\ r \in R. \tag{9}
\end{align}$$
