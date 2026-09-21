# Sets and Parameters

- $j = 1,\ldots,n$: customers.

- $i = 1,\ldots,m$: facilities (each is a production plant with associated warehouse).

- $t = 1,\ldots,T$: time periods in the planning horizon.

- $d_j$: total demand of customer $j$ over the planning horizon.

- $\sigma_t$: (nonnegative) seasonal factor for period $t$, with $\sum_{t=1}^{T} \sigma_t = 1$. Customer $j$’s demand in period $t$ equals $\sigma_t \, d_j$.

- $b_{it}$: production capacity at facility $i$ in period $t$.

- $c_{ijt}$: cost of supplying customer $j$ by facility $i$ in period $t$.

- $h_{it}$: unit inventory holding cost at facility $i$ in period $t$.

All parameters are nonnegative.

# Decision Variables

$$\begin{align*}
x_{ij} &\in \{0,1\} && i = 1,\ldots,m;\; j = 1,\ldots,n \quad (\text{1 if customer $j$ is assigned to facility $i$}) \\
I_{it} &\geq 0 && i = 1,\ldots,m;\; t = 1,\ldots,T \quad (\text{inventory at facility $i$ at the end of period $t$})
\end{align*}$$

# Objective

$$\begin{equation}
\text{minimize} \quad \sum_{t=1}^{T} \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ijt}\, x_{ij} \;+\; \sum_{t=1}^{T} \sum_{i=1}^{m} h_{it}\, I_{it} \tag{$P_0$}
\end{equation}$$

# Constraints

$$\begin{align}
\sigma_t \sum_{j=1}^{n} d_j\, x_{ij} \;+\; I_{it} &\leq b_{it} + I_{i,t-1}, && i = 1,\ldots,m;\; t = 1,\ldots,T \tag{1} \\
\sum_{i=1}^{m} x_{ij} &= 1, && j = 1,\ldots,n \notag \\
x_{ij} &\in \{0,1\}, && i = 1,\ldots,m;\; j = 1,\ldots,n \notag \\
I_{i0} &= 0, && i = 1,\ldots,m \notag \\
I_{it} &\geq 0, && i = 1,\ldots,m;\; t = 1,\ldots,T \notag
\end{align}$$
