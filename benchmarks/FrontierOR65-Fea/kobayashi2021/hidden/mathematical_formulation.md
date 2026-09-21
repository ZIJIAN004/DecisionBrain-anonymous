# Original Formulation: Cardinality-constrained Mean-CVaR Portfolio Optimization (CCMC)

*Source: Bilevel Cutting-plane Algorithm for Solving Cardinality-constrained Mean-CVaR Portfolio Optimization Problems, Ken Kobayashi, Yuichi Takano, Kazuhide Nakata, 2021.*

## Sets and Indices

$$\begin{align*}
\mathcal{N} &:= \{1, 2, \ldots, N\} && \text{index set of $N$ assets ($n \in \mathcal{N}$),}\\
\mathcal{S} &:= \{1, 2, \ldots, S\} && \text{index set of $S$ return scenarios ($s \in \mathcal{S}$).}
\end{align*}$$

## Parameters

$$\begin{align*}
k                 &\in \mathbb{Z}_{>0} && \text{cardinality limit (max number of assets held),}\\
\beta             &\in (0,1)           && \text{probability level for CVaR (set close to one),}\\
\gamma            &> 0                 && \text{$\ell_2$-regularization parameter,}\\
\boldsymbol{\mu}  &:= (\mu_1,\ldots,\mu_N)^\top && \text{vector of expected returns of assets,}\\
\bar{\mu}         &\in \mathbb{R}      && \text{required return level,}\\
\boldsymbol{A}    &\in \mathbb{R}^{M \times N},\ \boldsymbol{b} \in \mathbb{R}^{M} && \text{matrix/vector defining linear portfolio constraints $\boldsymbol{A}\boldsymbol{x}\le\boldsymbol{b}$,}\\
\boldsymbol{r}^{(s)} &:= (r_1^{(s)},\ldots,r_N^{(s)})^\top && \text{asset return vector in scenario $s$,}\\
p_s               &\ge 0,\ \boldsymbol{p}:=(p_1,\ldots,p_S)^\top && \text{occurrence probability of scenario $s$.}
\end{align*}$$

The feasible portfolio set and the cardinality feasible set are $$\begin{align}
\mathcal{X}     &:= \{\boldsymbol{x} \in \mathbb{R}^N \mid \boldsymbol{A}\boldsymbol{x} \le \boldsymbol{b},\ \boldsymbol{1}^\top \boldsymbol{x} = 1,\ \boldsymbol{x} \ge \boldsymbol{0}\}, \tag{1}\\
\mathcal{Z}_N^k &:= \left\{\boldsymbol{z} \in \{0,1\}^N \;\middle|\; \sum_{n \in \mathcal{N}} z_n \le k \right\},
\end{align}$$ where $\boldsymbol{A}\boldsymbol{x} \le \boldsymbol{b}$ is supposed to contain the expected return constraint $$\begin{equation}
\boldsymbol{\mu}^\top \boldsymbol{x} \ge \bar{\mu}. \tag{6}
\end{equation}$$

## Decision Variables

$$\begin{align*}
\boldsymbol{x} &:= (x_1,\ldots,x_N)^\top \in \mathbb{R}^N_{\ge 0} && \text{portfolio weights ($x_n$ = investment weight of asset $n$),}\\
\boldsymbol{z} &:= (z_1,\ldots,z_N)^\top \in \{0,1\}^N && \text{asset selection ($z_n=1$ iff asset $n$ is selected),}\\
a              &\in \mathbb{R} && \text{auxiliary variable corresponding to $\beta$-VaR,}\\
v              &\ge 0 && \text{auxiliary variable (tail-risk / CVaR contribution).}
\end{align*}$$

## Objective

$$\begin{equation}
\underset{a,\, v,\, \boldsymbol{x},\, \boldsymbol{z}}{\text{minimize}} \quad
  \frac{1}{2\gamma}\boldsymbol{x}^\top \boldsymbol{x} + a + v
\tag{7a}
\end{equation}$$

## Constraints

$$\begin{align}
\text{subject to} \quad
  & v \ge \frac{1}{1-\beta} \sum_{s \in \mathcal{S}} p_s
      \left[\, -(\boldsymbol{r}^{(s)})^\top \boldsymbol{x} - a \,\right]_+, \tag{7b}\\
  & z_n = 0 \;\Rightarrow\; x_n = 0 \quad (\forall n \in \mathcal{N}), \tag{7c}\\
  & \boldsymbol{x} \in \mathcal{X}, \quad \boldsymbol{z} \in \mathcal{Z}_N^k, \tag{7d}
\end{align}$$ where $[\xi]_+ := \max\{0,\xi\}$ denotes the positive part of $\xi$, $v$ is an auxiliary decision variable, and $\gamma>0$ is a regularization parameter.
