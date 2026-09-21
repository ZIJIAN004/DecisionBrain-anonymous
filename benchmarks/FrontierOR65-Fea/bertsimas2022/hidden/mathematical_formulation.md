# Original Formulation: Ridge-Regularized Sparse Portfolio Selection (SPS)

*Source: A Scalable Algorithm for Sparse Portfolio Selection, Dimitris Bertsimas and Ryan Cory-Wright, 2022.*

The problem the paper actually studies and solves to certifiable optimality is the ridge-regularized sparse portfolio selection model, Problem (4), introduced in Section 1.1 (“Problem Formulation and Main Contributions”) as the paper’s first main contribution. It augments the cardinality-constrained Markowitz model, Problem (2), with a ridge term $\tfrac{1}{2\gamma}\|\boldsymbol{x}\|_2^2$. Its notation ($\boldsymbol{x},\boldsymbol{\Sigma},\boldsymbol{\mu},\boldsymbol{A},\boldsymbol{l},\boldsymbol{u},k,\gamma$) is carried into every subsequent (algorithmic and experimental) section. The unregularized baseline (2), the convex mixed-integer quadratic reformulation (3), and the MISOCO reformulation (5) are listed under *Variants*.

## Sets and Parameters

- $n$ — number of securities in the universe; $[n] := \{1,\dots,n\}$.

- $m$ — number of generic linear inequality rows.

- $k \in \mathbb{Z}_{>0}$ — cardinality budget; maximum number of non-zero positions, with $k \ll n$.

- $\boldsymbol{\mu} \in \mathbb{R}^n$ — vector of expected marginal returns.

- $\boldsymbol{\Sigma} \in \mathbb{S}^n_+$ — positive semidefinite variance-covariance matrix of returns.

- $\sigma \geq 0$ — scalar trade-off parameter controlling the portfolio’s risk-return trade-off.

- $\gamma > 0$ — fixed ridge (concentration) regularization parameter.

- $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ — linear constraint matrix; $\boldsymbol{l} \in \mathbb{R}^m,\ \boldsymbol{u} \in \mathbb{R}^m$ — lower and upper bounds for the generic linear inequalities.

- $\boldsymbol{e} \in \mathbb{R}^n$ — vector of all ones.

## Decision Variables

- $\boldsymbol{x} \in \mathbb{R}^n_+$ — portfolio allocation vector (fraction of capital in each security; continuous, non-negative, i.e. no short selling).

The cardinality requirement is expressed directly on $\boldsymbol{x}$ via the $\ell_0$ “norm” $\|\boldsymbol{x}\|_0 = |\{ i : x_i \neq 0 \}|$. Binary indicators $\boldsymbol{z} \in \{0,1\}^n$ ($z_i = 1$ iff $x_i \neq 0$) are introduced only in the reformulations under *Variants*.

## Objective

$$\begin{equation}
  \min_{\boldsymbol{x} \in \mathbb{R}^n_+}\;
    \frac{\sigma}{2}\, \boldsymbol{x}^\top \boldsymbol{\Sigma} \boldsymbol{x}
    + \frac{1}{2\gamma}\, \|\boldsymbol{x}\|_2^2
    - \boldsymbol{\mu}^\top \boldsymbol{x}
  \tag{4-obj}
\end{equation}$$

## Constraints

$$\begin{align}
  \boldsymbol{l} \leq \boldsymbol{A}\boldsymbol{x} \leq \boldsymbol{u}, \qquad
  \boldsymbol{e}^\top \boldsymbol{x} = 1, \qquad
  \|\boldsymbol{x}\|_0 \leq k.
  \tag{4}
\end{align}$$ (The objective and constraints together constitute Problem (4) of the paper, tagged (4) in the source.)
