# Original Formulation: Cardinality Constrained Mean-Variance Portfolio Optimization ($\mathcal{P}_{CCMV}$)

*Source: An Efficient Global Optimal Method for Cardinality Constrained Portfolio Optimization, Wei Xu, Jie Tang, Ka Fai Cedric Yiu, Jian Wen Peng, 2023 (INFORMS Journal on Computing 36(2):690–704).*

## Sets and Indices

- $i = 1, 2, \ldots, n$ : index of the risky assets in the asset pool.

## Parameters

- $n$ : total number of risky assets in the asset pool.

- $k \in \mathbb{N}$ : restriction on the number of invested assets (cardinality).

- $b \in \mathbb{R}$ : lowest target (required) level of portfolio return.

- $r_i$ : expected return (yield) of asset $i$, with $\mathbf{r} = [r_1, r_2, \ldots, r_n]^\top \in \mathbb{R}^n$.

- $\mathbf{Q} \in \mathbb{R}^{n \times n}$ : covariance matrix of the asset returns ($n$-dimensional symmetric positive definite).

- $\mathbf{e} = [1, 1, \ldots, 1]^\top \in \mathbb{R}^n$ : column vector of all ones.

- $\delta(\cdot)$ : indicator function used to determine which assets are selected, given by $$\delta(x) = \begin{cases} 0, & x = 0, \\ 1, & x \neq 0. \end{cases}$$

## Decision Variables

- $x_i \in \mathbb{R}$ : portfolio weight of the $i$-th asset, $i = 1, 2, \ldots, n$, with $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top \in \mathbb{R}^n$ (real-valued; short-selling permitted).

## Objective

$$\begin{align}
(\mathcal{P}_{CCMV}) \quad \min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad & \mathbf{r}^\top \mathbf{x} \geq b, \tag{1}\\
& \mathbf{e}^\top \mathbf{x} = 1, \tag{1}\\
& \sum_{i=1}^{n} \delta(x_i) \leq k. \tag{1}
\end{align}$$
