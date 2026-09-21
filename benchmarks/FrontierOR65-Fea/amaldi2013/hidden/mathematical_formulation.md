# Original Formulation: Minimum Hyperplanes Clustering Problem (<span class="smallcaps">Min</span>-HCP)

*Source: Column Generation for the Minimum Hyperplanes Clustering Problem, Edoardo Amaldi, Kanika Dhyani, and Alberto Ceselli, 2013 (INFORMS Journal on Computing, 25(3), pp. 446–460).*

## Sets and Indices

- $i \in \{1,\dots,n\}$ : index of the points.

- $j \in \{1,\dots,K\}$ : index of the potential hyperplanes.

- $d$ : dimension of the ambient Euclidean space $\mathbb{R}^d$.

- $K = \lceil n/d \rceil$ : upper bound on the optimal number $k$ of hyperplanes (any $d$ points in $d$ dimensions can be fitted with a hyperplane).

## Parameters

- $\mathbf{a}_i \in \mathbb{R}^d$ : coordinates of the $i$-th point, $1 \le i \le n$.

- $\epsilon > 0$ : maximum allowed tolerance (Euclidean point-to-hyperplane distance).

- $M > 0$ : large enough constant, taken as the largest inter-point Euclidean distance: $$\begin{equation*}
      M = \max_{\substack{1 \le i_1,\, i_2 \le n \\ i_1 \neq i_2}} \sqrt{\sum_{l=1}^{d} (a_{i_1 l} - a_{i_2 l})^2}. \tag{10}
  \end{equation*}$$

## Decision Variables

- $\mathbf{w}_j \in \mathbb{R}^d$, $w_j^0 \in \mathbb{R}$ : parameters (normal vector and offset) of hyperplane $\mathscr{H}_j = \{\mathbf{p} \in \mathbb{R}^d \mid \mathbf{p}\,\mathbf{w}_j = w_j^0\}$, $1 \le j \le K$.

- $D_{ij} \in \{0,1\}$ : $=1$ if point $i$ is assigned to the $j$-th $\epsilon$-$h$-cluster.

- $y_j \in \{0,1\}$ : $=1$ if the $\epsilon$-$h$-cluster (hyperplane) $j$ appears in the solution.

## Objective

$$\begin{equation}
  \min \; \sum_{j=1}^{K} y_j \tag{2}
\end{equation}$$

## Constraints

$$\begin{align}
  \frac{-(\mathbf{a}_i \mathbf{w}_j - w_j^0)}{\|\mathbf{w}_j\|_2} &\le \epsilon + M(1 - D_{ij}),
    & 1 \le i \le n,\; 1 \le j \le K, \tag{3} \\[4pt]
  \frac{(\mathbf{a}_i \mathbf{w}_j - w_j^0)}{\|\mathbf{w}_j\|_2} &\le \epsilon + M(1 - D_{ij}),
    & 1 \le i \le n,\; 1 \le j \le K, \tag{4} \\[4pt]
  \sum_{j=1}^{K} D_{ij} &\ge 1, & 1 \le i \le n, \tag{5} \\[4pt]
  D_{ij} &\le y_j, & 1 \le i \le n,\; 1 \le j \le K, \tag{6} \\[4pt]
  \mathbf{w}_j \in \mathbb{R}^d,\; w_j^0 &\in \mathbb{R}, & 1 \le j \le K, \tag{7} \\[4pt]
  D_{ij} &\in \{0,1\}, & 1 \le i \le n,\; 1 \le j \le K, \tag{8} \\[4pt]
  y_j &\in \{0,1\}, & 1 \le j \le K. \tag{9}
\end{align}$$

## Linearized Distance Constraints (same formulation section)

$$\begin{align}
  -(\mathbf{a}_i \mathbf{w}_j - w_j^0) &\le \epsilon + M(1 - D_{ij}),
    & 1 \le i \le n,\; 1 \le j \le K, \tag{11} \\[4pt]
  (\mathbf{a}_i \mathbf{w}_j - w_j^0) &\le \epsilon + M(1 - D_{ij}),
    & 1 \le i \le n,\; 1 \le j \le K, \tag{12} \\[4pt]
  \|\mathbf{w}_j\|_2 &= 1, & 1 \le j \le K. \tag{13}
\end{align}$$
