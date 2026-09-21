# Original Formulation: Bilevel Second-Order Cone Program (BSOCP)

*Source: Benders Subproblem Decomposition for Bilevel Problems with Convex Follower, Byeon and Van Hentenryck, 2022.*

## Sets and Parameters

- $n_x, n_y$: dimensions of the leader and follower variable vectors.

- $m_x, m_y$: numbers of leader and follower constraints.

- $\mathcal{I} \subseteq \{1,\dots,n_x\}$: index set of integer leader variables.

- $\mathcal{J} \subseteq \mathcal{I}$: indices $i$ such that column $i$ of $A$ is nonzero (leader variables that appear in the follower problem).

- $c_x \in \mathbb{R}^{n_x}$, $c_y \in \mathbb{R}^{n_y}$: leader objective coefficients.

- $G_x \in \mathbb{R}^{m_x \times n_x}$, $G_y \in \mathbb{R}^{m_x \times n_y}$, $h \in \mathbb{R}^{m_x}$: leader constraint data.

- $A \in \mathbb{R}^{m_y \times n_x}$, $B \in \mathbb{R}^{m_y \times n_y}$, $b \in \mathbb{R}^{m_y}$, $d \in \mathbb{R}^{n_y}$: follower constraint and objective data.

- $\underline{x}_i, \overline{x}_i$: lower and upper bounds on $x_i$ (finite for $i \in \mathcal{J}$).

- $\mathcal{K}_x, \mathcal{K}_y$: Cartesian products of second-order cones and nonnegative orthants (the ambient cones for $x$ and $y$).

## Decision Variables

- $x \in \mathbb{R}^{n_x}$: leader (upper-level) decision variables.

- $y \in \mathbb{R}^{n_y}$: follower (lower-level) decision variables.

## Objective

$$\begin{equation}
\min_{x,\,y} \quad c_x^{\top} x + c_y^{\top} y \tag{1a}
\end{equation}$$

## Constraints (Bilevel)

$$\begin{align}
G_x x + G_y y & \;\geq\; h, \tag{1b} \\[2pt]
x & \;\in\; \mathcal{X} \;:=\; \Bigl\{ x \in \mathcal{K}_x \;:\;
      x_i \in [\underline{x}_i,\overline{x}_i] \cap \mathbb{Z},\;\forall i \in \mathcal{I} \Bigr\}, \tag{1c} \\[2pt]
y & \;\in\; \arg\min_{y' \in \mathcal{K}_y}
      \Bigl\{\, d^{\top} y' \;:\; A x + B y' \geq b \,\Bigr\}. \tag{1d}
\end{align}$$

Constraint (1d) enforces that $y$ is an optimal response of the follower to the leader decision $x$ (optimistic bilevel); the problem is an MISOCP-follower bilevel program. Under Assumption 2(b) integer bounded $x_i$ for $i \in \mathcal{J}$ may be encoded as binary without loss of generality.
