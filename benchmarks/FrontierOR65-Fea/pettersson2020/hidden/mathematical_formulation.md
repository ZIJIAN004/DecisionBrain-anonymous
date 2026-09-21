# Original Formulation: Multi-Objective Integer Program (MOIP)

**Source.** Pettersson & Ozlen (2020), “Multi-objective integer programming: Synergistic parallel approaches,” Section 2.2.

The paper states the abstract multi-objective integer program (MOIP) that the proposed parallel algorithms operate on; single-objective integer leaf subproblems of the same abstract form are what is ultimately handed to the solver. For the knapsack problem instance used in the computational experiments, the integer vector specializes to a binary vector.

## Notation

- $n$ — number of objective functions (all minimized).

- $c$ — dimension of the decision vector.

- $f_1(\mathbf{x}),\ldots,f_n(\mathbf{x})$ — linear objective functions.

- $A$, $C$ — constraint matrix and right-hand-side vector.

- $\mathbf{x} \in \mathbb{Z}^c$ — integer decision vector.

## Abstract MOIP

$$\begin{align}
\min\quad
& \bigl(f_1(\mathbf{x}),\; f_2(\mathbf{x}),\; \ldots,\; f_n(\mathbf{x})\bigr) \tag{1}\\
\text{s.t.}\quad
& A\,\mathbf{x} \;\le\; C, \tag{2}\\
& \mathbf{x} \in \mathbb{Z}^c. \tag{3}
\end{align}$$

A feasible objective vector $(z_1,\ldots,z_n)$ is *dominated* by $(z'_1,\ldots,z'_n)$ iff $z'_k\le z_k$ for all $k$ with at least one strict inequality; the target is the set of non-dominated objective vectors.

## Binary Knapsack Specialization

For the 0-1 multi-objective knapsack instances of Section 4.2, (3) is replaced by $$\begin{align}
& \mathbf{x} \in \{0,1\}^c. \tag{3'}
\end{align}$$ All $f_\ell(\mathbf{x}) = -\mathbf{p}_\ell^{\top}\mathbf{x}$ are linear (equivalently, maximizing profits), and the single linear constraint $A\mathbf{x}\le C$ is the capacity constraint $\sum_{i=1}^{c} w_i x_i \le W$.
