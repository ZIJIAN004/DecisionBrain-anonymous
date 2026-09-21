# Original Formulation: Benders Dual Decomposition – Generic MILP and Stochastic Capacitated Facility Location (SFL)

Rahmaniani, Ahmed, Crainic, Gendreau, Rei. “The Benders Dual Decomposition Method,” *Operations Research*, 2020.

## Generic MILP (Problem (1), Section 1)

The paper develops the Benders Dual Decomposition (BDD) method for a generic mixed-integer linear program of the form $$\begin{align}
\min_{y,x}\ & f^\top y + c^\top x \tag{1a} \\
\text{s.t.}\ & B y \ge b, \tag{1b} \\
& W x + T y \ge h, \tag{1c} \\
& y \in \mathbb{Z}_+^n, \quad x \in \mathbb{R}_+^m, \tag{1d}
\end{align}$$ where $f\in\mathbb{R}^n$, $B\in\mathbb{R}^{k\times n}$, $b\in\mathbb{R}^k$, $c\in\mathbb{R}^m$, $W\in\mathbb{R}^{\ell\times m}$, $h\in\mathbb{R}^\ell$, $T\in\mathbb{R}^{\ell\times n}$. The problem is feasible and bounded.

## Test Problem: Stochastic Capacitated Facility Location (SFL) (Appendix D.2)

**Sets and indices:** $N$ set of potential facility sites (indexed by $i$); $M$ set of customers (indexed by $j$); $S$ set of scenarios (indexed by $s$).

**Parameters:** $f_i$ fixed cost of opening facility $i$; $u_i$ capacity of facility $i$; $c_{ij}$ routing cost per unit of flow from $i$ to $j$; $p_s$ probability of scenario $s$; $d_j^s$ demand of customer $j$ in scenario $s$.

**Decision variables:** $y_i\in\{0,1\}$ equals $1$ iff facility $i$ is opened (first stage); $x_{ij}^s\ge 0$ is the flow from facility $i$ to customer $j$ under scenario $s$ (second stage).

**Formulation (D.4)–(D.7):** $$\begin{align}
\text{SFL}:\quad \min\ & \sum_{i\in N} f_i\, y_i
  + \sum_{s\in S} p_s \sum_{i\in N}\sum_{j\in M} c_{ij}\, x_{ij}^s \tag{D.4}\\
\text{s.t.}\ & \sum_{i\in N} x_{ij}^s \ge d_j^s,
  && \forall\, j\in M,\ s\in S \tag{D.5}\\
& \sum_{j\in M} x_{ij}^s \le u_i\, y_i,
  && \forall\, i\in N,\ s\in S \tag{D.6}\\
& \sum_{i\in N} u_i\, y_i \ge \max_{s\in S}\sum_{j\in M} d_j^s \tag{D.7}\\
& y_i \in \{0,1\},\ \ x_{ij}^s \ge 0,
  && \forall\, i\in N,\ j\in M,\ s\in S. \notag
\end{align}$$

**Interpretation.** Objective (D.4) combines first-stage facility opening costs with expected second-stage routing costs. Constraint (D.5) enforces demand satisfaction in every scenario; (D.6) bounds flows out of each facility by its capacity (and forces no flow from closed facilities); (D.7) is the complete-recourse property, ensuring that the total opened capacity covers the worst-case total demand.

Matching SFL to the generic form (1): the first-stage decisions are $y$ with $f = (f_i)_{i\in N}$ and constraint (D.7) taking the role of $By\ge b$; the second-stage recourse $x=(x_{ij}^s)$ and constraints (D.5)–(D.6) correspond to $Wx+Ty\ge h$.
