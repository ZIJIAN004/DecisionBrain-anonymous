# Original Formulation: Approximately Submodular Function Maximization (ASFM)

*Source: An efficient branch-and-cut algorithm for approximately submodular function maximization, Naoya Uematsu, Shunji Umetani, Yoshinobu Kawahara, 2019.*

## Sets and Parameters

$$\begin{align*}
N &= \{1,\dots,n\} && \text{ground set of elements (}|N|=n\text{)}\\
k &\in \mathbb{Z}_{>0},\ k \le n && \text{cardinality bound on the selected subset}\\
f : 2^{N} \to \mathbb{R} && \text{non-decreasing approximately submodular set function,}\\
&&& \text{with } f(\emptyset)=0,\ \text{evaluated by an oracle}\\
f(\{i\}\mid S) &:= f(S\cup\{i\}) - f(S) && \text{marginal gain of adding element } i \text{ to } S\\
F &= \{\, S \subseteq N : |S| \le k \,\} && \text{set of all feasible solutions (exponential cardinality)}\\
\gamma &\in (0,1] && \text{submodular ratio of } f,\ \text{defined by Eq.~(7)}\\
\bar{\gamma} &\in (0,1] && \text{computable upper bound on } \gamma\ (\bar{\gamma}\ge\gamma),\ \text{Eq.~(8)}
\end{align*}$$

The submodular ratio $\gamma$ and its upper bound $\bar{\gamma}$ are defined as $$\begin{align}
\gamma &= \min_{S,T \subseteq N} \frac{f(S) - f(S \cap T)}{f(S \cup T) - f(T)}, \tag{7}\\
\bar{\gamma} &= \min_{S \subseteq N} \frac{f(\{i\}\mid S)}{f(\{i\}\mid S \cup \{j\})}, \qquad i \notin S \cup \{j\}, \tag{8}
\end{align}$$ where $0/0 = 1$.

## Decision Variables

$$\begin{align*}
z &\in \mathbb{R} && \text{continuous variable representing the objective value}\\
y_i &\in \{0,1\},\ i \in N && y_i = 1 \text{ iff element } i \text{ is selected in the subset}
\end{align*}$$

## Objective

The ASFM problem is the maximization of a non-decreasing approximately submodular function under a cardinality constraint, first stated (Eq. (1)) as $\max_{S}\, f(S)$ s.t. $|S|\le k,\ S\subseteq N$. Section 3 reformulates it as the binary integer program (BIP) below, whose linear objective is $$\begin{align}
\text{maximize} \quad & z \tag{10}
\end{align}$$

## Constraints

$$\begin{align}
z &\le f(S) + f(\{j\}\mid S)\, y_j + \sum_{i \in N \setminus (S \cup \{j\})} \frac{1}{\gamma}\, f(\{i\}\mid S)\, y_i,
   && j \in N \setminus S,\ S \in F, \tag{10}\\
\sum_{i \in N} y_i &\le k, \notag\\
y_i &\in \{0,1\}, && i \in N. \notag
\end{align}$$
