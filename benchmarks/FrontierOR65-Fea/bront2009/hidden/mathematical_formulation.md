# Original Formulation: Choice-Based Deterministic Linear Program (CDLP)

*Source: A Column Generation Algorithm for Choice-Based Network Revenue Management, Bront, Méndez-Díaz, and Vulcano, 2009.*

## Sets and Parameters

- $N = \{1,\dots,n\}$: set of products (itinerary and fare-class combinations).

- $m$: number of resources (flight legs), indexed by $i = 1,\dots,m$.

- $L$: number of customer segments, indexed by $l = 1,\dots,L$.

- $T$: length of the booking horizon (number of discrete time periods).

- $S \subseteq N$: an offer set (a subset of products made available to arriving customers).

- $C_l \subseteq N$: consideration set of segment $l$ (overlap across segments is allowed).

- $c = (c_1,\dots,c_m)^{\top}$: initial capacity vector of the resources.

- $A = [a_{ij}] \in \{0,1\}^{m \times n}$: resource-product incidence matrix; $A_j$ denotes the $j$-th column.

- $r_j$: revenue collected from selling one unit of product $j$.

- $\lambda$: probability that a customer arrives in a given time period; $p_l$ is the conditional probability of segment $l$ given an arrival, with $\sum_l p_l = 1$; $\lambda_l = \lambda p_l$.

- $v_{lj} \geq 0$ for $j \in C_l$: preference weight of segment $l$ for product $j$, with $v_{l0} > 0$ the no-purchase weight.

- Under the MNL choice model, the probability that a segment-$l$ arrival chooses $j \in S$ is $P_{lj}(S) = v_{lj} / \bigl(\sum_{h \in C_l \cap S} v_{lh} + v_{l0}\bigr)$, and the aggregate purchase probability of product $j$ under $S$ is $P_j(S) = \sum_{l=1}^{L} p_l P_{lj}(S)$.

- Expected per-period revenue from $S$: $R(S) = \sum_{j \in S} r_j P_j(S)$.

- Resource consumption vector from $S$: $Q(S) = A\, P(S)$ where $P(S) = (P_1(S),\dots,P_n(S))^{\top}$.

## Decision Variables

- $t(S) \geq 0$ for every $S \subseteq N$: (continuous) number of time periods during which offer set $S$ is made available.

## Objective

$$\begin{equation}
V^{\mathrm{CDLP}} \;=\; \max \; \sum_{S \subseteq N} \lambda\, R(S)\, t(S) \tag{3}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{S \subseteq N} \lambda\, Q(S)\, t(S) & \;\leq\; c, \tag{3a} \\
\sum_{S \subseteq N} t(S) & \;\leq\; T, \tag{3b} \\
t(S) & \;\geq\; 0, \qquad \forall S \subseteq N. \tag{3c}
\end{align}$$

The formulation has one variable $t(S)$ for each of the $2^{n}-1$ nonempty subsets $S \subseteq N$, i.e. an exponential family of variables; the paper solves it via column generation.
