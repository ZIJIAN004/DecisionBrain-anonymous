# Original Formulation: Fixed Charge Transportation Problem (FCTP)

*Source: The Fixed Charge Transportation Problem: An Exact Algorithm Based on a New Integer Programming Formulation, Roberto Roberti, Enrico Bartolini, Aristide Mingozzi, 2014 (Management Science).*

The paper defines the FCTP it studies through the standard mixed integer programming formulation **(F0)**, written first in §1 with the variables $x_{ij}$ and $y_{ij}$ that state the problem. The exponential set-partitioning formulation (F1) introduced in §2 is the paper’s new reformulation and the basis of its branch-and-price algorithm; it is recorded under *Remarks* but is not the original problem definition.

## Sets and Indices

- $S = \{1, 2, \dots, m\}$: set of $m$ sources, indexed by $i$.

- $T = \{1, 2, \dots, n\}$: set of $n$ sinks, indexed by $j$.

- $A = \{(i,j) : i \in S,\, j \in T\}$: arc set of the complete bipartite graph $G = (S, T, A)$.

## Parameters

- $a_i$: integer supply available at source $i \in S$.

- $b_j$: integer demand required at sink $j \in T$.

- $c_{ij}$: unit (continuous) transportation cost on arc $(i,j) \in A$.

- $f_{ij}$: fixed cost incurred for using arc $(i,j) \in A$.

- $m_{ij} = \min(a_i, b_j)$, $(i,j) \in A$: arc capacity.

Balance assumption (without loss of generality): $\sum_{i \in S} a_i = \sum_{j \in T} b_j$.

## Decision Variables

- $x_{ij} \ge 0$: continuous quantity of goods transported along arc $(i,j) \in A$.

- $y_{ij} \in \{0,1\}$: equal to $1$ if and only if $x_{ij}$ is positive (i.e., arc $(i,j)$ is used).

## Objective

$$\begin{equation}
z_{F0} = \min \sum_{i \in S} \sum_{j \in T}
        \bigl( c_{ij}\, x_{ij} + f_{ij}\, y_{ij} \bigr)
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j \in T} x_{ij} &= a_i, & &\forall\, i \in S \tag{2}\\[4pt]
\sum_{i \in S} x_{ij} &= b_j, & &\forall\, j \in T \tag{3}\\[4pt]
x_{ij} &\le m_{ij}\, y_{ij}, & &\forall\, (i,j) \in A \tag{4}\\[4pt]
x_{ij} &\ge 0, & &\forall\, (i,j) \in A \tag{5}\\[4pt]
y_{ij} &\in \{0,1\}, & &\forall\, (i,j) \in A \tag{6}
\end{align}$$

Constraints (2) ship all of each source’s supply; constraints (3) meet each sink’s demand exactly; linking constraints (4) allow positive flow on an arc only when it is opened ($y_{ij}=1$) and bound it by $m_{ij}$; (5)–(6) give the variable domains.
