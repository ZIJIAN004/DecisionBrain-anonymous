# Original Formulation: Set Covering Problem (SCP)

*Source: Combination of Metaheuristic and Exact Algorithms for Solving Set Covering-Type Optimization Problems, İbrahim Muter, Ş. İlker Birbil, Güvenç Şahin, 2010 (INFORMS Journal on Computing, 22(4), pp. 603–619).*

The paper studies set covering-type optimization problems and states the problem it operates on (equation (1), §1) as the Set Covering Problem (SCP). This is the first formulation written and the one whose notation ($C,\;P,\;c_p,\;a_{ip},\;y_p$) is carried into every subsequent section (restricted SCP, its LP relaxation, the reduced cost, and the MetaOpt algorithm). The vehicle routing problem with time windows (VRPTW) of §4 is solved as an instance of this SCP; the experiments report objective values for this formulation. The instantiation of the symbols for VRPTW is given in **Remarks**.

## Sets and Parameters

- $C$ : a nonempty, finite set of elements to be covered; indexed by $i$. (VRPTW: the set of customers.)

- $P$ : the set of feasible subsets of $C$; indexed by $p$ (the “columns”). (VRPTW: the set of all feasible vehicle routes.)

- $c_p$ : cost of subset $p \in P$. (VRPTW: total travel distance of route $p$.)

- $a_{ip}$ : $a_{ip}=1$ if element $i \in p$, and $a_{ip}=0$ otherwise. (VRPTW: $1$ if customer $i$ is served by route $p$, $0$ otherwise.)

## Decision Variables

- $y_p \in \{0,1\}$ : $1$ iff subset (column) $p \in P$ is selected.

## Objective

$$\begin{equation*}
  \min \; \sum_{p \in P} c_p\, y_p \tag{1}
\end{equation*}$$

## Constraints

$$\begin{align*}
  \sum_{p \in P} a_{ip}\, y_p &\geq 1, & & i \in C, \tag{1} \\[2pt]
  y_p &\in \{0,1\}, & & p \in P. \tag{1}
\end{align*}$$ SCP finds the least costly collection of subsets in $P$ such that each element $i \in C$ belongs to at least one of the selected subsets. (The paper tags the objective, the covering constraints, and the integrality requirement together as block (1).)
