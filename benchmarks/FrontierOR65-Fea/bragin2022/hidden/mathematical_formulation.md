# Original Formulation: Generalized Assignment Problem (GAP)

*Source: Surrogate “Level-Based” Lagrangian Relaxation for Mixed-Integer Linear Programming, Mikhail A. Bragin & Emily L. Tucker, Scientific Reports 12:22417, 2022.*

The MILP that the paper actually benchmarks in its “Generalized Assignment Problems” section is the classical Generalized Assignment Problem. The paper states (p. 8) that “large-scale instances of GAPs are considered (formulation is available in subsection 4.2 of Supplementary Information),” and the experiments (Table 5) report objective (feasible cost) values for this formulation on the OR-library / Yagiura instances of types D and E with 20, 40, and 80 machines and 1600 jobs. The general separable MILP of the main body (eqs. (1)–(2)) and the SLBLR Lagrangian-relaxation machinery (eqs. (3)–(22)) are solution methodology, not the problem definition, and are intentionally excluded below.

## Sets and Indices

- $I$ : set of machines, indexed by $i = 1,\dots,|I|$.

- $J$ : set of jobs, indexed by $j = 1,\dots,|J|$.

## Parameters

- $c_{i,j}$ : cost of assigning job $j$ to machine $i$.

- $a_{i,j}$ : amount of resource consumed when job $j$ is processed on machine $i$.

- $b_{i}$ : resource capacity of machine $i$.

## Decision Variables

- $x_{i,j} \in \{0,1\}$ : equals $1$ if job $j$ is assigned to machine $i$, and $0$ otherwise, for all $i \in I,\ j \in J$.

## Objective

$$\begin{equation}
  \min_{x}\ \sum_{i \in I} \sum_{j \in J} c_{i,j}\, x_{i,j}
  \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  \sum_{i \in I} x_{i,j} &= 1, && \forall\, j \in J, \tag{2}\\[2pt]
  \sum_{j \in J} a_{i,j}\, x_{i,j} &\le b_{i}, && \forall\, i \in I, \tag{3}\\[2pt]
  x_{i,j} &\in \{0,1\}, && \forall\, i \in I,\ \forall\, j \in J. \tag{4}
\end{align}$$
