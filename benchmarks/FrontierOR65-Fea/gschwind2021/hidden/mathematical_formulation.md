# Original Formulation: Partitioning/Covering a Graph with a Minimum number of Relaxed Cliques (PGMRC / CGMRC)

*Source: A Branch-and-Price Framework for Decomposing Graphs into Relaxed Cliques, Timo Gschwind, Stefan Irnich, Fabio Furini, Roberto Wolfler Calvo, 2017.*

## Sets and Parameters

- $G = (V, E)$: undirected graph with vertex set $V$ and edge set $E$.

- $i \in V$: index of a vertex.

- $G[S] = (S,\, E \cap (S \times S))$: the vertex-induced subgraph of $S \subseteq V$.

- $S \subseteq V$: a subset of vertices that forms a *relaxed clique* (RC) of the prescribed type, with defining parameter $s \in \mathbb{N}$ (or $\gamma \in (0,1]$ for $\gamma$-quasi-cliques). The eight first-order RC families (Table 1): $k$-core, $s$-plex, $s$-clique, $s$-club, $\gamma$-quasi-clique, $s$-defective clique, $k$-block, $s$-bundle.

- $\mathscr{S} = \{\, S \subseteq V : S \text{ is a relaxed clique}\,\}$: the collection of *all* feasible relaxed cliques of the chosen type (for the connected variant, additionally $G[S]$ must be connected). $\mathscr{S}$ has, in general, exponential cardinality in $|V|$.

## Decision Variables

- $\lambda_S \in \{0,1\}$ for each $S \in \mathscr{S}$: equals $1$ if and only if the relaxed clique $S$ is part of the decomposition, and $0$ otherwise.

## Objective

$$\begin{align}
\min \quad & \sum_{S \in \mathscr{S}} \lambda_S \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad & \sum_{S \in \mathscr{S} : i \in S} \lambda_S = 1 \quad (\text{or } \geq 1) & & i \in V \tag{1b}\\
& \lambda_S \in \{0,1\} & & S \in \mathscr{S}. \tag{1c}
\end{align}$$

The objective (1a) minimizes the number of RCs in the decomposition; constraints (1b) are the partitioning constraints (“$=1$”: every vertex covered by exactly one RC, PGMRC) or the covering constraints (“$\geq 1$”: every vertex covered by at least one RC, CGMRC); constraints (1c) define the binary domain of the variables.
