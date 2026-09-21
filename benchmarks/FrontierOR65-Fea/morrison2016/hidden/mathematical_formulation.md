# Original Formulation: Graph Coloring via Set Covering (Mehrotra & Trick)

**Source.** Morrison, Sewell, Jacobson (2016), “Solving the Pricing Problem in a Branch-and-Price Algorithm for Graph Coloring using Zero-Suppressed Binary Decision Diagrams,” Section 2.3. The formulation is originally due to Mehrotra and Trick (1996).

## Sets and Indices

- $G = (V, E)$ — input graph with vertex set $V$ and edge set $E$.

- $\mathcal{S}$ — the family of all *maximal independent sets* of $G$ (each corresponds to a candidate color class).

- $v \in V$ — a vertex.

- $S \in \mathcal{S}$ — a maximal independent set.

## Decision Variables

$$x_S \in \{0,1\} \qquad \forall\, S \in \mathcal{S},$$ where $x_S = 1$ iff the maximal independent set $S$ is chosen as a color class in the coloring.

## Formulation

$$\begin{align}
\min\ & \sum_{S \in \mathcal{S}} x_S \tag{1} \\
\text{s.t.}\ & \sum_{S \in \mathcal{S}\,:\, v \in S} x_S \ \ge\ 1
   \qquad \forall\, v \in V, \tag{2} \\
& x_S \in \{0,1\} \qquad \forall\, S \in \mathcal{S}. \tag{3}
\end{align}$$

The objective (1) minimizes the number of color classes used. The covering constraints (2) require that every vertex appears in at least one selected maximal independent set. The integrality constraints (3) enforce binary selection of color classes.
