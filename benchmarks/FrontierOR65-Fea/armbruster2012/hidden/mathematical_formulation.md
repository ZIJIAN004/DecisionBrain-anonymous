# Original Formulation: Minimum Graph Bisection (MB)

*Source: LP and SDP branch-and-cut algorithms for the minimum graph bisection problem: a computational comparison, Michael Armbruster, Marzena Fügenschuh, Christoph Helmberg, Alexander Martin, 2012.*

## Sets and Indices

$$\begin{align*}
G = (V,E) \quad & \text{undirected graph with node set } V=\{1,\ldots,n\}\\
                & \text{and edge set } E \subseteq \{\{i,j\}:i,j\in V,\; i<j\}.\\
ij \in E        & \quad \text{shorthand for the edge } \{i,j\}.\\
\delta(S) := \{ij\in E : i\in S \wedge j\in V\setminus S\}
                & \quad \text{the cut induced by the partition } (S,\,V\setminus S).\\
C \subseteq E   & \quad \text{a cycle in } G \text{ (subgraph } (V_C,E_C)).\\
D \subseteq C   & \quad \text{an edge subset of cycle } C \text{ with } |D| \text{ odd.}
\end{align*}$$

## Parameters

$$\begin{align*}
f_i \in \mathbb{N}\cup\{0\}, \; i\in V & \quad \text{node (vertex) weight.}\\
w_{ij} \in \mathbb{R}, \; ij\in E      & \quad \text{edge cost; } w \text{ is the vector of edge costs.}\\
f(S) := \sum_{i\in S} f_i              & \quad \text{total weight of node subset } S.\\
F \in \mathbb{N}\cap\big[\lceil\tfrac12 f(V)\rceil,\, f(V)\big]
                                       & \quad \text{bisection capacity (upper bound on each cluster weight).}
\end{align*}$$ *W.l.o.g. $G$ contains a star $K_{1,n-1}$ with central node $1\in V$, adding edges $1j$ of cost zero if necessary; the star edges serve as the binary node variables indicating cluster membership.*

## Decision Variables

$$\begin{align*}
y_{ij} \in \{0,1\}, \; ij\in E \quad &
y_{ij}=1 \text{ if edge } ij \text{ is in the cut (endpoints in different clusters),}\\
& y_{ij}=0 \text{ otherwise.}
\end{align*}$$ In particular $y_{1i}$ ($i=2,\ldots,n$) indicates the cluster of node $i$ relative to the central star node $1$.

## Objective

$$\begin{align}
\text{minimize}\quad & w^{T} y \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{i=2}^{n} f_i\, y_{1i} \;\le\; F, \notag\\[2pt]
& f_1 + \sum_{i=2}^{n} f_i\,(1 - y_{1i}) \;\le\; F, \notag\\[2pt]
& \sum_{ij\in D} y_{ij} \;-\; \sum_{ij\in C\setminus D} y_{ij}
   \;\le\; |D| - 1,
   \quad D \subseteq C \subseteq E,\; |D|\text{ odd},\; C \text{ cycle in } G, \notag\\[2pt]
& y \in \{0,1\}^{E}. \notag
\end{align}$$ The first two inequalities are the capacity constraints bounding the weight of the cluster separated from node $1$ and of the cluster containing node $1$, respectively. The cycle inequalities, together with integrality, describe all cuts in $G$; the paper writes them inside the defining program (1) and separates them dynamically during branch-and-cut.
