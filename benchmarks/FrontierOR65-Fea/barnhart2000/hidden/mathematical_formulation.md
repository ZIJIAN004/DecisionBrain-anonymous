# Original Formulation: Origin-Destination Integer Multicommodity Flow (ODIMCF)

*Source: “Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer Multicommodity Flow Problems,” Barnhart, Hane, and Vance, Operations Research 48(2):318–326, 2000.*

## Sets and Parameters

- $G = (N, A)$: directed network with node set $N$ and arc set $A$.

- $K$: set of commodities; each commodity $k \in K$ is defined by an origin–destination pair.

- $q^k$: quantity (demand) of commodity $k$.

- $c^k_{ij}$: unit flow cost for commodity $k$ on arc $ij$.

- $d_{ij}$: capacity of arc $ij$, for $ij \in A$.

- $b^k_i = 1$ if $i$ is the origin of $k$, $-1$ if destination of $k$, $0$ otherwise.

## Decision Variables

- $x^k_{ij} \in \{0,1\}$: $1$ if the entire quantity $q^k$ of commodity $k$ is assigned to arc $ij$, $0$ otherwise.

## Objective

$$\begin{equation}
\min \sum_{k \in K} \sum_{ij \in A} c^k_{ij}\, q^k\, x^k_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k \in K} q^k\, x^k_{ij} &\le d_{ij}, & \forall\, ij \in A \tag{2} \\
\sum_{ij \in A} x^k_{ij} - \sum_{ji \in A} x^k_{ji} &= b^k_i, & \forall\, i \in N,\; \forall\, k \in K \tag{3} \\
x^k_{ij} &\in \{0,1\}, & \forall\, ij \in A,\; \forall\, k \in K \tag{4}
\end{align}$$
