# Original Formulation: Pickup and Delivery Problem with Time Windows (PDPTW)

*Source: “Branch and Cut and Price for the Pickup and Delivery Problem with Time Windows”, Stefan Ropke, Jean-François Cordeau, Transportation Science 43(3):267–286, 2009.*

This is the standard three-index mixed-integer program of Section 2.2, which the paper writes first as the definition of the PDPTW and whose notation ($x^k_{ij}, B^k_i, Q^k_i$) is carried into the rest of the paper. The set partitioning formulation of Section 2.3, used as the basis of the branch-and-cut-and-price algorithm, is recorded under **Variants**.

## Sets and Indices

- $n$ : number of transportation requests.

- $G=(N,A)$ : directed graph with node set $N$ and arc set $A$.

- $N=\{0,1,\dots,2n+1\}$ : node set. Node $0$ is the origin depot and node $2n+1$ is the destination depot.

- $P=\{1,\dots,n\}$ : set of pickup nodes.

- $D=\{n+1,\dots,2n\}$ : set of delivery nodes. Request $i$ is associated with pickup node $i$ and delivery node $n+i$.

- $K$ : set of (identical) vehicles.

## Parameters

- $q_i$ : load at node $i\in N$, with $q_0=q_{2n+1}=0$ and $q_i=-q_{n+i}$ for $i=1,\dots,n$.

- $d_i$ : nonnegative service duration at node $i$, with $d_0=d_{2n+1}=0$.

- $[a_i,b_i]$ : time window at node $i\in P\cup D$ ($a_i$ = earliest, $b_i$ = latest start of service); depot windows $[a_0,b_0]$ and $[a_{2n+1},b_{2n+1}]$ give earliest departure / latest return times.

- $Q$ : vehicle capacity (vehicles are identical).

- $c_{ij}$ : routing cost on arc $(i,j)\in A$.

- $t_{ij}$ : travel time on arc $(i,j)\in A$; $t_{ij}$ includes the service time $d_i$ at node $i$. Both $c_{ij}$ and $t_{ij}$ satisfy the triangle inequality.

## Decision Variables

- $x^k_{ij}\in\{0,1\}$ for $(i,j)\in A,\ k\in K$ : $=1$ iff vehicle $k$ travels directly from node $i$ to node $j$.

- $B^k_i\ge 0$ for $i\in N,\ k\in K$ : time at which vehicle $k$ begins service at node $i$.

- $Q^k_i\ge 0$ for $i\in N,\ k\in K$ : load of vehicle $k$ upon leaving node $i$.

## Objective

$$\begin{equation}
\min \sum_{k\in K}\sum_{i\in N}\sum_{j\in N} c_{ij}\, x^k_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k\in K}\sum_{j\in N} x^k_{ij} &= 1 && \forall\, i\in P \tag{2}\\[2pt]
\sum_{j\in N} x^k_{ij} - \sum_{j\in N} x^k_{n+i,\,j} &= 0 && \forall\, i\in P,\ k\in K \tag{3}\\[2pt]
\sum_{j\in N} x^k_{0j} &= 1 && \forall\, k\in K \tag{4}\\[2pt]
\sum_{j\in N} x^k_{ji} - \sum_{j\in N} x^k_{ij} &= 0 && \forall\, i\in P\cup D,\ k\in K \tag{5}\\[2pt]
\sum_{i\in N} x^k_{i,\,2n+1} &= 1 && \forall\, k\in K \tag{6}
\end{align}$$ $$\begin{align}
B^k_j &\ge (B^k_i + t_{ij})\, x^k_{ij} && \forall\, i\in N,\ j\in N,\ k\in K \tag{7}
\end{align}$$ $$\begin{align}
Q^k_j &\ge (Q^k_i + q_j)\, x^k_{ij} && \forall\, i\in N,\ j\in N,\ k\in K \tag{8}\\[2pt]
B^k_i + t_{i,\,n+i} &\le B^k_{n+i} && \forall\, i\in P,\ k\in K \tag{9}\\[2pt]
a_i \le B^k_i &\le b_i && \forall\, i\in N,\ k\in K \tag{10}\\[2pt]
\max\{0,q_i\} \le Q^k_i &\le \min\{Q,\,Q+q_i\} && \forall\, i\in N,\ k\in K \tag{11}\\[2pt]
x^k_{ij} &\in \{0,1\} && \forall\, i\in N,\ j\in N,\ k\in K \tag{12}
\end{align}$$
