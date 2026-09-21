# Original Formulation: Guillotine Two-Dimensional Knapsack Problem (G2KP)

*Source: Modeling Two-Dimensional Guillotine Cutting Problems via Integer Programming, Fabio Furini, Enrico Malaguti, Dimitri Thomopulos, 2016.*

## Sets and Indices

- $J$ : set of plates (the original rectangular panel together with the smaller rectangular residual plates obtained from sequences of guillotine cuts). The panel is indexed by $j=0$ and has dimensions $L,W$; each plate $j\in J$ has dimensions $(l_j,w_j)$ with $1\le w_j\le W$ and $1\le l_j\le L$.

- $\bar J\subseteq J$ : subset of plates whose dimensions equal one of the items to obtain; with a slight abuse of notation $\bar J$ also denotes the set of items. Without loss of generality $0\in\bar J$ (if the panel does not correspond to an item, set $u_0=0$).

- $O=\{h,v\}$ : set of possible cut orientations (horizontal and vertical), with $o\in O$ the generic orientation.

- $Q(j,o)$ : set of positions where plate $j$ can be cut with orientation $o\in O$. In the model, $Q(j,h)=\{1,\dots,w_j-1\}$ and $Q(j,v)=\{1,\dots,l_j-1\}$.

## Parameters

- $L,W$ : length and width of the original rectangular panel.

- $n$ : number of item types.

- $l_i,\,w_i$ : length and width of item $i$, $i=1,\dots,n$.

- $p_i$ : profit of item $i$, $i=1,\dots,n$.

- $u_i$ : number of available copies of item $i$, $i=1,\dots,n$.

- $a^o_{qkj}$ : coefficient taking value $1$ when a plate of type $k$ produces a plate of type $j$ by a cut at position $q$ with orientation $o$, and $0$ otherwise.

All problem data are assumed, without loss of generality, to be positive integers.

## Decision Variables

- $x^o_{qj}$ : integer variable, number of times a plate of type $j$ is cut at position $q$ through a guillotine cut with orientation $o$, for $j\in J,\ o\in O,\ q\in Q(j,o)$.

- $y_j$ : integer variable, number of plates of type $j$ kept as final items (equivalently, number of items of type $j$ obtained), for $j\in\bar J$.

## Objective

$$\begin{equation}
  \text{(PP-G2KP):}\qquad \max \sum_{j\in\bar J} p_j\, y_j \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  &\sum_{k\in J}\sum_{o\in O}\sum_{q\in Q(k,o)} a^o_{qkj}\, x^o_{qk}
     \;-\; \sum_{o\in O}\sum_{q\in Q(j,o)} x^o_{qj} \;-\; y_j \;\ge\; 0,
     && j\in\bar J,\ j\neq 0 \tag{2}\\[4pt]
  &\sum_{k\in J}\sum_{o\in O}\sum_{q\in Q(k,o)} a^o_{qkj}\, x^o_{qk}
     \;-\; \sum_{o\in O}\sum_{q\in Q(j,o)} x^o_{qj} \;\ge\; 0,
     && j\in J\setminus\bar J \tag{3}\\[4pt]
  &\sum_{o\in O}\sum_{q\in Q(0,o)} x^o_{q0} \;+\; y_0 \;\le\; 1
     && \tag{4}\\[4pt]
  &y_j \;\le\; u_j, && j\in\bar J \tag{5}\\[4pt]
  &x^o_{qj} \;\ge\; 0\ \text{integer}, && j\in J,\ o\in O,\ q\in Q(j,o) \tag{6}\\[4pt]
  &y_j \;\ge\; 0\ \text{integer}, && j\in\bar J \tag{7}
\end{align}$$
