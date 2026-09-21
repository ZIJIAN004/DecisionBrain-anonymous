# Original Formulation: Rank-One Quadratic Assignment Problem (QAP-R1)

*Source: The Rank-One Quadratic Assignment Problem, Yang Wang, Wei Yang, Abraham P. Punnen, Jingbo Tian, Aihua Yin, Zhipeng Lü, 2020 (INFORMS Journal on Computing).*

## Sets and Indices

- $N = \{1, 2, \ldots, n\}$: finite index set of rows / columns.

- $i \in N$: row index of the assignment matrix.

- $j \in N$: column index of the assignment matrix.

- $\mathbb{F}$: the family of all $n \times n$ matrices $X = (x_{ij})$ satisfying constraints <a href="#eq:row" data-reference-type="eqref" data-reference="eq:row">[eq:row]</a>–<a href="#eq:bin" data-reference-type="eqref" data-reference="eq:bin">[eq:bin]</a> (i.e., the set of $n \times n$ permutation / assignment matrices).

## Parameters

- $A = (a_{ij})$: $n \times n$ first quadratic cost matrix.

- $B = (b_{ij})$: $n \times n$ second quadratic cost matrix.

- $C = (c_{ij})$: $n \times n$ linear cost matrix.

## Decision Variables

- $x_{ij} \in \{0,1\}$: equals $1$ if row $i$ is assigned to column $j$, and $0$ otherwise. The matrix $X = (x_{ij})$ ranges over $\mathbb{F}$.

## Objective

$$\begin{equation}
\text{minimize}\quad
f(X) = \left(\sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij} x_{ij}\right)
       \left(\sum_{i=1}^{n}\sum_{j=1}^{n} b_{ij} x_{ij}\right)
       + \sum_{i=1}^{n}\sum_{j=1}^{n} c_{ij} x_{ij}
\tag{$f$}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j=1}^{n} x_{ij} &= 1, & i &= 1, \ldots, n, \tag{1}\label{eq:row}\\
\sum_{i=1}^{n} x_{ij} &= 1, & j &= 1, \ldots, n, \tag{2}\label{eq:col}\\
x_{ij} &\in \{0,1\}, & i,j &= 1, 2, \ldots, n. \tag{3}\label{eq:bin}
\end{align}$$

Equivalently, the problem is to find $X \in \mathbb{F}$ that minimizes $f(X)$, where $\mathbb{F}$ denotes the set of matrices satisfying <a href="#eq:row" data-reference-type="eqref" data-reference="eq:row">[eq:row]</a>–<a href="#eq:bin" data-reference-type="eqref" data-reference="eq:bin">[eq:bin]</a>.
