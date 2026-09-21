# Original Formulation: Trade–Container Assignment Problem in Construction Projects (TCAPCP)

*Source: A graph-based model and a heuristic for the trade–container assignment problem in construction projects (MIP TCAPCP$_{Per}$), M. Dienstknecht and D. Briskorn, European Journal of Operational Research 315(1), 324–337, 2024.*

## Sets and Indices

$$\begin{align*}
&C && \text{set of containers (single containers are the nodes of graph } G=(C,\mathcal{A})\text{).}\\
&\mathcal{J} && \text{set of trades.}\\
&\mathcal{A} && \text{set of arcs reflecting container adjacency:}\\
&&& \mathcal{A}=\{(c,c')\mid c\neq c'\in C,\ c'\in\mathcal{A}_c\},\ \text{with } c'\in\mathcal{A}_c\iff c\in\mathcal{A}_{c'}.\\
&C^p && \text{set of containers available in period } p:\ C^p=\{c\in C:\ s_c\le p\le f_c\}.\\
&\mathcal{J}^p && \text{set of trades active in period } p:\ \mathcal{J}^p=\{j\mid j\in\mathcal{J}:\ s_j\le p\le f_j\}.
\end{align*}$$

## Parameters

Each container $c\in C$ is a triple $(s_c,f_c,\mathcal{A}_c)$; each trade $j\in\mathcal{J}$ is a quadruple $(s_j,f_j,n_j,d_j^{max})$ of positive integers. $$\begin{align*}
&P && \text{number of periods the construction project spans.}\\
&s_c,f_c\in\{1,\dots,P\} && \text{start / end of on-site availability of container } c,\ s_c\le f_c.\\
&\mathcal{A}_c\subseteq C && \text{set of containers adjacent to } c.\\
&s_j,f_j\in\{1,\dots,P\} && \text{start / finish period of trade } j,\ s_j\le f_j.\\
&n_j && \text{number of containers trade } j \text{ requires in each period } [s_j,f_j].\\
&d_j^{max}\in\mathbb{N} && \text{maximum willingness of dispersion of trade } j \text{ (max.\ number of clusters).}
\end{align*}$$

## Decision Variables

$$\begin{align*}
&x_c^{j,p}\in\{0,1\} && 1 \text{ if trade } j \text{ is assigned to container } c \text{ in period } p.\\
&y_{(c,c')}^{j,p}\in\{0,1\} && 1 \text{ if containers } c \text{ and } c' \text{ belong to the same cluster of trade } j \text{ in period } p.\\
&z_c^{j,p}\in\{0,1\} && 1 \text{ if container } c \text{ is the source of a flow associated with trade } j \text{ in period } p.\\
&r^{j,p}\in\{0,1\} && 1 \text{ if trade } j \text{ is re-assigned in period } p.\\
&f_{(c,c')}^{j,p}\ge 0 && \text{flow associated with trade } j \text{ in period } p \text{ on arc } (c,c')\ \text{(continuous).}
\end{align*}$$

## Objective

$$\begin{align}
\min\ Z = \sum_{j\in\mathcal{J}}\ \sum_{p=s_j+1}^{f_j} r^{j,p} \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
&\sum_{c\in C^p} x_c^{j,p} = n_j
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p \tag{2}\\[2pt]
&\sum_{j\in\mathcal{J}^p} x_c^{j,p} \le 1
  && \forall\, p=1,\dots,P;\ c\in C^p \tag{3}\\[2pt]
&z_c^{j,p} \le x_c^{j,p}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c\in C^p \tag{4}\\[2pt]
&y_{(c,c')}^{j,p} \le x_c^{j,p}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c<c'\in C^p:(c,c')\in\mathcal{A} \tag{5}\\[2pt]
&y_{(c,c')}^{j,p} \le x_{c'}^{j,p}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c<c'\in C^p:(c,c')\in\mathcal{A} \tag{6}\\[2pt]
&z_c^{j,p}\, n_j
  + \sum_{\substack{(c',c)\in\mathcal{A}\\ c'\in C^p}} f_{(c',c)}^{j,p}
  - \sum_{\substack{(c,c')\in\mathcal{A}\\ c'\in C^p}} f_{(c,c')}^{j,p}
  \ge x_c^{j,p}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c\in C^p \tag{7}\\[2pt]
&f_{(c,c')}^{j,p} \le y_{(c,c')}^{j,p}\, n_j
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c<c'\in C^p:(c,c')\in\mathcal{A} \tag{8}\\[2pt]
&f_{(c',c)}^{j,p} \le y_{(c,c')}^{j,p}\, n_j
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c<c'\in C^p:(c,c')\in\mathcal{A} \tag{9}\\[2pt]
&\sum_{c\in C^p} z_c^{j,p} \le d_j^{max}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p \tag{10}\\[2pt]
&x_c^{j,p} - x_c^{j,p-1} \le r^{j,p}
  && \forall\, p=2,\dots,P;\ j\in\mathcal{J}^p:s_j<p;\ c\in C^p:c\in C^{p-1} \tag{11}\\[2pt]
&x_c^{j,p} \le r^{j,p}
  && \forall\, p=2,\dots,P;\ j\in\mathcal{J}^p:s_j<p;\ c\in\{C^p\setminus C^{p-1}\} \tag{12}\\[2pt]
&f_{(c,c')}^{j,p} \ge 0
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c,c'\in C^p:(c,c')\in\mathcal{A} \tag{13}\\[2pt]
&r^{j,p} \in \{0,1\}
  && \forall\, j\in\mathcal{J};\ p=2,\dots,P:s_j<p\le f_j \tag{14}\\[2pt]
&x_c^{j,p} \in \{0,1\}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c\in C^p \tag{15}\\[2pt]
&y_{(c,c')}^{j,p} \in \{0,1\}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c<c'\in C^p:(c,c')\in\mathcal{A} \tag{16}\\[2pt]
&z_c^{j,p} \in \{0,1\}
  && \forall\, p=1,\dots,P;\ j\in\mathcal{J}^p;\ c\in C^p \tag{17}
\end{align}$$
