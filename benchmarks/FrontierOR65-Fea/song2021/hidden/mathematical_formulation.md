# Original Formulation: Resource Loading Problem (RLP)

*Source: Polyhedral Results and Branch-and-Cut for the Resource Loading Problem, Guopeng Song, Tamás Kis, Roel Leus, 2020 (INFORMS Journal on Computing).*

The paper’s first explicit MIP defining the problem it studies is the *Pulse Formulation* of Section 3.1 (Equations (1)–(11)), whose notation ($y_{jt}, z_t, T_j$) is introduced when the problem is stated in Section 1 and is carried throughout the paper. This is the canonical original formulation below. The novel Execution-Interval formulation of Section 4.1 (the paper’s main contribution) is reproduced under *Variants*.

## Sets and Indices

$$\begin{align*}
&J && \text{set of orders (jobs), indexed by } j = 1,\ldots,n.\\
&t = 1,\ldots,H && \text{discrete time periods; } H \text{ is the length of the planning horizon.}
\end{align*}$$

## Parameters

$$\begin{align*}
&p_j && \text{work content of order } j \text{ (e.g., man-hours).}\\
&r_j && \text{release date of order } j,\quad 1 \le r_j \le d_j \le H.\\
&d_j && \text{due date of order } j.\\
&LB_j && \text{lower bound on the per-period execution intensity, } 0 < LB_j \le UB_j \le 1.\\
&UB_j && \text{upper bound on the per-period execution intensity.}\\
&w_j && \text{tardiness penalty per time period for order } j.\\
&\sigma && \text{unit cost of nonregular (overtime) capacity.}\\
&C_t && \text{regular workforce capacity available in period } t.
\end{align*}$$

## Decision Variables

$$\begin{align*}
&s_{jt} \in \{0,1\} && \text{$1$ if order $j$ starts in period $t$;}\quad j \in J,\; t = r_j,\ldots,H.\\
&f_{jt} \in \{0,1\} && \text{$1$ if order $j$ finishes in period $t$;}\quad j \in J,\; t = r_j,\ldots,H.\\
&y_{jt} \in [0,1] && \text{intensity (fraction of work content) of order $j$ in period $t$;}\quad j \in J,\; t = r_j,\ldots,H.\\
&T_j \ge 0 && \text{tardiness of order $j$;}\quad j \in J.\\
&z_t \ge 0 && \text{nonregular capacity used in period $t$;}\quad t = 1,\ldots,H.
\end{align*}$$

## Objective

$$\begin{align}
\min \quad \sum_{j \in J} w_j T_j + \sigma \sum_{t=1}^{H} z_t \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{t=r_j}^{H} s_{jt} = \sum_{t=r_j}^{H} f_{jt} = 1
    && \forall j \in J, \tag{2}\\[4pt]
& \sum_{k=r_j}^{t} f_{jk} \le \sum_{k=r_j}^{t} s_{jk}
    && \forall j \in J,\; t = r_j,\ldots,H, \tag{3}\\[4pt]
& T_j \ge \sum_{t=d_j}^{H} f_{jt}\,(t - d_j)
    && \forall j \in J, \tag{4}\\[4pt]
& LB_j\!\left( \sum_{k=r_j}^{t} s_{jk} - \sum_{k=r_j}^{t-1} f_{jk} \right)
    \le y_{jt}
    \le UB_j\!\left( \sum_{k=r_j}^{t} s_{jk} - \sum_{k=r_j}^{t-1} f_{jk} \right)
    && \forall j \in J,\; t = r_j,\ldots,H, \tag{5}\\[4pt]
& \sum_{t=r_j}^{H} y_{jt} = 1
    && \forall j \in J, \tag{6}\\[4pt]
& z_t \ge \sum_{j \in J} y_{jt}\, p_j - C_t
    && t = 1,\ldots,H, \tag{7}\\[4pt]
& y_{jt} \ge 0
    && \forall j \in J,\; t = r_j,\ldots,H, \tag{8}\\[4pt]
& z_t \ge 0
    && t = 1,\ldots,H, \tag{9}\\[4pt]
& T_j \ge 0
    && \forall j \in J, \tag{10}\\[4pt]
& s_{jt},\, f_{jt} \in \{0,1\}
    && \forall j \in J,\; t = r_j,\ldots,H. \tag{11}
\end{align}$$
