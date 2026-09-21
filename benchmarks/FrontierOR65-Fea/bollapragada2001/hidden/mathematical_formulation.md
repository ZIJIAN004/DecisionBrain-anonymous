# Original Formulation: Optimal Design of Truss Structures (Truss Design Problem)

*Source: Optimal Design of Truss Structures by Logic-Based Branch and Cut, S. Bollapragada, O. Ghattas, and J. N. Hooker, 2001 (Operations Research 49(1):42–51).*

## Sets and Indices

$$\begin{align*}
& i = 1, \dots, I        && \text{bars} \\
& j = 1, \dots, J        && \text{degrees of freedom (summed over all nodes)} \\
& \ell = 1, \dots, L     && \text{loading conditions} \\
& k = 1, \dots, K_i      && \text{discrete cross-sectional areas available for bar } i
\end{align*}$$

## Parameters

$$\begin{align*}
& I            && \text{number of bars} \\
& J            && \text{number of degrees of freedom (summed over all nodes)} \\
& L            && \text{number of loading conditions} \\
& K_i          && \text{number of discrete cross-sectional areas for bar } i \\
& h_i          && \text{length of bar } i \\
& A_{ik}       && k\text{-th discrete cross-sectional area of bar } i,\ \text{with } 0 \leqslant A_{i1} \leqslant \cdots \leqslant A_{iK_i} \\
& E_i          && \text{modulus of elasticity of bar } i \\
& p_{j\ell}    && \text{force imposed by load condition } \ell \text{ at degree of freedom } j \\
& b_{ij}       && \text{cosine of the angle between bar } i \text{ and degree of freedom } j \\
& c_i          && \text{cost per unit volume of bar } i \text{ (typically the weight density)} \\
& \sigma_i^L,\ \sigma_i^U && \text{minimum and maximum allowable stress in bar } i \\
& v_i^L,\ v_i^U && \text{limits on elongation (contraction if negative) of bar } i \\
& d_j^L,\ d_j^U && \text{limits on displacement for degree of freedom } j
\end{align*}$$

## Decision Variables

$$\begin{align*}
& A_i           && \text{cross-sectional area of bar } i \\
& s_{i\ell}     && \text{force in bar } i \text{ due to loading condition } \ell \\
& \sigma_{i\ell}&& \text{stress in bar } i \text{ due to loading condition } \ell \\
& v_{i\ell}     && \text{elongation (contraction if negative) of bar } i \text{ due to loading condition } \ell \\
& d_{j\ell}     && \text{node displacement along degree of freedom } j \text{ for loading condition } \ell
\end{align*}$$

## Objective

$$\begin{align}
\min \quad & \sum_{i=1}^{I} c_i h_i A_i \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{i=1}^{I} b_{ij}\, s_{i\ell} = p_{j\ell}, && \forall\, j,\, \ell
    && \text{(equilibrium equations)} \\
& \sum_{j=1}^{J} b_{ij}\, d_{j\ell} = v_{i\ell}, && \forall\, i,\, \ell
    && \text{(compatibility equations)} \\
& \frac{E_i}{h_i}\, A_i\, v_{i\ell} = s_{i\ell}, && \forall\, i,\, \ell
    && \text{(Hooke's law)} \\
& \sigma_{i\ell} = \frac{E_i}{h_i}\, v_{i\ell}, && \forall\, i,\, \ell
    && \text{(stress equations)} \\
& v_i^L \leqslant v_{i\ell} \leqslant v_i^U, && \forall\, i,\, \ell
    && \text{(elongation bounds)} \\
& \sigma_i^L \leqslant \sigma_{i\ell} \leqslant \sigma_i^U, && \forall\, i,\, \ell
    && \text{(stress bounds)} \\
& d_j^L \leqslant d_{j\ell} \leqslant d_j^U, && \forall\, j,\, \ell
    && \text{(displacement bounds)} \\
& \bigvee_{k=1}^{K_i} \left( A_i = A_{ik} \right), && \forall\, i
    && \text{(logical disjunction)} \tag{1}
\end{align}$$
