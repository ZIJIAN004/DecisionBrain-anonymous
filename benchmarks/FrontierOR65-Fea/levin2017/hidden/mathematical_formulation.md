# Original Formulation: Congestion-Aware System Optimal SAV Routing (SAV Routing Problem)

*Source: Congestion-aware system optimal route choice for shared autonomous vehicles, Michael W. Levin, Transportation Research Part C 82 (2017) 229–247.*

## Sets and Indices

$$\begin{align*}
&\mathcal{G} = (\mathcal{N}, \mathcal{A}) && \text{traffic network with node set } \mathcal{N} \text{ and link set } \mathcal{A};\ (i,j)\in\mathcal{A} \text{ is the link } i\to j \\
&\mathcal{Z} \subset \mathcal{N} && \text{set of centroids (origins / destinations of trips; vehicle parking)} \\
&\mathcal{A}_o \subseteq \mathcal{A} && \text{links not starting or ending at a centroid: } \mathcal{A}_o=\{(i,j):(i,j)\in\mathcal{A},\, i,j\in\mathcal{N}\setminus\mathcal{Z}\} \\
&\mathcal{A}_z \subseteq \mathcal{A} && \text{complement of } \mathcal{A}_o \text{ in } \mathcal{A} \text{ (centroid connectors)} \\
&\mathcal{A}_z^- = \{(i,j)\in\mathcal{A}_z : j\in\mathcal{Z}\} && \text{centroid connectors ending in a centroid} \tag{13} \\
&\mathcal{A}_z^+ = \{(i,j)\in\mathcal{A}_z : i\in\mathcal{Z}\} && \text{centroid connectors starting from a centroid} \tag{14} \\
&\Gamma_j^- && \text{set of incoming links to node } j \\
&\Gamma_j^+ && \text{set of outgoing links from node } j \\
&\mathcal{Z}^2 && \text{set of origin-destination centroid pairs } (r,s),\ r,s\in\mathcal{Z} \\
&t \in \{0,1,2,\dots,T\} && \text{discrete time steps (unit time step }=1) \\
&s \in \mathcal{Z} && \text{destination index of disaggregated flows}
\end{align*}$$

## Parameters

$$\begin{align*}
&L_{ij} && \text{length of link } (i,j) \\
&Q_{ij} && \text{capacity of link } (i,j) \quad (Q_{ij}=\infty \text{ for } (i,j)\in\mathcal{A}_z) \\
&v_{ij} && \text{free-flow speed of link } (i,j) \quad (L_{ij}/v_{ij} \text{ is a positive integer; } =1 \text{ for } (i,j)\in\mathcal{A}_z) \\
&w_{ij} && \text{congested wave speed of link } (i,j) \\
&K && \text{jam density } (KL_{ij} = \text{max number of vehicles that can occupy link } (i,j)) \\
&T && \text{length of analysis period (number of time steps)} \\
&p_i(0) && \text{number of vehicles parked at centroid } i \text{ at the start of the analysis period} \\
&d_r^s(t) && \text{person-trip demand from } r \text{ to } s \text{ departing at time } t
\end{align*}$$

## Decision Variables

$$\begin{align*}
&y_{ijk}^s(t) \ge 0 && \text{flow of vehicles destined for } s \text{ turning from link } (i,j) \text{ onto link } (j,k) \text{ at time } t \\
&y_{ij}^s(t) \ge 0 && \text{number of vehicles departing centroid } i \text{ via connector } (i,j)\in\mathcal{A}_z^+ \text{ toward } s \text{ at time } t \\
&N_{ij}^{Us}(t) \ge 0 && \text{upstream cumulative count for destination } s \text{ on link } (i,j) \text{ at time } t \\
&N_{ij}^{Ds}(t) \ge 0 && \text{downstream cumulative count for destination } s \text{ on link } (i,j) \text{ at time } t \\
&p_j(t) \ge 0 && \text{number of vehicles parked at centroid } j\in\mathcal{Z} \text{ at time } t \\
&e_r^s(t) \ge 0 && \text{number of travelers departing from } r \text{ to } s \text{ at time } t \\
&\omega_r^s(t) \ge 0 && \text{unserviced demand waiting at } r \text{ for destination } s \text{ at time } t
\end{align*}$$

## Objective

Minimize the total system travel time (TSTT), Eq. (30), used in the LP (33): $$\begin{align}
\min\ Z \;=\; \mathcal{T} \;=\; \sum_{(i,j)\in\mathcal{A}}\sum_{s\in\mathcal{Z}}\sum_{t=0}^{T}
\left( N_{ij}^{Us}(t) - N_{ij}^{Ds}(t) \right)
\;+\; \sum_{(r,s)\in\mathcal{Z}^2}\sum_{t=0}^{T}\omega_r^s(t)
\tag{30/33}
\end{align}$$ The first term is total vehicle-time on links (link occupancy summed over time); the second term is total traveler waiting time at origins.

## Constraints

$$\begin{align}
& N_{ij}^{Us}(t+1) = N_{ij}^{Us}(t) + \sum_{(j,k)\in\Gamma_j^+} y_{ijk}^s(t)
&& \forall (i,j)\in\mathcal{A}_o\cup\mathcal{A}_z^+,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T-1] \tag{34} \\
& N_{jk}^{Ds}(t+1) = N_{jk}^{Ds}(t) + \sum_{(i,j)\in\Gamma_j^-} y_{ijk}^s(t)
&& \forall (j,k)\in\mathcal{A}_o\cup\mathcal{A}_z^-,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T-1] \tag{35} \\
& \sum_{(j,k)\in\Gamma_j^+} y_{ijk}^s(t) \le N_{ij}^{Us}\!\left(t-\tfrac{L_{ij}}{v_{ij}}+1\right) - N_{ij}^{Ds}(t)
&& \forall (i,j)\in\mathcal{A}_o\cup\mathcal{A}_z^+,\ \forall s\in\mathcal{Z},\ \forall t\in\left[\tfrac{L_{ij}}{v_{ij}}-1,\,T\right] \tag{36} \\
& y_{ijk}^s(t) = 0
&& \forall (i,j)\in\mathcal{A}_o\cup\mathcal{A}_z^+,\ \forall (j,k)\in\Gamma_j^+,\ \forall s\in\mathcal{Z},\ \forall t\in\left[0,\,\tfrac{L_{ij}}{v_{ij}}-1\right) \tag{37} \\
& \sum_{(j,k)\in\Gamma_j^+}\sum_{s\in\mathcal{Z}} y_{ijk}^s(t) \le Q_{ij}
&& \forall (i,j)\in\mathcal{A}_o,\ \forall t\in[0,T] \tag{38} \\
& \sum_{(i,j)\in\Gamma_j^-}\sum_{s\in\mathcal{Z}} y_{ijk}^s(t) \le Q_{jk}
&& \forall (j,k)\in\mathcal{A}_o,\ \forall t\in[0,T] \tag{39} \\
& \sum_{(i,j)\in\Gamma_j^-}\sum_{s\in\mathcal{Z}} y_{ijk}^s(t)
\le \sum_{s\in\mathcal{Z}}\!\left( N_{jk}^{Us}\!\left(t-\tfrac{L_{jk}}{w_{jk}}+1\right) - N_{jk}^{Ds}(t) \right) + KL_{jk}
&& \forall (j,k)\in\mathcal{A}_o,\ \forall t\in\left[\tfrac{L_{jk}}{w_{jk}}-1,\,T\right] \tag{40} \\
& N_{ij}^{Us}(0) = 0 && \forall (i,j)\in\mathcal{A},\ \forall s\in\mathcal{Z} \tag{41} \\
& N_{ij}^{Ds}(0) = 0 && \forall (i,j)\in\mathcal{A},\ \forall s\in\mathcal{Z} \tag{42} \\
& p_j(t+1) = p_j(t) + \sum_{(i,j)\in\Gamma_j^-}\!\left( N_{ij}^{Uj}(t) - N_{ij}^{Dj}(t) \right)
- \sum_{(j,k)\in\Gamma_j^+}\sum_{s\in\mathcal{Z}} y_{jk}^s(t)
&& \forall j\in\mathcal{Z},\ \forall t\in[0,T-1] \tag{43} \\
& y_{ijk}^s(t) = 0
&& \forall (j,k)\in\mathcal{A}_z^-,\ \forall (i,j)\in\Gamma_j^-,\ s\neq k,\ \forall t\in[0,T] \tag{44} \\
& \sum_{(i,j)\in\Gamma_i^+}\sum_{s\in\mathcal{Z}} y_{ij}^s(t) \le p_i(t)
&& \forall i\in\mathcal{Z},\ \forall t\in[0,T] \tag{45} \\
& N_{ij}^{Us}(t+1) = N_{ij}^{Us}(t) + y_{ij}^s(t)
&& \forall (i,j)\in\mathcal{A}_z^+,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T-1] \tag{46} \\
& N_{ij}^{Ds}(t+1) = N_{ij}^{Us}(t)
&& \forall (i,j)\in\mathcal{A}_z^-,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T-1] \tag{47} \\
& \sum_{i\in\mathcal{Z}} p_i(0) = \sum_{i\in\mathcal{Z}} p_i(T) && \tag{48} \\
& e_r^s(t) \le \omega_r^s(t) && \forall (r,s)\in\mathcal{Z}^2,\ \forall t\in[0,T] \tag{49} \\
& e_r^s(t) \le \sum_{(r,j)\in\Gamma_r^+} y_{rj}^s(t) && \forall (r,s)\in\mathcal{Z}^2,\ \forall t\in[0,T] \tag{50} \\
& \omega_r^s(t+1) = \omega_r^s(t) + d_r^s(t) - e_r^s(t) && \forall (r,s)\in\mathcal{Z}^2,\ \forall t\in[0,T-1] \tag{51} \\
& \omega_r^s(T) = 0 && \forall (r,s)\in\mathcal{Z}^2 \tag{52} \\
& y_{ijk}^s(t) \ge 0 && \forall (i,j)\in\mathcal{A}_o\cup\mathcal{A}_z^+,\ \forall (j,k)\in\Gamma_j^+,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T] \tag{53} \\
& y_{rj}^s(t) \ge 0 && \forall (r,j)\in\mathcal{A}_z^+,\ \forall s\in\mathcal{Z},\ \forall t\in[0,T] \tag{54} \\
& e_r^s(t) \ge 0 && \forall (r,s)\in\mathcal{Z}^2,\ \forall t\in[0,T] \tag{55}
\end{align}$$
