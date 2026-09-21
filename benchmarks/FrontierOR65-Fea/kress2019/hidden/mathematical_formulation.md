# Original Formulation: Preemptive Crane Scheduling Problem with Seaside and Landside jobs (PCSP-SL)

*Source: An Exact Solution Approach for Scheduling Cooperative Gantry Cranes, Dominik Kress, Jan Dornseifer, Florian Jaehn, European Journal of Operational Research, 2019 (Appendix A).*

## Sets and Indices

- Slots $s \in \{0, 1, \ldots, S+1\}$: slot $0$ is the seaside handover point (I/O), slots $1,\ldots,S$ are the storage positions, slot $S+1$ is the landside handover point (I/O).

- Cranes $c \in \{w, l\}$: $w$ is the seaside crane, $l$ is the landside crane.

- Seaside containers $I = \{w_1, w_2, \ldots, w_n\}$, ordered by pick-up sequence ($w_i$ is picked up before $w_j$ for $i < j$); all originate at slot $0$.

- Landside containers $J = \{l_1, l_2, \ldots, l_m\}$, ordered in non-decreasing order of deadlines $d_j$.

- Time instants $t \in \{0, 1, \ldots, T\}$.

## Parameters

- $S \in \mathbb{N}$: number of storage slots in the block.

- $n \in \mathbb{N}$, $n \geq 1$: number of seaside containers ($|I| = n$).

- $m \in \mathbb{N}$: number of landside containers ($|J| = m$).

- $p \in \mathbb{N}$: number of time units required to lift (pick up) or drop a container.

- $T \in \mathbb{N}$: upper bound on the number of time slots needed to characterize an optimal solution.

- $\sigma_w = x_{w,0} \in \{0, \ldots, S\}$: initial position of the seaside crane.

- $\sigma_l = x_{l,0} \in \{1, \ldots, S+1\}$: initial position of the landside crane, with $\sigma_w < \sigma_l$.

- $s_i \in \{1, \ldots, S+1\}$: target slot of seaside container $w_i \in I$.

- $a_j \in \{1, \ldots, S\}$: source slot of landside container $l_j \in J$.

- $r_j \in \mathbb{N}$: earliest finish time of landside container $l_j \in J$.

- $d_j \in \mathbb{N}$: deadline of landside container $l_j \in J$.

- $\lambda_j$: number of landside jobs with the same deadline as $l_j$; $\lambda_{m+1} = 0$.

## Decision Variables

- $x_{c,t} \in \mathbb{R}_0^+$: position of crane $c \in \{w,l\}$ at time instant $t$.

- $C \in \mathbb{R}_0^+$: makespan (seaside makespan, to be minimized).

- $l^I_t \in \{0,1\}$: $1$ if crane $c=w$ starts lifting any container $w_i \in I$ at time $t$.

- $l^J_{t,i,s} \in \{0,1\}$: $1$ if crane $c=l$ starts lifting container $w_i$ at time $t$ in slot $s$.

- $l^J_{t,j} \in \{0,1\}$: $1$ if crane $c=l$ starts lifting container $l_j$ at time $t$.

- $d^I_{t,i,s} \in \{0,1\}$: $1$ if crane $c=w$ starts dropping container $w_i$ at time $t$ in slot $s$.

- $d^I_{t,i} \in \{0,1\}$: $1$ if crane $c=l$ starts dropping container $w_i$ at time $t$.

- $d^J_{t,j} \in \{0,1\}$: $1$ if crane $c=l$ starts dropping container $l_j$ at time $t$.

- $u_j \in \{0,1\}$: $1$ if landside container $l_j$ must be processed (its deadline $d_j \leq C$).

- $v_j \in \{0,1\}$: assures an additional landside container with smallest deadline larger than $C$ is processed (if it exists).

- $q_j \in \{0,1\}$: auxiliary modelling variable.

## Objective

$$\begin{align}
\min \quad & C \tag{A.1}
\end{align}$$

## Constraints

$$\begin{align}
& t \cdot d^I_{t,n,s_n} + p \leq C
   && \forall t \in \{1, \ldots, T\} \tag{A.2}\\[2pt]
& t \cdot \sum_{w_i \in I} d^I_{t,i} + p \leq C
   && \forall t \in \{1, \ldots, T\} \tag{A.3}\\[2pt]
& x_{c,0} = \sigma_c
   && \forall c \in \{w, l\} \tag{A.4}\\[2pt]
& x_{c,t-1} - 1 \leq x_{c,t} \leq x_{c,t-1} + 1
   && \forall c \in \{w, l\},\ t \in \{0, \ldots, T\} \tag{A.5}\\[2pt]
& x_{w,t} \leq x_{l,t} - 1
   && \forall t \in \{0, \ldots, T\} \tag{A.6}\\[2pt]
& x_{c,t} \leq S + 1
   && \forall c \in \{w, l\},\ t \in \{1, \ldots, T\} \tag{A.7}\\[2pt]
& \sum_{t=0}^{T} \sum_{s=1}^{S} t \cdot d^I_{t,i,s}
   \leq \sum_{t=0}^{T} \sum_{s=1}^{S} t \cdot d^I_{t,j,s}
   && \forall w_i, w_j \in I,\ i < j \tag{A.8}\\[2pt]
& x_{w,t'} \leq (1 - l^I_t) \cdot S
   && \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.9}
\end{align}$$

Position of the seaside crane while dropping a seaside container: $$\begin{align}
& \Bigl(1 - \sum_{w_i \in I} \sum_{s=1}^{S} d^I_{t,i,s}\Bigr) \cdot S
   + \sum_{w_i \in I} \sum_{s=1}^{S} s \cdot d^I_{t,i,s}
   \;\geq\; x_{w,t'} \;\geq\;
   \sum_{w_i \in I} \sum_{s=1}^{S} s \cdot d^I_{t,i,s} \notag\\
& \hspace{6cm} \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.10}
\end{align}$$

Position of the landside crane while lifting a seaside container: $$\begin{align}
& \Bigl(1 - \sum_{w_i \in I} \sum_{s=1}^{S} l^J_{t,i,s}\Bigr) \cdot (S+1)
   + \sum_{w_i \in I} \sum_{s=1}^{S} s \cdot l^J_{t,i,s}
   \;\geq\; x_{l,t'} \;\geq\;
   \sum_{w_i \in I} \sum_{s=1}^{S} s \cdot l^J_{t,i,s} \notag\\
& \hspace{6cm} \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.11}
\end{align}$$

Position of the landside crane while dropping a seaside container: $$\begin{align}
& \Bigl(1 - \sum_{w_i \in I} d^I_{t,i}\Bigr) \cdot (S+1)
   + \sum_{w_i \in I} s_i \cdot d^I_{t,i}
   \;\geq\; x_{l,t'} \;\geq\;
   \sum_{w_i \in I} s_i \cdot d^I_{t,i} \notag\\
& \hspace{6cm} \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.12}
\end{align}$$

Position of the landside crane while lifting a landside container: $$\begin{align}
& \Bigl(1 - \sum_{l_j \in J} l^J_{t,j}\Bigr) \cdot (S+1)
   + \sum_{l_j \in J} a_j \cdot l^J_{t,j}
   \;\geq\; x_{l,t'} \;\geq\;
   \sum_{l_j \in J} a_j \cdot l^J_{t,j} \notag\\
& \hspace{6cm} \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.13}
\end{align}$$

$$\begin{align}
& x_{l,t'} \geq \sum_{l_j \in J} (S+1) \cdot d^J_{t,j}
   && \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p\} \tag{A.14}
\end{align}$$

Landside crane does not simultaneously lift and drop in the same slot (the paper writes $d^I_{t,s}$, i.e. $d^I_{t,i}$): $$\begin{align}
& \sum_{w_i \in I} \sum_{s=1}^{S} l^J_{t',i,s} + \sum_{l_j \in J} l^J_{t',j}
   \leq 1 - \sum_{w_i \in I} d^I_{t,i} \notag\\
& \hspace{6cm} \forall t \in \{0, \ldots, T-p\},\ t' \in \{t, \ldots, t+p-1\} \tag{A.15}
\end{align}$$

$$\begin{align}
& 0 \leq \sum_{t'=0}^{t} \Bigl( l^I_{t'} - \sum_{w_i \in I} \sum_{s=1}^{S} d^I_{t',i,s} \Bigr) \leq 1
   && \forall t \in \{0, \ldots, T\} \tag{A.16}\\[2pt]
& 0 \leq \sum_{t'=0}^{t} \Bigl( \sum_{w_i \in I} \sum_{s=1}^{S} l^J_{t',i,s}
   + \sum_{l_j \in J} l^J_{t',j}
   - \sum_{w_i \in I} d^I_{t',i} - \sum_{l_j \in J} d^J_{t',j} \Bigr) \leq 1
   && \forall t \in \{0, \ldots, T\} \tag{A.17}\\[2pt]
& 0 \leq \sum_{t'=0}^{t} \sum_{w_i \in I} \Bigl( \sum_{s=1}^{S} l^J_{t',i,s} - d^I_{t',i} \Bigr) \leq 1
   && \forall t \in \{0, \ldots, T\} \tag{A.18}\\[2pt]
& 0 \leq \sum_{t'=0}^{t} \sum_{l_j \in J} \bigl( l^J_{t',j} - d^J_{t',j} \bigr) \leq 1
   && \forall t \in \{0, \ldots, T\} \tag{A.19}\\[2pt]
& \sum_{t=0}^{T} \Bigl( \sum_{s=1}^{S} l^J_{t,i,s} + d^I_{t,i,s_i} \Bigr) = 1
   && \forall w_i \in I \tag{A.20}\\[2pt]
& \sum_{t=0}^{T} \bigl( d^I_{t,i} + d^I_{t,i,s_i} \bigr) = 1
   && \forall w_i \in I \tag{A.21}\\[2pt]
& \sum_{t=0}^{T} l^J_{t,i,s} \leq \sum_{t=0}^{T} d^I_{t,i,s}
   && \forall w_i \in I,\ s \in \{1, \ldots, S\} \tag{A.22}
\end{align}$$

Handover ordering (the constant $T d$ acts as a big-$M$ term, written literally as in the paper): $$\begin{align}
& \Bigl(1 - \sum_{t=0}^{T} l^J_{t,i,s}\Bigr) \cdot T d + \sum_{t=0}^{T} t \cdot l^J_{t,i,s}
   \geq \sum_{t=0}^{T} t \cdot d^I_{t,i,s}
   && \forall w_i \in I,\ s \in \{1, \ldots, S\} \tag{A.23}\\[2pt]
& \sum_{t=0}^{T} t \cdot d^I_{t,i} \geq \sum_{t=0}^{T} \sum_{s=1}^{S} t \cdot l^J_{t,i,s}
   && \forall w_i \in I \tag{A.24}\\[2pt]
& \sum_{t=0}^{T} t \cdot d^J_{t,j} \geq \sum_{t=0}^{T} t \cdot l^J_{t,j}
   && \forall l_j \in J \tag{A.25}\\[2pt]
& (T+1) \cdot u_j \geq C - d_j + 0.5
   && \forall l_j \in J \tag{A.26}\\[2pt]
& u_j \leq \sum_{i=j+1}^{j + \lambda_{j+1}} v_i
   && \forall l_j \in J \tag{A.27}\\[2pt]
& q_j \geq 0.5 \cdot (u_j + v_j)
   && \forall l_j \in J \tag{A.28}\\[2pt]
& \sum_{t=0}^{T} d^J_{t,j} \geq q_j
   && \forall l_j \in J \tag{A.29}\\[2pt]
& q_j \cdot r_j \leq \sum_{t=0}^{T} t \cdot d^J_{t,j} + p \leq d_j
   && \forall l_j \in J \tag{A.30}
\end{align}$$

Variable domains: $$\begin{align}
& l^I_t \in \{0,1\}
   && \forall t \in \{0, \ldots, T\} \tag{A.31}\\[2pt]
& l^J_{t,i,s},\, d^I_{t,i,s} \in \{0,1\}
   && \forall t \in \{0, \ldots, T\},\ s \in \{1, \ldots, S\},\ w_i \in I \tag{A.32}\\[2pt]
& l^J_{t,j},\, d^J_{t,j} \in \{0,1\}
   && \forall t \in \{0, \ldots, T\},\ l_j \in J \tag{A.33}\\[2pt]
& d^I_{t,i} \in \{0,1\}
   && \forall t \in \{0, \ldots, T\},\ w_i \in I \tag{A.34}\\[2pt]
& x_{c,t} \in \mathbb{R}_0^+
   && \forall c \in \{w, l\},\ t \in \{0, \ldots, T\} \tag{A.35}\\[2pt]
& C \in \mathbb{R}_0^+ \tag{A.36}\\[2pt]
& u_j,\, v_j,\, q_j \in \{0,1\}
   && \forall l_j \in J \tag{A.37}\\[2pt]
& v_j = 0
   && \forall j \in \{m+1, \ldots, m + \lambda_m\} \tag{A.38}
\end{align}$$
