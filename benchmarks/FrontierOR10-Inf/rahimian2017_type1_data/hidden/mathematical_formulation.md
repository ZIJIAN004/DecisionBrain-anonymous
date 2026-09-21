# Original Formulation: Nurse Rostering Problem (NRP)

*Source: A Hybrid Integer Programming and Variable Neighbourhood Search Algorithm to Solve Nurse Rostering Problems, Erfan Rahimian, Kerem Akartunalı, John Levine, 2017.*

## Sets and Parameters

$D$  
set of days in the planning horizon.

$W$  
set of weekends in the planning horizon.

$I$  
set of nurses.

$T$  
set of shift types.

$R_t$  
set of shift types that cannot be assigned immediately after shift type $t \in T$.

$N_i$  
set of days that nurse $i \in I$ cannot be assigned a shift on.

$l_t$  
length of shift type $t \in T$ in minutes.

$m_{it}^{max}$  
maximum number of shifts of type $t \in T$ that can be assigned to nurse $i \in I$.

$b_i^{min}, b_i^{max}$  
minimum and maximum number of minutes that nurse $i \in I$ must be assigned.

$c_i^{min}, c_i^{max}$  
minimum and maximum number of consecutive shifts that nurse $i \in I$ must work. $c$ is the index of possible number of consecutive shifts.

$o_i^{min}$  
minimum number of consecutive days off that nurse $i \in I$ can be assigned. $b$ is the index of possible number of consecutive days off.

$a_i^{max}$  
maximum number of weekends that nurse $i \in I$ can work.

$q_{idt}$  
the incurred penalty if shift type $t \in T$ is not assigned to nurse $i \in I$ on day $d \in D$.

$p_{idt}$  
the incurred penalty if shift type $t \in T$ is assigned to nurse $i \in I$ on day $d \in D$.

$u_{dt}$  
preferred total number of nurses to whom is assigned shift type $t \in T$ on day $d \in D$.

$w_{dt}^{min}, w_{dt}^{max}$  
under-weight and over-weight relevant to the total coverage of shift type $t \in T$ on day $d \in D$.

## Decision Variables

$x_{idt}$  
$= 1$ if nurse $i \in I$ is assigned to shift type $t \in T$ on day $d \in D$, $= 0$ otherwise.

$k_{iw}$  
$= 1$ if nurse $i \in I$ works on weekend $w \in W$, $= 0$ otherwise.

$y_{dt}$  
total number of nurses below the preferred coverage for shift type $t \in T$ on day $d \in D$.

$z_{dt}$  
total number of nurses above the preferred coverage for shift type $t \in T$ on day $d \in D$.

$v_{idt}$  
total incurred penalty relevant to shift on/off requests of nurse $i \in I$ for shift type $t \in T$ on day $d \in D$.

## Objective

$$\begin{equation}
\min \sum_{i \in I} \sum_{d \in D} \sum_{t \in T} v_{idt}
   + \sum_{d \in D} \sum_{t \in T} w_{dt}^{min} y_{dt}
   + \sum_{d \in D} \sum_{t \in T} w_{dt}^{max} z_{dt}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{t \in T} x_{idt} \leq 1, && \forall i \in I,\, d \in D \tag{HC1} \\[4pt]
& x_{idt} + x_{i(d+1)u} \leq 1, && \forall i \in I,\, d \in \{1 \ldots |D|-1\},\, t \in T,\, u \in R_t \tag{HC2} \\[4pt]
& \sum_{d \in D} x_{idt} \leq m_{it}^{max}, && \forall i \in I,\, t \in T \tag{HC3} \\[4pt]
& b_i^{min} \leq \sum_{d \in D} \sum_{t \in T} l_t x_{idt} \leq b_i^{max}, && \forall i \in I \tag{HC4, HC5} \\[4pt]
& \sum_{j=d}^{d+c_i^{max}} \sum_{t \in T} x_{ijt} \leq c_i^{max}, && \forall i \in I,\, d \in \{1 \ldots |D| - c_i^{max}\} \tag{HC6} \\[4pt]
& \sum_{t \in T} x_{idt} + \left( c - 1 - \sum_{j=d+1}^{d+c} \sum_{t \in T} x_{ijt} \right) + \sum_{t \in T} x_{i(d+c+1)t} \geq 0,
   && \forall i \in I,\, c \in \{1 \ldots c_i^{min} - 1\},\, d \in \{1 \ldots |D| - (c+1)\} \tag{HC7} \\[4pt]
& \left(1 - \sum_{t \in T} x_{idt}\right) + \sum_{j=d+1}^{d+b} \sum_{t \in T} x_{ijt} + \sum_{t \in T} x_{i(d+b+1)t} \geq 0,
   && \forall i \in I,\, b \in \{1 \ldots o_i^{min} - 1\},\, d \in \{1 \ldots |D| - (b+1)\} \tag{HC8} \\[4pt]
& k_{iw} \leq \sum_{t \in T} x_{i(7w-1)t} + \sum_{t \in T} x_{i(7w)t} \leq 2 k_{iw}, && \forall i \in I,\, w \in W \tag{HC9} \\
& \sum_{w \in W} k_{iw} \leq a_i^{max}, && \forall i \in I \tag{HC9} \\[4pt]
& x_{int} = 0, && \forall i \in I,\, n \in N_i,\, t \in T \tag{HC10} \\[4pt]
& q_{idt}(1 - x_{idt}) + p_{idt} x_{idt} = v_{idt}, && \forall i \in I,\, d \in D,\, t \in T \tag{SC1} \\[4pt]
& \sum_{i \in I} x_{idt} - z_{dt} + y_{dt} = u_{dt}, && \forall d \in D,\, t \in T \tag{SC2} \\[4pt]
& x_{idt},\, k_{iw} \in \{0, 1\},\quad y_{dt},\, z_{dt},\, v_{idt} \in \mathbb{Z}, && \forall i \in I,\, d \in D,\, t \in T,\, w \in W \notag
\end{align}$$
