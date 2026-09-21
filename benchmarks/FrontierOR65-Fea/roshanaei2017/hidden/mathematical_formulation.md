# Original Formulation: Collaborative Operating Room Planning and Scheduling (CORPS)

*Source: Collaborative Operating Room Planning and Scheduling, Vahid Roshanaei, Curtiss Luong, Dionne M. Aleman, David R. Urbach, INFORMS Journal on Computing 29(3):558–580, 2017.*

## Sets and Indices

$$\begin{align*}
& p,\, s,\, h,\, d,\, r && \text{indices for patients, surgeons, hospitals, days, ORs}\\
& \mathcal{P} && \text{set of patients } (p=1,\dots,|\mathcal{P}|)\\
& \mathcal{S} && \text{set of surgeons } (s=1,\dots,|\mathcal{S}|)\\
& \mathcal{H} && \text{set of hospitals } (h=1,\dots,|\mathcal{H}|)\\
& \mathcal{D} && \text{set of days in the planning horizon } (d=1,\dots,|\mathcal{D}|)\\
& \mathcal{R}_h && \text{set of ORs in hospital } h \ (r=1,\dots,|\mathcal{R}_h|)\\
& \mathcal{P}_{hdr} && \text{patients operable in room } r \text{ of hospital } h \text{ with } \theta_p \geq d\\
& \Lambda_{s,d} && \text{patients belonging to surgeon } s \text{ with } \theta_p \geq d\\
& \Omega_p && \text{set of preferred (qualified) surgeons for patient } p\\
& \Delta_s && \text{set of days on which surgeon } s \text{ operates}\\
& \mathcal{C}_{ph} && \text{set of ORs at hospital } h \text{ eligible for patient } p
\end{align*}$$ SP-specific sets (Table 2), defined over the case set of one hospital-day: $$\begin{align*}
& \mathcal{C}_p && \text{qualified ORs for patient } p\\
& \mathcal{C}_{pk} && \text{qualified ORs shared between patients } p \text{ and } k\\
& \Omega_p && \text{qualified surgeons for patient } p\\
& \Omega_{pk} && \text{qualified surgeons shared between patients } p \text{ and } k\\
& \hat{\mathcal{P}}_{hd}^{(i)},\ \hat{\mathcal{S}}_{hd}^{(i)},\ \hat{\mathcal{R}}_{hd}^{(i)}
  && \text{patients, surgeons, ORs assigned by the MP to hospital } h, \text{ day } d \text{ at iteration } i
\end{align*}$$

## Parameters

$$\begin{align*}
& K_{hdr},\, C_{hdr} && \text{fixed and variable (overtime) costs of OR } r \text{ in hospital } h \text{ on day } d\\
& L_{shd} && \text{fixed cost of surgeon } s \text{ operating in hospital } h \text{ on day } d\\
& B_{hdr}\ (\text{SP: } B_r) && \text{regular time of OR } r \text{ in hospital } h \text{ on day } d\\
& T_{ps} && \text{total preparation + surgery + cleaning time of patient } p \text{ by surgeon } s\\
& F_p && \text{preparation time of patient } p\\
& G_p && \text{OR turnover (cleaning) time after patient } p\\
& E_{ps} && \text{surgical procedure time of patient } p \text{ by surgeon } s,\ \ E_{ps}=T_{ps}-(G_p+F_p)\\
& A_{sd}\ (\text{SP: } A_s) && \text{available time of surgeon } s \text{ on day } d\\
& U_p && \text{reward for optional patient } p \text{ if operated on in the current horizon}\\
& \theta_p && \text{due date of patient } p\\
& V_{hdr}\ (\text{SP: } V_r) && \text{maximum allowable overtime of OR } r\\
& \alpha && \text{surgeon schedule tightness coefficient, } \alpha \in [0,1]\\
& M && \text{a large positive number (SP)}
\end{align*}$$

## Decision Variables

Allocation Master Problem (MP): $$\begin{align*}
& x_{pshdr} \in \{0,1\} && 1 \text{ if patient } p \text{ is operated by surgeon } s \text{ in hospital } h \text{ on day } d \text{ in room } r\\
& y_{hdr} \in \{0,1\} && 1 \text{ if room } r \text{ of hospital } h \text{ on day } d \text{ is opened}\\
& z_{shd} \in \{0,1\} && 1 \text{ if surgeon } s \text{ is in hospital } h \text{ on day } d\\
& v_{hdr} \geq 0 && \text{overtime of OR } r \text{ in hospital } h \text{ on day } d
\end{align*}$$ Sequencing Subproblem (SP), per hospital-day $hd$: $$\begin{align*}
& x_{psr} \in \{0,1\} && 1 \text{ if patient } p \text{ is operated by surgeon } s \text{ in OR } r\\
& \eta_{pkr} \in \{0,1\} && 1 \text{ if patient } p \text{ is operated after patient } k \text{ in OR } r\\
& \pi_{pks} \in \{0,1\} && 1 \text{ if patient } p \text{ is operated after patient } k \text{ on surgeon } s\text{'s list}\\
& f_p \geq 0 && \text{finishing time of surgical case } p\\
& c_r \geq 0 && \text{completion time of OR } r\\
& v_r \geq 0 && \text{overtime of OR } r\\
& i_s \geq 0 && \text{starting time of surgeon } s\\
& e_s \geq 0 && \text{ending time of surgeon } s
\end{align*}$$

## Objective

**Allocation Master Problem (MP).** Minimize OR-opening, surgeon, and overtime costs, less the rewards earned by scheduling optional patients: $$\begin{equation}
\min \Bigg\{
\sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{R}_h} K_{hdr}\, y_{hdr}
+ \sum_{s \in \mathcal{S}} \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} L_{shd}\, z_{shd}
+ \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{R}_h} C_{hdr}\, v_{hdr}
- \sum_{p \in \mathcal{P} \,\mid\, \theta_p > |\mathcal{D}|} U_p
  \sum_{s \in \Omega_p} \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{C}_{ph}} x_{pshdr}
\Bigg\} \tag{MP}
\end{equation}$$

**Sequencing Subproblem (SP), for hospital-day $hd$.** Minimize OR overtime cost: $$\begin{equation}
\min\ \ \bar{v}_{hdr}^{(i)} = \sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)}} c_r\, v_r \tag{SP}
\end{equation}$$

## Constraints

#### Allocation Master Problem (MP).

$$\begin{align}
& \sum_{s \in \Omega_p} \sum_{h \in \mathcal{H}} \sum_{d \leq \theta_p} \sum_{r \in \mathcal{C}_{ph}} x_{pshdr} = 1
  && \forall\, p \in \mathcal{P} \mid \theta_p \leq |\mathcal{D}| \tag{1}\\
& \sum_{s \in \Omega_p} \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{C}_{ph}} x_{pshdr} \leq 1
  && \forall\, p \in \mathcal{P} \mid \theta_p > |\mathcal{D}| \tag{2}\\
& \sum_{h \in \mathcal{H}} z_{shd} \leq 1
  && \forall\, s \in \mathcal{S};\ d \in \Delta_s \tag{3}\\
& x_{pshdr} \leq z_{shd}
  && \forall\, p \in \mathcal{P};\ s \in \Omega_p;\ h \in \mathcal{H};\ d \in \Delta_s;\ r \in \mathcal{C}_{ph} \tag{4}\\
& x_{pshdr} \leq y_{hdr}
  && \forall\, p \in \mathcal{P};\ s \in \Omega_p;\ h \in \mathcal{H};\ d \in \Delta_s;\ r \in \mathcal{C}_{ph} \tag{5}\\
& \sum_{p \in \mathcal{P}_{hdr}} \sum_{s \in \Omega_p} T_{ps}\, x_{pshdr} \leq B_{hdr}\, y_{hdr} + v_{hdr}
  && \forall\, h \in \mathcal{H};\ d \in \Delta_s;\ r \in \mathcal{R}_h \tag{6}\\
& \sum_{p \in \Lambda_{s,d}} \sum_{r \in \mathcal{C}_{ph}} \big(\alpha E_{ps} + (1-\alpha) T_{ps}\big) x_{pshdr} \leq A_{sd}\, z_{shd}
  && \forall\, s \in \mathcal{S};\ h \in \mathcal{H};\ d \in \Delta_s \tag{7}\\
& V_{hdr} \geq v_{hdr} \geq 0
  && \forall\, h \in \mathcal{H};\ d \in \mathcal{D};\ r \in \mathcal{R}_h \tag{8}\\
& x_{pshdr},\, y_{hdr},\, z_{shd} \in \{0,1\}
  && \forall\, p \in \mathcal{P};\ s \in \Omega_p;\ h \in \mathcal{H};\ d \in \Delta_s;\ r \in \mathcal{C}_{ph} \notag
\end{align}$$

#### Sequencing Subproblem (SP), for hospital-day $hd$ at iteration $i$.

(All summation index sets are intersected with the MP-assigned sets $\hat{\mathcal{S}}_{hd}^{(i)}$, $\hat{\mathcal{R}}_{hd}^{(i)}$, $\hat{\mathcal{P}}_{hd}^{(i)}$.) $$\begin{align}
& \sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}
  \sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p} x_{psr} = 1
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)} \tag{9}\\
& f_p \geq F_p + \sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}
  \sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p} E_{ps}\, x_{psr}
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)} \tag{10}\\
& f_p \geq f_k + G_k + F_p + \sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p} E_{ps}\, x_{psr}
  - M\Big(3 - \eta_{pkr} - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}\!\! x_{psr}
  - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_k}\!\! x_{ksr}\Big)
  && \forall\, p,k \in \hat{\mathcal{P}}_{hd}^{(i)} \mid p<k;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_{pk} \tag{11}\\
& f_k \geq f_p + G_p + F_k + \sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_k} E_{ks}\, x_{ksr}
  - M\Big(2 + \eta_{pkr} - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}\!\! x_{psr}
  - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_k}\!\! x_{ksr}\Big)
  && \forall\, p,k \in \hat{\mathcal{P}}_{hd}^{(i)} \mid p<k;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_{pk} \tag{12}\\
& f_p \geq f_k + E_{ps}
  - M\Big(3 - \pi_{pks} - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p}\!\! x_{psr}
  - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_k}\!\! x_{ksr}\Big)
  && \forall\, p,k \in \hat{\mathcal{P}}_{hd}^{(i)} \mid p<k;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_{pk};\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_{pk} \tag{13}\\
& f_k \geq f_p + E_{ks}
  - M\Big(2 + \pi_{pks} - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p}\!\! x_{psr}
  - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_k}\!\! x_{ksr}\Big)
  && \forall\, p,k \in \hat{\mathcal{P}}_{hd}^{(i)} \mid p<k;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_{pk};\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_{pk} \tag{14}\\
& f_p + G_p - M\Big(1 - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}\!\! x_{psr}\Big) \leq B_r + v_r
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)};\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p \tag{15}\\
& e_s \geq f_p - M\Big(1 - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p}\!\! x_{psr}\Big)
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)} \cap \Lambda_s;\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p \tag{16}\\
& i_s \leq f_p - E_{ps} + M\Big(1 - \!\!\sum_{r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p}\!\! x_{psr}\Big)
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)} \cap \Lambda_s;\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p \tag{17}\\
& e_s - i_s \leq A_s
  && \forall\, s \in \hat{\mathcal{S}}_{hd}^{(i)} \tag{18}\\
& c_r \geq f_p + G_p - M\Big(1 - \!\!\sum_{s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p}\!\! x_{psr}\Big)
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)};\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p \tag{19}\\
& 0 \leq v_r \leq V_r
  && \forall\, r \in \hat{\mathcal{R}}_{hd}^{(i)} \tag{20}\\
& v_r \geq c_r - B_r
  && \forall\, r \in \hat{\mathcal{R}}_{hd}^{(i)} \tag{21}
\end{align}$$ SP variable domains (reproduced verbatim, including the paper’s $\pi/\eta$ index swap; see Remarks): $$\begin{align}
& x_{psr} \in \{0,1\}
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)};\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_p;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_p \notag\\
& \pi_{pkr} \in \{0,1\}
  && \forall\, p = 1,\dots,|\hat{\mathcal{P}}_{hd}^{(i)}|-1;\ p<k \leq |\hat{\mathcal{P}}_{hd}^{(i)}|;\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \cap \mathcal{C}_{pk} \notag\\
& \eta_{pks} \in \{0,1\}
  && \forall\, p = 1,\dots,|\hat{\mathcal{P}}_{hd}^{(i)}|-1;\ p<k \leq |\hat{\mathcal{P}}_{hd}^{(i)}|;\ s \in \hat{\mathcal{S}}_{hd}^{(i)} \cap \Omega_{pk} \notag\\
& f_p,\, e_s,\, i_s,\, c_r,\, v_r \geq 0
  && \forall\, p \in \hat{\mathcal{P}}_{hd}^{(i)};\ s \in \hat{\mathcal{S}}_{hd}^{(i)};\ r \in \hat{\mathcal{R}}_{hd}^{(i)} \notag
\end{align}$$
