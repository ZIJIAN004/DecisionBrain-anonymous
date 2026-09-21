# Original Formulation: Parallel Batch Scheduling to Minimize Total Flow Time

**Source.** Ozturk (2020), “A truncated column generation algorithm for the parallel batch scheduling problem to minimize total flow time,” Section 3 (straightforward MILP), equations (1)–(7).

**Problem.** $P\,/\,p\text{-Batch},\,r_j,\,p_j,\,v_j,\,\mathrm{Cap}\,/\,\sum C_j$.

## Sets and Indices

- $j = 1,\ldots,N$ — jobs.

- $k = 1,\ldots,N$ — batches (up to $N$ batches may be needed).

- $m = 1,\ldots,M$ — machines.

## Parameters

$r_j$ release date, $p_j$ processing time, $v_j$ size of job $j$; $\mathrm{Cap}$ batch/machine capacity; $Q$ a sufficiently large big-M constant.

## Decision Variables

$$\begin{align*}
& x_{jkm} \in \{0,1\} && \text{$=1$ iff job $j$ is processed in batch $k$ on machine $m$} \\
& p_{km}  \ge 0      && \text{processing duration of batch $k$ on machine $m$} \\
& S_{km}  \ge 0      && \text{start time of batch $k$ on machine $m$} \\
& C_j     \ge 0      && \text{completion (flow) time of job $j$}
\end{align*}$$

## Formulation

$$\begin{align}
\min\ & \sum_{j=1}^{N} C_j \tag{1}\\[2pt]
\text{s.t.}\
& \sum_{k=1}^{N}\sum_{m=1}^{M} x_{jkm} \;=\; 1
  && \forall\, j=1,\ldots,N \tag{2}\\
& \sum_{j=1}^{N} v_j\,x_{jkm} \;\le\; \mathrm{Cap}
  && \forall\, k=1,\ldots,N,\ m=1,\ldots,M \tag{3}\\
& p_{km} \;\ge\; p_j\,x_{jkm}
  && \forall\, j,k=1,\ldots,N,\ m=1,\ldots,M \tag{4}\\
& S_{km} \;\ge\; r_j\,x_{jkm}
  && \forall\, j,k=1,\ldots,N,\ m=1,\ldots,M \tag{5}\\
& S_{km} \;\ge\; S_{k-1,m} + p_{k-1,m}
  && \forall\, k=2,\ldots,N,\ m=1,\ldots,M \tag{6}\\
& C_j \;\ge\; (S_{km} + p_{km}) - Q(1 - x_{jkm})
  && \forall\, j,k=1,\ldots,N,\ m=1,\ldots,M \tag{7}\\
& x_{jkm} \in \{0,1\},\ p_{km},S_{km},C_j \ge 0
  && \forall\, j,k,m.\notag
\end{align}$$

\(1\) minimises total flow time. (2) assigns each job to exactly one batch on one machine. (3) enforces batch capacity. (4) makes the batch duration at least as long as the processing time of any job it contains. (5) makes the batch start no earlier than any of its jobs’ release dates. (6) enforces sequential non-overlap of consecutive batches on the same machine. (7) defines the completion time of each job through the start/duration of the batch that contains it, linearized by big-$Q$.
