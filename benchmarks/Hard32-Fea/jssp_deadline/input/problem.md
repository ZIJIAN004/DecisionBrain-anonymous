# Problem Description

A workshop contains M distinct machines and J jobs to be processed. Every job
consists of a fixed, ordered sequence of operations. Each operation must be
processed on a specific machine for a fixed number of time units. The machines
are shared by all jobs, and each machine can process at most one operation at any
moment. Processing is non-preemptive: once an operation starts on a machine it
runs for its full duration without interruption. The operations of each job must
be processed in their given order: an operation may start only after the
preceding operation of the same job has finished, and a job therefore cannot be
processed by two machines at the same time.

For every job, its completion time is the moment its last operation finishes. A
common deadline applies to the whole schedule: every job must complete by the
deadline, which is to say the makespan of the schedule, defined as the largest
completion time over all jobs, must not exceed the given deadline.

The planner must decide the time at which every operation starts on its machine,
for every job. Every operation of every job must be scheduled exactly once, at or
after time zero. The goal is to produce any feasible schedule whose makespan is
at most the deadline. The schedule's makespan is reported as the objective
value; the deadline is a hard constraint, not an objective to be traded off.
