# Problem Description

A finite set of jobs must each be assigned to exactly one machine drawn from a finite set of machines. Assigning a given job to a given machine incurs a known cost, specified for every (machine, job) pair. The total cost of an assignment is the sum, over all jobs, of the cost of the (machine, job) pair selected for that job. The goal is to choose an assignment that minimizes this total cost.

Processing a given job on a given machine consumes a known amount of that machine's resource, specified for every (machine, job) pair. Each machine has a fixed resource capacity. For every machine, the total resource consumed by all the jobs assigned to that machine must not exceed the resource capacity of that machine.

The decision for every (machine, job) pair is binary: it equals 1 if the job is assigned to that machine and 0 otherwise. Each job must be assigned to exactly one machine, so for every job exactly one (machine, job) pair takes the value 1. A machine may receive any number of jobs, provided the per-machine capacity requirement is satisfied.
