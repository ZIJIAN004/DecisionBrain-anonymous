# Problem Description

Given an undirected graph with a set of nodes and a set of edges, each node has a nonnegative integer weight and each edge has a real-valued cost. The input data specifies the number of nodes, the number of edges, the weight of every node, the two endpoints of every edge, the cost of every edge, and a bisection capacity. The bisection capacity is an integer that is at least the ceiling of half the total node weight and at most the total node weight; it serves as an upper bound on the combined weight of either group.

The task is to partition all nodes into exactly two groups so that each node belongs to exactly one group and the sum of node weights in each group does not exceed the bisection capacity. An edge is said to be "cut" when its two endpoints lie in different groups. The objective is to minimize the total cost of all cut edges.
