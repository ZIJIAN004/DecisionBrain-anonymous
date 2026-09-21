# Problem Description

A perfect graph is given, consisting of a set of vertices and a set of edges connecting pairs of vertices. The vertices are partitioned into a number of disjoint clusters, where every vertex belongs to exactly one cluster and each cluster contains at least one vertex. The input data specifies the total number of vertices, the complete list of edges (each defined by its two endpoint vertices), the number of clusters, and the membership of each vertex in its cluster.

Exactly one vertex must be selected from each cluster. The selected vertices, together with the edges of the original graph that connect pairs of selected vertices, form an induced subgraph. The selected vertices must then be colored so that no two selected vertices sharing an edge receive the same color.

The goal is to choose one vertex per cluster and assign one color to each chosen vertex, respecting the edge-conflict rule above, so as to minimize the total number of distinct colors used across all selected vertices.
