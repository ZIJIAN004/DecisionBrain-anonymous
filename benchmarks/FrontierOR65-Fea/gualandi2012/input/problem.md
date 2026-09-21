# Problem Description

Given an undirected graph with a specified set of vertices and a specified set of edges connecting pairs of vertices, the task is to assign a color to every vertex such that no two vertices connected by an edge share the same color, while using the fewest colors possible. The number of vertices, the number of edges, and the complete list of edges (each given as a pair of vertex indices) are provided as input data.

Each vertex must be assigned exactly one color, drawn from a set of available colors. For every edge in the graph, the two endpoint vertices must receive different colors. A coloring is evaluated by the number of distinct colors it uses across all vertices.

The objective is to minimize the number of distinct colors used in a feasible coloring, i.e., to find a vertex coloring that respects the adjacency-difference rule and uses as few colors as possible.
