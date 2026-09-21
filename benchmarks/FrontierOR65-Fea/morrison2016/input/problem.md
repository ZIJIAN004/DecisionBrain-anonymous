# Problem Description

Given an undirected graph with a specified set of vertices and a specified set of edges connecting pairs of vertices, solve graph coloring through its set-covering formulation. Select the fewest color classes such that the selected classes collectively cover every vertex.

Every selected color class must be a maximal independent set. It must be independent, so no two vertices in the class may be connected by an edge. It must also be maximal: every vertex outside the class must be adjacent to at least one vertex in the class, so no additional vertex can be added while preserving independence. The selected maximal independent sets must collectively cover every vertex at least once; different selected sets may overlap. The objective is to minimize the number of selected color classes.

The input data for an instance consists of the number of vertices, the number of edges, and the list of edges, where each edge is an unordered pair of distinct vertices. An instance may also provide auxiliary lower and upper bounds on the chromatic number.
