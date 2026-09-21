# Problem Description

Given a set of n points in d-dimensional Euclidean space, where each point is specified by its d real-valued coordinates, and a maximum allowed tolerance epsilon, the task is to cover the points using as few hyperplanes as possible. A hyperplane in d-dimensional space is any flat (d-1)-dimensional affine subspace, and the Euclidean distance from a point to a hyperplane is the standard shortest distance from that point to any point on the hyperplane.

Every point must be covered by at least one hyperplane, meaning its Euclidean distance to that hyperplane is at most epsilon. Each point may be covered by any number of hyperplanes, but must be covered by at least one. Hyperplanes are to be chosen freely in the ambient space (their positions and orientations are decisions of the problem).

The goal is to determine the minimum number of hyperplanes, together with their positions and orientations, such that every one of the n points lies within Euclidean distance epsilon of at least one chosen hyperplane.

In the required solution representation, each hyperplane is defined by a nonzero normal vector `w` and a scalar right-hand-side value `w0` using the equation `w dot p = w0`. Consequently, the Euclidean distance from a point `p` to that hyperplane is `abs(w dot p - w0) / ||w||_2`. The `w0` field in `solution.json` is this right-hand-side value; it is not the additive intercept `b` from the alternative convention `w dot p + b = 0`. An implementation that internally uses the latter convention must output `w0 = -b`.
