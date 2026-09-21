# Mathematical Formulation

Let `V = {0, 1, ..., n}` be the nodes, node `0` the depot and `1..n` the customers.
Let `d_i` be the demand of customer `i`, `[e_i, l_i]` its time window, `s` the
common service duration, `Q` the vehicle capacity and `K` the fleet limit. Travel
cost and travel time on arc `(i, j)` are

    c_ij = t_ij = floor(10 * sqrt((x_i - x_j)^2 + (y_i - y_j)^2))

with the instance's times already on the same scale.

Decision variables: `x_ij in {0,1}` indicates that some vehicle travels directly
from `i` to `j`; `a_i >= 0` is the time service begins at node `i`; `u_i` is the
load carried after serving `i`.

    minimise    sum_(i,j) c_ij x_ij                                        (1)

    subject to  sum_j x_ij = 1                     for all i in 1..n       (2)
                sum_i x_ij = 1                     for all j in 1..n       (3)
                sum_j x_0j = sum_i x_i0                                    (4)
                sum_j x_0j <= K                                            (5)
                a_j >= a_i + s_i + t_ij - M (1 - x_ij)  for all i, j != 0   (6)
                e_i <= a_i <= l_i                  for all i in 1..n       (7)
                a_0 = 0, and every return to the depot is at most l_0      (8)
                u_j >= u_i + d_j - Q (1 - x_ij)    for all i, j != 0       (9)
                d_j <= u_j <= Q                    for all j in 1..n      (10)

Constraint (5) is the fleet cap. It is a hard constraint, not a penalty: the
instances are chosen so that `K` equals `ceil(sum_i d_i / Q)`, the bin-packing
lower bound on the fleet, so (5) admits no slack.

Feasibility is decided by (2), (3), (5), (7), (8) and (10); the checker verifies
those directly on the reported routes, together with objective consistency
with (1).
