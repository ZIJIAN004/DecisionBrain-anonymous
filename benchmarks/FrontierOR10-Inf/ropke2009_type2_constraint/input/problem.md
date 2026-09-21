# Problem Description

A fleet of identical vehicles operates out of a single depot to fulfill a set of transportation requests. Each request specifies that a certain load must be picked up at one location and delivered to a different location. The number of requests and the number of available vehicles are given as input. Each vehicle has the same carrying capacity. The input data for each request includes the pickup and delivery coordinates, the load to be transported (a positive quantity), the time window at the pickup location (earliest and latest times at which service may begin), and the time window at the delivery location. The depot also has a time window representing the earliest departure and latest return times for all vehicles. For every pair of locations (including the depot), the routing cost and travel time are given. Travel times include the service duration at the origin location of each trip leg, and both routing costs and travel times satisfy the triangle inequality. Service durations at the depot are zero.

The dispatcher must assign each request to exactly one vehicle and determine the sequence in which that vehicle visits its assigned pickup and delivery locations. Every request must be served: no request may be left unassigned and no request may be split across vehicles. For each request, the same vehicle that performs the pickup must also perform the corresponding delivery, and the pickup must occur before the delivery on that vehicle's route. Each vehicle in the fleet departs from the depot and returns to the depot, including vehicles that serve no requests.

When a vehicle arrives at a location before the earliest allowable service time, it may wait until service is permitted. Service at any location must begin no later than the latest allowable time specified by that location's time window. After a vehicle begins service and the travel time to the next location elapses (recall that travel time already incorporates the service duration at the current location), the vehicle arrives at its next stop, where the same time window rules apply.

The load carried by each vehicle changes as pickups and deliveries are performed. When a pickup occurs, the vehicle's load increases by the request's load quantity; when the corresponding delivery occurs, the load decreases by the same amount. At no point along any vehicle's route may the carried load exceed the vehicle capacity, and at no point may it fall below zero.

The goal is to minimize the total routing cost, defined as the sum of the routing costs on all arcs traversed by all vehicles.


## Additional binding constraint
In addition to all requirements above, every vehicle's carried load must equal zero at every pickup, delivery, and intermediate state.
