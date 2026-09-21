# Problem Description

A courier operates a fleet of identical vehicles out of a single depot. The work
consists of transport requests. Each request names one pickup location and one
drop-off location: a vehicle must collect a given quantity of goods at the pickup
location and later unload that same quantity at the drop-off location. A request
cannot be split, and it cannot be handed over between vehicles: the vehicle that
collects the goods is the one that must deliver them, and it must visit the pickup
location before the drop-off location on its route.

Every location, including the depot, has a two-dimensional coordinate. The
distance and the travel time from one location to another are equal, and both are
the Euclidean distance between the two coordinate pairs rounded to the nearest
whole number. Every time in the instance -- the time windows and the service
durations -- is expressed in these same units, so all quantities are integers.

Every location has a time window and a service duration. A vehicle leaves the
depot at time zero. When it arrives at a location before that location's window
opens it waits until the window opens; arriving after the window has closed is not
allowed. Service takes the location's service duration, after which the vehicle
may depart. Every vehicle must be back at the depot no later than the end of the
depot's time window.

Each vehicle has a fixed carrying capacity. The load aboard a vehicle rises by the
request quantity when it serves a pickup and falls by the same quantity when it
serves the matching drop-off. The load must never be negative and must never
exceed the vehicle capacity at any point of the route.

The fleet is limited. At most a given number of vehicles may be used, and this
limit is a hard requirement rather than a cost term: a plan that serves every
request on time but needs one vehicle more than allowed is not a valid plan.

The planner must decide how many vehicles to dispatch, which requests each vehicle
handles, and in what order it visits the corresponding locations. Every request
must be served exactly once. The goal is to minimise the total distance travelled
by all vehicles.

The instance file names these quantities as follows. `num_nodes` is the number of locations including the depot and `num_requests` the number of transport requests; `coordinates`, `demand`, `time_window` and `service_time` are indexed by node, entry 0 being the depot; `requests` lists each request as a `[pickup, delivery]` pair of node indices; `capacity` is the vehicle capacity and `max_vehicles` the fleet limit. `name` is an identifier for the instance and carries no problem data.
