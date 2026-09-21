# Problem Description

A public transit company must simultaneously schedule vehicles and crew members to cover a set of timed bus trips across a network served by multiple depots. The instance specifies the total number of trips, each identified by an index and defined by its departure time in minutes from midnight, arrival time in minutes from midnight, origin, and destination, with trips ordered by increasing departure time. The number of depots and their identifiers, names, and geographic positions are given, along with a dictionary of all named locations and their coordinate pairs, placed within a rectangular area in kilometers. The bus lines operated, each with a from and to endpoint, define the route structure, and the number of lines and the number of trips per line per direction characterize the service pattern. A designated set of relief locations identifies where driver changeovers may take place. Travel times in minutes between every pair of locations and corresponding Euclidean distances in kilometers are provided. Vehicles traveling empty between assignments move at a fixed deadhead speed, while in-service operational speeds vary by time-of-day interval. Departure frequencies per line also vary by time-of-day interval, and a speed profile category determines which set of operational speeds applies. Five crew duty types are defined, each keyed by type name, with each type specifying an identifier, how many consecutive work segments it comprises, the minimum and maximum allowed duration of each work segment in minutes, the minimum required break duration between segments if applicable, the maximum total duty span, the maximum cumulative working time within a duty, and optional earliest or latest start or end times for the duty in minutes from midnight. Crew parameters specify the sign-on time when a driver begins at a depot, the sign-off time when ending at a depot, the extra time penalty when starting or ending at a non-depot relief location, whether that extra time includes the deadhead travel time to the depot, the maximum number of work segments per duty, whether a driver must be continuously present on the vehicle whenever it is outside the depot, and whether a driver may switch to a different vehicle during a break. Vehicle parameters indicate that each vehicle operates from and returns to a single home depot, that the fleet size at each depot is unlimited, that every depot may serve every trip, and a fictitious cost rate per minute that a vehicle sits idle outside its depot.

The company must assign every trip to exactly one vehicle and every work task to exactly one crew duty such that the vehicle schedules and crew schedules are mutually compatible. Each vehicle starts its day at its home depot, performs a sequence of trips connected by deadhead movements, and returns to the same depot at the end of the day. Two trips are compatible for consecutive service by the same vehicle if the earlier trip ends before the later trip begins and the vehicle can travel from the ending location of the earlier trip to the starting location of the later trip in time. Between two consecutive trips assigned to the same vehicle, if the idle time is long enough for the vehicle to travel from the ending location of the first trip to its home depot and then from the depot to the starting location of the next trip, the vehicle is assumed to return to the depot during that gap; such a connection is called a long arc. Otherwise the vehicle remains outside the depot during the idle period, forming a short arc. Every trip must have exactly one predecessor connection and exactly one successor connection in the overall schedule, where a predecessor or successor may be another trip or the depot source or sink respectively. Within each depot's sub-schedule, the flow is conserved at every trip node: the number of vehicle connections entering a trip from that depot equals the number leaving it for that depot. A trip is assigned to a particular depot if and only if that depot's vehicle schedule contains the arcs adjacent to that trip.

Crew members are organized into duties, each associated with a single depot, and a duty of a crew member only involves tasks on vehicles from that same depot. There are two kinds of tasks: trip tasks, which correspond one-to-one with trips, and deadhead tasks, which correspond to the vehicle's repositioning movements between consecutive trips or between a trip and its depot. A piece of work is a maximal sequence of consecutive tasks on one vehicle block without a break, bounded by relief points; its feasibility depends solely on its duration falling between the minimum and maximum piece length for the applicable duty type. Each duty consists of at most two pieces of work. When a duty has two pieces, they are separated by a break of at least the minimum break duration specified by the duty type. When a driver signs on at a depot, a sign-on time is added to the duty duration; when signing off at a depot, a sign-off time is added. When a duty begins or ends at a non-depot relief location, the extra time plus the deadhead travel time from that location to the depot is added to the duty duration instead. A duty need not start and end at its home depot, but a driver must be continuously present on the vehicle whenever it is outside the depot. Changeovers are permitted, meaning a driver may transfer to a different vehicle during a break, so the two pieces of work in a duty need not be on the same vehicle.

The crew schedule must be consistent with the vehicle schedule through the following linkages. A trip task is covered by a duty from a given depot if and only if the corresponding trip is assigned to a vehicle from that depot. For short arcs, the deadhead task between two consecutive trips is covered by a duty if and only if that short arc is used in the vehicle schedule. For deadhead tasks from the end of a trip to the depot, coverage by a duty from depot d is required if and only if either the trip is the last trip of its vehicle (connecting to the depot sink) or the vehicle's next connection from that trip is a long arc, since in both cases the vehicle returns to the depot. Symmetrically, the deadhead task from depot d to the start of a trip is covered if and only if either that trip is the vehicle's first trip from depot d or the vehicle's preceding connection to that trip is a long arc.

Each duty must conform to one of the five defined duty types. The tripper type consists of a single piece of work with no break. The four normal types — early, day, late, and split — each require exactly two pieces of work with a mandatory break. Each type imposes its own limits on minimum and maximum piece duration, minimum break length, maximum duty span, maximum cumulative working time, and allowable windows for when the duty may start and end.

The goal is to minimize the total combined cost of vehicle operations and crew duties. Vehicle costs consist of a fixed cost for each vehicle used plus a variable cost that accrues at the fictitious cost rate for every minute a vehicle is idle outside its depot. Crew costs combine fixed components such as wages with variable components such as overtime. In the standard experimental setting, each vehicle used counts as a cost of one and each driver duty counts as a cost of one, so the objective reduces to minimizing the total number of vehicles plus the total number of drivers.

# Data Instance Variable Explanation

- num_trips: total number of bus trips to be covered
- num_depots: number of vehicle/crew depots
- num_lines: number of bus lines operated
- trips_per_line_per_direction: number of trips scheduled per line per direction
- speed_type: speed profile category determining which operational speeds apply
- generation_seed: random seed used to generate this instance
- coordinate_area_km: dimensions of the rectangular geographic area in kilometers, as [width, height]
- locations: dictionary mapping each location name to its coordinate pair [x, y] in kilometers
- depots: list of depot objects, each containing:
  - depot_id: integer identifier of the depot
  - name: name of the depot (matching a key in locations)
  - coordinates: geographic position [x, y] in kilometers
- relief_locations: list of location names where driver changeovers may take place
- lines: list of bus line objects, each containing:
  - from: origin endpoint of the line
  - to: destination endpoint of the line
- trips: list of trip objects ordered by departure time, each containing:
  - trip_id: integer index of the trip
  - start_time: departure time in minutes from midnight
  - end_time: arrival time in minutes from midnight
  - start_location: origin location name
  - end_location: destination location name
- travel_times: nested dictionary of travel times in minutes, indexed by [origin][destination]
- distances_km: nested dictionary of Euclidean distances in kilometers, indexed by [origin][destination]
- deadhead_speed_km_per_hour: speed in km/h for empty vehicle repositioning moves
- operational_speeds_km_per_hour: dictionary mapping time-of-day interval strings to in-service speed in km/h
- frequency_table_minutes: dictionary mapping time-of-day interval strings to departure frequency in minutes per line
- duty_types: dictionary of crew duty types keyed by type name, each containing:
  - type_id: integer identifier of the duty type
  - num_pieces: number of consecutive work segments in a duty of this type
  - piece_length_min: minimum allowed duration of each work segment in minutes
  - piece_length_max: maximum allowed duration of each work segment in minutes
  - break_length_min: minimum required break duration between segments in minutes (null if not applicable)
  - duty_length_max: maximum total span of the duty in minutes (null if unconstrained)
  - work_time_max: maximum cumulative working time within the duty in minutes (null if unconstrained)
  - start_time_min: earliest allowed start time in minutes from midnight (null if unconstrained)
  - start_time_max: latest allowed start time in minutes from midnight (null if unconstrained)
  - end_time_max: latest allowed end time in minutes from midnight (null if unconstrained)
- crew_parameters: dictionary of crew scheduling parameters containing:
  - sign_on_time_depot_minutes: time added when a driver signs on at a depot
  - sign_off_time_depot_minutes: time added when a driver signs off at a depot
  - extra_time_non_depot_relief_minutes: extra time penalty when starting or ending at a non-depot relief location
  - extra_time_includes_deadhead_to_depot: whether the extra time includes deadhead travel time to the depot (boolean)
  - max_pieces_per_duty: maximum number of work segments per duty
  - continuous_attendance: whether a driver must stay on the vehicle whenever it is outside the depot (boolean)
  - changeovers_allowed: whether a driver may switch to a different vehicle during a break (boolean)
- vehicle_parameters: dictionary of vehicle scheduling parameters containing:
  - each_vehicle_has_own_depot: whether each vehicle operates from and returns to a single home depot (boolean)
  - unlimited_vehicles_per_depot: whether the fleet size at each depot is unlimited (boolean)
  - all_depots_can_serve_all_trips: whether every depot may serve every trip (boolean)
  - fictitious_cost_per_minute_empty_outside_depot: cost rate per minute for a vehicle idle outside its depot

# Data Instance
