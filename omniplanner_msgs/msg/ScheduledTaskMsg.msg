# One scheduled task from a temporal planner.
#
# Emitted alongside the plan rather than inside it: ActionSequenceMsg carries
# geometry only and has no place to put a timepoint, so a schedule published
# here can be inspected and validated without changing the executor contract.

# Robot the task is allocated to.
string robot_name

# Task type, e.g. "visit-location", "pick-object", "place-object".
string action

# DSG symbol the task acts on, lowercased as PDDL requires (e.g. "o8").
string target

# Seconds from the start of the plan. finish_time - start_time is the task's
# duration; a gap before start_time is travel.
float64 start_time
float64 finish_time
