# The schedule a temporal planner produced for one plan.
#
# GRSTAPS-X solves task planning, allocation and MILP scheduling together, so
# the timings here are the solver's own output, not an estimate derived after
# the fact. Precedence is implicit: a task whose PDDL precondition is another's
# effect can only start after that one finishes, which shows up as ordered
# timepoints.

std_msgs/Header header

# Matches the plan_id on the ActionSequenceMsg published for the same plan, so
# a schedule can be tied to the geometry it describes.
string plan_id

# Time at which the last task finishes; the quantity the planner minimises.
float64 makespan

# Every scheduled task, ordered by start_time.
omniplanner_msgs/ScheduledTaskMsg[] tasks
