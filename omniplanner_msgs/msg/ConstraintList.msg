# Persistent runtime constraints applied to all subsequent planning calls.
#
# Publish a fresh ConstraintList on the planner's `~/constraints` topic to
# replace the active constraint set; publish with an empty list to clear.
ConstraintFact[] constraints
