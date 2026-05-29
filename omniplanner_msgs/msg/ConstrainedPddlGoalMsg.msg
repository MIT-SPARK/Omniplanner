# A PDDL goal plus a set of runtime constraints to ground it under.
#
# `goal` is the underlying PDDL goal (same fields as a plain PddlGoalMsg).
# `constraints` is a list of ConstraintFact entries the grounder should respect
# when constructing the PDDL problem instance (e.g. exclude forbidden POIs/edges
# from connectivity).
PddlGoalMsg goal
ConstraintFact[] constraints
