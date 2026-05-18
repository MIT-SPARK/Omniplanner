# One PDDL fact representing a runtime constraint.
#
# `predicate` is the PDDL predicate name, e.g. "forbidden-poi" or "forbidden-edge".
# `symbols` is the ordered list of PDDL symbols that the predicate is applied to,
# e.g. ["p123"] for (forbidden-poi p123) or ["p1", "p2"] for (forbidden-edge p1 p2).
string predicate
string[] symbols
