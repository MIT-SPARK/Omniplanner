import logging
from functools import reduce

logger = logging.getLogger(__name__)


def extract_facts(goal, predicate):
    match goal:
        case tuple() | list():
            if len(goal) == 0:
                return ()
            elif goal[0] == predicate:
                return (goal,)
            else:
                children = (extract_facts(g, predicate) for g in goal)
                return reduce(lambda x, y: x + y, children)
        case _:
            return ()


def tokenize_lisp(string):
    return string.replace("\n", "").replace("(", "( ").replace(")", " )").split()


def get_lisp_ast(toks):
    t = toks.pop(0)
    if t == "(":
        exp = ()
        while toks[0] != ")":
            exp += (get_lisp_ast(toks),)
        toks.pop(0)
        return exp
    elif t != ")":
        return t
    else:
        raise SyntaxError("Unexpected )")


def lisp_string_to_ast(string):
    return get_lisp_ast(tokenize_lisp(string))


def ast_to_string(ast):
    match ast:
        case list() | tuple():
            elements = [ast_to_string(e) for e in ast]
            elements_str = " ".join(elements)
            return f"({elements_str})"
        case _:
            return str(ast)


# PDDL is case-insensitive, but spark_dsg is case sensitive.
# Here to convert back and forth, but this is *extremely* brittle.
def pddl_char_to_dsg_char(c):
    match c:
        case "r":
            return "R"
        case "o":
            return "O"
        case "p":
            # NOTE: this means we can only ground to 2d places, not 3d places
            return "P"
        case _:
            return c


# Regions are R(i) in Hydra's rooms layer but r(i) in hydra_multi's open-vocabulary
# regions (offline fusion); a PDDL region symbol names whichever one the graph has.
REGION_DSG_CHARS = ("R", "r")


def pddl_symbol_to_node_symbol(G, pddl_symbol: str):
    """The scene-graph node a PDDL symbol (``o12``, ``p3``, ``r0``) names in ``G``.

    Regions are normalized to the graph at planning time: ``r0`` is ``R(0)`` if the
    graph has it, else ``r(0)``. Raises ValueError for a token that is not a symbol.
    """
    import spark_dsg

    c, index = pddl_symbol[0], int(pddl_symbol[1:])
    ns = spark_dsg.NodeSymbol(pddl_char_to_dsg_char(c), index)
    if c == "r" and G is not None and not G.has_node(ns):
        for alt in REGION_DSG_CHARS:
            alt_ns = spark_dsg.NodeSymbol(alt, index)
            if G.has_node(alt_ns):
                return alt_ns
    return ns


def dsg_char_to_pddl_char(c: str) -> str:
    """Inverse of pddl_char_to_dsg_char. 'O' -> 'o', 'P' -> 'p', 'R' -> 'r'."""
    return c.lower()
