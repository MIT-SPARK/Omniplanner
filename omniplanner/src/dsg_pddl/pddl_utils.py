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
