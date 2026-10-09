import parse
import spark_dsg


def str_to_ns_value(s):
    p = parse.parse("{}({})", s)
    if p is None:
        raise ValueError(f"{s!r} is not a scene-graph node symbol like 'O(12)'")
    key = p.fixed[0]
    idx = int(p.fixed[1])
    ns = spark_dsg.NodeSymbol(key, idx)
    return ns.value
