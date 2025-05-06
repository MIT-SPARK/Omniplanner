import parse
import spark_dsg


def str_to_ns_value(s):
    p = parse.parse("{}({})", s)
    key = p.fixed[0]
    idx = int(p.fixed[1])
    ns = spark_dsg.NodeSymbol(key, idx)
    return ns.value
