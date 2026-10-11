"""Run a #%% script with only the named cells on. Usage: cells.py script.py gate1,gate2 [old=>new ...]. Every `name = True` gate not listed is set to False; figures go wherever the script writes them."""
import re
import sys
sys.path.insert(0, ".")
src = open(sys.argv[1]).read()
keep = set(sys.argv[2].split(","))
gates = re.findall(r"^(\w+) = True$", src, re.M)
assert keep <= set(gates), keep - set(gates)
for g in gates:
    if g not in keep:
        src = src.replace(f"{g} = True\n", f"{g} = False\n")
for rep in sys.argv[3:]:
    old, new = rep.split("=>")
    assert old in src, old
    src = src.replace(old, new)
exec(compile(src, sys.argv[1], "exec"), {"__name__": "__main__"})
