"""Figure registry: name -> function that produces it.

Names are <module>.<figure>; a name without a dot is the whole group, that
is every figure of that module, in declaration order.
"""
from .. import style as ts

ts.apply()

from . import (extra, memory, methodology, mitigation, noise, strategies,  # noqa: E402
               tables, tuning)

MODULES = [methodology, strategies, tuning, noise, mitigation, memory, tables,
           extra]

FIGURES = {}
for _m in MODULES:
    FIGURES.update(_m.FIGURES)

# Figures that evaluate the quantum model: they need qibo, qiboml and the run
# dependencies, and do not run in a few seconds like the others. They stay
# out of "all" and are requested by name or with --with-qibo.
NEEDS_QIBO = {
    "cap03.partitions",
    "noise.circle",
    "mitigation.regression",
    "tables.parameters",
}


def groups():
    """{group: [names]} in declaration order."""
    out = {}
    for name in FIGURES:
        out.setdefault(name.split(".")[0], []).append(name)
    return out


def resolve(names, with_qibo=False):
    """Expand groups and 'all' into the list of names to produce."""
    by_group = groups()
    selected = []
    for name in names:
        if name == "all":
            selected += list(FIGURES)
        elif name in by_group:
            selected += by_group[name]
        elif name in FIGURES:
            selected.append(name)
        else:
            raise KeyError(name)
    if not with_qibo:
        selected = [n for n in selected
                    if n in names or n not in NEEDS_QIBO]
    # Dedup, keeping the order.
    return list(dict.fromkeys(selected))
