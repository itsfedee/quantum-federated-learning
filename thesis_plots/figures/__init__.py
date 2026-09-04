"""Registro delle figure: nome -> funzione che la produce.

I nomi sono <modulo>.<figura>; un nome senza punto e' il gruppo intero, cioe'
tutte le figure di quel modulo, nell'ordine in cui sono dichiarate.
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

# Figure che valutano il modello quantistico: richiedono qibo, qiboml e le
# dipendenze delle run, e non girano in pochi secondi come le altre. Restano
# fuori da "all" e si chiedono per nome o con --with-qibo.
NEEDS_QIBO = {
    "cap03.partitions",
    "noise.circle",
    "mitigation.regression",
    "tables.parameters",
}


def groups():
    """{gruppo: [nomi]} nell'ordine di dichiarazione."""
    out = {}
    for name in FIGURES:
        out.setdefault(name.split(".")[0], []).append(name)
    return out


def resolve(names, with_qibo=False):
    """Espande gruppi e 'all' nella lista dei nomi da produrre."""
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
    # Dedup conservando l'ordine.
    return list(dict.fromkeys(selected))
