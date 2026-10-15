"""Quick comparisons between run folders, in the thesis style.

Meant for looking at a result that has just come out without writing a
registered figure: you give the folders, the palette picks the style. The
figures go to thesis_images/extra/, not among the ones in the document.

Three equivalent ways of using it:

    # 1. from the command line, nothing to edit
    python -m thesis_plots.compare confronto \\
        FedAvg=results/noise_and_mitigation_drift/iid/fedavg/noisy \\
        FedProx=results/noise_and_mitigation_drift/iid/fedprox/noisy

    # 2. importing it, from a notebook or a script
    from thesis_plots import compare
    compare({"FedAvg": "...", "FedProx": "..."}, "confronto")

    # 3. editing the block at the bottom of this file, then
    python -m thesis_plots.compare

With --ref the reference curves are added; they come out dashed and behind
the others: the centralized run, the noiseless one, whatever is needed.
"""
import argparse
import sys

from . import draw, loaders, palette, paths

# Colours and markers for the labels that are not strategy names.
FALLBACK_COLORS = [f"C{i}" for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]


def _stile(etichetta, i):
    """Colour, marker and offset of a curve.

    Strategies keep the colours they have throughout the thesis, so that a
    quick figure and one in the document read the same way.
    """
    nome = palette.STRATEGY_NAMES.get(etichetta.lower(), etichetta)
    if nome in palette.STRATEGY_COLORS:
        return (palette.STRATEGY_COLORS[nome],
                palette.STRATEGY_MARKERS[nome],
                palette.STRATEGY_OFFSETS[nome])
    return (FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
            i % 3)


def _da_glob(modello, max_round=None):
    """Curves from a group of files that sit in a shared folder.

    Some campaigns do not keep one folder per arm: the runs of the five
    strategies at 40 rounds all sit in strategy_comparison, and are told
    apart by the file name. Here the pattern is a path with a wildcard, for
    example .../fedavg_*.json, and the rounds are aggregated the way
    read_federated does on a folder.
    """
    import glob as _glob
    import json as _json
    from collections import defaultdict

    per_round = defaultdict(lambda: {m: [] for m in loaders.METRICS})
    for percorso in sorted(_glob.glob(modello)):
        if "aggregated" in percorso:
            continue
        with open(percorso, encoding="utf-8") as fh:
            dati = _json.load(fh)
        for riga in dati.get("rounds", []):
            r = riga["round"]
            if max_round is not None and r > max_round:
                break
            m = riga.get("eval_metrics_client") or riga.get("eval_metrics_server")
            if not m:
                continue
            per_round[r]["loss"].append(m["loss"])
            per_round[r]["accuracy"].append(m["accuracy"])
            per_round[r]["f1"].append(m.get("f1_score", m.get("f1", 0)))
    if not per_round:
        return {"rounds": []}
    return loaders._aggregate(per_round, sorted(per_round), lambda r: r)


def _carica(sorgente, centralizzato=False, max_round=None):
    """Read a folder or a group of files, or let already-read data through.

    The source can be a folder, or a path with a wildcard (.../fedavg_*.json)
    when the runs of several arms share the same folder. An empty source
    does not raise: the loaders return empty curves, which drawn would give
    a mute figure, so here it becomes a warning and the curve is skipped.
    """
    if not isinstance(sorgente, str):
        return sorgente
    try:
        if "*" in sorgente or "?" in sorgente:
            dati = _da_glob(sorgente, max_round)
        elif centralizzato:
            dati = loaders.read_centralized(sorgente)
        else:
            dati = loaders.read_federated(sorgente, max_round=max_round)
    except Exception as exc:                       # broken or unreadable source
        print(f"  skipped {sorgente}: {exc}")
        return None
    if not dati.get("rounds"):
        print(f"  skipped {sorgente}: no run read")
        return None
    return dati


def compare(scenari, nome, refs=None, metrics=("loss", "accuracy", "f1"),
            max_round=None, out=None):
    """Compare several run folders in a multi-panel figure.

    scenari: dict {label: folder} (or {label: already-read data}).
    nome:    file name, without extension.
    refs:    dict {label: folder} for the reference curves, which come out
             dashed and behind the others. A label containing "centr" is
             read as a centralized run.
    metrics: which panels to draw, e.g. ("loss",) for the first one only.
    out:     destination folder; default thesis_images/extra/.

    Returns the path of the figure, or None if there was nothing to draw.
    """
    curve = []

    for etichetta, sorgente in (refs or {}).items():
        centralizzato = "centr" in etichetta.lower()
        dati = _carica(sorgente, centralizzato, max_round)
        if dati is not None:
            colore = palette.CENTRALIZED if centralizzato else palette.NOISELESS
            curve.append((etichetta, dati, colore, "--", None, 0))

    for i, (etichetta, sorgente) in enumerate(scenari.items()):
        dati = _carica(sorgente, "centr" in etichetta.lower(), max_round)
        if dati is None:
            continue
        colore, marcatore, sfasamento = _stile(etichetta, i)
        curve.append((etichetta, dati, colore, "-", marcatore, sfasamento))

    if not curve:
        print("No curve to draw: check the paths.")
        return None

    destinazione = f"{paths.ensure(out or paths.EXTRA)}/{nome}"
    draw.scenario_panels(curve, destinazione, metrics=metrics)
    return destinazione


def _coppia(testo):
    """Split 'Label=folder' into its two parts."""
    if "=" not in testo:
        raise argparse.ArgumentTypeError(
            f"expected label=folder, got {testo!r}")
    etichetta, _, cartella = testo.partition("=")
    return etichetta, cartella


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m thesis_plots.compare",
        description="Quick comparison between run folders, in the thesis style.")
    parser.add_argument("nome", nargs="?",
                        help="name of the produced file, without extension")
    parser.add_argument("scenari", nargs="*", type=_coppia, default=[],
                        help="one or more label=folder pairs")
    parser.add_argument("--ref", action="append", type=_coppia, default=[],
                        metavar="LABEL=FOLDER",
                        help="reference curve, dashed (repeatable)")
    parser.add_argument("--metrics", nargs="+", default=["loss", "accuracy", "f1"],
                        help="panels to draw. Default: loss accuracy f1")
    parser.add_argument("--max-round", type=int, default=None,
                        help="truncate the longer runs at this round")
    parser.add_argument("--out", default=None,
                        help="destination folder. Default: thesis_images/extra")
    args = parser.parse_args(argv)

    if not args.nome:
        # No arguments: use the block below, meant to be edited.
        return esempio()

    if not args.scenari:
        parser.error("at least one label=folder pair is required")

    percorso = compare(dict(args.scenari), args.nome, refs=dict(args.ref),
                       metrics=tuple(args.metrics), max_round=args.max_round,
                       out=args.out)
    return 0 if percorso else 1


def esempio():
    """The comparison to edit, for those who prefer working on the file.

    Run with `python -m thesis_plots.compare` without arguments.

    Here the five strategies without noise: their runs share a single folder
    and are told apart by the file name, so the source is a path with a
    wildcard and not a folder. The centralized run, which has a folder of
    its own, serves as the dashed reference.
    """
    base = f"{paths.STRATEGY_FED}/iid"
    percorso = compare(
        {
            "FedAvg": f"{base}/fedavg_*.json",
            "FedProx": f"{base}/fedprox_*.json",
            "FedAdagrad": f"{base}/fedadagrad_*.json",
            "FedAdam": f"{base}/fedadam_*.json",
            "FedYogi": f"{base}/fedyogi_*.json",
        },
        "confronto_strategie",
        refs={"Centralized": paths.STRATEGY_CEN},
        max_round=40,
    )
    return 0 if percorso else 1


if __name__ == "__main__":
    sys.exit(main())
