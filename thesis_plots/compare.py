"""Confronti rapidi fra cartelle di run, con lo stile della tesi.

Serve per guardare un risultato appena uscito senza scrivere una figura
registrata: si danno le cartelle, lo stile lo sceglie la palette. Le figure
escono in thesis_images/extra/, non fra quelle impaginate.

Tre modi di usarlo, equivalenti:

    # 1. da riga di comando, niente da editare
    python -m thesis_plots.compare confronto \\
        FedAvg=results/noise_and_mitigation_drift/iid/fedavg/noisy \\
        FedProx=results/noise_and_mitigation_drift/iid/fedprox/noisy

    # 2. importandolo, da un notebook o da uno script
    from thesis_plots import compare
    compare({"FedAvg": "...", "FedProx": "..."}, "confronto")

    # 3. editando il blocco in fondo a questo file, poi
    python -m thesis_plots.compare

Con --ref si aggiungono le curve di riferimento, che escono tratteggiate e
dietro alle altre: il centralizzato, il noiseless, quello che serve.
"""
import argparse
import sys

from . import draw, loaders, palette, paths

# Colori e marcatori per le etichette che non sono nomi di strategia.
FALLBACK_COLORS = [f"C{i}" for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]


def _stile(etichetta, i):
    """Colore, marcatore e sfasamento di una curva.

    Le strategie tengono i colori che hanno in tutta la tesi, cosi' una
    figura al volo e una impaginata si leggono allo stesso modo.
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
    """Curve da un gruppo di file che stanno in una cartella condivisa.

    Alcune campagne non tengono una cartella per braccio: le run delle
    cinque strategie a 40 round stanno tutte in strategy_comparison, e si
    distinguono per il nome del file. Qui il modello e' un percorso con un
    jolly, per esempio .../fedavg_*.json, e i round si aggregano come fa
    read_federated su una cartella.
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
    """Legge una cartella o un gruppo di file, o lascia passare dati gia' letti.

    La sorgente puo' essere una cartella, oppure un percorso con un jolly
    (.../fedavg_*.json) quando le run di piu' bracci condividono la stessa
    cartella. Una sorgente vuota non solleva: i loader tornano curve vuote,
    che disegnate darebbero una figura muta, quindi qui diventa un avviso e
    la curva si salta.
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
    except Exception as exc:                       # sorgente storta o illeggibile
        print(f"  saltata {sorgente}: {exc}")
        return None
    if not dati.get("rounds"):
        print(f"  saltata {sorgente}: nessuna run letta")
        return None
    return dati


def compare(scenari, nome, refs=None, metrics=("loss", "accuracy", "f1"),
            max_round=None, out=None):
    """Confronta piu' cartelle di run in una figura a pannelli.

    scenari: dict {etichetta: cartella} (o {etichetta: dati gia' letti}).
    nome:    nome del file, senza estensione.
    refs:    dict {etichetta: cartella} per le curve di riferimento, che
             escono tratteggiate e dietro alle altre. Un'etichetta che
             contiene "centr" si legge come run centralizzata.
    metrics: quali pannelli disegnare, es. ("loss",) per il solo primo.
    out:     cartella di destinazione; default thesis_images/extra/.

    Restituisce il percorso della figura, o None se non c'era nulla da
    disegnare.
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
        print("Nessuna curva da disegnare: controlla i percorsi.")
        return None

    destinazione = f"{paths.ensure(out or paths.EXTRA)}/{nome}"
    draw.scenario_panels(curve, destinazione, metrics=metrics)
    return destinazione


def _coppia(testo):
    """Converte 'Etichetta=cartella' nei due pezzi."""
    if "=" not in testo:
        raise argparse.ArgumentTypeError(
            f"atteso etichetta=cartella, ricevuto {testo!r}")
    etichetta, _, cartella = testo.partition("=")
    return etichetta, cartella


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m thesis_plots.compare",
        description="Confronto rapido fra cartelle di run, nello stile della tesi.")
    parser.add_argument("nome", nargs="?",
                        help="nome del file prodotto, senza estensione")
    parser.add_argument("scenari", nargs="*", type=_coppia, default=[],
                        help="una o piu' coppie etichetta=cartella")
    parser.add_argument("--ref", action="append", type=_coppia, default=[],
                        metavar="ETICHETTA=CARTELLA",
                        help="curva di riferimento, tratteggiata (ripetibile)")
    parser.add_argument("--metrics", nargs="+", default=["loss", "accuracy", "f1"],
                        help="pannelli da disegnare. Default: loss accuracy f1")
    parser.add_argument("--max-round", type=int, default=None,
                        help="tronca le run piu' lunghe a questo round")
    parser.add_argument("--out", default=None,
                        help="cartella di destinazione. Default: thesis_images/extra")
    args = parser.parse_args(argv)

    if not args.nome:
        # Nessun argomento: si usa il blocco qui sotto, da editare.
        return esempio()

    if not args.scenari:
        parser.error("serve almeno una coppia etichetta=cartella")

    percorso = compare(dict(args.scenari), args.nome, refs=dict(args.ref),
                       metrics=tuple(args.metrics), max_round=args.max_round,
                       out=args.out)
    return 0 if percorso else 1


def esempio():
    """Il confronto da editare, per chi preferisce lavorare sul file.

    Si lancia con `python -m thesis_plots.compare` senza argomenti.

    Qui le cinque strategie senza rumore: le loro run condividono una sola
    cartella e si distinguono per il nome del file, percio' la sorgente e'
    un percorso con un jolly e non una cartella. Il centralizzato, che sta
    in una cartella sua, fa da riferimento tratteggiato.
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
