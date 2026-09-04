"""Lettura dei JSON delle run e statistiche di base.

Ogni funzione qui compariva, identica, in tre o quattro script di plot. La
differenza fra le copie era la sola forma delle virgolette, quindi si e'
tenuta una sola versione per ciascuna.

Due formati di file, due famiglie di funzioni:

    federato       {"info": ..., "rounds": [{"round", "eval_metrics_client",
                                             "eval_metrics_server", ...}]}
    centralizzato  {"info": ..., "epochs": [{"epoch", "eval_loss", ...}]}

Le metriche lato client si chiamano "f1", quelle lato server "f1_score".
Salvo dove indicato si legge sempre il client: e' la valutazione su cui si
basano le figure della tesi.
"""
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import median_abs_deviation

from . import palette

METRICS = palette.METRICS_CLIENT


# =====================================================================
# Statistiche
# =====================================================================

def mad(v):
    """Deviazione assoluta mediana di un vettore."""
    v = np.asarray(v, dtype=float)
    return float(np.median(np.abs(v - np.median(v))))


def median_mad(matrix, axis=0):
    """Mediana e MAD lungo l'asse dei seed. Funziona anche su un vettore."""
    matrix = np.asarray(matrix, dtype=float)
    med = np.median(matrix, axis=axis)
    return med, np.median(np.abs(matrix - med), axis=axis)


def sign_test(k, n):
    """p bilaterale del test dei segni con k successi su n."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


# =====================================================================
# Elenchi di file
# =====================================================================

def run_files(folder):
    """JSON delle run in una cartella, escluse le eventuali aggregate."""
    return [f for f in sorted(glob.glob(f"{folder}/*.json"))
            if "aggregated" not in f]


def seed_of(path):
    """Seed dal nome del file, che finisce sempre con _seed<N>.json."""
    return int(str(path).split("seed")[1].split(".")[0])


# =====================================================================
# Curve aggregate sui seed (mediana e MAD gia' calcolate)
# =====================================================================

def read_federated(folder, source="eval_metrics_client", max_round=None):
    """Mediana e MAD per round delle tre metriche, su tutti i seed.

    max_round tronca le run piu' lunghe: il mitigato OU gira per 40 round, ma
    va confrontato coi 10 delle altre curve.
    """
    merged = defaultdict(lambda: {m: [] for m in METRICS})
    for f in run_files(folder):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data["rounds"]:
            r = entry["round"]
            if max_round is not None and r > max_round:
                continue
            m = entry.get(source) or {}
            if not m:
                continue
            merged[r]["loss"].append(m["loss"])
            merged[r]["accuracy"].append(m["accuracy"])
            merged[r]["f1"].append(m.get("f1_score", m.get("f1", 0)))
    return _aggregate(merged, sorted(merged), lambda r: r)


def read_centralized(folder, epochs_per_round=2):
    """Come read_federated, ma sulle epoche, riscalate in round."""
    merged = defaultdict(lambda: {m: [] for m in METRICS})
    for f in sorted(glob.glob(f"{folder}/lr_*_seed*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data["epochs"]:
            if entry.get("eval_loss") is None:
                continue
            e = entry["epoch"]
            merged[e]["loss"].append(entry["eval_loss"])
            merged[e]["accuracy"].append(entry.get("eval_accuracy", 0))
            merged[e]["f1"].append(entry.get("eval_f1", 0))
    epochs = sorted(merged)
    return _aggregate(merged, epochs, lambda e: e / epochs_per_round)


def _aggregate(merged, keys, to_round):
    out = {"rounds": [to_round(k) for k in keys]}
    for metric in METRICS:
        arr = [np.array(merged[k][metric]) for k in keys]
        out[f"{metric}_median"] = [float(np.median(a)) for a in arr]
        out[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr]
    return out


# =====================================================================
# Valori finali, un valore per seed
# =====================================================================

def final_metrics(folder):
    """Ultimo round con valutazione client, un valore per seed e metrica."""
    out = {m: [] for m in METRICS}
    for f in run_files(folder):
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        for r in reversed(rounds):
            m = r.get("eval_metrics_client") or {}
            if m.get("loss") is not None:
                out["loss"].append(m["loss"])
                out["accuracy"].append(m["accuracy"])
                out["f1"].append(m.get("f1_score", m.get("f1", 0)))
                break
    return {k: np.array(v) for k, v in out.items()}


def final_loss(folder):
    """Loss all'ultimo round con valutazione client, un valore per seed."""
    return final_metrics(folder)["loss"]


def accuracy_by_seed(folder):
    """Dizionario seed -> {round: accuracy}, per appaiare due bracci."""
    out = {}
    for f in run_files(folder):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        out[d["info"]["seed"]] = {
            r["round"]: (r.get("eval_metrics_client") or {}).get("accuracy")
            for r in d["rounds"]}
    return out


# =====================================================================
# Matrici grezze (seed, round)
# =====================================================================

def client_metrics(files, n_rounds=None, metrics=None):
    """Metriche client per seed, con ricaduta sul server dove mancano.

    Al round 0 i client non hanno ancora valutato: la ricaduta sul server e'
    quello che fa entrare il round 0 nella curva.
    """
    metrics = metrics or METRICS
    data = {m: [] for m in metrics}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        rounds = d["rounds"] if n_rounds is None else d["rounds"][:n_rounds]
        for m in metrics:
            vals = []
            for r in rounds:
                ec = r.get("eval_metrics_client") or {}
                if m in ec:
                    vals.append(ec[m])
                else:
                    es = r.get("eval_metrics_server") or {}
                    vals.append(es.get("f1_score" if m == "f1" else m, np.nan))
            data[m].append(vals)
    return data


def client_arms(folder, seeds, pattern="fedavg_etal0.3_seed{seed}.json",
                n_rounds=11):
    """Come client_metrics, ma scegliendo i file per seed invece che a glob."""
    folder = Path(folder)
    files = [folder / pattern.format(seed=s) for s in seeds]
    return client_metrics([f for f in files if f.exists()], n_rounds)


def centralized_arms(folder, seeds, n_epochs=10):
    """Riferimento centralizzato, stesse chiavi di client_arms."""
    folder = Path(folder)
    cmap = {"loss": "eval_loss", "accuracy": "eval_accuracy", "f1": "eval_f1"}
    data = {m: [] for m in METRICS}
    for s in seeds:
        fp = folder / f"lr_0.3_seed{s}.json"
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        for m in METRICS:
            data[m].append([e[cmap[m]] for e in d["epochs"][:n_epochs + 1]])
    return data


def client_curves(files, key="loss"):
    """Matrice (seed, round) di una metrica client, sui round comuni.

    Si tengono i numeri di round veri e la loro intersezione fra i seed: la
    valutazione client parte dal round 1, quindi indicizzare con arange()
    trasterebbe la curva di un round.
    """
    per_seed = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        d = {r["round"]: (r.get("eval_metrics_client") or {}).get(key)
             for r in rounds}
        d = {k: v for k, v in d.items() if v is not None}
        if d:
            per_seed.append(d)
    if not per_seed:
        return np.array([]), np.array([])
    common = sorted(set.intersection(*(set(d) for d in per_seed)))
    M = np.array([[d[r] for r in common] for d in per_seed], dtype=float)
    return np.array(common), M


# =====================================================================
# Metriche lato server, per il confronto fra strategie a 40 round
# =====================================================================

def server_curves(base_dir, pattern, seeds, max_round, metrics=None):
    """Matrici (seed, round) delle metriche server, troncate ai round comuni."""
    metrics = metrics or palette.METRICS_SERVER
    base_dir = Path(base_dir)
    all_data = {m: [] for m in metrics}
    for seed in seeds:
        fp = base_dir / pattern.format(seed=seed)
        if not fp.exists():
            print(f"  WARN: missing {fp}")
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        seed_data = {m: [] for m in metrics}
        for r in d["rounds"]:
            if r["round"] > max_round:
                break
            for m in metrics:
                seed_data[m].append(r["eval_metrics_server"][m])
        for m in metrics:
            all_data[m].append(seed_data[m])
    return _trim(all_data, metrics)


def centralized_curves(base_dir, seeds, max_epoch, metrics=None):
    """Come server_curves, sul centralizzato."""
    metrics = metrics or palette.METRICS_SERVER
    base_dir = Path(base_dir)
    cmap = {"loss": "eval_loss", "accuracy": "eval_accuracy",
            "f1_score": "eval_f1"}
    all_data = {m: [] for m in metrics}
    for seed in seeds:
        fp = base_dir / f"lr_0.3_seed{seed}.json"
        if not fp.exists():
            print(f"  WARN: missing {fp}")
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        seed_data = {m: [] for m in metrics}
        for e in d["epochs"]:
            if e["epoch"] > max_epoch:
                break
            for m in metrics:
                seed_data[m].append(e[cmap[m]])
        for m in metrics:
            all_data[m].append(seed_data[m])
    return _trim(all_data, metrics)


def _trim(all_data, metrics):
    if not all_data["loss"]:
        return None
    min_len = min(len(l) for l in all_data["loss"])
    return {m: (np.arange(min_len),
                np.array([l[:min_len] for l in all_data[m]]))
            for m in metrics}


def server_final(base_dir, pattern, seeds, final_round, metrics=None):
    """Metriche server a un round preciso, un valore per seed."""
    metrics = metrics or palette.METRICS_SERVER
    base_dir = Path(base_dir)
    result = {m: [] for m in metrics}
    for seed in seeds:
        fp = base_dir / pattern.format(seed=seed)
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        for r in d["rounds"]:
            if r["round"] == final_round:
                for m in metrics:
                    result[m].append(r["eval_metrics_server"][m])
                break
    return {m: np.array(v) for m, v in result.items()}


def server_last(base_dir, pattern, seeds, metrics=None):
    """Metriche server all'ultimo round disponibile, un valore per seed."""
    metrics = metrics or palette.METRICS_SERVER
    base_dir = Path(base_dir)
    result = {m: [] for m in metrics}
    for seed in seeds:
        fp = base_dir / pattern.format(seed=seed)
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            last = json.load(fh)["rounds"][-1]
        for m in metrics:
            result[m].append(last["eval_metrics_server"][m])
    return {m: np.array(v) for m, v in result.items()}


def centralized_final(base_dir, seeds, final_epoch, metrics=None):
    """Metriche centralizzate a un'epoca precisa, un valore per seed."""
    metrics = metrics or palette.METRICS_SERVER
    base_dir = Path(base_dir)
    cmap = {"loss": "eval_loss", "accuracy": "eval_accuracy",
            "f1_score": "eval_f1"}
    result = {m: [] for m in metrics}
    for seed in seeds:
        fp = base_dir / f"lr_0.3_seed{seed}.json"
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        for e in d["epochs"]:
            if e["epoch"] == final_epoch:
                for m in metrics:
                    result[m].append(e[cmap[m]])
                break
    return {m: np.array(v) for m, v in result.items()}


# =====================================================================
# Ricalibrazioni della CDR
# =====================================================================

def _n_maps(round_entry):
    """Ricalibrazioni e controlli di un round, sommati sui client."""
    n = checks = 0
    for key in ("cdr_train_per_client", "cdr_per_client"):
        for e in (round_entry.get(key) or []):
            n += e.get("n_maps", 0)
            checks += e.get("n_checks") or 0
    return n, checks


def total_calibrations(files):
    """Totale per seed, ricostruito sommando n_maps su round e client.

    NON si usa info["cdr_total_calibrations"]: la strategia scrive quel campo
    solo se il conteggio e' > 0 (custom_strategy.py). Un seed che non
    ricalibra mai - il seed 4 alle soglie da 0.05 in su - non ha il campo, e
    leggerlo direttamente lo farebbe sparire dal campione proprio dov'e' piu'
    basso, gonfiando la mediana. La somma di n_maps coincide col contatore
    quando questo esiste, ed e' 0 quando non esiste.
    """
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        out.append(sum(_n_maps(r)[0] for r in rounds))
    return np.array(out, dtype=float)


def calibration_map(files):
    """Per seed, coppia (ricalibrazioni, controlli).

    Il denominatore e' il numero di volte in cui il criterio di affidabilita'
    e' stato davvero valutato, non il numero di round: ogni client controlla
    piu' volte per round (11 nello sweep), quindi normalizzare per
    client-round sbaglierebbe di un ordine di grandezza. Il contatore non e'
    cumulativo lato client - il mitigatore viene ricreato a ogni round -
    quindi si somma.
    """
    out = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        n = checks = 0
        for r in rounds:
            a, b = _n_maps(r)
            n += a
            checks += b
        out[seed_of(f)] = (n, checks)
    return out


def cumulative_calibrations(files):
    """Matrice (seed, round) delle ricalibrazioni cumulate."""
    curves = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        per_round = [_n_maps(r)[0] for r in rounds if r.get("round", 0) != 0]
        if per_round:
            curves.append(np.cumsum(per_round))
    if not curves:
        return np.array([]), np.array([])
    n = min(len(c) for c in curves)
    return np.arange(1, n + 1), np.array([c[:n] for c in curves], dtype=float)


def centralized_calibrations(folder):
    """Per seed, coppia (ricalibrazioni, controlli) dello sweep centralizzato.

    Dizionario e non array: il risparmio si calcola appaiando i seed, e i due
    rami non sono garantiti completi. Qui n_checks e' gia' cumulativo lungo le
    epoche, quindi il totale e' l'ultimo valore registrato, non la somma.
    """
    out = {}
    for f in sorted(glob.glob(f"{folder}/*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        checks = [e["n_checks"] for e in d["epochs"]
                  if e.get("n_checks") is not None]
        out[seed_of(f)] = (d["info"]["total_calibrations"],
                           checks[-1] if checks else 0)
    return out


def declared_calibrations(folder, seeds):
    """cdr_total_calibrations dichiarato nell'info, un valore per seed.

    Usata solo dagli sweep vecchi, dove il campo c'e' sempre.
    """
    folder = Path(folder)
    vals = []
    for seed in seeds:
        fp = folder / f"fedavg_etal0.3_seed{seed}.json"
        if fp.exists():
            with open(fp, encoding="utf-8") as fh:
                d = json.load(fh)
            vals.append(d.get("info", {}).get("cdr_total_calibrations", 0))
    return np.array(vals)


# =====================================================================
# Tuning degli iperparametri
# =====================================================================

def parse_tuning_name(path, strategies):
    """(strategia, configurazione, seed) dal nome del file di tuning."""
    head, _, seed = Path(path).stem.rpartition("_seed")
    for strat in sorted(strategies, key=len, reverse=True):
        if head.startswith(strat + "_"):
            return strat, head[len(strat) + 1:], int(seed)
    return head, "", int(seed)


def load_tuning(folder, seeds, strategies, metrics=None, keep=None):
    """{strategia: {configurazione: [run, ...]}} dalle metriche server."""
    metrics = metrics or palette.METRICS_SERVER
    by_strategy = defaultdict(lambda: defaultdict(list))
    for fp in sorted(Path(folder).glob("*.json")):
        strat, config, seed = parse_tuning_name(fp, strategies)
        if seed not in seeds or (keep is not None and strat not in keep):
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        run = {m: [r["eval_metrics_server"][m] for r in d["rounds"]]
               for m in metrics}
        run["rounds"] = [r["round"] for r in d["rounds"]]
        by_strategy[strat][config].append(run)
    return by_strategy


def pretty_config(config):
    """Etichetta LaTeX di una configurazione, es. eta0.2_etal0.15."""
    labels = []
    for p in config.split("_"):
        if p.startswith("etal"):
            labels.append(f"$\\eta_l={p[4:]}$")
        elif p.startswith("eta"):
            labels.append(f"$\\eta={p[3:]}$")
        elif p.startswith("mu"):
            labels.append(f"$\\mu={p[2:]}$")
    return ", ".join(labels)


# =====================================================================
# Pesi
# =====================================================================

def load_weights(path):
    """Vettore piatto dei parametri salvati in un .npz."""
    return np.concatenate([np.asarray(v).ravel()
                           for v in np.load(path).values()])


def wrap(d):
    """Differenza angolare avvolta in (-pi, pi], componente per componente."""
    return np.arctan2(np.sin(d), np.cos(d))
