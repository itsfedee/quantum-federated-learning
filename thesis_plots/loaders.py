"""Reading of the run JSONs and basic statistics.

Every function here appeared, identical, in three or four plotting scripts.
The only difference between the copies was the quoting style, so a single
version of each was kept.

Two file formats, two families of functions:

    federated      {"info": ..., "rounds": [{"round", "eval_metrics_client",
                                             "eval_metrics_server", ...}]}
    centralized    {"info": ..., "epochs": [{"epoch", "eval_loss", ...}]}

Client-side metrics are called "f1", server-side ones "f1_score". Unless
stated otherwise the client side is always read: it is the evaluation the
thesis figures are based on.
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
# Statistics
# =====================================================================

def mad(v):
    """Median absolute deviation of a vector."""
    v = np.asarray(v, dtype=float)
    return float(np.median(np.abs(v - np.median(v))))


def median_mad(matrix, axis=0):
    """Median and MAD along the seed axis (works on a vector too)."""
    matrix = np.asarray(matrix, dtype=float)
    med = np.median(matrix, axis=axis)
    return med, np.median(np.abs(matrix - med), axis=axis)


def sign_test(k, n):
    """Two-sided p of the sign test with k successes out of n."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


# =====================================================================
# File lists
# =====================================================================

def run_files(folder):
    """Run JSONs in a folder, excluding any aggregated ones."""
    return [f for f in sorted(glob.glob(f"{folder}/*.json"))
            if "aggregated" not in f]


def seed_of(path):
    """Seed from the file name, which always ends with _seed<N>.json."""
    return int(str(path).split("seed")[1].split(".")[0])


# =====================================================================
# Curves aggregated over seeds (median and MAD already computed)
# =====================================================================

def read_federated(folder, source="eval_metrics_client", max_round=None):
    """Median and MAD per round of the three metrics, over all seeds.

    max_round truncates the longer runs.
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
    """Like read_federated, but over epochs, rescaled to rounds."""
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
# Final values, one value per seed
# =====================================================================

def final_metrics(folder):
    """Last round with a client evaluation, one value per seed and metric."""
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
    """Loss at the last round with a client evaluation, one value per seed."""
    return final_metrics(folder)["loss"]


def accuracy_by_seed(folder):
    """Dictionary seed -> {round: accuracy}, to pair two arms."""
    out = {}
    for f in run_files(folder):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        out[d["info"]["seed"]] = {
            r["round"]: (r.get("eval_metrics_client") or {}).get("accuracy")
            for r in d["rounds"]}
    return out


# =====================================================================
# Raw (seed, round) matrices
# =====================================================================

def client_metrics(files, n_rounds=None, metrics=None):
    """Client metrics per seed, falling back to the server where missing."""
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
    """Like client_metrics, but picking the files by seed instead of by glob."""
    folder = Path(folder)
    files = [folder / pattern.format(seed=s) for s in seeds]
    return client_metrics([f for f in files if f.exists()], n_rounds)


def centralized_arms(folder, seeds, n_epochs=10):
    """Centralized reference, same keys as client_arms."""
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
    """(seed, round) matrix of a client metric, on the common rounds.

    The true round numbers and their intersection across seeds are kept: the
    client evaluation starts at round 1, so indexing with arange() would
    shift the curve by one round.
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
# Server-side metrics, for the 40-round strategy comparison
# =====================================================================

def server_curves(base_dir, pattern, seeds, max_round, metrics=None):
    """(seed, round) matrices of the server metrics, cut to the common rounds."""
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
    """Like server_curves, on the centralized runs."""
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
    """Server metrics at a given round, one value per seed."""
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
    """Server metrics at the last available round, one value per seed."""
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
    """Centralized metrics at a given epoch, one value per seed."""
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
# CDR recalibrations
# =====================================================================

def _n_maps(round_entry):
    """Recalibrations and checks of one round, summed over the clients."""
    n = checks = 0
    for key in ("cdr_train_per_client", "cdr_per_client"):
        for e in (round_entry.get(key) or []):
            n += e.get("n_maps", 0)
            checks += e.get("n_checks") or 0
    return n, checks


def total_calibrations(files):
    """Total per seed, rebuilt by summing n_maps over rounds and clients.

    info["cdr_total_calibrations"] is NOT used: the strategy writes that
    field only when the count is > 0 (custom_strategy.py). A seed that never
    recalibrates - seed 4 at thresholds from 0.05 upwards - has no such
    field, and reading it directly would drop it from the sample exactly
    where it is lowest, inflating the median. The sum of n_maps coincides
    with the counter when it exists, and is 0 when it does not.
    """
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rounds = json.load(fh)["rounds"]
        out.append(sum(_n_maps(r)[0] for r in rounds))
    return np.array(out, dtype=float)


def calibration_map(files):
    """Per seed, the pair (recalibrations, checks).

    The denominator is the number of times the reliability criterion was
    actually evaluated, not the number of rounds: every client checks
    several times per round (11 in the sweep), so normalising per
    client-round would be off by an order of magnitude. The counter is not
    cumulative on the client side - the mitigator is recreated every round -
    so it is summed.
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
    """(seed, round) matrix of the cumulative recalibrations."""
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
    """Per seed, the pair (recalibrations, checks) of the centralized sweep.

    A dictionary and not an array: the saving is computed by pairing seeds,
    and the two arms are not guaranteed to be complete. Here n_checks is
    already cumulative along the epochs, so the total is the last recorded
    value, not the sum.
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
    """cdr_total_calibrations as declared in the info, one value per seed.

    Used only by the old sweeps, where the field is always present.
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
# Hyperparameter tuning
# =====================================================================

def parse_tuning_name(path, strategies):
    """(strategy, configuration, seed) from a tuning file name."""
    head, _, seed = Path(path).stem.rpartition("_seed")
    for strat in sorted(strategies, key=len, reverse=True):
        if head.startswith(strat + "_"):
            return strat, head[len(strat) + 1:], int(seed)
    return head, "", int(seed)


def load_tuning(folder, seeds, strategies, metrics=None, keep=None):
    """{strategy: {configuration: [run, ...]}} from the server metrics."""
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
    """LaTeX label of a configuration, e.g. eta0.2_etal0.15."""
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
# Weights
# =====================================================================

def load_weights(path):
    """Flat vector of the parameters saved in an .npz."""
    return np.concatenate([np.asarray(v).ravel()
                           for v in np.load(path).values()])


def wrap(d):
    """Angular difference wrapped into (-pi, pi], component by component."""
    return np.arctan2(np.sin(d), np.cos(d))
