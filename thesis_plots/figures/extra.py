"""Figure esplorative, fuori dalla tesi.

Erano cinque script a se' che salvavano PNG nella radice del repository e che
puntavano a cartelle spostate da tempo (THRESH_SWEEP_FEDERATED_OU e' sotto
MEMORY_RESULTS_NEW/ da agosto, quindi giravano a vuoto). Qui i percorsi sono
aggiornati, l'output finisce tutto in thesis_images/extra/ e la resa e' quella
condivisa dal resto delle figure invece dei tre rcParams copiati a mano.

Tre di queste sono superate da figure della tesi:

    loss_thresholds        -> memory.quality_panels (04_thresh_vs_quality)
    threshold_comparison   -> mitigation.static_threshold
    noiseless_noisy_mit_ou -> mitigation.panels (04_fedavg_mit_drift)

restano perche' il taglio e' diverso, non perche' servano al documento.
"""
import json
from pathlib import Path

import numpy as np

from .. import loaders, noise_model, palette, paths
from .. import style as ts

# Loss del riferimento noiseless, usata come linea orizzontale.
REFERENCE_LOSS = 0.35314588745435077

SWEEP = Path(paths.SWEEP_OU)
THRESH_DIR = {0.005: "thresh_0005", 0.015: "thresh_0015", 0.03: "thresh_003",
              0.05: "thresh_005", 0.1: "thresh_01", 0.2: "thresh_02"}
MODE_LABELS = {"no_memory": "No memory", "ema": "EMA (inter-round)"}


def _out(name):
    return f"{paths.ensure(paths.EXTRA)}/{name}"


def _losses(directory, seeds):
    """Loss per seed, dal client con ricaduta sul server."""
    out = []
    for s in seeds:
        fp = Path(directory) / f"fedavg_etal0.3_seed{s}.json"
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        losses = []
        for r in data["rounds"]:
            client = r.get("eval_metrics_client") or {}
            losses.append(client["loss"] if "loss" in client
                          else r["eval_metrics_server"]["loss"])
        out.append(losses)
    return out


def loss_thresholds():
    """Loss client su 40 round, tre soglie a confronto, EMA e senza memoria.

    Il seed 4 e' escluso: e' degenere, non ricalibra mai.
    """
    seeds = [1, 2, 3, 5, 6, 7]
    thresholds = {0.015: "tab:green", 0.1: "tab:blue", 0.2: "tab:red"}

    for mode, mode_label in MODE_LABELS.items():
        fig, ax = ts.single()
        for thresh, color in sorted(thresholds.items()):
            all_losses = _losses(SWEEP / THRESH_DIR[thresh] / mode, seeds)
            if not all_losses:
                continue
            min_len = min(len(l) for l in all_losses)
            mat = np.array([l[:min_len] for l in all_losses])
            rounds = np.arange(min_len)
            med, dev = loaders.median_mad(mat)
            ax.plot(rounds, med, color=color, lw=ts.LINEWIDTH,
                    label=f"threshold = {thresh}")
            ax.fill_between(rounds, med - dev, med + dev, alpha=0.2,
                            color=color)
        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel("Eval loss (client)")
        ax.set_title(f"{mode_label}: eval loss per threshold")
        ax.legend(frameon=False, fontsize=8)
        ts.save(fig, _out(f"loss_thresholds_{mode}"))


def pareto():
    """Costo contro qualita': calibrazioni totali contro loss finale."""
    seeds = palette.SEEDS
    colors = {"no_memory": "tab:red", "ema": "tab:blue"}
    markers = {"no_memory": "o", "ema": "s"}

    fig, ax = ts.single()
    for mode, mode_label in MODE_LABELS.items():
        thresh_vals, median_cals, median_loss = [], [], []
        for thresh in sorted(THRESH_DIR, reverse=True):
            d = SWEEP / THRESH_DIR[thresh] / mode
            if not d.exists():
                continue
            losses = [l[-1] for l in _losses(d, seeds)]
            cals = loaders.total_calibrations(loaders.run_files(d))
            if not losses:
                continue
            thresh_vals.append(thresh)
            median_cals.append(np.median(cals))
            median_loss.append(np.median(losses))

        ax.plot(median_cals, median_loss, marker=markers[mode],
                color=colors[mode], label=mode_label, lw=ts.LINEWIDTH,
                markersize=5, zorder=5)
        for t, x, y in zip(thresh_vals, median_cals, median_loss):
            ax.annotate(f"{t}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7,
                        color=colors[mode])

    ax.set_xlabel("Total calibrations (median over 7 seeds)")
    ax.set_ylabel("Final loss (median over 7 seeds)")
    ax.set_title("Cost-quality tradeoff: calibrations vs final loss")
    ax.legend(frameon=False, fontsize=8)
    ts.save(fig, _out("pareto_cals_vs_loss"))


def threshold_comparison():
    """Soglia CDR 0.015 contro 0.03, FedAvg mitigato IID a p = 0.005."""
    seeds = palette.SEEDS
    arms = [
        (r"$\tau = 0.015$ (default)",
         paths.static_noise(paths.FEDAVG_IID, "mitigated", 0.005), "tab:blue"),
        (r"$\tau = 0.03$", paths.FEDAVG_TH_003, "tab:orange"),
    ]

    fig, ax = ts.single()
    for label, directory, color in arms:
        # Solo i round 1-10: il round 0 non ha valutazione client.
        curves = [l[1:11] for l in _losses(directory, seeds)]
        min_len = min(len(c) for c in curves)
        mat = np.array([c[:min_len] for c in curves])
        rounds = np.arange(1, min_len + 1)
        ax.plot(rounds, mat.mean(axis=0), label=label, color=color,
                lw=ts.LINEWIDTH)
        ax.fill_between(rounds, mat.mean(axis=0) - mat.std(axis=0),
                        mat.mean(axis=0) + mat.std(axis=0), alpha=0.2,
                        color=color)

    ax.axhline(REFERENCE_LOSS, color="green", ls="--", lw=0.8,
               label=f"Noiseless ref = {REFERENCE_LOSS:.4f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss (eval client)")
    ax.set_title("FedAvg mitigated IID: confronto soglia CDR")
    ax.legend(frameon=False, fontsize=8)
    ts.save(fig, _out("confronto_threshold_015_003"))


def noiseless_noisy_mitigated_ou():
    """FedAvg IID: noiseless, noisy sotto deriva OU e mitigato, 10 round."""
    seeds = palette.SEEDS
    arms = [
        ("Noiseless", f"{paths.FEDAVG_IID}/noiseless", "tab:green"),
        ("Noisy (OU drift)", paths.NOISY_OU_10R, "tab:red"),
        ("Mitigated (OU drift)", paths.MITIGATED_OU_10R, "tab:blue"),
    ]

    fig, ax = ts.single()
    for label, directory, color in arms:
        curves = _losses(directory, seeds)
        min_len = min(min(len(l) for l in curves), 11)
        mat = np.array([l[:min_len] for l in curves])
        rounds = np.arange(min_len)
        med, dev = loaders.median_mad(mat)
        ax.plot(rounds, med, color=color, lw=ts.LINEWIDTH, label=label)
        ax.fill_between(rounds, med - dev, med + dev, alpha=0.2, color=color)

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel("Eval loss (client)")
    ax.set_title("FedAvg IID: noiseless vs noisy vs mitigated (OU drift)")
    ax.legend(frameon=False, fontsize=8)
    ts.save(fig, _out("noiseless_noisy_mitigated_ou"))


def drift_random_walk(walk_seed=1, sigma=0.0005, base=0.007, n_rounds=10,
                      n_clients=5):
    """Deriva a random walk, un pannello per client.

    Il processo e' quello di noise_model con theta = 0: senza richiamo verso il
    valore base la varianza cresce coi round invece di restare stazionaria.
    """
    pauli, readout = noise_model.trajectories(
        walk_seed, base=base, scale=0.002, sigma=sigma, theta=0.0,
        n_rounds=n_rounds, n_clients=n_clients)
    rounds = np.arange(n_rounds + 1)

    fig, axes = ts.grid(2, 3, height=4.0)
    axes = axes.ravel()
    axes[-1].set_visible(False)
    for pid in range(n_clients):
        ax = axes[pid]
        ax.plot(rounds, pauli[pid], label="Pauli prob", marker=".", ms=3)
        ax.plot(rounds, readout[pid], label="Readout prob", marker=".", ms=3)
        ax.set_xlabel("Server round")
        ax.set_ylabel("P")
        ax.set_title(f"Client {pid}")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(f"Noise drift (walk_sigma={sigma}, base={base})", fontsize=10)
    ts.save(fig, _out("noise_drift"))


def all_extra():
    loss_thresholds()
    pareto()
    threshold_comparison()
    noiseless_noisy_mitigated_ou()
    drift_random_walk()


FIGURES = {
    "extra.loss_thresholds": loss_thresholds,
    "extra.pareto": pareto,
    "extra.threshold_comparison": threshold_comparison,
    "extra.noiseless_noisy_mitigated_ou": noiseless_noisy_mitigated_ou,
    "extra.drift_random_walk": drift_random_walk,
}
