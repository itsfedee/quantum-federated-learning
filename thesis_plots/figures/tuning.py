"""Tuning degli iperparametri: una figura per strategia, la griglia 2x2 e FedAvg.

Erano quattro script quasi identici, uno per campagna e uno per taglio della
figura. Qui la campagna e' un profilo e il taglio una funzione.

Nota: le figure della campagna vecchia (tuning_non_iid/, senza NEW_) escono ora
nello stile della tesi come tutte le altre, invece che nel formato 14x3.5 con
cui erano state prodotte la prima volta.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import loaders, palette, paths
from .. import style as ts

SEEDS = [1001, 1002, 1003]
METRICS = palette.METRICS_SERVER
STRATEGIES = list(palette.STRATEGY_NAMES)

# La configurazione scelta dal tuning, evidenziata con linea piu' spessa.
# Cambia fra IID e non-IID solo per FedAdam.
SELECTED = {
    "fedavg": "etal0.3",
    "fedprox": "mu0.03_etal0.3",
    "fedadagrad": "eta0.3_etal0.2",
    "fedyogi": "eta0.1_etal0.1",
}

PROFILES = {
    "iid": {
        "dir": paths.TUNING_IID,
        "prefix": "new_tuning_iid",
        "pdf_name": "04_{strat}_tuning_iid",
        "grid_out": "06_full_tuning_iid",
        "fedadam": "eta0.2_etal0.15",
    },
    "non_iid": {
        "dir": paths.TUNING_NON_IID,
        "prefix": "new_tuning_non_iid",
        "grid_out": "06_tuning_non_iid",
        "fedavg_out": "06_tuning_non_iid_fedavg",
        "fedadam": "eta0.1_etal0.1",
    },
    # Campagna precedente, tenuta per confronto.
    "non_iid_old": {
        "dir": paths.TUNING_NON_IID_OLD,
        "prefix": "tuning_non_iid",
        "fedadam": "eta0.1_etal0.1",
    },
}

# Griglia comune del tuning: dieci combinazioni (primo parametro, eta_l), le
# stesse per tutte le strategie, nell'ordine del sort alfabetico delle config.
CONFIG_GRID = [
    (0.001, 0.01), (0.01, 0.15), (0.03, 0.3), (0.05, 0.1), (0.1, 0.01),
    (0.1, 0.1), (0.1, 0.3), (0.2, 0.05), (0.2, 0.15), (0.3, 0.2),
]
GRID_STRATEGIES = ["fedadagrad", "fedadam", "fedyogi", "fedprox"]


def _selected(profile):
    return dict(SELECTED, fedadam=PROFILES[profile]["fedadam"])


def _out_dir(profile):
    return Path(paths.ensure(f"{PROFILES[profile]['dir']}/plot"))


def _matrix(runs, metric):
    """Matrice (seed, round) di una metrica, sui round comuni ai seed."""
    min_len = min(len(r[metric]) for r in runs)
    return np.array([r[metric][:min_len] for r in runs])


def _style(is_selected):
    """Spessore, opacita' della banda e ordine di disegno della curva."""
    return ((2.5, 0.25, 10) if is_selected else (1.0, 0.10, 1))


def per_strategy(profile="iid"):
    """Una figura a tre pannelli per strategia, una curva per configurazione."""
    cfg = PROFILES[profile]
    out = _out_dir(profile)
    selected = _selected(profile)
    data = loaders.load_tuning(cfg["dir"], SEEDS, STRATEGIES)

    pdf_dir = None
    if cfg.get("pdf_name"):
        # I PDF per la tesi stanno in una sottocartella, col nome del documento.
        pdf_dir = Path(paths.ensure(out / "pdf"))

    for strat, configs in data.items():
        fig, axes = ts.panels(3)
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))

        for i, (config, runs) in enumerate(sorted(configs.items())):
            lw, alpha, zorder = _style(config == selected.get(strat))
            label = loaders.pretty_config(config)
            if config == selected.get(strat):
                label += "  (selected)"
            for ax, m in zip(axes, METRICS):
                mat = _matrix(runs, m)
                rounds = np.arange(mat.shape[1])
                med, dev = loaders.median_mad(mat)
                ax.plot(rounds, med, color=cmap[i], lw=lw, label=label,
                        zorder=zorder)
                ax.fill_between(rounds, med - dev, med + dev, alpha=alpha,
                                color=cmap[i], zorder=zorder)

        for ax, m in zip(axes, METRICS):
            ax.set_xlabel("Round $t$")
            ax.set_ylabel(palette.METRIC_LABELS[m])
            ax.tick_params(direction="in")
            if m == "loss":
                ax.set_ylim(bottom=0.28)

        handles, _ = axes[0].get_legend_handles_labels()
        ts.legend_above(fig, axes[0], ncol=min(len(handles), 5))

        if pdf_dir is not None:
            fig.savefig(pdf_dir / f"{cfg['pdf_name'].format(strat=strat)}.pdf",
                        bbox_inches="tight")
        ts.save(fig, str(out / f"{cfg['prefix']}_{strat}"))
        print(f"Saved {palette.STRATEGY_NAMES.get(strat, strat)}")


def grid(profile="non_iid"):
    """Pannello 2x2 della loss, quattro strategie, legenda unica condivisa."""
    cfg = PROFILES[profile]
    out = _out_dir(profile)
    selected = _selected(profile)
    data = loaders.load_tuning(cfg["dir"], SEEDS, STRATEGIES,
                               keep=GRID_STRATEGIES)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(CONFIG_GRID), 10)))
    fig, axes = ts.grid(2, 2)
    axes = axes.flatten()

    handles, labels = [], []
    for ax_idx, (ax, strat) in enumerate(zip(axes, GRID_STRATEGIES)):
        for i, (config, runs) in enumerate(sorted(data[strat].items())):
            lw, alpha, zorder = _style(config == selected.get(strat))
            mat = _matrix(runs, "loss")
            rounds = np.arange(mat.shape[1])
            med, dev = loaders.median_mad(mat)
            line, = ax.plot(rounds, med, color=cmap[i], lw=lw, zorder=zorder)
            ax.fill_between(rounds, med - dev, med + dev, alpha=alpha,
                            color=cmap[i], zorder=zorder)
            # Le maniglie della legenda si prendono dalla prima strategia: la
            # griglia di configurazioni e' la stessa per tutte e quattro.
            if ax_idx == 0:
                v1, v2 = CONFIG_GRID[i]
                handles.append(line)
                labels.append(f"$({v1},\\ {v2})$")

        ax.set_xlabel(r"Round $t$")
        ax.set_ylabel("Loss")
        ax.set_title(palette.STRATEGY_NAMES[strat])
        ax.set_ylim(bottom=0.28)

    ts.legend_above(fig, None, ncol=5, handles=handles, labels=labels)
    ts.save(fig, str(out / cfg["grid_out"]))


def fedavg(profile="non_iid"):
    """FedAvg: un solo pannello, sweep di eta_l."""
    cfg = PROFILES[profile]
    out = _out_dir(profile)
    selected = _selected(profile)["fedavg"]

    configs = defaultdict(list)
    for fp in sorted(Path(cfg["dir"]).glob("fedavg_*.json")):
        _, config, seed = loaders.parse_tuning_name(fp, ["fedavg"])
        if seed not in SEEDS:
            continue
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)
        configs[config].append([r["eval_metrics_server"]["loss"]
                                for r in d["rounds"]])

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))
    fig, ax = ts.single()

    for i, (config, runs) in enumerate(sorted(configs.items())):
        lw, alpha, zorder = _style(config == selected)
        label = f"$\\eta_l={config[4:]}$"      # config e' sempre etal<valore>
        if config == selected:
            label += "  (selected)"
        min_len = min(len(r) for r in runs)
        mat = np.array([r[:min_len] for r in runs])
        rounds = np.arange(min_len)
        med, dev = loaders.median_mad(mat)
        ax.plot(rounds, med, color=cmap[i], lw=lw, label=label, zorder=zorder)
        ax.fill_between(rounds, med - dev, med + dev, alpha=alpha,
                        color=cmap[i], zorder=zorder)

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0.28)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    # La legenda e' ancorata a mano, fuori dal layout constrained: senza questi
    # due aggiustamenti finisce sopra il bordo superiore della figura.
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    ts.save(fig, str(out / cfg["fedavg_out"]))


def all_tuning():
    """Tutte le figure di tuning, nelle tre campagne."""
    per_strategy("iid")
    per_strategy("non_iid_old")
    grid("iid")
    grid("non_iid")
    fedavg("non_iid")


FIGURES = {
    "tuning.per_strategy_iid": lambda: per_strategy("iid"),
    "tuning.per_strategy_non_iid_old": lambda: per_strategy("non_iid_old"),
    "tuning.grid_iid": lambda: grid("iid"),
    "tuning.grid_non_iid": lambda: grid("non_iid"),
    "tuning.fedavg": lambda: fedavg("non_iid"),
}
