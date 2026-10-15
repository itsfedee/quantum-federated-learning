"""Hyperparameter tuning: one figure per strategy, the 2x2 grid and FedAvg.

These were four nearly identical scripts, one per campaign and one per cut
of the figure. Here the campaign is a profile and the cut a function.

Note: the figures of the old campaign (tuning_non_iid/, without NEW_) now
come out in the thesis style like all the others, instead of the 14x3.5
format they were first produced in.
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

# The configuration chosen by the tuning, highlighted with a thicker line.
# It changes between IID and non-IID only for FedAdam.
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
    # Previous campaign, kept for comparison.
    "non_iid_old": {
        "dir": paths.TUNING_NON_IID_OLD,
        "prefix": "tuning_non_iid",
        "fedadam": "eta0.1_etal0.1",
    },
}

# Common tuning grid: ten (first parameter, eta_l) combinations, the same for
# every strategy, in the alphabetical sort order of the configs.
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
    """(seed, round) matrix of a metric, on the rounds common to the seeds."""
    min_len = min(len(r[metric]) for r in runs)
    return np.array([r[metric][:min_len] for r in runs])


def _style(is_selected):
    """Line width, band opacity and drawing order of the curve."""
    return ((2.5, 0.25, 10) if is_selected else (1.0, 0.10, 1))


def per_strategy(profile="iid"):
    """A three-panel figure per strategy, one curve per configuration."""
    cfg = PROFILES[profile]
    out = _out_dir(profile)
    selected = _selected(profile)
    data = loaders.load_tuning(cfg["dir"], SEEDS, STRATEGIES)

    pdf_dir = None
    if cfg.get("pdf_name"):
        # The PDFs for the thesis go to the 4.1 folder, with the document's
        # name; the working pair stays in the campaign folder.
        pdf_dir = Path(paths.ensure(paths.FINAL_041))

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
    """2x2 panel of the loss, four strategies, single shared legend."""
    cfg = PROFILES[profile]
    # The grids and the FedAvg sweep are appendix figures.
    out = Path(paths.ensure(paths.FINAL_06))
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
            # The legend handles are taken from the first strategy: the grid
            # of configurations is the same for all four.
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
    """FedAvg: a single panel, eta_l sweep."""
    cfg = PROFILES[profile]
    out = Path(paths.ensure(paths.FINAL_06))
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
        label = f"$\\eta_l={config[4:]}$"      # config is always etal<value>
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
    # The legend is anchored by hand, outside the constrained layout: without
    # these two adjustments it ends up above the top edge of the figure.
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    ts.save(fig, str(out / cfg["fedavg_out"]))


def all_tuning():
    """Every tuning figure, in the three campaigns."""
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
