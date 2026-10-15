"""CDR threshold sweep and memory of the map.

The sweep exists in two campaigns, which differ by the process that moves
the noise: Ornstein-Uhlenbeck at 40 rounds and random walk at 10. What
changes are the available thresholds, the third memory variant (ema_full
against warm_start), the tail window for the plateau statistics and the
marker spacing. Everything else is identical, so the regime is a parameter
and not a copy of the script.

The figures of the OU sweep go to thesis_images/4_3_FINAL/ and its tables to
thesis_images/tables/; the random walk campaign, which stays out of the
thesis, keeps both in NEW_THESIS_IMAGES_RW/.
"""
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import draw, loaders, noise_model, palette, paths
from .. import style as ts

SEEDS = palette.SEEDS
NOISE_LEVELS = [0.003, 0.006, 0.01, 0.014, 0.018, 0.022]


@dataclass
class Regime:
    """Profile of a sweep campaign."""
    name: str
    sweep: str
    thresholds: list
    thresh_dir: dict
    variants: list
    labels: dict
    quality_thresholds: tuple
    tail: int           # tail rounds on which quality and stability are read
    mark_every: int
    duration: int       # rounds of the campaign, for the rates in the table
    out: str            # folder of the figures in the document
    extra: str = None   # folder of the diagnostic cuts; None: the same
    tables: str = None  # folder of the .tex/.csv; None: the same as the figures
    seeds: list = None  # None: no filter, the folders already hold 7 seeds
    centralized: bool = False   # the centralized block exists in the table


REGIMES = {
    "OU": Regime(
        name="OU",
        sweep=paths.SWEEP_OU,
        thresholds=[0.005, 0.015, 0.03, 0.05, 0.1, 0.2],
        thresh_dir={0.005: "thresh_0005", 0.015: "thresh_0015",
                    0.03: "thresh_003", 0.05: "thresh_005",
                    0.1: "thresh_01", 0.2: "thresh_02"},
        variants=["no_memory", "ema", "ema_full"],
        labels={"no_memory": "No memory", "ema": "EMA inter-round",
                "ema_full": "EMA full"},
        quality_thresholds=(0.015, 0.05, 0.1, 0.2),
        tail=10,
        mark_every=8,
        duration=40,
        out=paths.MEMORY_OUT_OU,
        extra=paths.EXTRA,
        tables=paths.MEMORY_TABLES_OU,
        centralized=True,
    ),
    "RW": Regime(
        name="RW",
        sweep=paths.SWEEP_RW,
        # 0.0 is excluded: a zero threshold means recalibrating at every check.
        thresholds=[0.005, 0.015, 0.03, 0.05, 0.2],
        thresh_dir={0.005: "thresh_0005", 0.015: "thresh_0015",
                    0.03: "thresh_003", 0.05: "thresh_005",
                    0.2: "thresh_02"},
        variants=["no_memory", "ema", "warm_start"],
        labels={"no_memory": "No memory", "ema": "EMA inter-round",
                "warm_start": "Warm start"},
        quality_thresholds=(0.015, 0.03, 0.05, 0.2),
        tail=5,             # over 10 rounds the tail cannot be 10
        mark_every=2,
        duration=10,
        out=paths.MEMORY_OUT_RW,
        extra=paths.MEMORY_OUT_RW,
        tables=paths.MEMORY_TABLES_RW,
        seeds=list(range(1, 8)),
    ),
}

# Centralized threshold sweep with memory, under the same OU drift as the
# federated one (sigma = 0.001, theta = 0.3, beta = 0.2) but over 30 epochs
# instead of 40 rounds: it is an earlier campaign. Thresholds 0.03 and 0.05
# also appear in memory_centralized_ou_sweep with exactly the same seven
# seeds, and 0.015 is duplicated in memory_centralized/{ema,no_memory}
# without a subfolder: here a single copy of each is read, and from the
# ou_sweep only 0.08 and 0.12, which live only there.
CEN_MEMORY_DIR = {
    0.005: f"{paths.CEN_MEMORY}/thresh_0005",
    0.015: f"{paths.CEN_MEMORY}/thresh_0015",
    0.03: f"{paths.CEN_MEMORY}/thresh_003",
    0.05: f"{paths.CEN_MEMORY}/thresh_005",
    0.08: f"{paths.CEN_MEMORY_OU_SWEEP}/thresh_008",
    0.12: f"{paths.CEN_MEMORY_OU_SWEEP}/thresh_012",
    0.2: f"{paths.CEN_MEMORY}/thresh_02",
}
CEN_DURATION = 30


def sweep_files(reg, threshold, variant):
    """Sweep JSONs for (threshold, variant), filtered on the profile's seeds."""
    files = loaders.run_files(f"{reg.sweep}/{reg.thresh_dir[threshold]}/{variant}")
    if reg.seeds is None:
        return files
    keep = set(reg.seeds)
    return [f for f in files if loaders.seed_of(f) in keep]


# The three cuts of the sweep that enter 4.3: everything else this module
# produces is diagnostic and goes among the exploratory figures.
IN_THESIS = {
    "04_threshold_sweep",
    "04_calibration_savings_vs_round",
    "04_weight_distance",
}


def _out(reg, name):
    base = reg.out if name in IN_THESIS else (reg.extra or reg.out)
    return f"{paths.ensure(base)}/{name}"


# =====================================================================
# Cost: recalibrations against threshold and against round
# =====================================================================

def threshold_sweep(reg, logscale=True):
    """Total recalibrations against threshold, one curve per memory variant.

    Categorical x axis: the thresholds are equally spaced regardless of their
    value, as in the figures already in the thesis. On a numeric axis the
    first three would pile up, and a logarithm would make the spacing hard
    to read. The y axis is logarithmic: the counts span two orders of
    magnitude and on a linear scale the loose thresholds get squashed onto
    zero, which is where the difference between the memory variants lies.
    The MAD band is clipped at 0.5 so that the lower edge does not drop to
    values that cannot be represented.
    """
    fig, ax = ts.single()
    x = np.arange(len(reg.thresholds))
    for v in reg.variants:
        totals = [loaders.total_calibrations(sweep_files(reg, t, v))
                  for t in reg.thresholds]
        med = np.array([np.median(c) for c in totals])
        dev = np.array([loaders.mad(c) for c in totals])
        ax.plot(x, med, color=palette.VAR_COLOR[v], lw=ts.LINEWIDTH,
                label=reg.labels[v], marker=palette.VAR_MARKER[v], ms=5)
        draw.band(ax, x, med, dev, palette.VAR_COLOR[v],
                  floor=0.5 if logscale else 0.0)
    if logscale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in reg.thresholds])
    ax.set_xlabel(r"CDR threshold $\epsilon_{\mathrm{th}}$")
    ax.set_ylabel("Total calibrations")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ts.save(fig, _out(reg, "04_threshold_sweep_median_mad"))


def threshold_sweep_inset(reg, zoom_from=0.05):
    """Same figure on a linear y axis, with an inset on the loose thresholds.

    This is the version in the 4.3 document (checked against the delivered
    PDF): the main axis stays linear and the informative part (thresholds >=
    0.05, where the memory halves the calibrations) is read in the box. The
    upper limit of the inset follows the highest MAD band of the region, not
    a fixed value: at 0.05 no_memory reaches 227 in the OU regime and a ylim
    of 200 would cut it.
    """
    fig, ax = ts.single()
    x = np.arange(len(reg.thresholds))
    zoom = [i for i, t in enumerate(reg.thresholds) if t >= zoom_from]
    axins = ax.inset_axes([0.50, 0.45, 0.47, 0.50])
    top = 0.0
    for v in reg.variants:
        totals = [loaders.total_calibrations(sweep_files(reg, t, v))
                  for t in reg.thresholds]
        med = np.array([np.median(c) for c in totals])
        dev = np.array([loaders.mad(c) for c in totals])
        color = palette.VAR_COLOR[v]
        marker = palette.VAR_MARKER[v]
        ax.plot(x, med, color=color, lw=ts.LINEWIDTH, label=reg.labels[v],
                marker=marker, ms=5)
        draw.band(ax, x, med, dev, color)
        axins.plot(x, med, color=color, lw=ts.LINEWIDTH, marker=marker, ms=4)
        draw.band(axins, x, med, dev, color)
        top = max(top, float((med + dev)[zoom].max()))

    axins.set_xlim(zoom[0] - 0.3, zoom[-1] + 0.3)
    axins.set_ylim(0, 1.05 * top)
    axins.set_xticks(zoom)
    axins.set_xticklabels([str(reg.thresholds[i]) for i in zoom])
    axins.tick_params(labelsize=7, direction="in")
    ax.indicate_inset_zoom(axins, edgecolor="0.5", lw=0.6, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in reg.thresholds])
    ax.set_xlabel(r"CDR threshold $\epsilon_{\mathrm{th}}$")
    ax.set_ylabel("Total calibrations")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ts.save(fig, _out(reg, "04_threshold_sweep"))


def cumulative(reg):
    """Cumulative recalibrations against round, one curve per threshold (EMA)."""
    fig, ax = ts.single()
    colors = plt.cm.viridis(np.linspace(0, .88, len(reg.thresholds)))
    for i, (t, c) in enumerate(zip(reg.thresholds, colors)):
        rounds, M = loaders.cumulative_calibrations(sweep_files(reg, t, "ema"))
        if not len(M):
            continue
        med = np.median(M, axis=0)
        dev = np.array([loaders.mad(M[:, j]) for j in range(M.shape[1])])
        ax.plot(rounds, med, color=c, lw=ts.LINEWIDTH,
                marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                ms=3.2, markevery=(i, reg.mark_every),
                label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
        draw.band(ax, rounds, med, dev, c, floor=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative calibrations")
    ax.set_xlim(1, None)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ts.save(fig, _out(reg, "04_cumulative_calibrations"))


def savings(reg):
    """Calibrations saved by the memory, cumulative, per threshold.

    The difference no_memory - EMA paired by seed is plotted, not the two
    totals: being a difference it lies in the same range for every
    threshold, while the totals span three orders of magnitude and squash
    the loose thresholds. The y axis is thus directly the benefit of the
    memory.
    """
    fig, ax = ts.single()
    for i, t in enumerate(reg.thresholds):
        _, A = loaders.cumulative_calibrations(sweep_files(reg, t, "no_memory"))
        rounds, B = loaders.cumulative_calibrations(sweep_files(reg, t, "ema"))
        if not len(A) or not len(B):
            continue
        n = min(A.shape[0], B.shape[0])
        m = min(A.shape[1], B.shape[1])
        D = A[:n, :m] - B[:n, :m]          # paired seed by seed
        med = np.median(D, axis=0)
        dev = np.array([loaders.mad(D[:, j]) for j in range(m)])
        c = f"C{i}"
        ax.plot(rounds[:m], med, color=c, lw=ts.LINEWIDTH,
                marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                ms=3.2, markevery=(i, reg.mark_every),
                label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
        draw.band(ax, rounds[:m], med, dev, c)
    ax.axhline(0, color="0.6", lw=.6, zorder=0)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Saved calibrations")
    ax.set_xlim(1, None)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ts.save(fig, _out(reg, "04_calibration_savings_vs_round"))


# =====================================================================
# Quality: client metrics against round, one curve per threshold
# =====================================================================

def quality_panels(reg, variant="ema"):
    """Client-side loss, accuracy and F1 against the round, one curve per threshold.

    Three panels side by side, same round scale but independent ordinates:
    loss and classification metrics live on different ranges.
    """
    keys = [("loss", "Client evaluation loss"),
            ("accuracy", "Client accuracy"),
            ("f1", "Client $F_1$")]
    fig, axes = ts.panels(3, sharex=True)
    for ax, (key, ylabel) in zip(axes, keys):
        for i, t in enumerate(reg.quality_thresholds):
            rounds, M = loaders.client_curves(sweep_files(reg, t, variant), key)
            if not len(M):
                continue
            med = np.median(M, axis=0)
            dev = np.array([loaders.mad(M[:, j]) for j in range(M.shape[1])])
            c = f"C{i}"
            ax.plot(rounds, med, color=c, lw=ts.LINEWIDTH,
                    marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                    ms=3.2, markevery=(i, reg.mark_every),
                    label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
            draw.band(ax, rounds, med, dev, c)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xlabel("Communication round $t$", fontsize=8.5)
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 10, 20, 30, 40])
        # Few ticks and small labels: in a panel barely wider than two inches
        # the default spacing bleeds into the next one.
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=8)
    # Single legend above the panels: inside a narrow panel four entries
    # would cover the curves, and repeating it three times would be
    # redundant.
    fig.legend(handles=axes[0].get_lines(), frameon=False, fontsize=8.5,
               ncol=4, loc="outside upper center", columnspacing=1.6,
               handlelength=1.8)
    ts.save(fig, _out(reg, "04_thresh_vs_quality"))


def quality_stats(reg, threshold, variant):
    """Cost, quality and stability of one sweep point, one value per seed.

    The two stability measures are not redundant: sigma is the typical
    spread around the plateau, spike the tail, that is the worst regression
    observed in the window with respect to the best loss reached up to that
    round since the start of the run - how much the single episode in which
    the CDR map stays stale for too long costs. The running minimum
    accumulates over the whole trajectory, but the maximum is read only in
    the window, like everything else in the table.
    """
    files = sweep_files(reg, threshold, variant)
    _, L = loaders.client_curves(files, "loss")
    _, A = loaders.client_curves(files, "accuracy")
    _, F = loaders.client_curves(files, "f1")
    regression = L - np.minimum.accumulate(L, axis=1)
    tail = reg.tail
    return {
        "cal": loaders.total_calibrations(files),
        "loss": np.median(L[:, -tail:], axis=1),
        "acc": np.median(A[:, -tail:], axis=1),
        "f1": np.median(F[:, -tail:], axis=1),
        "sd": L[:, -tail:].std(axis=1, ddof=1),
        "spike": regression[:, -tail:].max(axis=1),
    }


# =====================================================================
# Weight distance and noise drift (OU only)
# =====================================================================

# Series of the distance figure: pipeline, regime, label, colour, marker.
WEIGHT_SERIES = [
    ("fed", "noisy", "Federated noisy", "C0", "o"),
    ("fed", "mitigated", "Federated mitigated", "C1", "s"),
    ("cen", "noisy", "Centralized noisy", "C4", "D"),
    ("cen", "mitigated", "Centralized mitigated", "C2", "^"),
]


def parameter_paths(kind, regime, p, seed):
    """Reference (noiseless) and run weights, for one configuration."""
    if kind == "fed":
        return (f"{paths.FEDAVG_IID}/noiseless/weights/"
                f"FedAvg_etal0.3_seed{seed}.npz",
                f"{paths.static_noise(paths.FEDAVG_IID, regime, p)}/weights/"
                f"FedAvg_etal0.3_seed{seed}.npz")
    return (f"{paths.CENTRALIZED}/weights/lr_0.3_seed{seed}.npz",
            f"{paths.CENTRALIZED}/{regime}/p{p}/nshots_1000/weights/"
            f"lr_0.3_seed{seed}.npz")


def weight_distances(kind, regime, p):
    """Distances from the noiseless solution, one value per available seed."""
    import os
    out = []
    for seed in SEEDS:
        ref, run = parameter_paths(kind, regime, p, seed)
        if os.path.exists(ref) and os.path.exists(run):
            out.append(np.linalg.norm(
                loaders.wrap(loaders.load_weights(ref)
                             - loaders.load_weights(run))))
    return out


def weight_distance(reg=None):
    """Weight distance from the noiseless solution, four pipelines.

    Same quantity as the first column of tab_parameters: per-parameter
    difference wrapped on the torus and aggregated in Euclidean norm, median
    over the seven seeds with a MAD band. Categorical x axis because the
    noise levels are not equally spaced (0.003 is 0.003 away from the next,
    the others 0.004).
    """
    reg = reg or REGIMES["OU"]
    fig, ax = ts.single()
    x = np.arange(len(NOISE_LEVELS))
    for kind, regime, label, color, marker in WEIGHT_SERIES:
        med, dev = [], []
        for p in NOISE_LEVELS:
            d = weight_distances(kind, regime, p)
            med.append(np.median(d) if d else np.nan)
            dev.append(loaders.mad(d) if d else np.nan)
        med, dev = np.array(med), np.array(dev)
        ax.plot(x, med, color=color, lw=ts.LINEWIDTH, marker=marker, ms=5,
                label=label)
        draw.band(ax, x, med, dev, color, floor=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in NOISE_LEVELS])
    ax.set_xlabel(r"Noise level $p$")
    ax.set_ylabel(r"Weight distance $\|\boldsymbol{\theta} - "
                  r"\boldsymbol{\theta}_{\mathrm{noiseless}}\|$")
    ax.set_ylim(bottom=0)
    # Inner legend: the curves rise from left to right and the top-left
    # corner stays free even with the widest band.
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ts.save(fig, _out(reg, "04_weight_distance"))


# Parameters of the federated OU campaign, read from the run info.
DRIFT = dict(base=0.005, scale=0.002, sigma=0.001, theta=0.3)


def noise_drift(reg=None, seed=1, n_rounds=10, n_clients=5):
    """OU drift of the noise, one curve per client, Pauli and readout side by side."""
    reg = reg or REGIMES["OU"]
    pauli, readout = noise_model.trajectories(
        noise_model.WALK_SEED_MAP[seed], n_rounds=n_rounds,
        n_clients=n_clients, **DRIFT)
    rounds = np.arange(n_rounds + 1)

    fig, axes = ts.panels(2, sharex=True, sharey=True)
    for ax, data, label in ((axes[0], pauli, r"Pauli $p(t)$"),
                            (axes[1], readout, r"Readout $r(t)$")):
        for pid in range(n_clients):
            ax.plot(rounds, data[pid], color=f"C{pid}", lw=1.3,
                    marker=palette.LINE_MARKERS[pid % len(palette.LINE_MARKERS)],
                    ms=3.5, label=f"Client {pid}")
        # The base value is the centre the OU process pulls the drift
        # towards, not the clients' starting point: the static spread already
        # scatters them at round 0.
        ax.axhline(DRIFT["base"], color="0.6", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_xlim(0, n_rounds)
        ax.tick_params(labelsize=8)
    fig.legend(handles=axes[0].get_lines()[:n_clients], frameon=False,
               fontsize=8.5, ncol=n_clients, loc="outside upper center",
               columnspacing=1.6, handlelength=1.8)
    ts.save(fig, _out(reg, "04_noise_drift_clients"))


# =====================================================================
# Strategies under OU drift, one figure per strategy
# =====================================================================

REGIME_ARMS = (("noiseless", "Noiseless", "tab:blue", "s"),
               ("noisy", "Noisy ($p=0.005$)", "tab:red", "o"),
               ("mitigated", "Mitigated ($p=0.005$)", "tab:green", "^"))


def strategy_sources(reg, strat, regime):
    """JSON folder for (strategy, regime) under OU drift.

    Every strategy has the same shape. Of the mitigated FedAvg arm the first
    10 rounds are used: it is the 0.015 threshold of the sweep, which runs
    for 40.
    """
    if regime == "noiseless":
        return f"{paths.IID}/{strat}/noiseless"
    return f"{paths.DRIFT_OU}/iid/{strat}/{regime}"


def strategy_drift(reg=None):
    """A 1x3 panel per strategy: noiseless, noisy and mitigated under drift.

    Legend anchored by hand instead of by the constrained layout, and PNG at
    200 dpi: it is the rendering this series was first produced with, kept
    so as not to change files that already sit in the campaign folder.
    """
    reg = reg or REGIMES["OU"]
    outdir = Path(paths.ensure(f"{reg.extra or reg.out}/strategies_drift_plots"))
    for strat in palette.STRATEGY_NAMES:
        arms = {label: (loaders.client_metrics(
                            loaders.run_files(strategy_sources(reg, strat, regime)),
                            n_rounds=11), color, marker)
                for regime, label, color, marker in REGIME_ARMS}
        fig, axes = ts.panels(3)
        for arm_idx, (label, (data, color, marker)) in enumerate(arms.items()):
            mevery = (palette.MARKER_OFFSETS[arm_idx % len(palette.MARKER_OFFSETS)], 3)
            for ax, m in zip(axes, palette.METRICS_CLIENT):
                if not data[m]:
                    continue
                n = min(len(v) for v in data[m])
                mat = np.array([v[:n] for v in data[m]])
                if m == "accuracy":
                    mat = mat * 100
                med, dev = loaders.median_mad(mat)
                ax.plot(np.arange(n), med, color=color, ls="-", lw=1.8,
                        label=label, marker=marker, markersize=ts.MARKERSIZE,
                        markevery=mevery)
                ax.fill_between(np.arange(n), med - dev, med + dev,
                                alpha=0.15, color=color)
        for ax, m in zip(axes, palette.METRICS_CLIENT):
            ax.set_xlabel("Communication round $t$")
            ax.set_ylabel(palette.METRIC_LABELS_PCT[m])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8,
                   ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        out = outdir / f"{strat}_ou_drift"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  strategies_drift_plots/{strat}_ou_drift.pdf / .png")


# =====================================================================
# CDR sweep of the earlier campaigns (folders at the same level as the repo)
# =====================================================================

CDR_MODES = ["no_memory", "ema", "warm_start"]
CDR_LABELS = {"no_memory": "No memory", "ema": "EMA", "warm_start": "Warm start"}
CDR_THRESHOLDS = [0.005, 0.015, 0.03, 0.05, 0.2]
CDR_THRESH_DIR = {0.005: "thresh_0005", 0.015: "thresh_0015",
                  0.03: "thresh_003", 0.05: "thresh_005", 0.2: "thresh_02"}
CDR_NOISE_LEVELS = [0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]


# These four figures were born before the shared style, with only the three
# rcParams below on top of matplotlib's defaults. Reproducing them inside
# rc_context keeps them identical to the PDFs already included in the
# document.
CDR_RC = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.labelsize": "medium", "axes.titlesize": "large",
    "xtick.labelsize": "medium", "ytick.labelsize": "medium",
    "legend.fontsize": "medium", "lines.linewidth": 1.5,
    "lines.markersize": 6.0, "axes.grid": False,
}


def _mean_std(values):
    return np.mean(values), np.std(values)


def _median_mad(values):
    return np.median(values), loaders.mad(values)


def _cdr_curve(ax, x_values, base_dir, x_map, stat):
    for mode in CDR_MODES:
        centers, spreads, valid = [], [], []
        for xv in x_values:
            d = Path(base_dir) / x_map[xv] / mode
            if not d.exists():
                continue
            vals = loaders.declared_calibrations(d, SEEDS)
            if len(vals) == 0:
                continue
            c, s = stat(vals)
            centers.append(c)
            spreads.append(s)
            valid.append(xv)
        x, mu, sigma = np.array(valid), np.array(centers), np.array(spreads)
        ax.plot(x, mu, color=palette.VAR_COLOR[mode], lw=1.2,
                label=CDR_LABELS[mode], marker=palette.VAR_MARKER[mode], ms=5)
        ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.15,
                        color=palette.VAR_COLOR[mode])


def cdr_sweep():
    """Recalibrations against threshold and against noise, pre-OU campaigns.

    Two statistics for each sweep: mean with standard deviation and median
    with MAD. The first is the historical one, the second the one used by
    the rest of the chapter.
    """
    out = Path(paths.ensure(paths.CAP04))
    stats = [(_mean_std, "mean_std"), (_median_mad, "median_mad")]

    with plt.rc_context(CDR_RC):
        _cdr_sweep_figures(out, stats)


def _cdr_sweep_figures(out, stats):
    for stat, suffix in stats:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        _cdr_curve(ax, CDR_THRESHOLDS, paths.THRESH_SWEEP_VARWALK,
                   CDR_THRESH_DIR, stat)
        ax.set_xlabel(r"CDR threshold $\tau$")
        ax.set_ylabel("Total calibrations")
        ax.set_xscale("log")
        ax.set_xticks(CDR_THRESHOLDS)
        ax.set_xticklabels([str(t) for t in CDR_THRESHOLDS], fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(bottom=-5)
        _cdr_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(out / f"04_threshold_sweep_{suffix}.pdf", bbox_inches="tight")
        fig.savefig(out / f"threshold_sweep_{suffix}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved threshold sweep ({suffix})")

    noise_dir = {p: f"p{str(p).replace('.', '')}" for p in CDR_NOISE_LEVELS}
    for stat, suffix in stats:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        _cdr_curve(ax, CDR_NOISE_LEVELS, paths.NOISE_SWEEP_TH015, noise_dir, stat)
        ax.set_xlabel(r"Base noise level $p$")
        ax.set_ylabel("Total calibrations")
        _cdr_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(out / f"04_noise_sweep_{suffix}.pdf", bbox_inches="tight")
        fig.savefig(out / f"noise_sweep_{suffix}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved noise sweep ({suffix})")


def _cdr_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.15))


# =====================================================================

def all_figures(regime="OU"):
    """The figures of one campaign, in the order they used to be produced."""
    reg = REGIMES[regime]
    print(f">>> Regime {reg.name}, output in {reg.out}/")
    # The inset version is the one in the document, the plain one stays for
    # comparison.
    threshold_sweep_inset(reg)
    threshold_sweep(reg)
    savings(reg)
    cumulative(reg)
    quality_panels(reg)
    if reg.name == "OU":
        # Weight distances and noise trajectories do not depend on the memory
        # sweep: they are produced once, in the OU profile.
        weight_distance(reg)
        noise_drift(reg)


FIGURES = {
    "memory.ou": lambda: all_figures("OU"),
    "memory.thresh_inset": lambda: [threshold_sweep_inset(REGIMES[r])
                                    for r in ("OU", "RW")],
    "memory.rw": lambda: all_figures("RW"),
    "memory.strategy_drift": strategy_drift,
    "memory.cdr_sweep": cdr_sweep,
}
