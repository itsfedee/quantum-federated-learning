"""Comparison between federated strategies.

Two groups:

  comparison  ch. 4.1, the five strategies at 40 rounds against the
              centralized run, IID and non-IID, plus the eta_l sweep of FedAvg
  under_noise ch. 4.2 and appendix, the same strategies under static noise,
              mitigation and drift (random walk and mean-reverting)

The first group reads the server-side metrics of the 40-round runs, the
second the client-side ones of the 10-round campaigns: they are two different
campaigns, not two readings of the same one.
"""
from pathlib import Path

import numpy as np

from .. import draw, loaders, palette, paths
from .. import style as ts

SEEDS = palette.SEEDS
MAX_ROUND = 40
METRICS = palette.METRICS_SERVER

# Configuration chosen by the tuning for each strategy. FedAdam is the only
# one that changes between IID and non-IID.
STRATEGY_FILES = {
    "FedAvg":     "fedavg_etal0.3_seed{seed}.json",
    "FedProx":    "fedprox_mu0.03_etal0.3_seed{seed}.json",
    "FedAdagrad": "fedadagrad_eta0.3_etal0.2_seed{seed}.json",
    "FedAdam":    None,
    "FedYogi":    "fedyogi_eta0.1_etal0.1_seed{seed}.json",
}
ADAM_IID = "fedadam_eta0.2_etal0.15_seed{seed}.json"
ADAM_NON_IID = "fedadam_eta0.1_etal0.1_seed{seed}.json"

TUNING_SEEDS = [1001, 1002, 1003]
ETA_L_VALUES = [0.001, 0.005, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35]
TUNING_COLORS = [
    "#d62728",  # 0.001 - red
    "#ff7f0e",  # 0.005 - orange
    "#e377c2",  # 0.01  - pink
    "#17becf",  # 0.15  - cyan
    "#2ca02c",  # 0.2   - green
    "#1f77b4",  # 0.25  - blue
    "#9467bd",  # 0.3   - purple
    "#8c564b",  # 0.35  - brown
]

STATIC_P005 = "scaled_p0.005_r0.005_s0.002"


# =====================================================================
# 4.1 - comparison at 40 rounds
# =====================================================================

def _as_pct(metric, med, dev):
    """Accuracy and F1 in percentage points, the loss stays as it is."""
    if palette.METRIC_INFO[metric]["pct"]:
        return med * 100, dev * 100
    return med, dev


def _centralized_reference(ax, x, med, dev):
    """Centralized reference: grey dashes behind the other curves."""
    ax.plot(x, med, color="0.3", lw=1.5, ls="--", label="Centralized", zorder=0)
    ax.fill_between(x, med - dev, med + dev, alpha=0.08, color="0.5", zorder=0)


def _comparison_panel(axes, dist, adam_pattern, centr_data):
    sim_dir = Path(paths.STRATEGY_FED) / dist

    for name, pattern in STRATEGY_FILES.items():
        if name == "FedAdam":
            pattern = adam_pattern
        data = loaders.server_curves(sim_dir, pattern, SEEDS, MAX_ROUND)
        if data is None:
            continue
        for ax, m in zip(axes, METRICS):
            rounds, mat = data[m]
            med, dev = loaders.median_mad(mat)
            med, dev = _as_pct(m, med, dev)
            draw.curve(ax, rounds, med, dev, name,
                       palette.STRATEGY_COLORS[name],
                       palette.STRATEGY_MARKERS[name],
                       marker_offset=palette.STRATEGY_OFFSETS[name])

    if centr_data is not None:
        for ax, m in zip(axes, METRICS):
            # The centralized run records two epochs per round: one epoch in
            # two is sampled to bring it onto the round scale.
            _, mat = centr_data[m]
            sampled = mat[:, ::2]
            n_points = min(sampled.shape[1], MAX_ROUND + 1)
            sampled = sampled[:, :n_points]
            med, dev = loaders.median_mad(sampled)
            med, dev = _as_pct(m, med, dev)
            _centralized_reference(ax, np.arange(n_points), med, dev)

    for ax, m in zip(axes, METRICS):
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(palette.METRIC_INFO[m]["ylabel"])
        ax.set_xlim(0, MAX_ROUND)
        ax.tick_params(direction="in")
        if m == "loss":
            ax.set_ylim(bottom=0.28)

    ts.legend_above(axes[0].figure, axes[0], ncol=6)


def comparison():
    """The three figures of 4.1: IID, non-IID and the eta_l sweep."""
    out = Path(paths.ensure(paths.FIG_STRATEGY))
    centr = loaders.centralized_curves(paths.STRATEGY_CEN, SEEDS, 80)

    for dist, adam, name in (("iid", ADAM_IID, "04_strategy_comparison_iid"),
                             ("non_iid", ADAM_NON_IID,
                              "04_strategy_comparison_non_iid")):
        fig, axes = ts.panels(3)
        _comparison_panel(axes, dist, adam, centr)
        ts.save(fig, str(out / name))

    fig, axes = ts.panels(3)
    for i, eta_l in enumerate(ETA_L_VALUES):
        pattern = f"fedavg_etal{eta_l}_seed{{seed}}.json"
        data = loaders.server_curves(paths.STRATEGY_TUNING, pattern,
                                     TUNING_SEEDS, MAX_ROUND)
        if data is None:
            continue
        for ax, m in zip(axes, METRICS):
            rounds, mat = data[m]
            med, dev = loaders.median_mad(mat)
            draw.curve(ax, rounds, med, dev, f"$\\eta_l = {eta_l}$",
                       TUNING_COLORS[i], None)
    for ax, m in zip(axes, METRICS):
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(palette.METRIC_LABELS[m])
        ax.tick_params(direction="in")
        if m == "loss":
            ax.set_ylim(bottom=0.28)
    ts.legend_above(fig, axes[0], ncol=4)
    # The eta_l sweep is not in the document: it sits among the exploratory
    # figures.
    ts.save(fig, f"{paths.ensure(paths.EXTRA)}/04_fedavg_iid_tuning")


# =====================================================================
# 4.2 and appendix - the same strategies under noise
# =====================================================================

def _strategy_arms(folder_of):
    """The five strategy curves, with colour, marker and offset."""
    return [(name,
             loaders.read_federated(folder_of(name.lower())),
             palette.STRATEGY_COLORS[name],
             "-",
             palette.STRATEGY_MARKERS[name],
             palette.STRATEGY_OFFSETS[name])
            for name in palette.STRATEGY_COLORS]


def under_noise():
    """The strategy comparison figures under noise, drift and mitigation."""
    out = Path(paths.ensure(paths.FIG_NOISE))

    centr = loaders.read_centralized(paths.CENTRALIZED)
    noiseless_iid = loaders.read_federated(f"{paths.FEDAVG_IID}/noiseless")
    noiseless_ni = loaders.read_federated(f"{paths.FEDAVG_NON_IID}/noiseless")

    ref_iid = [("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
               ("FedAvg (noiseless)", noiseless_iid, "C0", "--", None, 0)]
    ref_ni = [("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
              ("FedAvg (noiseless)", noiseless_ni, "C0", "--", None, 0)]

    # --- static noise and mitigation, p = 0.005 -------------------------
    def static(base, regime):
        return lambda strat: f"{base}/{strat}/{regime}/{STATIC_P005}"

    # The two figures of the mitigated arm sit in the appendix, with the
    # chapter 6 prefix; the other two in 4.2 like the rest of the section.
    mit_out = Path(paths.ensure(paths.FINAL_06))
    # The random walk variants are not in the document.
    extra = paths.ensure(paths.EXTRA)
    for base, refs, dist in ((paths.IID, ref_iid, "iid"),
                             (paths.NON_IID, ref_ni, "noniid")):
        for regime in ("noisy", "mitigated"):
            dest = (mit_out / f"06_strategy_comparison_mitigated_{dist}_p005"
                    if regime == "mitigated"
                    else out / f"04_strategy_comparison_noisy_{dist}_p005")
            draw.scenario_panels(
                refs + _strategy_arms(static(base, regime)),
                str(dest))

    # --- random walk drift ---------------------------------------------
    for sub, refs, dist in ((f"{paths.DRIFT_RW}/noisy_iid", ref_iid, "iid"),
                            (f"{paths.DRIFT_RW}/noisy_non_iid", ref_ni, "noniid")):
        draw.scenario_panels(
            refs + _strategy_arms(lambda s, sub=sub: f"{sub}/{s}"),
            f"{extra}/04_strategy_comparison_drift_{dist}_p005_RW")

    # --- mean-reverting walk drift --------------------------------------
    # This is the drift in the 4.2 document: the name stays without suffix.
    draw.scenario_panels(
        ref_iid + _strategy_arms(
            lambda s: f"{paths.DRIFT_OU}/iid/{s}/noisy"),
        str(out / "04_strategy_comparison_drift_iid_p005"))
    draw.scenario_panels(
        ref_ni + _strategy_arms(
            lambda s: f"{paths.DRIFT_OU}/non_iid/{s}/noisy"),
        str(out / "04_strategy_comparison_drift_noniid_p005"))

    # --- FedAvg non-IID: noiseless vs static vs drift --------------------
    noisy_ni = loaders.read_federated(
        paths.static_noise(paths.FEDAVG_NON_IID, "noisy", 0.005))
    for drift_folder, suffix, dest in (
            (f"{paths.DRIFT_OU}/non_iid/fedavg/noisy", "", str(out)),
            (f"{paths.DRIFT_RW}/noisy_non_iid/fedavg", "_RW", extra)):
        draw.scenario_panels([
            ("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
            ("Noiseless", noiseless_ni, "C0", "-", "o", 0),
            ("Noisy (static, $p=0.005$)", noisy_ni, palette.NOISY, "-", "s", 1),
            ("Noisy (drift, $p=0.005$)",
             loaders.read_federated(drift_folder), "#67000d", "-", "^", 2),
        ], f"{dest}/04_fedavg_noiseless_vs_noisy_vs_drift_noniid{suffix}")


FIGURES = {
    "strategies.comparison": comparison,
    "strategies.under_noise": under_noise,
}
