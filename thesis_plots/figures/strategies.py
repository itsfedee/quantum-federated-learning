"""Confronto fra strategie federate.

Due gruppi:

  comparison  cap. 4.1, le cinque strategie a 40 round contro il centralizzato,
              IID e non-IID, piu' lo sweep di eta_l di FedAvg
  under_noise cap. 4.2 e appendice, le stesse strategie sotto rumore statico,
              mitigazione e deriva (random walk e Ornstein-Uhlenbeck)

Il primo gruppo legge le metriche lato server delle run a 40 round, il secondo
quelle lato client delle campagne a 10 round: sono due campagne diverse, non
due letture della stessa.
"""
from pathlib import Path

import numpy as np

from .. import draw, loaders, palette, paths
from .. import style as ts

SEEDS = palette.SEEDS
MAX_ROUND = 40
METRICS = palette.METRICS_SERVER

# Configurazione scelta dal tuning per ciascuna strategia. FedAdam e' l'unica
# che cambia fra IID e non-IID.
STRATEGY_FILES = {
    "FedAvg":     "fedavg_etal0.3_seed{seed}.json",
    "FedProx":    "fedprox_mu0.03_etal0.3_seed{seed}.json",
    "FedAdagrad": "fedadagrad_eta0.3_etal0.2_seed{seed}.json",
    "FedAdam":    None,
    "FedYogi":    "fedyogi_eta0.1_etal0.1_seed{seed}.json",
}
ADAM_IID = "fedadam_eta0.2_etal0.15_seed{seed}.json"
ADAM_NON_IID = "fedadam_eta0.1_etal0.1_seed{seed}.json"

TUNING_SEEDS = [14, 71, 130]
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
# 4.1 - confronto a 40 round
# =====================================================================

def _centralized_reference(ax, x, med, dev):
    """Riferimento centralizzato: tratteggio grigio dietro alle altre curve."""
    ax.plot(x, med, color="0.3", lw=1.5, ls="--", label="Centralized", zorder=0)
    ax.fill_between(x, med - dev, med + dev, alpha=0.08, color="0.5", zorder=0)


def _comparison_panel(axes, dist, adam_pattern, centr_data):
    sim_dir = Path(paths.STRATEGY_FED) / dist / "simulations" / "simulation_experiments"

    for name, pattern in STRATEGY_FILES.items():
        if name == "FedAdam":
            pattern = adam_pattern
        data = loaders.server_curves(sim_dir, pattern, SEEDS, MAX_ROUND)
        if data is None:
            continue
        for ax, m in zip(axes, METRICS):
            rounds, mat = data[m]
            med, dev = loaders.median_mad(mat)
            draw.curve(ax, rounds, med, dev, name,
                       palette.STRATEGY_COLORS[name],
                       palette.STRATEGY_MARKERS[name],
                       marker_offset=palette.STRATEGY_OFFSETS[name])

    if centr_data is not None:
        for ax, m in zip(axes, METRICS):
            # Il centralizzato registra due epoche per round: si campiona una
            # epoca su due per riportarlo sulla scala dei round.
            _, mat = centr_data[m]
            sampled = mat[:, ::2]
            n_points = min(sampled.shape[1], MAX_ROUND + 1)
            sampled = sampled[:, :n_points]
            med, dev = loaders.median_mad(sampled)
            _centralized_reference(ax, np.arange(n_points), med, dev)

    for ax, m in zip(axes, METRICS):
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(palette.METRIC_LABELS[m])
        ax.set_xlim(0, MAX_ROUND)
        ax.tick_params(direction="in")
        if m == "loss":
            ax.set_ylim(bottom=0.28)

    ts.legend_above(axes[0].figure, axes[0], ncol=6)


def comparison():
    """Le tre figure di 4.1: IID, non-IID e sweep di eta_l."""
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
    ts.save(fig, str(out / "04_fedavg_iid_tuning"))


# =====================================================================
# 4.2 e appendice - le stesse strategie sotto rumore
# =====================================================================

def _strategy_arms(folder_of):
    """Le cinque curve di strategia, con colore, marcatore e sfasamento."""
    return [(name,
             loaders.read_federated(folder_of(name.lower())),
             palette.STRATEGY_COLORS[name],
             "-",
             palette.STRATEGY_MARKERS[name],
             palette.STRATEGY_OFFSETS[name])
            for name in palette.STRATEGY_COLORS]


def under_noise():
    """Le figure di confronto fra strategie sotto rumore, deriva e mitigazione."""
    out = Path(paths.ensure(paths.FIG_NOISE))

    centr = loaders.read_centralized(paths.CENTRALIZED)
    noiseless_iid = loaders.read_federated(f"{paths.FEDAVG_IID}/noiseless")
    noiseless_ni = loaders.read_federated(f"{paths.FEDAVG_NON_IID}/noiseless")

    ref_iid = [("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
               ("FedAvg (noiseless)", noiseless_iid, "C0", "--", None, 0)]
    ref_ni = [("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
              ("FedAvg (noiseless)", noiseless_ni, "C0", "--", None, 0)]

    # --- rumore statico e mitigazione, p = 0.005 -----------------------
    # Sotto non-IID FedAvg tiene i suoi file in noisy/p0.005 e
    # mitigated/p0.005, le altre strategie in scaled_p0.005_r0.005_s0.002.
    def static(base, regime, fedavg_sub=None):
        def folder(strat):
            if strat == "fedavg" and fedavg_sub:
                return f"{base}/fedavg/{regime}/{fedavg_sub}"
            return f"{base}/{strat}/{regime}/{STATIC_P005}"
        return folder

    for base, refs, dist, fedavg_sub in (
            (paths.IID, ref_iid, "iid", None),
            (paths.NON_IID, ref_ni, "noniid", "p0.005")):
        for regime in ("noisy", "mitigated"):
            draw.scenario_panels(
                refs + _strategy_arms(static(base, regime, fedavg_sub)),
                str(out / f"04_strategy_comparison_{regime}_{dist}_p005"))

    # --- deriva random walk --------------------------------------------
    for sub, refs, dist in ((f"{paths.DRIFT_RW}/noisy_iid", ref_iid, "iid"),
                            (f"{paths.DRIFT_RW}/noisy_non_iid", ref_ni, "noniid")):
        draw.scenario_panels(
            refs + _strategy_arms(lambda s, sub=sub: f"{sub}/{s}"),
            str(out / f"04_strategy_comparison_drift_{dist}_p005"))

    # --- deriva Ornstein-Uhlenbeck --------------------------------------
    # FedAvg IID non sta in strategies_OU_drift_10r: le sue run OU a 10 round
    # sono quelle del braccio noisy dello sweep sulla memoria.
    def ou_iid(strat):
        if strat == "fedavg":
            return paths.NOISY_OU_10R
        return f"{paths.DRIFT_OU}/iid/{strat}/noisy"

    draw.scenario_panels(ref_iid + _strategy_arms(ou_iid),
                         str(out / "04_strategy_comparison_drift_iid_p005_OU"))
    draw.scenario_panels(
        ref_ni + _strategy_arms(
            lambda s: f"{paths.DRIFT_OU}/non_iid/{s}/noisy"),
        str(out / "04_strategy_comparison_drift_noniid_p005_OU"))

    # --- FedAvg non-IID: noiseless vs statico vs deriva ------------------
    noisy_ni = loaders.read_federated(f"{paths.FEDAVG_NON_IID}/noisy/p0.005")
    for drift_folder, suffix in (
            (f"{paths.DRIFT_RW}/noisy_non_iid/fedavg", ""),
            (f"{paths.DRIFT_OU}/non_iid/fedavg/noisy", "_OU")):
        draw.scenario_panels([
            ("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
            ("Noiseless", noiseless_ni, "C0", "-", "o", 0),
            ("Noisy (static, $p=0.005$)", noisy_ni, palette.NOISY, "-", "s", 1),
            ("Noisy (drift, $p=0.005$)",
             loaders.read_federated(drift_folder), "#67000d", "-", "^", 2),
        ], str(out / f"04_fedavg_noiseless_vs_noisy_vs_drift_noniid{suffix}"))


FIGURES = {
    "strategies.comparison": comparison,
    "strategies.under_noise": under_noise,
}
