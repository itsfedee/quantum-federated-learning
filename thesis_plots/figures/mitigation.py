"""Chapter 4.3.1: CDR mitigation.

  fedavg          noiseless / noisy / mitigated, static and under drift
  panels          the same curves read from the raw client values, with
                  centralized and noiseless as fixed references
  metrics_vs_p    final metrics against noise level
  loss_vs_p       final loss against p and accuracy gain per round
  regression      regression panels of the CDR map
  static_threshold threshold 0.015 against 0.03 under static noise
  recovery        on-screen table: how much of the loss the CDR recovers
"""
from pathlib import Path

import numpy as np
from scipy.stats import median_abs_deviation

from .. import draw, loaders, palette, paths
from .. import style as ts

SEEDS = palette.SEEDS
BASE = paths.FEDAVG_IID

NOISE_LEVELS = [0.003, 0.005, 0.006, 0.007, 0.01, 0.014, 0.018, 0.022]
SPEEDUP_LEVELS = [0.005, 0.007, 0.014]
REGIME_STYLE = (("noisy", "Noisy", palette.NOISY, "s"),
                ("mitigated", "Mitigated", palette.MITIGATED, "^"))


# =====================================================================
# Curves over time
# =====================================================================

def fedavg():
    """FedAvg noiseless / noisy / mitigated: static, RW drift and OU drift."""
    out = Path(paths.ensure(paths.FIG_MITIGATION_431))
    centr = loaders.read_centralized(paths.CENTRALIZED)
    noiseless = loaders.read_federated(f"{BASE}/noiseless")

    draw.scenario_panels([
        ("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
        ("Noiseless", noiseless, "C0", "-", "o", 0),
        ("Noisy ($p=0.005$)",
         loaders.read_federated(paths.static_noise(BASE, "noisy", 0.005)),
         palette.NOISY, "-", "s", 1),
        ("Mitigated ($p=0.005$)",
         loaders.read_federated(paths.static_noise(BASE, "mitigated", 0.005)),
         palette.MITIGATED, "-", "^", 2),
    ], str(out / "04_fedavg_noiseless_noisy_mitigated_p005"))

    # Under drift the noiseless run becomes a dashed reference: the curves
    # that matter are the two under drift.
    refs = [("Centralized", centr, palette.CENTRALIZED, "--", None, 0),
            ("Noiseless", noiseless, "C0", "--", None, 0)]

    draw.scenario_panels(refs + [
        ("Noisy (drift)",
         loaders.read_federated(f"{paths.DRIFT_RW}/noisy_iid/fedavg"),
         palette.NOISY, "-", "s", 1),
        ("Mitigated (drift)",
         loaders.read_federated(f"{paths.DRIFT_RW}/mit_iid/fedavg"),
         palette.MITIGATED, "-", "^", 2),
    ], f"{paths.ensure(paths.EXTRA)}/04_fedavg_noisy_vs_mitigated_drift_p005_RW")

    # Same configuration, but with Ornstein-Uhlenbeck drift instead of the
    # random walk: the noise returns towards the base value with theta = 0.3
    # instead of wandering away. The mitigated arm comes from the 40-round
    # sweep, cut to the first 10 because it is the only mitigated OU run at
    # this threshold.
    draw.scenario_panels(refs + [
        ("Noisy (drift)", loaders.read_federated(paths.NOISY_OU_10R),
         palette.NOISY, "-", "s", 1),
        ("Mitigated (drift)",
         loaders.read_federated(paths.MITIGATED_OU_10R, max_round=10),
         palette.MITIGATED, "-", "^", 2),
    ], str(out / "04_fedavg_noisy_vs_mitigated_drift_p005"))


def panels():
    """04_fedavg_mit and 04_fedavg_mit_drift, from the raw client values.

    They are the alternative cut of the 4.3 curves, the per-client one: they
    do not enter the thesis and sit among the exploratory figures.
    """
    out = Path(paths.ensure(paths.EXTRA))
    centr = loaders.centralized_arms(paths.CENTRALIZED, SEEDS, n_epochs=10)
    noiseless = loaders.client_arms(f"{BASE}/noiseless", SEEDS)

    arms = {
        "static": (paths.static_noise(BASE, "noisy", 0.005),
                   paths.static_noise(BASE, "mitigated", 0.005),
                   "04_fedavg_mit"),
        "drift": (paths.NOISY_OU_10R, paths.MITIGATED_OU_10R,
                  "04_fedavg_mit_drift"),
    }
    for noisy_dir, mit_dir, name in arms.values():
        draw.arm_panels({
            "Centralized": {"data": centr, "color": "gray", "ls": "--",
                            "lw": 1.5},
            "Noiseless": {"data": noiseless, "color": "tab:blue",
                          "marker": "s"},
            "Noisy ($p=0.005$)": {"data": loaders.client_arms(noisy_dir, SEEDS),
                                  "color": "tab:red", "marker": "o"},
            "Mitigated ($p=0.005$)": {
                "data": loaders.client_arms(mit_dir, SEEDS),
                "color": "tab:green", "marker": "^"},
        }, out / name)


# =====================================================================
# Final metrics against noise level
# =====================================================================

def metrics_vs_p():
    """Final FedAvg metrics against p, noisy and mitigated.

    Every point is the median over the seven seeds of the value at the last
    round, with a MAD band; the noiseless run is a horizontal line, as it
    does not depend on p.
    """
    noiseless = loaders.final_metrics(f"{BASE}/noiseless")
    series = {regime: [loaders.final_metrics(paths.static_noise(BASE, regime, p))
                       for p in NOISE_LEVELS]
              for regime, _, _, _ in REGIME_STYLE}

    fig, axes = ts.panels(3)
    for ax, metric in zip(axes, palette.METRICS_CLIENT):
        info = palette.METRIC_INFO[metric]
        scale = 100 if info["pct"] else 1

        ref = noiseless[metric] * scale
        ref_med = float(np.median(ref))
        ref_mad = float(median_abs_deviation(ref))
        ax.axhline(ref_med, color="C0", ls="--", lw=ts.LINEWIDTH,
                   label="Noiseless", zorder=0)
        ax.fill_between(NOISE_LEVELS, ref_med - ref_mad, ref_med + ref_mad,
                        color="C0", alpha=0.15, lw=0, zorder=0)

        for regime, label, color, marker in REGIME_STYLE:
            med = np.array([np.median(d[metric])
                            for d in series[regime]]) * scale
            dev = np.array([median_abs_deviation(d[metric])
                            for d in series[regime]]) * scale
            ax.plot(NOISE_LEVELS, med, color=color, lw=ts.LINEWIDTH,
                    marker=marker, ms=ts.MARKERSIZE, label=label)
            ax.fill_between(NOISE_LEVELS, med - dev, med + dev, color=color,
                            alpha=0.15, lw=0)

        # Axis in units of 1e-3: at two inches of width the labels "0.003"
        # and "0.007" overlap, the integer ones do not.
        ax.set_xlabel(r"Noise level $p$ [$10^{-3}$]")
        ax.set_ylabel(info["ylabel"])
        ax.set_xticks([0.005, 0.010, 0.015, 0.020])
        ax.set_xticklabels(["5", "10", "15", "20"])
        ax.tick_params(direction="in")

    handles, labels = axes[0].get_legend_handles_labels()
    ts.legend_above(fig, axes[0], ncol=len(labels))
    ts.save(fig, f"{paths.ensure(paths.FIG_MITIGATION_431)}"
                 f"/04_fedavg_metrics_vs_noise_level")


def loss_vs_p():
    """Final loss against p, and accuracy gain round by round.

    Right panel: accuracy difference between mitigated and noisy, paired seed
    by seed before aggregating, so that the band measures the spread of the
    gain and not that of the two separate curves.
    """
    fig, axes = ts.panels(2)

    ax = axes[0]
    ref = loaders.final_loss(f"{BASE}/noiseless")
    ref_med = float(np.median(ref))
    ref_mad = float(median_abs_deviation(ref))
    ax.axhline(ref_med, color="C0", ls="--", lw=ts.LINEWIDTH,
               label="Noiseless", zorder=0)
    ax.fill_between(NOISE_LEVELS, ref_med - ref_mad, ref_med + ref_mad,
                    color="C0", alpha=0.15, lw=0, zorder=0)

    for regime, label, color, marker in REGIME_STYLE:
        vals = [loaders.final_loss(paths.static_noise(BASE, regime, p))
                for p in NOISE_LEVELS]
        med = np.array([np.median(v) for v in vals])
        dev = np.array([median_abs_deviation(v) for v in vals])
        ax.plot(NOISE_LEVELS, med, color=color, lw=ts.LINEWIDTH, marker=marker,
                ms=ts.MARKERSIZE, label=label)
        ax.fill_between(NOISE_LEVELS, med - dev, med + dev, color=color,
                        alpha=0.15, lw=0)

    ax.set_xlabel(r"Noise level $p$")
    ax.set_ylabel("Loss")
    ax.set_xticks([0.005, 0.014, 0.022])
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax = axes[1]
    for p in SPEEDUP_LEVELS:
        noisy = loaders.accuracy_by_seed(paths.static_noise(BASE, "noisy", p))
        mit = loaders.accuracy_by_seed(paths.static_noise(BASE, "mitigated", p))
        seeds = sorted(set(noisy) & set(mit))
        rounds = sorted(r for r in noisy[seeds[0]]
                        if r > 0 and noisy[seeds[0]][r] is not None
                        and mit[seeds[0]].get(r) is not None)
        diff = np.array([[100 * (mit[s][r] - noisy[s][r]) for r in rounds]
                         for s in seeds])
        med = np.median(diff, axis=0)
        dev = median_abs_deviation(diff, axis=0)
        ax.plot(rounds, med, color=palette.NOISE_COLORS[p], lw=ts.LINEWIDTH,
                marker="o", ms=ts.MARKERSIZE, label=rf"$p = {p}$")
        ax.fill_between(rounds, med - dev, med + dev,
                        color=palette.NOISE_COLORS[p], alpha=0.15, lw=0)

    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel(r"Round $t$")
    ax.set_ylabel(r"$\Delta$ Accuracy [%]")
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ts.save(fig, f"{paths.ensure(paths.FIG_MITIGATION_431)}"
                 f"/04_mitigation_vs_p")


# =====================================================================
# Regression of the CDR map
# =====================================================================

REGRESSION_LEVELS = [0.005, 0.014, 0.035]
REGRESSION_SEED = 1


def cdr_training_points(p, seed, scale=0.002, client_id=0,
                        n_training_samples=120, nshots_cdr=30000, cdr_seed=40,
                        nshots_model=1000, x_input=None):
    """Training points of the CDR map and coefficients of the fit.

    Builds the model with the noise channel at level p, evaluates it once on
    an arbitrary point to make the mitigator build the map, and returns its
    training points together with the line that interpolates them. Needs
    qibo: the import sits in here because every other figure of this module
    is produced without it.
    """
    import torch

    from qibo_qfl_pt.task import (build_noise_model, create_model, set_seed)

    set_seed(seed)
    noise_model, readout_prob = build_noise_model(
        pauli_base=p, readout_base=p, partition_id=client_id, scale=scale,
    )
    single = np.array([[1 - readout_prob, readout_prob],
                       [readout_prob, 1 - readout_prob]])
    mitigation_config = {
        "threshold": 0.1, "min_iterations": 500, "method": "CDR",
        "method_kwargs": {
            "n_training_samples": n_training_samples,
            "nshots": nshots_cdr, "seed": cdr_seed,
            "readout": {"response_matrix": np.kron(single, single)},
        },
    }
    model = create_model(model_type="quantum", nshots=nshots_model,
                         noise_model=noise_model,
                         mitigation_config=mitigation_config)

    x = torch.tensor(x_input or [[0.5, 0.3]], dtype=torch.float64)
    with torch.no_grad():
        model(x)

    mitigator = model.q_model.decoding.mitigator
    train_data = mitigator._training_data
    popt = mitigator._mitigation_map_popt
    return (np.array(train_data["noisy"]), np.array(train_data["noise-free"]),
            float(popt[0]), float(popt[1]))


def regression():
    """CDR training points, noisy against noise-free, and fit line.

    Three panels, one per noise level. Needs qibo.
    """
    fig, axes = ts.panels(3)
    for ax, p in zip(axes, REGRESSION_LEVELS):
        noisy, noisefree, a, b = cdr_training_points(p, REGRESSION_SEED)

        ax.scatter(noisy, noisefree, color="blue", alpha=0.5, s=4)
        x_fit = np.linspace(noisy.min(), noisy.max(), 100)
        ax.plot(x_fit, a * x_fit + b, color="red", linewidth=1.0)

        # R^2 of the fit actually applied, not of a recomputed regression.
        pred = a * noisy + b
        ss_res = float(((noisefree - pred) ** 2).sum())
        ss_tot = float(((noisefree - noisefree.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        label = "\n".join([rf"$y = {a:.2f}x {b:+.3f}$", rf"$R^2 = {r2:.3f}$"])
        ax.text(0.04, 0.96, label, transform=ax.transAxes, fontsize=7,
                va="top", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.7", linewidth=0.6))

        ax.set_title(rf"$p = {p}$")
        ax.grid(False)
        # Labels smaller than the base text: at two inches of panel the
        # vertical one, being long, would take up the whole height.
        ax.set_xlabel("Noisy exp. value", fontsize=7.5)
        ax.set_ylabel("Noiseless exp. value", fontsize=7.5)
        print(f"  p={p}: a={a:.3f} b={b:.3f} R2={r2:.4f} ({len(noisy)} points)",
              flush=True)

    ts.save(fig, f"{paths.ensure(paths.FIG_MITIGATION)}"
                 f"/04_cdr_regression_panels")


# =====================================================================
# CDR threshold under static noise
# =====================================================================

STATIC_ARMS = [
    (0.015, f"{paths.STATIC_CALIBRATIONS}/thresh_0015", "C0", "o"),
    (0.03, paths.FEDAVG_TH_003, palette.NOISY, "s"),
]


def static_threshold():
    """Client loss of static mitigated FedAvg, threshold 0.015 against 0.03.

    Same configuration in both arms (p = 0.005, sigma = 0, no_memory, 10
    rounds, 7 seeds, same init_seed and data_seed): only the CDR threshold
    changes.
    """
    fig, ax = ts.single()
    rounds = None
    for thr, folder, color, marker in STATIC_ARMS:
        rounds, M = loaders.client_curves(loaders.run_files(folder), "loss")
        med = np.median(M, axis=0)
        dev = np.array([loaders.mad(M[:, j]) for j in range(M.shape[1])])
        ax.plot(rounds, med, color=color, lw=ts.LINEWIDTH, marker=marker,
                ms=ts.MARKERSIZE,
                label=rf"$\epsilon_{{\mathrm{{th}}}} = {thr}$")
        ax.fill_between(rounds, med - dev, med + dev, color=color, alpha=0.15,
                        lw=0)
        print(f"  eps={thr}: final loss {med[-1]:.4f} (MAD {dev[-1]:.4f})")

    ax.set_xlabel(r"Round $t$")
    ax.set_ylabel("Client evaluation loss")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ts.save(fig, f"{paths.ensure(paths.EXTRA)}"
                 f"/04_fedavg_static_loss_thresh_0015_vs_003")


# =====================================================================
# On-screen table
# =====================================================================

def recovery():
    """How much of the loss due to noise the mitigation recovers."""
    def fmt(arr):
        return f"{np.median(arr):.4f} +/- {median_abs_deviation(arr):.4f}"

    print("\n" + "=" * 110)
    print(f'{"Strategy":<14} {"Noiseless Loss":<22} {"Noisy Loss":<22}'
          f' {"Mitigated Loss":<22} {"Recovery (%)":<14}')
    print("-" * 110)

    for name, strat in palette.STRATEGY_NAMES.items():
        base = f"{paths.IID}/{name}"
        l_nl = loaders.final_loss(f"{base}/noiseless")
        l_ny = loaders.final_loss(paths.static_noise(base, "noisy", 0.005))
        l_mt = loaders.final_loss(paths.static_noise(base, "mitigated", 0.005))
        gap = np.median(l_ny) - np.median(l_nl)
        rec = (np.median(l_ny) - np.median(l_mt)) / gap * 100 if gap > 0 else 0
        print(f"{strat:<14} {fmt(l_nl):<22} {fmt(l_ny):<22} {fmt(l_mt):<22}"
              f" {rec:<14.1f}")

    print("=" * 110)


FIGURES = {
    "mitigation.fedavg": fedavg,
    "mitigation.panels": panels,
    "mitigation.metrics_vs_p": metrics_vs_p,
    "mitigation.loss_vs_p": loss_vs_p,
    "mitigation.regression": regression,
    "mitigation.static_threshold": static_threshold,
    "mitigation.recovery": recovery,
}
