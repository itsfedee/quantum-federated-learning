"""Capitolo 4.3.1: mitigazione CDR.

  fedavg          noiseless / noisy / mitigated, statico e con deriva
  panels          le stesse curve lette dai valori client grezzi, con
                  centralizzato e noiseless come riferimenti fissi
  metrics_vs_p    metriche finali contro livello di rumore
  loss_vs_p       loss finale contro p e guadagno di accuratezza per round
  regression      pannelli di regressione della mappa CDR
  static_threshold soglia 0.015 contro 0.03 sotto rumore statico
  recovery        tabella a schermo: quanto della perdita recupera la CDR
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
# Curve nel tempo
# =====================================================================

def fedavg():
    """FedAvg noiseless / noisy / mitigated: statico, deriva RW e deriva OU."""
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

    # Con deriva il noiseless passa a riferimento tratteggiato: le curve che
    # contano sono le due sotto deriva.
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

    # Stessa configurazione, ma con deriva di Ornstein-Uhlenbeck invece del
    # random walk: il rumore torna verso il valore base con theta = 0.3 invece
    # di allontanarsene. Il mitigato viene dallo sweep a 40 round, troncato ai
    # primi 10 perche' e' l'unica run OU mitigata a questa soglia.
    draw.scenario_panels(refs + [
        ("Noisy (drift)", loaders.read_federated(paths.NOISY_OU_10R),
         palette.NOISY, "-", "s", 1),
        ("Mitigated (drift)",
         loaders.read_federated(paths.MITIGATED_OU_10R, max_round=10),
         palette.MITIGATED, "-", "^", 2),
    ], str(out / "04_fedavg_noisy_vs_mitigated_drift_p005"))


def panels():
    """04_fedavg_mit e 04_fedavg_mit_drift, dai valori client grezzi.

    Sono il taglio alternativo delle curve di 4.3, quello per cliente: non
    entrano nella tesi e stanno fra le figure esplorative.
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
# Metriche finali contro livello di rumore
# =====================================================================

def metrics_vs_p():
    """Metriche finali di FedAvg contro p, noisy e mitigato.

    Ogni punto e' la mediana sui sette seed del valore all'ultimo round, con
    banda MAD; il noiseless e' una riga orizzontale, non dipendendo da p.
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

        # Asse in unita' di 1e-3: a due pollici di larghezza le etichette
        # "0.003" e "0.007" si sovrappongono, quelle intere no.
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
    """Loss finale contro p, e guadagno di accuratezza round per round.

    Pannello destro: differenza di accuratezza fra mitigato e noisy, appaiata
    seed per seed prima di aggregare, cosi' la banda misura la dispersione del
    guadagno e non quella delle due curve separate.
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
# Regressione della mappa CDR
# =====================================================================

REGRESSION_LEVELS = [0.005, 0.014, 0.035]
REGRESSION_SEED = 1


def cdr_training_points(p, seed, scale=0.002, client_id=0,
                        n_training_samples=120, nshots_cdr=30000, cdr_seed=40,
                        nshots_model=1000, x_input=None):
    """Punti di training della mappa CDR e coefficienti del fit.

    Costruisce il modello con il canale di rumore a livello p, lo valuta una
    volta su un punto qualsiasi per far costruire la mappa al mitigatore, e
    ne restituisce i punti di training insieme alla retta che li interpola.
    Richiede qibo: l'import sta qui dentro perche' tutte le altre figure di
    questo modulo si producono senza.
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
    """Punti di training della CDR, noisy contro noise-free, e retta di fit.

    Tre pannelli, uno per livello di rumore. Richiede qibo.
    """
    fig, axes = ts.panels(3)
    for ax, p in zip(axes, REGRESSION_LEVELS):
        noisy, noisefree, a, b = cdr_training_points(p, REGRESSION_SEED)

        ax.scatter(noisy, noisefree, color="blue", alpha=0.5, s=4)
        x_fit = np.linspace(noisy.min(), noisy.max(), 100)
        ax.plot(x_fit, a * x_fit + b, color="red", linewidth=1.0)

        # R^2 del fit effettivamente applicato, non di una regressione
        # ricalcolata.
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
        # Etichette piu' piccole del testo di base: a due pollici di pannello
        # quella verticale, lunga, occuperebbe l'intera altezza.
        ax.set_xlabel("Noisy exp. value", fontsize=7.5)
        ax.set_ylabel("Noiseless exp. value", fontsize=7.5)
        print(f"  p={p}: a={a:.3f} b={b:.3f} R2={r2:.4f} ({len(noisy)} punti)",
              flush=True)

    ts.save(fig, f"{paths.ensure(paths.FIG_MITIGATION)}"
                 f"/04_cdr_regression_panels")


# =====================================================================
# Soglia CDR sotto rumore statico
# =====================================================================

STATIC_ARMS = [
    (0.015, f"{paths.STATIC_CALIBRATIONS}/thresh_0015", "C0", "o"),
    (0.03, paths.FEDAVG_TH_003, palette.NOISY, "s"),
]


def static_threshold():
    """Loss client di FedAvg mitigato statico, soglia 0.015 contro 0.03.

    Stessa configurazione nei due bracci (p = 0.005, sigma = 0, no_memory, 10
    round, 7 seed, stessi init_seed e data_seed): cambia solo la soglia CDR.
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
        print(f"  eps={thr}: loss finale {med[-1]:.4f} (MAD {dev[-1]:.4f})")

    ax.set_xlabel(r"Round $t$")
    ax.set_ylabel("Client evaluation loss")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ts.save(fig, f"{paths.ensure(paths.EXTRA)}"
                 f"/04_fedavg_static_loss_thresh_0015_vs_003")


# =====================================================================
# Tabella a schermo
# =====================================================================

def recovery():
    """Quanto della perdita dovuta al rumore recupera la mitigazione."""
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
