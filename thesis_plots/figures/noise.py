"""Capitolo 4.2: effetto del rumore su FedAvg.

  noise_levels    FedAvg a tre livelli di rumore statico
  seed_isolation  una fonte di variabilita' per pannello
  drift           noiseless vs rumore statico vs deriva (random walk e OU)
  circle          mappa delle probabilita' predette, quattro regimi
"""
from pathlib import Path

import numpy as np

from .. import draw, loaders, palette, paths
from .. import style as ts

BASE = paths.FEDAVG_IID

# Le due varianti della figura sulla deriva differiscono solo per la sorgente
# del braccio con deriva: random walk contro Ornstein-Uhlenbeck (theta = 0.3).
# Il resto della configurazione e' lo stesso: p = 0.005, sigma = 0.001, 10
# round, 7 seed.
DRIFT_SOURCES = [
    (f"{paths.DRIFT_RW}/noisy_iid/fedavg", ""),
    (paths.NOISY_OU_10R, "_OU"),
]

SEED_SOURCES = [
    (f"{paths.SEED_ISOLATION}/init_only", "Model initialization"),
    (f"{paths.SEED_ISOLATION}/sampling_only", "Client sampling"),
    (f"{paths.SEED_ISOLATION}/shots_only", "Shot noise"),
    (f"{paths.SEED_ISOLATION}/partition_noniid", "Data partition (non-IID)"),
]


def noise_levels():
    """FedAvg noiseless e a tre livelli di rumore statico, col centralizzato."""
    scenarios = [
        ("Centralized", loaders.read_centralized(paths.CENTRALIZED),
         palette.CENTRALIZED, "--", None, 0),
        ("Noiseless", loaders.read_federated(f"{BASE}/noiseless"),
         "C0", "-", "o", 0),
    ]
    for i, (p, marker) in enumerate(zip((0.005, 0.007, 0.014), "s^D")):
        scenarios.append((f"$p={p}$",
                          loaders.read_federated(paths.static_noise(BASE, "noisy", p)),
                          palette.NOISE_COLORS[p], "-", marker, i + 1))
    draw.scenario_panels(
        scenarios, f"{paths.ensure(paths.FIG_NOISE)}/04_fedavg_noiselevels")


def seed_isolation():
    """Una fonte di variabilita' per pannello, banda MAD sui seed."""
    fig, axes = ts.grid(2, 2)
    for ax, (folder, title) in zip(axes.ravel(), SEED_SOURCES):
        data = loaders.read_federated(folder)
        med = np.array(data["loss_median"])
        dev = np.array(data["loss_mad"])
        ax.plot(data["rounds"], med, color="black", lw=ts.LINEWIDTH)
        ax.fill_between(data["rounds"], med - dev, med + dev,
                        color="C0", alpha=0.25, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Round $t$")
        ax.set_ylabel("Loss")
        ax.tick_params(direction="in")
        print(f"  {title}: MAD finale {dev[-1]:.5f}")
    ts.save(fig, f"{paths.ensure(paths.FIG_NOISE)}/seed_isolation_median_mad")


def drift():
    """Noiseless vs rumore statico vs deriva, nelle due varianti di processo."""
    out = paths.ensure(paths.FIG_NOISE)
    # Le prime tre curve non dipendono dal tipo di deriva: si leggono una volta.
    common = [
        ("Centralized", loaders.read_centralized(paths.CENTRALIZED),
         palette.CENTRALIZED, "--", None, 0),
        ("Noiseless", loaders.read_federated(f"{BASE}/noiseless"),
         "C0", "-", "o", 0),
        ("Noisy (static, $p=0.005$)",
         loaders.read_federated(paths.static_noise(BASE, "noisy", 0.005)),
         palette.NOISY, "-", "s", 1),
    ]
    for folder, suffix in DRIFT_SOURCES:
        draw.scenario_panels(
            common + [("Noisy (drift, $p=0.005$)",
                       loaders.read_federated(folder), "#67000d", "-", "^", 2)],
            f"{out}/04_noiseless_vs_noisy_vs_drift{suffix}")


# =====================================================================
# Mappa delle probabilita' predette
# =====================================================================

CIRCLE_SEED = 7
CIRCLE_NSHOTS = 1000
CIRCLE_POINTS = 1200
CIRCLE_GRID_SEED = 11

# (etichetta, cartella dei pesi, livello di rumore usato per la valutazione)
CIRCLE_PANELS = [
    ("Noiseless", f"{BASE}/noiseless", None),
    (r"$p = 0.005$", paths.static_noise(BASE, "noisy", 0.005), 0.005),
    (r"$p = 0.007$", paths.static_noise(BASE, "noisy", 0.007), 0.007),
    (r"$p = 0.014$", paths.static_noise(BASE, "noisy", 0.014), 0.014),
]


def circle():
    """Probabilita' predette dal modello federato, seed 7, quattro regimi.

    Per ciascun regime si caricano i pesi finali di quella run e si valuta il
    modello attraverso il canale corrispondente, cosi' la figura mostra
    insieme i due effetti del rumore: lo spostamento della soluzione appresa e
    la compressione dei valori attesi verso 0.5.

    Richiede qibo: gli import stanno dentro la funzione perche' tutte le altre
    figure si producono senza.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import qibo

    qibo.config.log.setLevel("ERROR")
    from qibo import set_backend

    set_backend("numpy")
    import qibo_qfl_pt.patches  # noqa: F401
    from qibo_qfl_pt.task import (set_seed, create_model, set_weights,
                                  build_noise_model)

    rng = np.random.default_rng(CIRCLE_GRID_SEED)
    X = rng.uniform(-1, 1, size=(CIRCLE_POINTS, 2))
    XT = torch.tensor(X, dtype=torch.float64)

    def predictions(weights_path, p):
        noise = None
        if p is not None:
            # partition_id 0 e scale 0: il pannello mostra il livello nominale,
            # non la dispersione fra client.
            noise, _ = build_noise_model(pauli_base=p, readout_base=p,
                                         partition_id=0, scale=0.0)
        set_seed(CIRCLE_SEED)
        model = create_model(model_type="quantum", nshots=CIRCLE_NSHOTS,
                             noise_model=noise)
        w = list(np.load(weights_path).values())[0]
        set_weights(model, [w] if np.ndim(w) == 1 else list(w))
        set_seed(CIRCLE_SEED)
        with torch.no_grad():
            return model(XT).squeeze().numpy()

    fig, axes = ts.grid(2, 2, height=5.4)
    theta = np.linspace(0, 2 * np.pi, 400)

    sc = None
    for ax, (label, folder, p) in zip(axes.ravel(), CIRCLE_PANELS):
        probs = predictions(
            f"{folder}/weights/FedAvg_etal0.3_seed{CIRCLE_SEED}.npz", p)
        sc = ax.scatter(X[:, 0], X[:, 1], c=probs, cmap="RdBu_r", vmin=0,
                        vmax=1, s=3, linewidths=0)
        ax.plot(np.cos(theta), np.sin(theta), color="0.3", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_aspect("equal")
        print(f"  {label}: media {probs.mean():.3f}, "
              f"escursione {probs.min():.3f}-{probs.max():.3f}", flush=True)

    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Predicted probability")
    ts.save(fig, f"{paths.ensure(paths.FIG_NOISE)}/04_circle_predictions")


FIGURES = {
    "noise.levels": noise_levels,
    "noise.seed_isolation": seed_isolation,
    "noise.drift": drift,
    "noise.circle": circle,
}
