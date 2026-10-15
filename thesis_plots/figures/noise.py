"""Chapter 4.2: effect of noise on FedAvg.

  noise_levels    FedAvg at three static noise levels
  seed_isolation  one source of variability per panel
  drift           noiseless vs static noise vs drift (random walk and OU)
  circle          map of the predicted probabilities, four regimes
"""
from pathlib import Path

import numpy as np

from .. import draw, loaders, palette, paths
from .. import style as ts

BASE = paths.FEDAVG_IID

# The two variants of the drift figure differ only by the source of the
# drifting arm: random walk against Ornstein-Uhlenbeck (theta = 0.3). The
# rest of the configuration is the same: p = 0.005, sigma = 0.001, 10 rounds,
# 7 seeds.
DRIFT_SOURCES = [
    (paths.NOISY_OU_10R, "", paths.FIG_NOISE),      # in the 4.2 document
    (f"{paths.DRIFT_RW}/noisy_iid/fedavg", "_RW", paths.EXTRA),
]

SEED_SOURCES = [
    (f"{paths.SEED_ISOLATION}/init_only", "Model initialization"),
    (f"{paths.SEED_ISOLATION}/sampling_only", "Client sampling"),
    (f"{paths.SEED_ISOLATION}/shots_only", "Shot noise"),
    (f"{paths.SEED_ISOLATION}/partition_noniid", "Data partition (non-IID)"),
]


def noise_levels():
    """FedAvg noiseless and at three static noise levels, with the centralized run."""
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
    """One source of variability per panel, MAD band over the seeds."""
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
        print(f"  {title}: final MAD {dev[-1]:.5f}")
    ts.save(fig, f"{paths.ensure(paths.FIG_NOISE)}/seed_isolation_median_mad")


def drift():
    """Noiseless vs static noise vs drift, in the two process variants."""
    # The first three curves do not depend on the drift type: read once.
    common = [
        ("Centralized", loaders.read_centralized(paths.CENTRALIZED),
         palette.CENTRALIZED, "--", None, 0),
        ("Noiseless", loaders.read_federated(f"{BASE}/noiseless"),
         "C0", "-", "o", 0),
        ("Noisy (static, $p=0.005$)",
         loaders.read_federated(paths.static_noise(BASE, "noisy", 0.005)),
         palette.NOISY, "-", "s", 1),
    ]
    for folder, suffix, dest in DRIFT_SOURCES:
        draw.scenario_panels(
            common + [("Noisy (drift, $p=0.005$)",
                       loaders.read_federated(folder), "#67000d", "-", "^", 2)],
            f"{paths.ensure(dest)}/04_noiseless_vs_noisy_vs_drift{suffix}")


# =====================================================================
# Map of the predicted probabilities
# =====================================================================

CIRCLE_SEED = 7
CIRCLE_NSHOTS = 1000
CIRCLE_POINTS = 1200
CIRCLE_GRID_SEED = 11

# (label, weights folder, noise level used for the evaluation)
CIRCLE_PANELS = [
    ("Noiseless", f"{BASE}/noiseless", None),
    (r"$p = 0.005$", paths.static_noise(BASE, "noisy", 0.005), 0.005),
    (r"$p = 0.007$", paths.static_noise(BASE, "noisy", 0.007), 0.007),
    (r"$p = 0.014$", paths.static_noise(BASE, "noisy", 0.014), 0.014),
]


def circle():
    """Probabilities predicted by the federated model, seed 7, four regimes.

    For each regime the final weights of that run are loaded and the model
    is evaluated through the corresponding channel, so the figure shows the
    two effects of the noise together: the shift of the learned solution and
    the compression of the expected values towards 0.5.

    Needs qibo: the imports sit inside the function because every other
    figure is produced without it.
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
            # partition_id 0 and scale 0: the panel shows the nominal level,
            # not the spread between clients.
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
        print(f"  {label}: mean {probs.mean():.3f}, "
              f"range {probs.min():.3f}-{probs.max():.3f}", flush=True)

    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Predicted probability")
    ts.save(fig, f"{paths.ensure(paths.FIG_NOISE)}/04_circle_predictions")


FIGURES = {
    "noise.levels": noise_levels,
    "noise.seed_isolation": seed_isolation,
    "noise.drift": drift,
    "noise.circle": circle,
}
