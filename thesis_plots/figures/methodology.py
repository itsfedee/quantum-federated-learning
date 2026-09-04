"""Figure del capitolo 3: partizioni Dirichlet e deriva del rumore.

Entrambe esistevano solo come PDF, senza codice che le producesse. Qui sono
ricostruite dai parametri che le generano, ritrovati confrontando i numeri
stampati nelle figure originali: le partizioni con alpha = 1.8 e seed dati 2
riproducono esattamente i conteggi nei titoli (119/16, 43/10, 83/46, 89/11,
65/18), e la deriva usa i walk seed veri delle sette run.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from .. import noise_model, palette, paths
from .. import style as ts

# --- partizioni ------------------------------------------------------------
ALPHA = 1.8
DATA_SEED = 2
N_CLIENTS = 5

# --- deriva ----------------------------------------------------------------
# Parametri della campagna del capitolo 3, diversi da quelli dello sweep sulla
# memoria (dove la base e' 0.005).
BASE = 0.007
SCALE = 0.002
SIGMA = 0.001
THETA = 0.3
N_ROUNDS = 10
# Un pannello per seed della campagna. Si tiene la sola realizzazione del
# seed 5; la mappa completa e' in noise_model.WALK_SEED_MAP.
SEEDS_SHOWN = [5]


def partitions():
    """Le cinque partizioni Dirichlet, una per client."""
    from qibo_qfl_pt.task import load_data_client

    fig, axes = ts.panels(N_CLIENTS, height=1.7)
    for pid, ax in enumerate(axes):
        x, y = load_data_client(partition_id=pid, ndata=500, iid=False,
                                num_partitions=N_CLIENTS, alpha=ALPHA,
                                seed=DATA_SEED, client_eval=False, testing=True)
        x = np.asarray(x)
        y = np.asarray(y).ravel()
        inside, outside = y == 1, y == 0
        ax.scatter(x[inside, 0], x[inside, 1], s=2, color="C0", label="Inside")
        ax.scatter(x[outside, 0], x[outside, 1], s=2, color="C3", label="Outside")
        # Il cerchio unitario e' il confine della classe.
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), ls="--", lw=0.6, color="0.5")
        ax.set_title(f"Client {pid} ({inside.sum()}/{outside.sum()})")
        ax.set_xlabel("$x_1$")
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        if pid == 0:
            ax.set_ylabel("$x_2$")
        else:
            ax.set_yticklabels([])
    ts.legend_above(fig, axes[0], ncol=2)
    ts.save(fig, f"{paths.ensure(paths.CAP03)}/03_dirichlet_partitions")


def drift():
    """Deriva OU del rumore, Pauli e readout affiancati, una curva per client."""
    for seed in SEEDS_SHOWN:
        walk_seed = noise_model.WALK_SEED_MAP[seed]
        pauli, readout = noise_model.trajectories(
            walk_seed, base=BASE, scale=SCALE, sigma=SIGMA, theta=THETA,
            n_rounds=N_ROUNDS, n_clients=N_CLIENTS)
        rounds = np.arange(N_ROUNDS + 1)
        fig, axes = ts.panels(2, height=2.3)
        for ax, data, lab in ((axes[0], pauli, r"Pauli $p(t)$"),
                              (axes[1], readout, r"Readout $r(t)$")):
            for pid in range(N_CLIENTS):
                # Marcatori sfalsati fra client: con cinque curve che si
                # intrecciano il colore da solo non basta a seguirle.
                ax.plot(rounds, data[pid], color=f"C{pid}",
                        marker=palette.CLIENT_MARKERS[pid], ms=ts.MARKERSIZE,
                        markevery=(pid % 2, 2), label=f"Client {pid}")
            ax.axhline(BASE, ls="--", lw=0.6, color="0.5", zorder=0)
            ax.set_xlabel(r"Round $t$")
            ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
        ts.legend_above(fig, axes[0], ncol=N_CLIENTS)
        ts.save(fig, f"{paths.ensure(paths.CAP03)}"
                     f"/03_noise_drift_ou_sigma001_seed{seed}")


FIGURES = {
    "cap03.partitions": partitions,
    "cap03.drift": drift,
}
