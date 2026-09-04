"""Ricostruzione delle traiettorie di rumore usate nelle run.

build_noise_model (qibo_qfl_pt/task.py) e' deterministico e stateless: il
generatore riparte da [walk_seed, partition_id] a ogni round e ripercorre gli
stessi passi. Le traiettorie si riproducono quindi qui, senza rilanciare
niente e senza dipendere da qibo.

La parte statica usa un generatore separato, seminato col solo partition_id,
che sparpaglia i client attorno al valore base.
"""
import numpy as np

# Mappa seed della run -> walk seed, come in parallel_experiments.
WALK_SEED_MAP = {1: 1, 2: 11, 3: 12, 4: 20, 5: 21, 6: 22, 7: 24}


def trajectories(walk_seed, base=0.005, scale=0.002, sigma=0.001, theta=0.3,
                 n_rounds=10, n_clients=5):
    """Rumore Pauli e readout per client e per round.

    theta e' il richiamo del processo di Ornstein-Uhlenbeck verso il valore di
    partenza: dX = theta*(0 - X) + sigma*dW. Con theta = 0 il processo degenera
    nel random walk delle campagne piu' vecchie, dove la varianza cresce coi
    round invece di restare stazionaria.
    """
    pauli = np.zeros((n_clients, n_rounds + 1))
    readout = np.zeros((n_clients, n_rounds + 1))
    for pid in range(n_clients):
        rng = np.random.default_rng(seed=pid)
        p0 = base + rng.uniform(-scale, scale)
        r0 = base + rng.uniform(-scale, scale)
        pauli[pid, 0], readout[pid, 0] = p0, r0
        for t in range(1, n_rounds + 1):
            steps = np.random.default_rng([walk_seed, pid]).normal(
                0.0, sigma, size=(t, 2))
            po = ro = 0.0
            for k in range(t):
                po += theta * (0.0 - po) + steps[k, 0]
                ro += theta * (0.0 - ro) + steps[k, 1]
            pauli[pid, t] = np.clip(p0 + po, 0, 1)
            readout[pid, t] = np.clip(r0 + ro, 0, 1)
    return pauli, readout
