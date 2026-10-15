"""Reconstruction of the noise trajectories used in the runs.

build_noise_model (qibo_qfl_pt/task.py) is deterministic and stateless: the
generator restarts from [walk_seed, partition_id] at every round and retraces
the same steps. The trajectories are therefore reproduced here, without
relaunching anything and without depending on qibo.

The static part uses a separate generator, seeded with partition_id only,
which scatters the clients around the base value.
"""
import numpy as np

# Map run seed -> walk seed, as in parallel_experiments.
WALK_SEED_MAP = {1: 1, 2: 11, 3: 12, 4: 20, 5: 21, 6: 22, 7: 24}


def trajectories(walk_seed, base=0.005, scale=0.002, sigma=0.001, theta=0.3,
                 n_rounds=10, n_clients=5):
    """Pauli and readout noise per client and per round.

    theta is the pull of the Ornstein-Uhlenbeck process towards the starting
    value: dX = theta*(0 - X) + sigma*dW. With theta = 0 the process
    degenerates into the random walk of the older campaigns, where the
    variance grows with the rounds instead of staying stationary.
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
