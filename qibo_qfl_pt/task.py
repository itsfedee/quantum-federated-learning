"""Quantum Federated Learning with Qiboml (PyTorch) and Flower.

Contents, in order:
  1. Circuit registry      CircuitSpec, trainable layers, CIRCUITS
  2. Dataset registry      DatasetSpec, generators, DATASETS
  3. Seed                  set_seed
  4. Noise                 readout_matrix, build_noise_model, build_client_config
  5. Model                 ClassicalModel, QMLModel, create_model
  6. Weights               get_weights, set_weights
  7. Training/evaluation   train_model, evaluate_model
  8. Data loading          generate_data, load_data_server, load_data_client
  9. Flower messages       get_partition_id, get_server_round
"""

import os
import random
from dataclasses import dataclass
from functools import reduce
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import qibo
from qibo import Circuit, gates, set_backend, construct_backend
from qibo.symbols import Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import NoiseModel, PauliError, ReadoutError
from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
import qiboml.interfaces.pytorch as pt
from qiboml.operations.differentiation import PSR
from qiboml.models.encoding import QuantumEncoding

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

qibo.config.log.setLevel("ERROR")
torch.set_default_dtype(torch.float64)
set_backend("numpy")

import qibo_qfl_pt.patches


# =====================================================================
# 1. Circuit registry
# =====================================================================
# A circuit is the block [encoder, trainable layer] repeated nlayers times,
# measured with an observable. To add a model: write its layer (and encoder,
# if needed) and add one entry to CIRCUITS. It is then selected by name with
# the "circuit" run-config key.

@dataclass(frozen=True)
class CircuitSpec:
    nqubits: int          # number of qubits
    nlayers: int          # number of layers
    layer: Callable       # Circuit, one trainable layer (called once per layer)
    encoder: Callable     # QuantumEncoding
    observable: Callable  # symbolic expression of the observable
    nfeatures: int | None = None  # features the encoder reads; None = one per qubit

    @property
    def features(self):
        """Number of components of x the encoder consumes.
        PhaseEncoding reads x[i] for every qubit, so none means nqubits. An
        encoder that packs several features on one qubit must declare its own.
        """
        return self.nqubits if self.nfeatures is None else self.nfeatures



def layer_rx_rz(nqubits):
    """RX and RZ on every qubit, then a chain of CNOT."""
    circ = Circuit(nqubits)
    for q in range(nqubits):
        circ.add(gates.RX(q, theta=np.random.uniform(0, np.pi / 2)))
        circ.add(gates.RZ(q, theta=np.random.uniform(0, np.pi / 2)))
    for q in range(nqubits - 1):
        circ.add(gates.CNOT(q, q + 1))
    return circ


# Define an Encoder that maps k features on 1 qubit
class SingleQubitEncoding(QuantumEncoding):
    """k features on one qubit (alternating Ry and  Rz rotations)"""

    def __init__(self, nfeatures: int):
        super().__init__(nqubits=1)
        self.nfeatures = nfeatures

    def __call__(self, x):
        if len(x) != self.nfeatures:
            raise ValueError(f"expected {self.nfeatures} features, got {len(x)}")
        circuit = self.circuit.copy()
        q = self.qubits[0]
        for i, xi in enumerate(x):
            gate = gates.RY if i % 2 == 0 else gates.RZ
            circuit.add(gate(q, theta=xi, trainable=False))
        return circuit


CIRCUITS = {
    "2q": CircuitSpec(
        nqubits=2,
        nlayers=3,
        layer=lambda: layer_rx_rz(nqubits=2),
        encoder=lambda: PhaseEncoding(nqubits=2),
        observable=lambda: (Z(0) + Z(1)) / 2,
    ),
}


# =====================================================================
# 2. Dataset registry
# =====================================================================
# Every dataset is a function load(split, n, seed) returning
#   x: shape (n, nfeatures), values in [-1, 1]
#   y: shape (n,), values in {0, 1}
# split is "train", "val" or "test". To add a dataset: write its function and add one entry to DATASETS.
# It is then selected by name with the "dataset" run-config key.

@dataclass(frozen=True)
class DatasetSpec:
    nfeatures: int  # number of features of each point
    load: Callable  # (split, n, seed) -> (x, y)


def circle(split, n, seed):
    """Uniform points in the square [-1, 1]^2, label 1 inside the unit circle."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    y = np.linalg.norm(x, axis=1) <= 1.0
    return x, y


DATASETS = {
    "circle": DatasetSpec(nfeatures=2, load=circle),
}

def check_shapes(circuit, dataset):
    """Fail early if dataset and encoder disagree on the number of features."""

    want = CIRCUITS[circuit].features
    have = DATASETS[dataset].nfeatures
    if have != want:
        raise ValueError(
            f"dataset '{dataset}' has {have} features, circuit '{circuit}' "
            f"encodes {want}: the remaining ones would be dropped silently."
        )

# =====================================================================
# 3. Seed
# =====================================================================

def set_seed(seed, backend=None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if backend is not None:
        backend.set_seed(seed)
    else:
        qibo.get_backend().set_seed(seed)

    from qibo.models.error_mitigation import SIMULATION_BACKEND, CLIFFORD_BACKEND
    SIMULATION_BACKEND().set_seed(seed)
    CLIFFORD_BACKEND().set_seed(seed)


# =====================================================================
# 4. Noise
# =====================================================================

WALK_SEED = 7


def readout_matrix(readout_prob, nqubits):
    """Symmetric bit-flip readout matrix on nqubits qubits (2^n x 2^n)."""
    single = np.array([
        [1 - readout_prob, readout_prob],
        [readout_prob, 1 - readout_prob],
    ])
    return reduce(np.kron, [single] * nqubits)


def build_noise_model(pauli_base, readout_base, partition_id, scale,
                      server_round=0, walk_sigma=0.0, walk_seed=WALK_SEED,
                      drift_type="rw", walk_theta=0.1, nqubits=2):

    """ Build local Pauli noise channel + Readout error in the general drifting case. 
    If walk = 0 the noise model is static. """

    # --- static part ---
    rng = np.random.default_rng(seed=partition_id)
    pauli_prob = pauli_base + rng.uniform(-scale, scale)
    readout_prob = readout_base + rng.uniform(-scale, scale)

    # --- dynamic part: per-client drift across the rounds ---
    # Stateless reconstruction: the rng always restarts from
    # [WALK_SEED, partition_id], so the first (server_round - 1) steps are the
    # same as in the previous rounds and the offset advances by one increment
    # per round.
    if walk_sigma > 0 and server_round > 0:
        walk_rng = np.random.default_rng([walk_seed, partition_id])
        noise_steps = walk_rng.normal(0.0, walk_sigma, size=(server_round, 2))

        if drift_type == "rw":
            # Random walk: offset = cumulative sum of the steps
            pauli_prob += noise_steps[:, 0].sum()
            readout_prob += noise_steps[:, 1].sum()
        elif drift_type == "ou":
            # Ornstein-Uhlenbeck: mean-reverting around the base value
            # dX = theta*(0 - X) + sigma*dW
            pauli_offset = 0.0
            readout_offset = 0.0
            for t in range(server_round):
                pauli_offset += walk_theta * (0.0 - pauli_offset) + noise_steps[t, 0]
                readout_offset += walk_theta * (0.0 - readout_offset) + noise_steps[t, 1]
            pauli_prob += pauli_offset
            readout_prob += readout_offset

    pauli_prob = float(np.clip(pauli_prob, 0, 1))
    readout_prob = float(np.clip(readout_prob, 0, 1))

    noise = NoiseModel()

    # local Pauli noise
    for q in range(nqubits):
        noise.add(
            PauliError([("X", pauli_prob), ("Y", pauli_prob), ("Z", pauli_prob)]),
            qubits=q,
        )

    # Readout noise
    noise.add(
        ReadoutError(readout_matrix(readout_prob, nqubits)),
        gate=gates.M,
        qubits=list(range(nqubits)),
    )
    return noise, readout_prob


def build_client_config(run_config, partition_id, server_round):
    """ Compute the noise model and mitigation config for a specific client. """

    mode = run_config.get("mode", "noiseless")
    nshots = run_config.get("nshots", 1000)
    if nshots == "none":
        nshots = None
    else:
        nshots = int(nshots)

    if mode == "noiseless":
        return None, None, nshots

    nqubits = CIRCUITS[run_config.get("circuit", "2q")].nqubits

    noise_model, readout_prob = build_noise_model(
        pauli_base=run_config["base-pauli"],
        readout_base=run_config["base-readout"],
        partition_id=partition_id,
        scale=run_config["scale"],
        server_round=server_round,
        walk_sigma=run_config.get("walk-sigma", 0.0),
        walk_seed=int(run_config.get("walk-seed", WALK_SEED)),
        drift_type=run_config.get("drift-type", "rw"),
        walk_theta=float(run_config.get("walk-theta", 0.1)),
        nqubits=nqubits,
    )

    mitigation_config = None
    if mode == "mitigated":
        response_matrix = readout_matrix(readout_prob, nqubits)
        mitigation_config = {
            "threshold": run_config.get("cdr-threshold", 0.015),
            "min_iterations": run_config.get("cdr-min-iterations", 500),
            "method": "CDR",
            "method_kwargs": {
                "n_training_samples": run_config.get("cdr-n-training-samples", 120),
                "nshots": run_config.get("cdr-nshots", 30000),
                "seed": run_config.get("cdr-seed", 40),
                "readout": {"response_matrix": response_matrix},
            },
        }
    return noise_model, mitigation_config, nshots


# =====================================================================
# 5. Model
# =====================================================================

HIDDEN_HYBRID = 6
HIDDEN_CLASSICAL = 9


class ClassicalModel(nn.Module):
    def __init__(self, hidden=HIDDEN_CLASSICAL):
        super().__init__()
        self.hybrid = False
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return self.net(x)


class QMLModel(nn.Module):
    """Post-MLP version: quantum -> MLP -> output."""
    def __init__(self, q_model, hybrid=False):
        super().__init__()
        self.q_model = q_model
        self.hybrid = hybrid
        if hybrid:
            self.post_net = nn.Sequential(
                nn.Linear(1, HIDDEN_HYBRID, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(HIDDEN_HYBRID, 1, dtype=torch.float64),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = torch.tanh(x) * torch.tensor(np.pi, dtype=torch.float64)
        x = torch.stack([self.q_model(xi).reshape(-1) for xi in x])
        if self.hybrid:
            x = x.double()
            x = self.post_net(x)
        else:
            x = (x + 1) / 2
        return torch.clamp(x, 1e-7, 1 - 1e-7)


def create_model(model_type="quantum", noise_model=None, nshots=None,
                 mitigation_config=None, hidden_classical=None,
                 circuit="2q", nlayers=None):

    """ Build the ML model (classical, quantum or hybrid) used in the federated pipeline. """

    if model_type == "classical":
        h = hidden_classical if hidden_classical is not None else HIDDEN_CLASSICAL
        return ClassicalModel(hidden=h)

    # quantum or hybrid: everything that defines the circuit comes from its spec
    spec = CIRCUITS[circuit]
    nlayers = nlayers or spec.nlayers
    encoding = spec.encoder()
    observable = SymbolicHamiltonian(spec.observable(), nqubits=spec.nqubits)
    decoding = Expectation(
        observable=observable,
        nqubits=spec.nqubits,
        backend=construct_backend("numpy"),
        density_matrix=False if noise_model is None else True,
        noise_model=noise_model,
        nshots=nshots,
        mitigation_config=mitigation_config,
    )

    circuit_structure = []
    for _ in range(nlayers):
        circuit_structure.extend([encoding, spec.layer()])

    q_model = pt.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding,
        differentiation=PSR,
    )
    hybrid = (model_type == "hybrid")
    return QMLModel(q_model, hybrid=hybrid)


# =====================================================================
# 6. Weights (for Flower)
# =====================================================================

def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=p.dtype, device=p.device)


# =====================================================================
# 7. Training and evaluation
# =====================================================================

def train_model(model, x_train, y_train, epochs=5, lr=0.2, batch_size=8,
                verbose=True, partition_id=None,
                global_weights=None, proximal_mu=0.0):
    """Train the model on the local data with SGD (+ proximal term for FedProx)."""

    # -- Setup: loss, optimizer ------------------------------
    use_logits = isinstance(model, ClassicalModel)
    loss_fn = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    if model.hybrid:  # test: different learning rates for the hybrid model
        param_groups = [
            {"params": list(model.q_model.parameters()), "lr": lr},
            {"params": list(model.post_net.parameters()), "lr": lr * 0.8},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": lr}]

    optimizer = torch.optim.SGD(param_groups)

    # -- Setup: global weights for the proximal term (FedProx) --------------
    if proximal_mu > 0 and global_weights is not None:
        global_tensors = [torch.tensor(w, dtype=torch.float64) for w in global_weights]
    else:
        global_tensors = None

    # -- Training loop ----------------------------------------
    history = {"loss": [], "accuracy": [], "f1": []}

    for epoch in range(epochs):

        # -- Forward + backward on shuffled mini-batches --
        model.train()
        idx = torch.randperm(len(x_train))
        batch_losses = []

        for i in range(0, len(x_train), batch_size):
            b = idx[i : i + batch_size]
            optimizer.zero_grad()

            loss = loss_fn(model(x_train[b]).squeeze(-1).float(), y_train[b].float())

            # FedProx: add the proximal term (mu/2) * ||w - w_global||^2
            if global_tensors is not None:
                prox_term = sum(
                    torch.sum((p - g) ** 2) for p, g in zip(model.parameters(), global_tensors)
                )
                loss = loss + (proximal_mu / 2) * prox_term

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # -- End of epoch metrics --
        epoch_loss = np.mean(batch_losses)
        history["loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            out = model(x_train).squeeze(-1)
            probs = torch.sigmoid(out) if use_logits else out
            preds = (probs >= 0.5).float()
            acc = (preds == y_train).float().mean().item()
            f1 = f1_score(y_train.numpy(), preds.numpy(), average="macro", zero_division=0)
        history["accuracy"].append(acc)
        history["f1"].append(f1)

    return history


def evaluate_model(model, x, y, verbose=False):
    """Evaluate the model: returns (loss, accuracy, f1)."""

    use_logits = isinstance(model, ClassicalModel)
    loss_fn = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    model.eval()
    with torch.no_grad():
        out = model(x).squeeze(-1)
        loss = loss_fn(out.float(), y.float()).item()
        probs = torch.sigmoid(out) if use_logits else out
        preds = (probs >= 0.5).float()
        acc = (preds == y).float().mean().item()
        f1 = f1_score(y.numpy(), preds.numpy(), average="macro", zero_division=0)

    if verbose:
        print(f"  loss: {loss:.4f}  acc: {acc:.4f}  f1: {f1:.4f}")

    return loss, acc, f1


# =====================================================================
# 8. Data loading
# =====================================================================

# Cache of the partitioned training set, one per process. last_seed holds the
# (seed, dataset) pair the cached partitioner was built with.
partitioner = None
last_seed = None


def generate_data(n_data, seed, dataset="circle", split="train"):
    """n_data points of a dataset of the registry, as a dict of numpy arrays."""
    x, y = DATASETS[dataset].load(split, n_data, seed)
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == DATASETS[dataset].nfeatures, (
        f"dataset '{dataset}' declares {DATASETS[dataset].nfeatures} features "
        f"but returned x with shape {x.shape}"
    )
    return {
        "point": x,
        "label": np.asarray(y).astype(np.int32),
    }


def load_data_server(ndata=500, testing=True, seed=40, dataset="circle"):
    current_seed = seed if testing else seed + 1
    split = "test" if testing else "val"
    data_dict = generate_data(ndata, current_seed, dataset, split)
    x = torch.tensor(data_dict["point"], dtype=torch.float64)
    y = torch.tensor(data_dict["label"].squeeze(), dtype=torch.float64)
    return x, y


def load_data_client(partition_id, ndata=500, iid=True, num_partitions=5,
                     alpha=0.25, seed=42, client_eval=False, testing=True,
                     dataset="circle"):

    global partitioner, last_seed
    if partitioner is None or last_seed != (seed, dataset):
        last_seed = (seed, dataset)
        train_dataset = Dataset.from_dict(generate_data(ndata, seed, dataset, split="train"))
        if iid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                alpha=alpha,
                partition_by="label",
                seed=seed,
            )
        partitioner.dataset = train_dataset

    part = partitioner.load_partition(partition_id)
    part.set_format("numpy")
    x_train = torch.tensor(np.array(part["point"], dtype=np.float64), dtype=torch.float64)
    y_train = torch.tensor(np.array(part["label"], dtype=np.float64).squeeze(), dtype=torch.float64)

    if client_eval:
        eval_seed = partition_id + 1000 if testing else partition_id + 2000
        split = "test" if testing else "val"
        eval_dict = generate_data(200, eval_seed, dataset, split)
        x_eval = torch.tensor(eval_dict["point"], dtype=torch.float64)
        y_eval = torch.tensor(eval_dict["label"], dtype=torch.float64)
        return x_eval, y_eval
    return x_train, y_train


# =====================================================================
# 9. Flower messages
# =====================================================================

def get_partition_id(msg, context):
    """Extract the partition_id from the server message or the node config."""
    config_record = msg.content.get("config")
    if config_record and "partition_id" in config_record:
        return config_record["partition_id"]
    return context.node_config.get("partition-id", 0)


def get_server_round(msg, context):
    """Extract the server round from the server message (0 if absent)."""
    config_record = msg.content.get("config")
    if config_record and "server_round" in config_record:
        return int(config_record["server_round"])
    return 0