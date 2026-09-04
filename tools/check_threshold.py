"""Misura |noisy_ref - exact_ref| per calibrare il threshold del CDR.

Strategia: creare il modello CON mitigation ma threshold altissimo (999)
così CDR non parte, ma il mitigator costruisce la reference circuit.
Poi ri-misurare la reference con rumore e calcolare lo shift.
"""

import numpy as np
import torch
from qibo_qfl_pt.task import (
    build_noise_model, create_model, set_seed, NQUBITS,
)
from qibo.symbols import Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.backends import construct_backend

# ── Parametri (gli stessi delle tue run) ─────────────────────────────
PAULI_BASE = 0.007
READOUT_BASE = 0.007
SCALE = 0.002
WALK_SIGMA = 0.0
WALK_SEED = 7
NUM_CLIENTS = 5
NSHOTS_CHECK = 30000   # shots usati dal mitigator per il check
N_REPEATS = 20         # ripetizioni per stimare media e std dello shift

SEED = 1

# ── Setup ────────────────────────────────────────────────────────────
set_seed(SEED)
numpy_backend = construct_backend("numpy")
observable = SymbolicHamiltonian((Z(0) + Z(1)) / 2, nqubits=NQUBITS)

print(f"Parametri: pauli={PAULI_BASE}, readout={READOUT_BASE}, "
      f"scale={SCALE}, walk_sigma={WALK_SIGMA}")
print(f"Shots per check: {NSHOTS_CHECK}, ripetizioni: {N_REPEATS}")
print(f"Shot noise floor (1/sqrt(N)): {1/np.sqrt(NSHOTS_CHECK):.6f}")
print()

# ── Per ogni client ──────────────────────────────────────────────────
for client_id in range(NUM_CLIENTS):
    noise_model, readout_prob = build_noise_model(
        pauli_base=PAULI_BASE, readout_base=READOUT_BASE,
        partition_id=client_id, scale=SCALE,
        server_round=0, walk_sigma=WALK_SIGMA, walk_seed=WALK_SEED,
    )

    # Mitigation config con threshold=999 → CDR non scatta,
    # ma il mitigator costruisce la reference circuit
    single = np.array([[1 - readout_prob, readout_prob],
                        [readout_prob, 1 - readout_prob]])
    response_matrix = np.kron(single, single)
    fake_mitigation_config = {
        "threshold": 999.0,
        "min_iterations": 999999,
        "method": "CDR",
        "method_kwargs": {
            "n_training_samples": 120,
            "nshots": NSHOTS_CHECK,
            "seed": 40,
            "readout": {"response_matrix": response_matrix},
        },
    }

    model = create_model(
        model_type="quantum", noise_model=noise_model,
        nshots=NSHOTS_CHECK, mitigation_config=fake_mitigation_config,
    )

    # Una forward pass per triggerare calculate_reference_expval
    x_dummy = torch.tensor([[-0.5, 0.5]], dtype=torch.float64)
    with torch.no_grad():
        _ = model(x_dummy)

    # Leggi reference circuit e valore esatto dal mitigator
    mitigator = model.q_model.decoding.mitigator
    ref_circuit = mitigator._reference_circuit
    exact_ref = float(mitigator._reference_value)

    if ref_circuit is None:
        print(f"Client {client_id}: reference circuit non costruita!")
        continue

    # Aggiungi misura se non presente
    from qibo import gates as g
    ref_with_meas = ref_circuit.copy()
    if not ref_with_meas.measurements:
        ref_with_meas.add(g.M(*range(NQUBITS)))

    # Ri-misura la reference con rumore (N_REPEATS volte)
    shifts = []
    for _ in range(N_REPEATS):
        noisy_circuit = noise_model.apply(ref_with_meas)
        noisy_result = numpy_backend.execute_circuit(
            noisy_circuit, nshots=NSHOTS_CHECK,
        )
        noisy_ref = observable.expectation_from_samples(
            noisy_result.frequencies(),
        )
        shifts.append(abs(float(noisy_ref) - exact_ref))

    mean_shift = np.mean(shifts)
    std_shift = np.std(shifts)

    print(f"Client {client_id}: "
          f"|noisy - exact| = {mean_shift:.6f} ± {std_shift:.6f}  "
          f"(exact_ref = {exact_ref:.6f})")

print()
print("[0.2, 0.8] dummy data")
print("── Interpretazione ──")
print(f"Shot noise floor (1σ):  {1/np.sqrt(NSHOTS_CHECK):.6f}")
print(f"Shot noise floor (2σ):  {2/np.sqrt(NSHOTS_CHECK):.6f}")
print(f"Shot noise floor (3σ):  {3/np.sqrt(NSHOTS_CHECK):.6f}")
print()
print("Il threshold deve stare tra il floor (2-3σ) e lo shift medio.")
print("Se threshold < shift → CDR si attiva sempre (bene)")
print("Se threshold > shift → CDR NON si attiva (mappa identità!)")
