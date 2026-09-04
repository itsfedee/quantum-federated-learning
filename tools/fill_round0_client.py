"""
Riempie eval_metrics_client del round 0 per tutti i JSON in una cartella (ricorsivo).
Valuta il modello iniziale su ogni partizione client e aggrega.
"""
import json
import glob
import sys
import os
import numpy as np
import torch

from qibo_qfl_pt.task import (
    create_model, set_seed, set_weights, evaluate_model,
    load_data_client, build_noise_model,
)


# Defaults per mitigation config (override via kwargs di fill_round0)
CDR_DEFAULTS = {
    "n_training_samples": 120,
    "nshots_cdr": 30000,
    "cdr_seed": 40,
    "min_iterations": 500,
}


def fill_round0(json_path, dry_run=False, **overrides):
    """Riempie eval_metrics_client del round 0.

    overrides supportati (prendono il valore dal JSON se non specificati):
        cdr_threshold, n_training_samples, nshots_cdr, cdr_seed, min_iterations
    """
    with open(json_path) as f:
        data = json.load(f)

    info = data["info"]
    rounds = data["rounds"]

    # Check se round 0 ha già eval_metrics_client
    r0 = rounds[0]
    if r0["round"] != 0:
        print(f"  SKIP {json_path}: round 0 non trovato")
        return False
    if r0.get("eval_metrics_client") and r0["eval_metrics_client"].get("loss") is not None:
        print(f"  SKIP {json_path}: round 0 già ha eval_metrics_client")
        return False

    # Parametri dal JSON
    seed = info["seed"]
    init_seed = info.get("init_seed", seed)
    data_seed = info.get("data_seed", seed)
    num_clients = info.get("num_clients", 5)
    iid = info.get("iid", True)
    alpha = info.get("alpha", 1.8)
    mode = info.get("training_mode", "noiseless")
    nshots = info.get("nshots")
    if nshots == "none" or nshots is None:
        nshots = None
    else:
        nshots = int(nshots)
    model_type = info.get("model_type", "quantum")
    hidden_classical = info.get("hidden_classical")

    # Noise config
    base_pauli = info.get("base_pauli", 0.0)
    base_readout = info.get("base_readout", 0.0)
    scale = info.get("noise_scale", info.get("scale", 0.0))

    # CDR config: JSON -> default -> override
    cdr_threshold = overrides.get("cdr_threshold",
                       info.get("cdr_threshold", 0.015))
    n_training_samples = overrides.get("n_training_samples",
                            info.get("n_training_samples", CDR_DEFAULTS["n_training_samples"]))
    nshots_cdr = overrides.get("nshots_cdr",
                    info.get("nshots_cdr", CDR_DEFAULTS["nshots_cdr"]))
    cdr_seed = overrides.get("cdr_seed",
                  info.get("cdr_seed", CDR_DEFAULTS["cdr_seed"]))
    min_iterations = overrides.get("min_iterations",
                        info.get("min_iterations", CDR_DEFAULTS["min_iterations"]))

    # Valuta su tutti i client
    all_losses, all_accs, all_f1s = [], [], []

    for pid in range(num_clients):
        # Seed per riproducibilità
        set_seed(seed)

        # Noise model per questo client
        noise_model = None
        mitigation_config = None
        if mode in ("noisy", "mitigated"):
            noise_model, readout_prob = build_noise_model(
                pauli_base=base_pauli, readout_base=base_readout,
                partition_id=pid, scale=scale, server_round=0,
            )
            if mode == "mitigated":
                single = np.array([
                    [1 - readout_prob, readout_prob],
                    [readout_prob, 1 - readout_prob],
                ])
                response_matrix = np.kron(single, single)
                mitigation_config = {
                    "threshold": cdr_threshold,
                    "min_iterations": min_iterations,
                    "method": "CDR",
                    "method_kwargs": {
                        "n_training_samples": n_training_samples,
                        "nshots": nshots_cdr,
                        "seed": cdr_seed,
                        "readout": {"response_matrix": response_matrix},
                    },
                }

        # Crea modello con init_seed
        set_seed(init_seed)
        model = create_model(
            model_type=model_type, nshots=nshots,
            noise_model=noise_model, mitigation_config=mitigation_config,
            hidden_classical=hidden_classical,
        )

        # Carica dati client per eval
        x_eval, y_eval = load_data_client(
            partition_id=pid, ndata=500, iid=iid,
            num_partitions=num_clients, alpha=alpha,
            seed=data_seed, client_eval=True, testing=True,
        )

        # Resetta seed per shot noise
        set_seed(seed)

        loss, acc, f1 = evaluate_model(model, x_eval, y_eval)
        all_losses.append(loss)
        all_accs.append(acc)
        all_f1s.append(f1)

    # Aggrega (media su tutti i client)
    eval_client = {
        "loss": float(np.mean(all_losses)),
        "accuracy": float(np.mean(all_accs)),
        "f1": float(np.mean(all_f1s)),
    }

    if dry_run:
        print(f"  DRY  {json_path}: {eval_client}")
        return False

    # Scrivi
    r0["eval_metrics_client"] = eval_client
    with open(json_path, "w") as f:
        json.dump(data, f, indent=3)
    print(f"  DONE {json_path}: {eval_client}")
    return True


if __name__ == "__main__":
    BASE = "results/fixed_training_set_results/noiseless_noisy_mit_fixed_results/federated/iid/fedavg"

    NOISE_LEVELS = ["0.003", "0.005", "0.007", "0.014"]

    folders = [
        f"{BASE}/noiseless",
    ]
    for p in NOISE_LEVELS:
        folders.append(f"{BASE}/noisy/scaled_p{p}_r{p}_s0.002")
        folders.append(f"{BASE}/mitigated/scaled_p{p}_r{p}_s0.002")

    dry_run = "--dry-run" in sys.argv

    for folder in folders:
        files = sorted(glob.glob(f"{folder}/**/*.json", recursive=True))
        if not files:
            files = sorted(glob.glob(f"{folder}/*.json"))

        print(f"\n{'='*60}")
        print(f"Folder: {folder} ({len(files)} files)")
        if dry_run:
            print("DRY RUN\n")

        filled = 0
        for f in files:
            if "aggregated" in f or "weights" in f:
                continue
            try:
                if fill_round0(f, dry_run=dry_run):
                    filled += 1
            except Exception as e:
                print(f"  ERROR {f}: {e}")

        print(f"Filled {filled} files")

    print("\nDone.")
