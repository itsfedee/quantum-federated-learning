"""Quantum Federated Learning Server with Qiboml (PyTorch) and Flower."""

import os
import gc
import numpy as np
from pathlib import Path

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from qibo_qfl_pt.custom_strategy import fedavg, fedadam, fedyogi, fedadagrad, fedprox
from qibo_qfl_pt.task import create_model, evaluate_model, get_weights, set_weights, load_data_server, set_seed


app = ServerApp()

strategies = {
    "FedAvg": (fedavg, None),
    "FedAdam": (fedadam, "eta"),
    "FedAdagrad": (fedadagrad, "eta"),
    "FedYogi": (fedyogi, "eta"),
    "FedProx": (fedprox, "mu"),
}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # ── 1. Leggo i parametri di configurazione ──────────────────────────

    # Param di run (numero client, numero round, ...)
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    iid_flag = context.run_config["iid"]
    fraction_train = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    local_epochs = context.run_config["local-epochs"]

    seed = context.run_config["seed"]
    init_seed = int(context.run_config.get("init-seed", seed))
    data_seed = int(context.run_config.get("data-seed", seed))
    alpha = context.run_config["alpha"]

    model_type = context.run_config.get("model-type", "quantum")
    hidden_classical = int(context.run_config.get("hidden-classical", 9))

    # Param della strategia (nome, LR, ...)
    strategy_name = context.run_config["strategy"]
    eta = context.run_config.get("eta", 0.1)
    eta_l = context.run_config.get("eta_l", 0.1)
    mu = context.run_config.get("mu", 0.1)

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class, param_name = strategies[strategy_name]

    # Parametri di noise (quantum)
    mode = context.run_config.get("mode", "noiseless")

    base_pauli = context.run_config.get("base-pauli", 0.0)
    base_readout = context.run_config.get("base-readout", 0.0)
    noise_scale = context.run_config.get("scale", 0.0)
    walk_sigma = context.run_config.get("walk-sigma", 0.0)
    walk_seed = int(context.run_config.get("walk-seed", 7))
    drift_type = context.run_config.get("drift-type", "rw")
    walk_theta = context.run_config.get("walk-theta", 0.1)

    nshots = context.run_config.get("nshots", 1000)
    nshots_info = None if nshots == "none" else int(nshots)

    noise_info = None
    if mode != "noiseless":
        noise_info = {
            "base_pauli": base_pauli,
            "base_readout": base_readout,
            "noise_scale": noise_scale,
            "sigma_drift": walk_sigma,
            "seed_walk": walk_seed if walk_sigma > 0 else None,
            # Il processo di drift va registrato: senza questo campo una run
            # archiviata non è ricostruibile (rw e ou hanno statistiche molto
            # diverse a parità di sigma). theta serve solo per l'ou.
            "drift_type": drift_type if walk_sigma > 0 else None,
            "walk_theta": walk_theta if (walk_sigma > 0 and drift_type == "ou") else None,
        }


    # ── 2. Creo la strategia ────────────────────────────────────────────
    save_path = context.run_config.get("save-path", "federated_results/iid/default")
    seed_label = context.run_config.get("seed-label", "") or None

    strategy_kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "local_epochs": local_epochs,
        "seed": seed,
        "init_seed": init_seed,
        "data_seed": data_seed,
        "sampling_seed": int(context.run_config.get("sampling-seed", seed)),
        "run_id": context.run_config.get("run-id", None),
        "save_path": save_path,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "alpha": alpha,
        "iid": iid_flag,
        "eta_l": eta_l,
        "noise_info": noise_info,
        "nshots": nshots_info,
        "training_mode": mode,
        "model_type": model_type,
        "seed_label": seed_label,
    }

    # --- CDR memory mode: "no_memory" | "warm_start" | "ema" ---
    cdr_mode = str(context.run_config.get("cdr-mode", "no_memory")).lower()
    _mode_map = {
        "no_memory":  {"cdr_memory": False, "cdr_beta": 0.0},
        "warm_start": {"cdr_memory": True,  "cdr_beta": 0.0},
        "ema":        {"cdr_memory": True,
                       "cdr_beta": float(context.run_config.get("cdr-beta", 0.5))},
    }
    if cdr_mode not in _mode_map:
        raise ValueError(f"cdr-mode '{cdr_mode}' non valido: usa no_memory|warm_start|ema")
    strategy_kwargs.update(_mode_map[cdr_mode])
    strategy_kwargs["cdr_mode"] = cdr_mode
    cdr_threshold = context.run_config.get("cdr-threshold", None)
    if cdr_threshold is not None:
        strategy_kwargs["cdr_threshold"] = float(cdr_threshold)

    if param_name == "eta":
        strategy_kwargs["eta"] = eta
        strategy_kwargs["eta_l"] = eta_l
    elif param_name == "mu":
        strategy_kwargs["mu"] = mu

    strategy = strategy_class(**strategy_kwargs)

    # ── 3. Creo il modello e i pesi iniziali ────────────────────────────
    set_seed(init_seed)
    model = create_model(model_type=model_type, hidden_classical=hidden_classical)
    arrays = ArrayRecord(get_weights(model))
    del model


    # ── 4. Definisco la valutazione globale ─────────────────────────────
    def global_evaluate(server_round: int, arrays: ArrayRecord, testing: bool) -> MetricRecord:
        """Evaluate on centralized validation or test set."""
        x_val, y_val = load_data_server(ndata=context.run_config["n-eval-data"], testing=testing)
        model = create_model(model_type=model_type, hidden_classical=hidden_classical)
        set_weights(model, arrays.to_numpy_ndarrays())
        loss, acc, f1 = evaluate_model(model, x_val, y_val)
        print(f"Server eval - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
        del model
        gc.collect()
        return MetricRecord({"accuracy": acc, "loss": loss, "f1_score": f1})

    def global_train_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate global model on the full (non-partitioned) training set."""
        x_train, y_train = load_data_server(ndata=context.run_config["n-train-data"], testing=False)
        model = create_model(model_type=model_type, hidden_classical=hidden_classical)
        set_weights(model, arrays.to_numpy_ndarrays())
        loss, acc, f1 = evaluate_model(model, x_train, y_train)
        print(f"Server train eval - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
        del model
        gc.collect()
        return MetricRecord({"accuracy": acc, "loss": loss, "f1_score": f1})

    # ── 5. Avvio la simulazione federata ────────────────────────────────
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=lambda sr, arr: global_evaluate(sr, arr, testing=context.run_config.get("testing", True)),
        train_evaluate_fn=global_train_evaluate,
    )

    # ── 6. Salvataggio pesi finali ──────────────────────────────────────
    if context.run_config.get("save-weights", True):
        if param_name == "eta":
            param_str = f"_eta{eta}"
        elif param_name == "mu":
            param_str = f"_mu{mu}"
        else:
            param_str = ""

        weights_dir = str(Path(save_path) / "weights")
        os.makedirs(weights_dir, exist_ok=True)
        wt_seed_label = seed_label if seed_label else f"seed{seed}"
        filename = f"{strategy_name}{param_str}_etal{eta_l}_{wt_seed_label}.npz"
        np.savez(os.path.join(weights_dir, filename), *result.arrays.to_numpy_ndarrays())

    gc.collect()
