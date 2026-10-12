"""Quantum Federated Learning Client with Qiboml (PyTorch) and Flower."""

import gc
import traceback

import torch
import numpy as np

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.client import ClientApp
from flwr.common import Context

from qibo_qfl_pt.task import (
    create_model, build_noise_model, build_client_config, train_model, evaluate_model,
    get_weights, set_weights, get_partition_id, load_data_client, set_seed, get_server_round
)
from qibo_qfl_pt.patches import inject_cdr_map

app = ClientApp()


# ---------------------------------------------------------------------------
#  Helper: pull the CDR parameters out of the quantum model's mitigator
# ---------------------------------------------------------------------------

def _extract_cdr_metrics(model, include_training_data=False):
    """Read the CDR parameters (a, b) from the mitigator, when there is one.

    Returns un dict (possibilmente vuoto) con le metriche CDR.
    """
    metrics = {}
    try:
        mitigator = getattr(
            getattr(getattr(model, "q_model", None), "decoding", None),
            "mitigator", None,
        )
        if mitigator is None:
            return metrics

        popt = getattr(mitigator, "_mitigation_map_popt", None)
        if popt is None:
            return metrics

        a = popt[0].item() if hasattr(popt[0], "item") else float(popt[0])
        b = popt[1].item() if hasattr(popt[1], "item") else float(popt[1])
        metrics["cdr_a"] = float(a)
        metrics["cdr_b"] = float(b)

        n_maps = getattr(mitigator, "_n_maps_computed", None)
        if n_maps is not None:
            metrics["cdr_n_maps"] = int(n_maps)

        n_checks = getattr(mitigator, "_n_checks", None)
        if n_checks is not None:
            metrics["cdr_n_checks"] = int(n_checks)

        if include_training_data:
            td = getattr(mitigator, "_training_data", None)
            if td:
                metrics["cdr_noisy"] = [float(v) for v in td["noisy"]]
                metrics["cdr_noisefree"] = [float(v) for v in td["noise-free"]]

        # Diagnostic check log (patch 4)
        check_log = getattr(mitigator, "_check_log", None)
        if check_log:
            metrics["_check_log"] = check_log

    except Exception:
        traceback.print_exc()

    return metrics


# ---------------------------------------------------------------------------
#  Train
# ---------------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # ── 1. Read the configuration parameters ────────────────────────────
    seed = int(context.run_config["seed"])
    data_seed = int(context.run_config.get("data-seed", seed))
    set_seed(seed)

    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", True)
    lr = context.run_config["eta_l"]

    model_type = context.run_config.get("model-type", "quantum")
    hidden_classical = int(context.run_config.get("hidden-classical", 9))

    partition_id = get_partition_id(msg, context)
    server_round = get_server_round(msg, context)

    # ── 2. Create the model ─────────────────────────────────────────────
    noise_model, mitigation_config, nshots = build_client_config(
        context.run_config, partition_id, server_round,
    )
    
    model = create_model(
        model_type=model_type, nshots=nshots,
        noise_model=noise_model, mitigation_config=mitigation_config,
        hidden_classical=hidden_classical,
    )

    # ── 3. Receive the global parameters from the server ────────────────
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_weights(model, ndarrays)

    # CDR memory: inject (a, b) from the previous round if the server sent them
    config = msg.content.get("config", {})
    if "cdr-a-init" in config and "cdr-b-init" in config:
        inject_cdr_map(model, config["cdr-a-init"], config["cdr-b-init"])

    # Intra-round EMA: when enabled, the internal refit blends with the previous map
    intra_beta = float(context.run_config.get("cdr-intra-beta", 0))
    if intra_beta > 0:
        mitigator = getattr(
            getattr(getattr(model, "q_model", None), "decoding", None),
            "mitigator", None,
        )
        if mitigator is not None:
            mitigator._intra_ema_beta = intra_beta

    global_params = [p.clone().detach() for p in model.parameters()]

    # FedProx: read mu from the config (the other strategies do not send it → 0.0)
    proximal_mu = config.get("proximal-mu", 0.0)
    global_weights = [w.copy() for w in ndarrays] if proximal_mu > 0 else None

    # ── 4. Load the local data ──────────────────────────────────────────
    x_train, y_train = load_data_client(
        partition_id,
        ndata=context.run_config["n-train-data"],
        iid=context.run_config["iid"],
        num_partitions=context.run_config["num-clients"],
        alpha=context.run_config["alpha"],
        seed=data_seed,
    )

    # ── 5. Local training ───────────────────────────────────────────────
    history = train_model(
        model, x_train, y_train, lr=lr,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        partition_id=partition_id,
        global_weights=global_weights, proximal_mu=proximal_mu,
    )

    # ── 6. Compute the metrics ──────────────────────────────────────────
    metrics = {"num-examples": len(x_train)}
    if history["loss"]:
        metrics["train_loss"] = history["loss"][-1]
    if history["accuracy"]:
        metrics["train_acc"] = history["accuracy"][-1]
    if history["f1"]:
        metrics["train_f1"] = history["f1"][-1]

    # Drift: L2 distance between the updated local parameters and the global ones
    drift = sum(
        (p - gp).pow(2).sum() for p, gp in zip(model.parameters(), global_params)
    ).sqrt().item()
    metrics["drift"] = drift
    metrics["partition_id"] = partition_id

    # CDR logging (keys renamed with a _train suffix)
    cdr = _extract_cdr_metrics(model)
    if "cdr_a" in cdr:
        metrics["cdr_a_train"] = cdr["cdr_a"]
        metrics["cdr_b_train"] = cdr["cdr_b"]
        metrics["cdr_client"] = int(partition_id)
    if "cdr_n_maps" in cdr:
        metrics["cdr_n_maps"] = cdr["cdr_n_maps"]
    if "cdr_n_checks" in cdr:
        metrics["cdr_n_checks"] = cdr["cdr_n_checks"]

    # Pull out check_log (it cannot go in the MetricRecord, it is a list of dicts)
    check_log = cdr.pop("_check_log", None)

    # ── 7. Send the updated parameters back to the server ───────────────
    content = RecordDict({
        "arrays": ArrayRecord(get_weights(model)),
        "metrics": MetricRecord(metrics),
    })
    if check_log:
        import json as _json
        content["cdr_check_log"] = ConfigRecord({"data": _json.dumps(check_log)})

    del model, x_train, y_train, history
    gc.collect()

    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # ── 1. Read the configuration parameters ────────────────────────────
    seed = int(context.run_config["seed"])
    data_seed = int(context.run_config.get("data-seed", seed))
    set_seed(seed)

    model_type = context.run_config.get("model-type", "quantum")
    hidden_classical = int(context.run_config.get("hidden-classical", 9))

    partition_id = get_partition_id(msg, context)
    server_round = get_server_round(msg, context)

    # ── 2. Create the model ─────────────────────────────────────────────
    noise_model, mitigation_config, nshots = build_client_config(
        context.run_config, partition_id, server_round,
    )
    model = create_model(
        model_type=model_type, nshots=nshots,
        noise_model=noise_model, mitigation_config=mitigation_config,
        hidden_classical=hidden_classical,
    )

    # ── 3. Receive the global parameters from the server ────────────────
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_weights(model, ndarrays)

    # CDR memory: inject (a, b) from the previous round if the server sent them
    config = msg.content.get("config", {})
    if "cdr-a-init" in config and "cdr-b-init" in config:
        inject_cdr_map(model, config["cdr-a-init"], config["cdr-b-init"])

    # Intra-round EMA
    intra_beta = float(context.run_config.get("cdr-intra-beta", 0))
    if intra_beta > 0:
        mitigator = getattr(
            getattr(getattr(model, "q_model", None), "decoding", None),
            "mitigator", None,
        )
        if mitigator is not None:
            mitigator._intra_ema_beta = intra_beta

    # ── 4. Load the evaluation data ─────────────────────────────────────
    x_eval, y_eval = load_data_client(
        partition_id,
        ndata=context.run_config["n-train-data"],
        iid=context.run_config["iid"],
        num_partitions=context.run_config["num-clients"],
        alpha=context.run_config["alpha"],
        seed=data_seed,
        client_eval=True,
        testing=context.run_config.get("testing", True),
    )

    # ── 5. Evaluation ───────────────────────────────────────────────────
    loss, acc, f1 = evaluate_model(model, x_eval, y_eval)

    # ── 6. Compute the metrics ──────────────────────────────────────────
    metrics = {"num-examples": len(x_eval), "loss": loss, "accuracy": acc, "f1": f1}

    cdr = _extract_cdr_metrics(model, include_training_data=True)
    cdr_scatter = {}
    check_log = None
    if cdr:
        cdr["cdr_client"] = int(partition_id)
        # Scatter data goes in a separate record (Flower cannot aggregate lists)
        cdr_scatter = {k: cdr.pop(k) for k in ["cdr_noisy", "cdr_noisefree"] if k in cdr}
        check_log = cdr.pop("_check_log", None)
        metrics.update(cdr)

    # ── 7. Send the metrics back to the server ──────────────────────────
    content = RecordDict({
        "metrics": MetricRecord(metrics),
    })
    if cdr_scatter:
        content["cdr_scatter"] = ConfigRecord(cdr_scatter)
    if check_log:
        import json as _json
        content["cdr_check_log"] = ConfigRecord({"data": _json.dumps(check_log)})

    del model, x_eval, y_eval
    gc.collect()

    return Message(content=content, reply_to=msg)
