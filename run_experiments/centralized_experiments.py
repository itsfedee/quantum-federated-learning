"""
Esperimenti centralizzati (singolo modello, no federazione).
Sweep su noise levels, salva JSON con metriche e pesi.

Uso:
  python centralized_experiments.py --workers 8
  python centralized_experiments.py --mode noisy --workers 4
  python centralized_experiments.py --mode mitigated --workers 4
"""

import os
import json
import time
import gc
import argparse
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Importa da qibo_qfl_pt (patch + funzioni condivise)
import qibo_qfl_pt.patches  # noqa: F401 — applica i patch PSR e CDR
from qibo_qfl_pt.task import (
    set_seed, create_model, get_weights, NQUBITS,
)

import qibo
from qibo import set_backend, construct_backend
from qibo.noise import NoiseModel, PauliError, ReadoutError, gates


# =====================================================================
# Funzioni specifiche per il centralizzato
# =====================================================================

def generate_data(ndata=100, seed=42):
    """Genera dati circle-in-square come tensori PyTorch."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(ndata, 2))
    y = (np.linalg.norm(x, axis=1) <= 1.0).astype(np.float64)
    return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def build_noise_model_centralized(pauli_prob, readout_prob):
    """Noise model senza partition_id/scale (centralizzato)."""
    noise = NoiseModel()
    for q in range(NQUBITS):
        noise.add(
            PauliError([("X", pauli_prob), ("Y", pauli_prob), ("Z", pauli_prob)]),
            qubits=q,
        )
    single = np.array([
        [1 - readout_prob, readout_prob],
        [readout_prob, 1 - readout_prob],
    ])
    readout_matrix = np.kron(single, single)
    noise.add(ReadoutError(readout_matrix), gate=gates.M, qubits=[0, 1])
    return noise


def drifted_probs(pauli_base, readout_base, epoch, walk_sigma,
                  walk_seed=7, drift_type="rw", walk_theta=0.1):
    """Valori di rumore all'epoca `epoch`, ricostruiti stateless come nel federato.

    Il generatore riparte sempre da [walk_seed], quindi i primi (epoch-1) passi
    sono identici a quelli delle epoche precedenti: il drift evolve di un solo
    incremento per epoca. epoch=0 restituisce i valori base.
    drift_type: "rw" random walk, "ou" mean-reverting verso il valore base.
    """
    rng = np.random.default_rng([walk_seed])
    p, r = float(pauli_base), float(readout_base)
    for _ in range(int(epoch)):
        eps_p, eps_r = rng.normal(0.0, walk_sigma, 2)
        if drift_type == "ou":
            p = p + walk_theta * (pauli_base - p) + eps_p
            r = r + walk_theta * (readout_base - r) + eps_r
        else:
            p += eps_p
            r += eps_r
        p = min(max(p, 0.0), 1.0)
        r = min(max(r, 0.0), 1.0)
    return p, r


def _set_decoding_noise(model, noise_model):
    """Sostituisce il noise model dentro la decoding del QuantumModel."""
    dec = getattr(getattr(model, "q_model", model), "decoding", None)
    if dec is None:
        raise RuntimeError("decoding non trovata sul modello")
    dec.noise_model = noise_model


def build_mitigation_config(readout_prob, threshold=0.015, with_memory=False, memory_beta=0.2):
    """Config CDR per mitigazione."""
    single = np.array([[1 - readout_prob, readout_prob],
                       [readout_prob, 1 - readout_prob]])
    response_matrix = np.kron(single, single)
    cfg = {
        "threshold": threshold,
        "min_iterations": 500,
        "method": "CDR",
        "method_kwargs": {
            "n_training_samples": 120,
            "nshots": 30000,
            "seed": 40,
            "readout": {"response_matrix": response_matrix},
        },
    }
    if with_memory:
        cfg["with_memory"] = True
        cfg["memory_beta"] = memory_beta
    return cfg


def get_trainable_params(model):
    return np.concatenate([p.detach().cpu().numpy().flatten()
                           for p in model.parameters()])


def train_and_evaluate(model, ndata_train, seed, epochs, lr, batch_size,
                       ndata_eval=200, seed_eval=40, job_tag="",
                       noise_update_fn=None):
    """Traina e ritorna metriche per epoca + pesi finali."""
    x_train, y_train = generate_data(ndata=ndata_train, seed=seed)
    x_eval, y_eval = generate_data(ndata=ndata_eval, seed=seed_eval)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Accesso al mitigatore per logging calibrazioni
    mitigator = getattr(
        getattr(getattr(model, "q_model", None), "decoding", None),
        "mitigator", None,
    )
    prev_n_maps = 0

    epochs_data = []
    t0 = time.time()

    # Epoca 0: valutazione modello non trainato
    model.eval()
    with torch.no_grad():
        y_prob = model(x_eval).squeeze(-1)
        eval_loss_0 = loss_fn(y_prob.float(), y_eval.float()).item()
    y_pred = (y_prob >= 0.5).float()
    acc_0 = float((y_pred == y_eval).float().mean().item())
    f1_0 = float(f1_score(y_eval.numpy(), y_pred.numpy(), average="macro"))
    epochs_data.append({
        "epoch": 0,
        "train_loss": None,
        "eval_loss": eval_loss_0,
        "eval_accuracy": acc_0,
        "eval_f1": f1_0,
    })

    for epoch in range(epochs):
        # Drift del rumore: aggiorna il noise model per questa epoca
        drift_info = None
        if noise_update_fn is not None:
            drift_info = noise_update_fn(model, epoch + 1)

        model.train()
        idx = torch.randperm(len(x_train))
        batch_losses = []
        for i in range(0, len(x_train), batch_size):
            b = idx[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[b]).squeeze(-1).float(), y_train[b].float())
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            y_prob = model(x_eval).squeeze(-1)
            if (epoch + 1) % 5 == 0:
                print(f"  y_prob stats: min={y_prob.min():.4e}, max={y_prob.max():.4e}, "
                    f"mean={y_prob.mean():.4f}, "
                    f"saturated_low={(y_prob < 1e-5).float().mean().item():.2%}, "
                    f"saturated_high={(y_prob > 1-1e-5).float().mean().item():.2%}")
            eval_loss = loss_fn(y_prob.float(), y_eval.float()).item()
        y_pred = (y_prob >= 0.5).float()
        acc = float((y_pred == y_eval).float().mean().item())
        f1 = float(f1_score(y_eval.numpy(), y_pred.numpy(), average="macro"))

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": acc,
            "eval_f1": f1,
        }
        if drift_info is not None:
            row["pauli_prob"], row["readout_prob"] = drift_info

        # Logging calibrazioni dal mitigatore
        if mitigator is not None:
            cur_n_maps = getattr(mitigator, "_n_maps_computed", 0)
            row["n_maps_epoch"] = cur_n_maps - prev_n_maps
            row["n_maps_cumulative"] = cur_n_maps
            row["n_checks"] = getattr(mitigator, "_n_checks", 0)
            # (a, b) correnti
            popt = getattr(mitigator, "_mitigation_map_popt", None)
            if popt is not None:
                row["cdr_a"] = float(popt[0])
                row["cdr_b"] = float(popt[1])
            # check_log con D, (a,b), step, esito — flush per epoca
            check_log = getattr(mitigator, "_check_log", None)
            if check_log:
                row["check_log"] = list(check_log)
                mitigator._check_log = []  # reset per prossima epoca
            prev_n_maps = cur_n_maps

        epochs_data.append(row)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tag = f"{job_tag}" if job_tag else ""
            print(f"    {tag} Epoch {epoch+1}/{epochs}  eval loss: {eval_loss:.4f}  eval acc: {acc:.4f}", flush=True)

    # Totale calibrazioni
    total_calibrations = getattr(mitigator, "_n_maps_computed", 0) if mitigator else 0

    elapsed = time.time() - t0
    weights = get_trainable_params(model)
    return epochs_data, weights, elapsed, total_calibrations


# =====================================================================
# Singolo job
# =====================================================================

def run_single_job(job):
    """Esegue un singolo esperimento centralizzato."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    qibo.config.log.setLevel("ERROR")
    set_backend("numpy")
    set_seed(job["seed"])

    if job["mode"] == "noiseless":
        print(f"  [START] noiseless seed={job['seed']}", flush=True)
    else:
        print(f"  [START] {job['mode']} p={job['pauli_prob']} seed={job['seed']}", flush=True)

    noise_model = None
    mitigation_config = None

    if job["mode"] in ("noisy", "mitigated"):
        noise_model = build_noise_model_centralized(job["pauli_prob"], job["readout_prob"])

    if job["mode"] == "mitigated":
        mitigation_config = build_mitigation_config(
            job["readout_prob"],
            threshold=job.get("cdr_threshold", 0.015),
            with_memory=job.get("with_memory", False),
            memory_beta=job.get("memory_beta", 0.2),
        )

    # Drift per epoca (solo noisy/mitigated, se walk_sigma > 0)
    noise_update_fn = None
    walk_sigma = float(job.get("walk_sigma", 0.0) or 0.0)
    if job["mode"] in ("noisy", "mitigated") and walk_sigma > 0:
        p0, r0 = job["pauli_prob"], job["readout_prob"]
        wseed = int(job.get("walk_seed", 7))
        dtype_ = job.get("drift_type", "rw")
        theta = float(job.get("walk_theta", 0.1))
        mit_cfg = mitigation_config  # riferimento condiviso: mutato in place

        def noise_update_fn(model, epoch):
            p_t, r_t = drifted_probs(p0, r0, epoch, walk_sigma, wseed, dtype_, theta)
            _set_decoding_noise(model, build_noise_model_centralized(p_t, r_t))
            if mit_cfg is not None:
                single = np.array([[1 - r_t, r_t], [r_t, 1 - r_t]])
                mit_cfg["method_kwargs"]["readout"]["response_matrix"] = np.kron(single, single)
            return (p_t, r_t)

    nshots = job["nshots"]
    model = create_model(
        model_type=job.get("model_type", "quantum"),
        nshots=nshots,
        noise_model=noise_model,
        mitigation_config=mitigation_config,
        hidden_classical=job.get("hidden_classical"),
    )

    if job["mode"] == "noiseless":
        tag = f"[noiseless s{job['seed']}]"
    else:
        tag = f"[{job['mode']} p={job['pauli_prob']} s{job['seed']}]"

    epochs_data, weights, elapsed, total_calibrations = train_and_evaluate(
        model,
        ndata_train=job["ndata"],
        seed=job["data_seed"],
        epochs=job["epochs"],
        lr=job["lr"],
        batch_size=job["batch_size"],
        job_tag=tag,
        noise_update_fn=noise_update_fn,
    )

    # Salva JSON
    result = {
        "info": {
            "mode": job["mode"],
            "seed": job["seed"],
            "pauli_prob": job["pauli_prob"],
            "readout_prob": job["readout_prob"],
            "nshots": job["nshots"],
            "epochs": job["epochs"],
            "lr": job["lr"],
            "batch_size": job["batch_size"],
            "ndata": job["ndata"],
            "cdr_threshold": job.get("cdr_threshold", 0.015),
            "with_memory": job.get("with_memory", False),
            "memory_beta": job.get("memory_beta", 0.2) if job.get("with_memory") else None,
            "walk_sigma": job.get("walk_sigma", 0.0),
            "walk_seed": job.get("walk_seed", 7),
            "drift_type": job.get("drift_type", "rw"),
            "walk_theta": job.get("walk_theta", 0.1),
            "total_calibrations": total_calibrations,
            "elapsed_seconds": elapsed,
        },
        "epochs": epochs_data,
    }

    save_dir = Path(job["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"lr_{job['lr']}_seed{job['seed']}"
    json_path = save_dir / f"{file_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=3)

    # Salva pesi
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.savez(weights_dir / f"{file_prefix}.npz", weights=weights)

    label = f"{job['mode']} lr={job['lr']} p={job['pauli_prob']} seed={job['seed']}"
    print(f"  [DONE]  {label} ({elapsed:.0f}s)")

    del model
    gc.collect()
    return ("ok", label, elapsed)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, nargs="+", default=["mitigated"],
                        choices=["noiseless", "noisy", "mitigated"],
                        help="Uno o più modi. Es: --mode noisy mitigated")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, nargs="+", default=[0.3])
    parser.add_argument("--ndata", type=int, default=500)
    parser.add_argument("--nshots", type=str, default="1000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--data-seed", type=int, default=2,
                        help="Seed per la generazione dei dati (fisso). Default: 2.")
    parser.add_argument("--model-type", type=str, default="quantum",
                        choices=["quantum", "hybrid", "classical"],
                        help="Tipo di modello: quantum, hybrid, classical.")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Numero di hidden units per il modello classico.")
    parser.add_argument("--cdr-threshold", type=float, default=0.015,
                        help="Soglia CDR per mitigazione. Default: 0.015.")
    parser.add_argument("--with-memory", action="store_true", default=False,
                        help="Abilita EMA sulla ricalibrazione CDR.")
    parser.add_argument("--memory-beta", type=float, default=0.2,
                        help="Coefficiente EMA per la memoria CDR. Default: 0.2.")
    parser.add_argument("--walk-sigma", type=float, default=0.001,
                        help="Sigma del drift per epoca (0 = niente drift). Default: 0.001.")
    parser.add_argument("--walk-seed", type=int, default=None,
                        help="Seed per il drift. Default: None (usa il seed della run).")
    parser.add_argument("--drift-type", type=str, default="ou", choices=["rw", "ou"])
    parser.add_argument("--walk-theta", type=float, default=0.3,
                        help="Mean reversion (solo drift-type=ou). Default: 0.3.")
    parser.add_argument("--save-root", type=str, default="results/fixed_training_set_results/noiseless_noisy_mit_fixed_results/centralized/quantum")
    args = parser.parse_args()
    args.nshots = None if args.nshots.lower() == "none" else int(args.nshots)

    print(f">>> Platform: {platform.system()}")
    print(f">>> Workers: {args.workers}")
    print(f">>> Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f">>> LR values: {args.lr}")
    print(f">>> Seeds: {args.seeds}")
    print(f">>> Nshots: {args.nshots}")
    print(f">>> Model type: {args.model_type}")
    if args.walk_sigma > 0:
        print(f">>> Drift: {args.drift_type}, sigma={args.walk_sigma}, seed={args.walk_seed}")
    print()

    jobs = []

    # -----------------------------------------------------------------
    # TEST LEARNING RATE — 9 run: 3 seed x {noiseless, noisy, mitigated}
    #
    # Serve a verificare perche' nel centralizzato la curva mitigata sia
    # PIU' LONTANA dal noiseless di quella noisy. Ipotesi: la CDR divide
    # per il fattore di smorzamento e cosi' amplifica anche il rumore di
    # shot, allargando la palla di rumore dell'ottimizzatore a lr fisso.
    # Se e' cosi', abbassando lr la distanza del mitigato deve scendere
    # mentre quella del noisy (gia' smorzata e stabile) cambia poco.
    #
    # Il noiseless DEVE girare allo stesso lr degli altri due: definisce
    # il punto di riferimento da cui si misurano le distanze.
    #
    # Layout di salvataggio identico a parameters_centralized/lr_0.3,
    # cosi' gli script di analisi esistenti funzionano senza modifiche.
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Selettore: quale campagna costruire.
    #   "test_lr"   -> test sul learning rate, 9 run a p = 0.022 (il vecchio
    #                  blocco, con seed e livello di rumore cablati)
    #   "rw_memory" -> memoria CDR sotto random walk, 42 run
    # -----------------------------------------------------------------
    JOBS_MODE = "rw_memory"

    TEST_ROOT = f"parameters_centralized/TEST_lr{args.lr[0]}"
    TEST_P = 0.022          # livello dove l'effetto e' massimo
    TEST_SEEDS = [1, 2, 3]

    base_job = {
        "data_seed": args.data_seed,
        "nshots": args.nshots,
        "epochs": args.epochs,
        "lr": args.lr[0],
        "batch_size": args.batch_size,
        "ndata": args.ndata,
        "model_type": args.model_type,
        "cdr_threshold": args.cdr_threshold,
        "with_memory": False,
        "memory_beta": 0.0,
        "walk_sigma": 0.0,      # nessun drift: confronto statico
        "drift_type": "ou",
        "walk_theta": 0.3,
    }

    for seed in (TEST_SEEDS if JOBS_MODE == "test_lr" else []):
        # Riferimento noiseless, stesso lr delle altre due.
        jobs.append({
            **base_job,
            "mode": "noiseless",
            "seed": seed,
            "walk_seed": seed,
            "pauli_prob": 0.0,
            "readout_prob": 0.0,
            "save_path": TEST_ROOT,
        })
        for mode in ("noisy", "mitigated"):
            jobs.append({
                **base_job,
                "mode": mode,
                "seed": seed,
                "walk_seed": seed,
                "pauli_prob": TEST_P,
                "readout_prob": TEST_P,
                "save_path": f"{TEST_ROOT}/{mode}/p{TEST_P}/nshots_{args.nshots}",
            })

    # -----------------------------------------------------------------
    # Memoria CDR sotto random walk: 3 soglie x 2 config x 7 seed = 42 run.
    #
    # sigma = 0.0005 e non 0.001: su 30 epoche un random walk a 0.001 porta
    # il rumore fuori dal dominio fisico per una frazione consistente delle
    # traiettorie, che finiscono clippate a zero. E' lo stesso sigma del
    # braccio federato RW a 30 round, cosi' i due lati della tabella dei
    # risparmi vedono la stessa intensita' di deriva.
    #
    # La configurazione con memoria e' l'EMA intra-round a beta 0.2: nel
    # centralizzato with_memory attiva esattamente quella, il mitigatore media
    # la mappa nuova con la precedente invece di sovrascriverla.
    # -----------------------------------------------------------------

    RW_ROOT = "FINAL_RW_CENTRALIZED"
    RW_P = 0.005
    RW_THRESHOLDS = [0.015, 0.05, 0.2]
    RW_SEEDS = [1, 2, 3, 4, 5, 6, 7]

    if JOBS_MODE == "rw_memory":
        rw_base = {
            "data_seed": args.data_seed,
            "nshots": args.nshots,
            "epochs": 30,
            "lr": args.lr[0],
            "batch_size": args.batch_size,
            "ndata": args.ndata,
            "model_type": args.model_type,
            "mode": "mitigated",
            "pauli_prob": RW_P,
            "readout_prob": RW_P,
            "walk_sigma": 0.0005,
            "drift_type": "rw",
            "walk_theta": 0.3,      # ignorato dal random walk
        }
        for thresh in RW_THRESHOLDS:
            label = str(thresh).replace(".", "")
            for with_memory, sub in ((False, "no_memory"), (True, "ema")):
                for seed in RW_SEEDS:
                    jobs.append({
                        **rw_base,
                        "seed": seed,
                        "walk_seed": seed,
                        "cdr_threshold": thresh,
                        "with_memory": with_memory,
                        "memory_beta": 0.2 if with_memory else None,
                        "save_path": f"{RW_ROOT}/thresh_{label}/{sub}",
                    })

    total = len(jobs)
    print(f"Totale jobs: {total}\n")

    start_global = time.time()
    results = []

    if args.workers > 1:
        print(f">>> POOL: {total} jobs su {args.workers} workers\n")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    status, label, elapsed = future.result()
                    results.append((status, label, elapsed))
                    print(f"  >>> Progress: {i}/{total}")
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    results.append(("error", str(e), 0))
    else:
        print(f">>> SEQUENZIALE: {total} jobs\n")
        for i, job in enumerate(jobs, 1):
            try:
                status, label, elapsed = run_single_job(job)
                results.append((status, label, elapsed))
                print(f"  >>> Progress: {i}/{total}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("error", str(e), 0))

    elapsed_global = time.time() - start_global
    ok = sum(1 for r in results if r[0] == "ok")
    skip = sum(1 for r in results if r[0] == "skip")
    fail = sum(1 for r in results if r[0] not in ("ok", "skip"))

    print(f"\n{'='*70}")
    print(f"COMPLETATO in {elapsed_global:.0f}s ({elapsed_global/3600:.2f} ore)")
    print(f"  OK:    {ok}/{total}")
    print(f"  SKIP:  {skip}/{total}")
    print(f"  FAIL:  {fail}/{total}")
    print(f"{'='*70}")
