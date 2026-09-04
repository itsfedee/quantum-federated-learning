import subprocess
import os
import sys
import time
import random
import argparse
import tempfile
import shutil
import platform
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from qibo_qfl_pt.experiment_utils import ExperimentPath

# =====================================================================
# Platform detection (fatta una volta sola, top-level)
# =====================================================================

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# =====================================================================
# Argomenti da riga di comando
# =====================================================================
# Esempi d'uso:
#   python parallel_experiments.py --strategy FedProx --workers 4
#   python parallel_experiments.py --strategy FedAdam --workers 6


# =====================================================================
# Strategie e iperparametri
# Top-level: i worker spawn-ati su Windows li importano.
# =====================================================================

iid_strategies = {
    "FedAvg":     (None,  None,   "eta_l", 0.3,    [1, 2, 3, 4, 5, 6, 7]),
  #  "FedProx":    ("mu",  0.03,   "eta_l", 0.3,    [1, 2, 3, 4, 5, 6, 7]),
  #  "FedAdagrad": ("eta", 0.3,    "eta_l", 0.2,    [1, 2, 3, 4, 5, 6, 7]),
 # "FedAdam":    ("eta", 0.2,    "eta_l", 0.15,   [1, 2, 3, 4, 5, 6, 7]),
   # "FedYogi":    ("eta", 0.1,    "eta_l", 0.1,    [1, 2, 3, 4, 5, 6, 7]),
}

non_iid_strategies = {
    "FedAvg":     (None,  None,   "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
    "FedProx":    ("mu",  0.03,   "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
    "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   [1, 2, 3, 4, 5, 6, 7]),
    "FedAdam":    ("eta", 0.1,    "eta_l", 0.1,   [1, 2, 3, 4, 5, 6, 7]),
    "FedYogi":    ("eta", 0.1,    "eta_l", 0.1,   [1, 2, 3, 4, 5, 6, 7]),
}


# =====================================================================
# Helper cross-platform per kill processi e cleanup
# =====================================================================

def kill_proc_tree(pid):
    """Killa un processo e tutti i suoi discendenti.
    Cross-platform. Su Windows usa taskkill /T, su Unix usa il process group."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    else:
        # Su Unix abbiamo creato un nuovo session group con start_new_session=True,
        # quindi tutti i discendenti condividono il process group ID = pid.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            # Se dopo SIGTERM e' ancora vivo, SIGKILL secco
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # gia' morto, ottimo
        except (ProcessLookupError, PermissionError):
            pass


def final_cleanup():
    """Pulizia finale: chiude ray e killa eventuali orfani.
    Cross-platform."""
    # Tentativo soft via comando ray
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
    except Exception:
        pass

    # Hard kill OS-specifico per orfani sopravvissuti
    if IS_WINDOWS:
        for image in ("ray.exe", "raylet.exe", "gcs_server.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        # pkill ritorna != 0 se non trova nulla, ed e' OK: capture_output ingoia tutto
        for pattern in ("raylet", "gcs_server", "ray::"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def cleanup_orphan_ray_dirs(label="startup"):
    """Rimuove le Ray temp dir orfane lasciate da run interrotte.

    Il cleanup per-job (la rmtree nel finally di run_single_job) non viene
    eseguito se il processo muore di SIGKILL, se cade la sessione ssh o se
    la macchina si riavvia: le dir restano nella temp dir e col tempo
    riempiono il disco. Su una macchina condivisa questo danneggia tutti,
    quindi puliamo all'avvio e alla fine di ogni sweep.

    Sicurezza: non tocca niente se ci sono ancora processi flwr/ray vivi
    (potrebbe essere uno sweep parallelo), e salta le dir di altri utenti.
    """
    tmp_root = tempfile.gettempdir()

    if not IS_WINDOWS:
        try:
            alive = subprocess.run(
                ["pgrep", "-f", "flwr|raylet|gcs_server"],
                capture_output=True, timeout=10,
            )
            if alive.returncode == 0:
                print(f">>> Cleanup {label}: processi flwr/ray ancora attivi, "
                      f"pulizia di {tmp_root} saltata per sicurezza.")
                return
        except Exception:
            pass

    uid = None if IS_WINDOWS else os.getuid()
    removed = 0
    freed = 0

    try:
        entries = os.listdir(tmp_root)
    except OSError:
        return

    for name in entries:
        if not (name.startswith("ray_") or name.startswith("warmup_dummy_")):
            continue
        path = os.path.join(tmp_root, name)
        if not os.path.isdir(path):
            continue
        try:
            # Non toccare le temp dir di altri utenti della macchina.
            if uid is not None and os.stat(path).st_uid != uid:
                continue
        except OSError:
            continue

        size = 0
        for root, _dirs, files in os.walk(path):
            for fname in files:
                try:
                    size += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass

        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            removed += 1
            freed += size

    if removed:
        print(f">>> Cleanup {label}: rimosse {removed} Ray temp dir orfane da "
              f"{tmp_root} ({freed / 1e9:.1f} GB liberati)")
    try:
        usage = shutil.disk_usage(tmp_root)
        print(f">>> Spazio libero su {tmp_root}: {usage.free / 1e9:.1f} GB "
              f"/ {usage.total / 1e9:.1f} GB")
    except Exception:
        pass


def raise_fd_limit_if_possible():
    """Su Unix con tanti workers serve alzare il limit dei file descriptor.
    Su Windows non serve (Windows non ha questo limite)."""
    if IS_WINDOWS:
        return
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f">>> File descriptor limit alzato: {soft} -> {target}")
    except Exception as e:
        print(f">>> WARN: impossibile alzare fd limit ({e}). Se hai workers>16, "
              f"fai 'ulimit -n 65536' prima di lanciare lo script.")


# =====================================================================
# Funzione per eseguire un singolo job
# Top-level: deve essere importabile dai worker (spawn su Windows).
# =====================================================================

def run_single_job(job, timeout_sec, stagger_max=0.0):
    # Stagger casuale per evitare che tutti i worker bombardino il SuperLink
    # nello stesso istante. Il SuperLink locale ci mette ~15s a essere
    # operativo a freddo, e il client flwr CLI ha un timeout di 15s sulla
    # connessione iniziale: con N client simultanei almeno uno sfora.
    if stagger_max > 0:
        time.sleep(random.uniform(0, stagger_max))

    seed = job["seed"]
    parts = [
        f'strategy="{job["strategy"]}"',
        f'seed={seed}',
        f'data-seed={job.get("data_seed", 2)}',
        f'init-seed={job.get("init_seed", seed)}',
        f'sampling-seed={job.get("sampling_seed", seed)}',
        f'mode="{job["mode"]}"',
        f'model-type="{job["model_type"]}"',
        f'base-pauli={job["base_pauli"]}',
        f'base-readout={job["base_readout"]}',
        f'scale={job["scale"]}',
        f'nshots="{job["nshots"]}"' if isinstance(job["nshots"], str) else (f'nshots={job["nshots"]}' if job["nshots"] is not None else 'nshots="none"'),
        f'save-path="{job["save_path"]}"',

    ]
    if job.get("num_rounds"):
        parts.append(f'num-server-rounds={job["num_rounds"]}')
    if job.get("cdr_threshold") is not None:
        parts.append(f'cdr-threshold={job["cdr_threshold"]}')
    if job.get("seed_label"):
        parts.append(f'seed-label="{job["seed_label"]}"')
    if job["srv_name"] is not None:
        parts.append(f'{job["srv_name"]}={job["srv_val"]}')
    if job["cli_name"] is not None:
        parts.append(f'{job["cli_name"]}={job["cli_val"]}')
    if job.get("hidden_classical"):
        parts.append(f'hidden-classical={job["hidden_classical"]}')
    if job.get("num_clients"):
        parts.append(f'num-clients={job["num_clients"]}')
        parts.append(f'num-partitions={job["num_clients"]}')
    if job.get("n_train_data"):
        parts.append(f'n-train-data={job["n_train_data"]}')
    if job.get("fraction_fit") is not None:
        parts.append(f'fraction-fit={job["fraction_fit"]}')
    if job.get("fraction_evaluate") is not None:
        parts.append(f'fraction-evaluate={job["fraction_evaluate"]}')
    if job.get("walk_sigma") is not None:
        parts.append(f'walk-sigma={job["walk_sigma"]}')
    if job.get("walk_seed") is not None:
        parts.append(f'walk-seed={job["walk_seed"]}')
    if job.get("drift_type"):
        parts.append(f'drift-type="{job["drift_type"]}"')
    if job.get("walk_theta") is not None:
        parts.append(f'walk-theta={job["walk_theta"]}')
    if job.get("cdr_mode"):
        parts.append(f'cdr-mode="{job["cdr_mode"]}"')
    if job.get("cdr_beta") is not None and job.get("cdr_mode") == "ema":
        parts.append(f'cdr-beta={job["cdr_beta"]}')
    if job.get("cdr_intra_beta") is not None and job["cdr_intra_beta"] > 0:
        parts.append(f'cdr-intra-beta={job["cdr_intra_beta"]}')
    if job["distribution"] == "non_iid":
        parts.append('iid=false')
        parts.append('alpha=1.8')

    config_string = " ".join(parts)

    # Ambiente custom per questo job.
    # NON impostiamo FLWR_HOME per-job: il SuperLink locale managed di
    # Flower e' uno solo per macchina, condiviso fra tutti i flwr run
    # paralleli, ed e' progettato per gestire run concorrenti.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['RAY_DEDUP_LOGS'] = '0'
    env['RAY_memory_monitor_refresh_ms'] = '0'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'

    # RAY_TMPDIR univoco per non incrociare i Ray work dirs fra job.
    ray_tmp = tempfile.mkdtemp(prefix=f"ray_{job['strategy']}_seed{seed}_")
    env['RAY_TMPDIR'] = ray_tmp

    job_label = (
        f"{job['strategy']} seed={seed} {job['mode']} "
        f"{job['noise_label']} nshots={job['nshots']}"
    )
    start_time = time.time()
    print(f"  [START] {job_label}")

    # Argomenti Popen cross-platform.
    # Su Unix start_new_session=True crea un nuovo process group, cosi'
    # possiamo killare tutti i discendenti via os.killpg.
    popen_kwargs = dict(
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    if not IS_WINDOWS:
        popen_kwargs["start_new_session"] = True

    tag = f"[{job['strategy']} s{seed}]"
    proc = None
    try:
        try:
            proc = subprocess.Popen(
                ["flwr", "run", ".", "--run-config", config_string, "--stream"],
                **popen_kwargs,
            )
            deadline = time.time() + timeout_sec
            for line in proc.stdout:
                line = line.rstrip()
                # Stampa solo righe utili: round, metriche, errori
                if any(k in line for k in ("[ROUND", "Server eval", "Server train eval", "ERROR", "aggregate_", "FAIL")):
                    print(f"  {tag} {line}", flush=True)
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(proc.args, timeout_sec)
            proc.wait()
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                print(f"  [DONE]  {job_label} ({elapsed:.0f}s)")
                return ("ok", job_label, elapsed)
            else:
                print(f"  [FAIL]  {job_label} ({elapsed:.0f}s) returncode={proc.returncode}")
                return ("fail", job_label, elapsed)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"  [TIMEOUT] {job_label} ({elapsed:.0f}s)")
            if proc and proc.pid:
                kill_proc_tree(proc.pid)
            return ("timeout", job_label, elapsed)

    finally:
        # Sicurezza: se il processo per qualche motivo e' ancora vivo, killalo
        if proc and proc.poll() is None:
            kill_proc_tree(proc.pid)
        # Pausa per rilascio file handles (su Windows e' indispensabile,
        # su Unix non fa male)
        time.sleep(2)
        shutil.rmtree(ray_tmp, ignore_errors=True)


# =====================================================================
# Setup + esecuzione: TUTTO sotto __main__ (spawn su Windows).
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Se specificato, lancia solo quella strategia. Altrimenti tutte.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Quante run lanciare in parallelo. Default: 1."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Timeout per singola run, in secondi. Default: 21600 (6 ore)."
    )
    args = parser.parse_args()

    print(f">>> Platform: {platform.system()} ({platform.machine()})")

    # Pulizia dei residui lasciati da run precedenti interrotte, prima di
    # occupare altro spazio con questo sweep.
    cleanup_orphan_ray_dirs("avvio")

    # Su Linux con tanti workers conviene alzare il fd limit
    if args.workers >= 8:
        raise_fd_limit_if_possible()

    # Stagger massimo all'avvio dei job nel pool. Per workers=1 non serve.
    # Per workers>=2 vogliamo distribuire l'avvio su qualche secondo cosi'
    # i client non bombardano il SuperLink simultaneamente.
    pool_stagger = 0.0 if args.workers == 1 else 5.0 * args.workers

    # Validazione strategia
    if args.strategy is not None:
        if args.strategy not in iid_strategies:
            print(f"ERRORE: strategia '{args.strategy}' non riconosciuta.")
            print(f"Strategie valide: {list(iid_strategies.keys())}")
            sys.exit(1)
        strategies_to_use = [args.strategy]
        print(f"\n>>> Modalità singola strategia: {args.strategy}")
    else:
        strategies_to_use = list(iid_strategies.keys())
        print(f"\n>>> Modalità completa: tutte le strategie")

    print(f">>> Workers paralleli: {args.workers}")
    print(f">>> Timeout per run: {args.timeout}s")
    print(f">>> Pool stagger max: {pool_stagger:.1f}s\n")

    # =================================================================
    # Configurazione
    # =================================================================

    SEEDS = [1, 2, 3, 4, 5, 6, 7]

    NOISE_BASE_005 = {"base_pauli": 0.005, "base_readout": 0.005, "scale": 0.002,
                      "walk_sigma": 0.0, "walk_seed": 7}
    WALK_SEED_MAP = {1: 1, 2: 11, 3: 12, 4: 20, 5: 21, 6: 22, 7: 24}
    WALK_SEED_MAP_10 = {1: 1, 2: 11, 3: 12, 4: 20, 5: 21, 6: 22, 7: 24,
                        8: 30, 9: 31, 10: 32}

    runs = []

    # =================================================================
    # 5 strategie IID noiseless noshots — 7 seed, 40 round, HP scelti
    # =================================================================

    all_strategies_iid = {
        "FedAvg":     (None,  None,  "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   [1, 2, 3, 4, 5, 6, 7]),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.15,  [1, 2, 3, 4, 5, 6, 7]),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.1,   [1, 2, 3, 4, 5, 6, 7]),
    }







    # =================================================================
    # FedAvg IID mitigated statico p=0.005, soglie alte (0.03, 0.05)
    # COMPLETATE — non rilanciare
    # =================================================================

    # for thresh in [0.03, 0.05]:
    #     thresh_label = str(thresh).replace(".", "")
    #     runs.append({
    #         "distribution": "iid", "mode": "mitigated",
    #         "model_type": "quantum",
    #         **NOISE_BASE_005,
    #         "nshots": 1000,
    #         "num_rounds": 10,
    #         "cdr_threshold": thresh,
    #         "save_path_override": f"static_calibrations/thresh_{thresh_label}",
    #         "strategies": {
    #             "FedAvg": (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7]),
    #         },
    #     })

    # =================================================================
    # Drift sigma sweep: FedAvg IID mitigated, p=0.005, soglie 0.03/0.05
    # sigma_w in {0.003, 0.01} — crossover characterization
    # =================================================================

    # for sigma_w in [0.003, 0.01]:
    #     sigma_label = str(sigma_w).replace(".", "")
    #     for thresh in [0.03, 0.05]:
    #         thresh_label = str(thresh).replace(".", "")
    #         runs.append({
    #             "distribution": "iid", "mode": "mitigated",
    #             "model_type": "quantum",
    #             **NOISE_BASE_005,
    #             "walk_sigma": sigma_w,
    #             "walk_seed_from_seed": True,
    #             "walk_seed_map": WALK_SEED_MAP,
    #             "nshots": 1000,
    #             "num_rounds": 10,
    #             "cdr_threshold": thresh,
    #             "save_path_override": f"drift_sigma_sweep/sigma_{sigma_label}/thresh_{thresh_label}",
    #             "strategies": {
    #                 "FedAvg": (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7]),
    #             },
    #         })

    # =================================================================
    # Sweep soglie OU federato, 40 round, FedAvg IID mitigated
    # 6 soglie × 2 arm × 7 seed = 84 run
    # =================================================================

    ou_sweep_thresholds = [0.005, 0.015, 0.03, 0.05, 0.1, 0.2]


    # =================================================================
    # Retry ema_full corrotti in THRESH_SWEEP_FEDERATED_OU — 11 run
    # thresh_0015/seed3, thresh_01/seed5,6,7, thresh_02/seed1-7
    # =================================================================

    ema_full_retries = {
        0.015: [3],
        0.1:   [5, 6, 7],
        0.2:   [1, 2, 3, 4, 5, 6, 7],
    }

    for thresh, seeds in ema_full_retries.items():
        thresh_label = str(thresh).replace(".", "")
        runs.append({
            "distribution": "iid", "mode": "mitigated",
            "model_type": "quantum",
            **NOISE_BASE_005,
            "walk_sigma": 0.001,
            "walk_seed_from_seed": True,
            "walk_seed_map": WALK_SEED_MAP,
            "drift_type": "ou",
            "walk_theta": 0.3,
            "nshots": 1000,
            "num_rounds": 40,
            "cdr_threshold": thresh,
            "cdr_mode": "ema",
            "cdr_beta": 0.2,
            "cdr_intra_beta": 0.2,
            "save_path_override": f"THRESH_SWEEP_FEDERATED_OU/thresh_{thresh_label}/ema_full",
            "strategies": {
                "FedAvg": (None, None, "eta_l", 0.3, seeds),
            },
        })

    # =================================================================
    # 4 strategie (no FedAvg) IID, drift OU, 10 round, p=0.005
    # noisy + mitigated (thresh 0.015, no_memory)
    # 4 strat × 2 mode × 7 seed = 56 run
    # =================================================================

    other_strategies = {
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   [1, 2, 3, 4, 5, 6, 7]),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.15,  [1, 2, 3, 4, 5, 6, 7]),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.1,   [1, 2, 3, 4, 5, 6, 7]),
    }

    # Noisy (no mitigation) con drift OU, 10 round
    runs.append({
        "distribution": "iid", "mode": "noisy",
        "model_type": "quantum",
        **NOISE_BASE_005,
        "walk_sigma": 0.001,
        "walk_seed_from_seed": True,
        "walk_seed_map": WALK_SEED_MAP,
        "drift_type": "ou",
        "walk_theta": 0.3,
        "nshots": 1000,
        "num_rounds": 10,
        "save_path_override": "strategies_OU_drift_10r/{strategy}/noisy",
        "strategies": other_strategies,
    })

    # Mitigated con drift OU, 10 round
    runs.append({
        "distribution": "iid", "mode": "mitigated",
        "model_type": "quantum",
        **NOISE_BASE_005,
        "walk_sigma": 0.001,
        "walk_seed_from_seed": True,
        "walk_seed_map": WALK_SEED_MAP,
        "drift_type": "ou",
        "walk_theta": 0.3,
        "nshots": 1000,
        "num_rounds": 10,
        "cdr_threshold": 0.015,
        "cdr_mode": "no_memory",
        "save_path_override": "strategies_OU_drift_10r/{strategy}/mitigated",
        "strategies": other_strategies,
    })

    # =================================================================
    # TEST RUMORE OMOGENEO — controllo per la 4.3.2
    #
    # Stessa configurazione delle run noisy a p=0.022 che abbiamo già
    # (FedAvg IID, 10 round, 5 client, nshots=1000, eta_l=0.3, no drift),
    # ma con scale=0: tutti i client hanno lo STESSO rumore invece di
    # riceverlo distribuito attorno alla base.
    #
    # Serve a isolare l'eterogeneità dal rumore: se la distanza dei pesi
    # dal noiseless scende ai livelli del centralizzato (~0.24 rad invece
    # di ~0.98), la diffusione è causata dalla disomogeneità fra client e
    # non dal rumore in sé. Il riferimento noiseless è già disponibile e
    # non va rifatto: stessi seed, stessa configurazione.
    # =================================================================

    # Per eseguire solo questa run usa RUN_ONLY = "homogeneous" piu' sotto.

    homogeneous_noise_run = {
        "distribution": "iid", "mode": "noisy",
        "model_type": "quantum",
        "base_pauli": 0.022, "base_readout": 0.022,
        "scale": 0.0,                       # <-- rumore identico su tutti i client
        "walk_sigma": 0.0,                  # nessun drift, confronto statico
        "nshots": 1000,
        "num_rounds": 10,
        "save_path_override": (
            "NEW_HOMOGENEUS_NOISE"
        ),
        "strategies": {"FedAvg": (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7])},
    }
    runs.append(homogeneous_noise_run)

    # =================================================================
    # DRIFT OU NON-IID — noisy e mitigated per tutte e cinque le strategie
    #
    # Speculare ai due blocchi IID sopra: stessa base di rumore, stesso OU
    # (sigma=0.001, theta=0.3), 10 round, soglia 0.015 con no_memory per il
    # mitigato. Cambia solo la distribuzione dei dati, cosi' il confronto
    # iid/non-iid e' controllato.
    #
    # Le strategie usano non_iid_strategies, che ha iperparametri propri
    # (eta/eta_l diversi da quelli IID): sono i valori gia' tarati per il
    # non-IID, non vanno allineati a quelli IID.
    #
    # distribution="non_iid" aggiunge da solo iid=false e alpha=1.8 alla
    # config string (riga ~256).
    # =================================================================

    non_iid_drift_runs = []
    for mode in ("noisy", "mitigated"):
        run = {
            "distribution": "non_iid", "mode": mode,
            "model_type": "quantum",
            **NOISE_BASE_005,
            "walk_sigma": 0.001,
            "walk_seed_from_seed": True,
            "walk_seed_map": WALK_SEED_MAP,
            "drift_type": "ou",
            "walk_theta": 0.3,
            "nshots": 1000,
            "num_rounds": 10,
            "save_path_override": f"strategies_OU_drift_10r/non_iid/{{strategy}}/{mode}",
            "strategies": non_iid_strategies,
        }
        if mode == "mitigated":
            run["cdr_threshold"] = 0.015
            run["cdr_mode"] = "no_memory"
        non_iid_drift_runs.append(run)
    runs.extend(non_iid_drift_runs)

    # =================================================================
    # DRIFT FORTE — regime in cui le ricalibrazioni sono dominate dal drift
    #
    # A sigma=0.001 il drift NON contribuisce: la statistica di scatto D e'
    # identica al caso statico (17.5% contro 18.3% oltre soglia a 0.05).
    # Qui alziamo sigma per ottenere un regime in cui il drift domina, da
    # confrontare con il controllo statico gia' disponibile a soglia 0.05
    # (tasso 1.27) e con l'OU a sigma=0.001 (tasso 1.28).
    #
    # ATTENZIONE, da dichiarare in tesi: un OU con theta=0.3 ha deviazione
    # stazionaria sigma/sqrt(2*theta). Su base 0.005 questo significa che il
    # processo passa sotto zero circa il 10% del tempo a sigma=0.003 e il 22%
    # a sigma=0.005, e viene quindi troncato. Non sono configurazioni
    # fisicamente realistiche: servono a dimostrare che il meccanismo risponde
    # al drift quando il drift e' abbastanza grande, non a modellare hardware.
    #
    # Soglia 0.05 e 10 round: il tasso e' stazionario (verificato sulle run a
    # 40 round), quindi l'orizzonte breve misura la stessa grandezza.
    # =================================================================

    STRONG_DRIFT_SIGMAS = [0.003, 0.005]

    strong_drift_runs = []
    for sigma_w in STRONG_DRIFT_SIGMAS:
        sigma_label = str(sigma_w).replace(".", "")
        strong_drift_runs.append({
            "distribution": "iid", "mode": "mitigated",
            "model_type": "quantum",
            **NOISE_BASE_005,
            "walk_sigma": sigma_w,
            "walk_seed_from_seed": True,
            "walk_seed_map": WALK_SEED_MAP,
            "drift_type": "ou",        # esplicito: senza, il default e' "rw"
            "walk_theta": 0.3,
            "nshots": 1000,
            "num_rounds": 10,
            "cdr_threshold": 0.05,
            "cdr_mode": "no_memory",
            "save_path_override": f"STRONG_DRIFT_OU/sigma_{sigma_label}/thresh_005",
            "strategies": {
                "FedAvg": (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7]),
            },
        })
    runs.extend(strong_drift_runs)

    # =================================================================
    # Terzo arm della campagna RW: ema_full sullo sweep di soglia
    # 5 soglie x 7 seed = 35 run
    #
    # ema_full significa EMA su entrambi i livelli: fra un round e il
    # successivo (cdr_beta) e dentro il singolo refit (cdr_intra_beta),
    # come nell'omologo OU. Configurazione identica agli altri due bracci
    # della stessa campagna RW, walk_seed compresi, cosi' i tre bracci
    # vedono la stessa traiettoria di rumore e restano appaiabili per seed.
    #
    # sigma = 0.0005 e non 0.001: su 30 round un random walk a 0.001 porta
    # un terzo delle traiettorie a rumore nullo per clipping, e i client
    # diventerebbero di fatto noiseless.
    # =================================================================

    rw_ema_full_runs = []
    for thresh in [0.005, 0.015, 0.03, 0.05, 0.2]:
        thresh_label = str(thresh).replace(".", "")
        rw_ema_full_runs.append({
            "distribution": "iid", "mode": "mitigated",
            "model_type": "quantum",
            **NOISE_BASE_005,
            "walk_sigma": 0.0005,
            "walk_seed_from_seed": True,
            "walk_seed_map": WALK_SEED_MAP,
            "drift_type": "rw",
            "nshots": 1000,
            "num_rounds": 30,
            "cdr_threshold": thresh,
            "cdr_mode": "ema",
            "cdr_beta": 0.2,
            "cdr_intra_beta": 0.2,
            "save_path_override":
                f"FINAL_RW_SWEEP/thresh_{thresh_label}/ema_full",
            "strategies": {
                "FedAvg": (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7]),
            },
        })
    runs.extend(rw_ema_full_runs)

    # -----------------------------------------------------------------
    # Selettore: quale sottoinsieme eseguire.
    #   "all"           -> tutte le run definite sopra
    #   "homogeneous"   -> test a rumore omogeneo (7 run)
    #   "strong_drift"  -> test a drift forte (14 run)
    #   "non_iid_drift" -> drift OU non-IID, 5 strategie x 7 seed x 2 modi (70 run)
    #   "rw_ema_full"   -> terzo arm della campagna RW, 5 soglie x 7 seed (35 run)
    # -----------------------------------------------------------------
    RUN_ONLY = "rw_ema_full"

    if RUN_ONLY == "homogeneous":
        runs = [homogeneous_noise_run]
    elif RUN_ONLY == "strong_drift":
        runs = strong_drift_runs
    elif RUN_ONLY == "non_iid_drift":
        runs = non_iid_drift_runs
    elif RUN_ONLY == "rw_ema_full":
        runs = rw_ema_full_runs

    # =================================================================
    # Costruzione lista jobs (flatten)
    # =================================================================

    jobs = []
    for run in runs:
        noise_label = ExperimentPath.noise_label(
            run["base_pauli"], run["base_readout"], run["scale"]
        )
        # Supporta sia dict che list di tuple per le strategie
        strat_items = run["strategies"].items() if isinstance(run["strategies"], dict) else run["strategies"]
        for strategy, (srv_name, srv_val, cli_name, cli_val, seeds) in strat_items:
            override = run.get("save_path_override", "")
            save_path = override.format(strategy=strategy.lower()) if override else ExperimentPath.build(
                distribution=run["distribution"],
                strategy=strategy,
                mode=run["mode"],
                noise=noise_label,
                nshots=run["nshots"],
            )
            seed_overrides = run.get("seed_overrides", {})
            seed_label_prefix = run.get("seed_label_prefix", "")
            seed_label_source = run.get("seed_label_source", "")

            fixed_training = run.get("fixed_training_set", True)

            for seed in seeds:
                job = {
                    "strategy": strategy,
                    "seed": seed,
                    "mode": run["mode"],
                    "model_type": run.get("model_type", "quantum"),
                    "base_pauli": run["base_pauli"],
                    "base_readout": run["base_readout"],
                    "scale": run["scale"],
                    "nshots": run["nshots"],
                    "save_path": save_path,
                    "srv_name": srv_name,
                    "srv_val": srv_val,
                    "cli_name": cli_name,
                    "cli_val": cli_val,
                    "distribution": run["distribution"],
                    "noise_label": noise_label,
                    "data_seed": 2 if fixed_training else seed,
                    "num_rounds": run.get("num_rounds"),
                    "cdr_threshold": run.get("cdr_threshold"),
                    "hidden_classical": run.get("hidden_classical"),
                    "num_clients": run.get("num_clients"),
                    "n_train_data": run.get("n_train_data"),
                    "fraction_fit": run.get("fraction_fit"),
                    "fraction_evaluate": run.get("fraction_evaluate"),
                    "walk_sigma": run.get("walk_sigma"),
                    "walk_seed": run.get("walk_seed_map", {}).get(seed, seed) if run.get("walk_seed_from_seed") else run.get("walk_seed"),
                    "drift_type": run.get("drift_type"),
                    "walk_theta": run.get("walk_theta"),
                    "cdr_mode": run.get("cdr_mode"),
                    "cdr_beta": run.get("cdr_beta"),
                    "cdr_intra_beta": run.get("cdr_intra_beta"),
                }

                # Seed isolation: i seed in overrides sono fissi,
                # quelli NON in overrides variano con seed.
                if seed_overrides:
                    job["init_seed"] = seed_overrides.get("init_seed", seed)
                    job["data_seed"] = seed_overrides.get("data_seed", seed)
                    job["sampling_seed"] = seed_overrides.get("sampling_seed", seed)

                if seed_label_prefix:
                    # Determina il valore del seed che varia per il label
                    if seed_label_source == "init_seed":
                        vary_val = job.get("init_seed", seed)
                    elif seed_label_source == "data_seed":
                        vary_val = job.get("data_seed", seed)
                    elif seed_label_source == "sampling_seed":
                        vary_val = job.get("sampling_seed", seed)
                    else:
                        vary_val = seed
                    job["seed_label"] = f"{seed_label_prefix}{vary_val}"

                jobs.append(job)

    total = len(jobs)
    print(f"Totale jobs da eseguire: {total}\n")

    # =================================================================
    # Esecuzione: warmup dummy + pool parallelo
    # =================================================================

    print(f"{'='*70}")
    print(f"Avvio {total} jobs ({args.workers} workers paralleli dopo warmup)")
    print(f"{'='*70}\n")

    start_global = time.time()
    results = []
    completed = 0

    if total == 0:
        sys.exit(0)

    try:
        # ---- Warmup dummy ----
        # Un job noiseless da 1 round che scalda il SuperLink di Flower
        # (~15s a freddo) senza sprecare un job vero. Il risultato non
        # viene salvato (save_path punta a una dir temporanea cancellata
        # subito dopo).
        warmup_tmp = tempfile.mkdtemp(prefix="warmup_dummy_")
        warmup_job = {
            "strategy": "FedAvg",
            "seed": 1,
            "mode": "noiseless",
            "model_type": "quantum",
            "base_pauli": 0.0,
            "base_readout": 0.0,
            "scale": 0.0,
            "nshots": "none",
            "save_path": warmup_tmp,
            "srv_name": None,
            "srv_val": None,
            "cli_name": "eta_l",
            "cli_val": 0.3,
            "distribution": "iid",
            "noise_label": "noiseless",
            "data_seed": 2,
            "num_rounds": 1,
            "cdr_threshold": None,
            "hidden_classical": None,
            "num_clients": None,
            "n_train_data": None,
            "fraction_fit": None,
            "fraction_evaluate": None,
            "walk_sigma": None,
            "walk_seed": None,
            "drift_type": None,
            "walk_theta": None,
            "cdr_mode": None,
            "cdr_beta": None,
            "cdr_intra_beta": None,
        }
        print(">>> WARMUP: job dummy noiseless 1-round per scaldare il SuperLink")
        warmup_status, warmup_label, warmup_elapsed = run_single_job(
            warmup_job, timeout_sec=300, stagger_max=0.0
        )
        print(f"  >>> Warmup done ({warmup_status}) in {warmup_elapsed:.0f}s")
        # Cleanup: il warmup non deve lasciare traccia
        shutil.rmtree(warmup_tmp, ignore_errors=True)

        # ---- Pool: TUTTI i job in parallelo ----
        pool_jobs = jobs

        if pool_jobs and args.workers > 1:
            print(f">>> POOL: {len(pool_jobs)} jobs restanti su {args.workers} workers\n")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_single_job, job, args.timeout, pool_stagger): job
                    for job in pool_jobs
                }
                for future in as_completed(futures):
                    completed += 1
                    try:
                        status, label, elapsed = future.result()
                        results.append((status, label, elapsed))
                        print(f"  >>> Progress: {completed}/{total} done")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        results.append(("error", str(e), 0))
        elif pool_jobs:
            # workers=1: tutto in sequenza, niente pool
            print(f">>> SEQUENZIALE: {len(pool_jobs)} jobs restanti su 1 worker\n")
            for job in pool_jobs:
                status, label, elapsed = run_single_job(job, args.timeout, stagger_max=0.0)
                results.append((status, label, elapsed))
                completed += 1
                print(f"  >>> Progress: {completed}/{total} done")

        elapsed_global = time.time() - start_global

        # =================================================================
        # Sommario finale
        # =================================================================

        ok = sum(1 for r in results if r[0] == "ok")
        fail = sum(1 for r in results if r[0] == "fail")
        timeout_count = sum(1 for r in results if r[0] == "timeout")
        error = sum(1 for r in results if r[0] == "error")

        print(f"\n{'='*70}")
        print(f"COMPLETATO in {elapsed_global:.0f}s ({elapsed_global/3600:.2f} ore)")
        print(f"  OK:      {ok}/{total}")
        print(f"  FAIL:    {fail}/{total}")
        print(f"  TIMEOUT: {timeout_count}/{total}")
        print(f"  ERROR:   {error}/{total}")
        print(f"{'='*70}\n")

        if fail > 0 or timeout_count > 0 or error > 0:
            print("Jobs falliti/timeout:")
            for status, label, elapsed in results:
                if status in ("fail", "timeout", "error"):
                    print(f"  [{status.upper()}] {label}")

    finally:
        # Pulizia finale, ESEGUITA SEMPRE — anche se l'utente fa Ctrl+C,
        # anche se qualcosa esplode dentro il pool.
        print("\n>>> Pulizia finale (Ray + processi orfani)...")
        final_cleanup()
        # I processi Ray sono morti: ora le temp dir rimaste sono tutte
        # orfane e si possono rimuovere senza rischi.
        cleanup_orphan_ray_dirs("finale")
        print(">>> Done.")
