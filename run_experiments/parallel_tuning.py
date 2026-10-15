"""
Hyperparameter tuning: a grid of (server, client) learning parameters for every
strategy, on noiseless runs evaluated on the validation set (testing=false).

Usage:
  python parallel_tuning.py --distribution non_iid --workers 8
  python parallel_tuning.py --distribution iid --strategy FedProx --workers 4
  python parallel_tuning.py --model-type classical --classical-variant all --workers 8
  python parallel_tuning.py --circuit 2q --dataset circle --dry-run

The results land in <save-root>_<distribution>/, with a subfolder for every
setup that is not the default one (see build_save_path).
"""

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

# =====================================================================
# Platform detection
# =====================================================================

IS_WINDOWS = platform.system() == "Windows"


# =====================================================================
# Hyperparameter grid for the tuning
# =====================================================================
# Every entry of `configs` is (server param, value, client param, value).

SEEDS = [1001, 1002, 1003]

# FedAvg has no server parameter: only the client learning rate is scanned.
FEDAVG_ETA_L = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

# The other strategies scan the same (server value, eta_l) pairs. The server
# parameter is mu for FedProx and eta for the adaptive strategies.
SERVER_CLIENT_GRID = [
    (0.001, 0.01),
    (0.01,  0.15),
    (0.03,  0.3),
    (0.05,  0.1),
    (0.1,   0.01),
    (0.1,   0.1),
    (0.1,   0.3),
    (0.2,   0.05),
    (0.2,   0.15),
    (0.3,   0.2),
]

configs = {
    "FedAvg":     [(None, None, "eta_l", eta_l) for eta_l in FEDAVG_ETA_L],
    "FedProx":    [("mu", srv, "eta_l", eta_l) for srv, eta_l in SERVER_CLIENT_GRID],
    "FedAdagrad": [("eta", srv, "eta_l", eta_l) for srv, eta_l in SERVER_CLIENT_GRID],
    "FedAdam":    [("eta", srv, "eta_l", eta_l) for srv, eta_l in SERVER_CLIENT_GRID],
    "FedYogi":    [("eta", srv, "eta_l", eta_l) for srv, eta_l in SERVER_CLIENT_GRID],
}

# Classical model variants: name -> (hidden units, number of clients, folder label)
CLASSICAL_VARIANTS = {
    "3h":       (3, 5,  "3_hidden"),
    "9h":       (9, 5,  "9_hidden"),
    "9h_20cli": (9, 20, "9_hidden_20clients"),
}


# =====================================================================
# Destination folder
# =====================================================================

def build_save_path(save_root, dist, model_type, variant_label=None, circuit=None, dataset=None):
    """Folder of one tuning setup.

    The default setup (quantum model, circuit and dataset of the pyproject)
    goes in <save_root>_<dist>/. Every other setup gets its own subfolder:
    the file names only contain strategy, hyperparameters and seed, so two
    setups in the same folder would overwrite each other.
    """
    path = f"{save_root}_{dist}"
    if model_type == "classical":
        path += f"/classical_{variant_label}"
    elif model_type == "hybrid":
        path += "/hybrid"
    setup = []
    if circuit is not None:
        setup.append(f"circuit-{circuit}")
    if dataset is not None:
        setup.append(f"dataset-{dataset}")
    if setup:
        path += "/" + "_".join(setup)
    return path


# =====================================================================
# Cross-platform helpers to kill processes and clean up
# =====================================================================

def kill_proc_tree(pid):
    """Kill a process and all its descendants."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except (ProcessLookupError, PermissionError):
            pass


def final_cleanup():
    """Final cleanup: shut ray down and kill any orphans left behind."""
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
    except Exception:
        pass

    if IS_WINDOWS:
        for image in ("ray.exe", "raylet.exe", "gcs_server.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        for pattern in ("raylet", "gcs_server", "ray::"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


# =====================================================================
# Function that runs a single tuning job
# Top level: it must be importable by the workers (spawn on Windows).
# =====================================================================

def run_single_job(job, timeout_sec, stagger_max=0.0):
    if stagger_max > 0:
        time.sleep(random.uniform(0, stagger_max))

    # Fixed for every tuning run: noiseless, exact expectation values,
    # evaluated on the validation set, fixed training set, no weights saved.
    config_parts = [
        f'strategy="{job["strategy"]}"',
        f'seed={job["seed"]}',
        'data-seed=2',
        f'save-path="{job["save_path"]}"',
        'save-weights=false',
        'testing=false',
        'mode="noiseless"',
        'nshots="none"',
        f'model-type="{job["model_type"]}"',
        f'init-seed={job["seed"]}',
        f'sampling-seed={job["seed"]}',
    ]
    if job.get("num_rounds"):
        config_parts.append(f'num-server-rounds={job["num_rounds"]}')
    if job.get("hidden_classical"):
        config_parts.append(f'hidden-classical={job["hidden_classical"]}')

    if job.get("num_clients"):
        config_parts.append(f'num-clients={job["num_clients"]}')
        config_parts.append(f'num-partitions={job["num_clients"]}')
    if job["distribution"] == "non_iid":
        config_parts.append('iid=false')
        config_parts.append('alpha=1.8')

    if job["srv_name"] is not None:
        config_parts.append(f'{job["srv_name"]}={job["srv_val"]}')
    config_parts.append(f'{job["cli_name"]}={job["cli_val"]}')

    # Circuit and dataset: passed only when given, otherwise the pyproject decides
    if job.get("circuit"):
        config_parts.append(f'circuit="{job["circuit"]}"')
    if job.get("dataset"):
        config_parts.append(f'dataset="{job["dataset"]}"')

    config_string = " ".join(config_parts)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['RAY_DEDUP_LOGS'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'

    ray_tmp = tempfile.mkdtemp(prefix=f"ray_tuning_{job['strategy']}_seed{job['seed']}_")
    env['RAY_TMPDIR'] = ray_tmp

    job_label = (
        f"{job['strategy']} {job['srv_name']}={job['srv_val']} "
        f"{job['cli_name']}={job['cli_val']} seed={job['seed']}"
    )
    start_time = time.time()
    print(f"  [START] {job_label}")

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

    tag = f"[{job['strategy']} s{job['seed']}]"
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
                if any(k in line for k in ("[ROUND", "Server eval", "ERROR", "aggregate_", "FAIL")):
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
        if proc and proc.poll() is None:
            kill_proc_tree(proc.pid)
        time.sleep(2)
        shutil.rmtree(ray_tmp, ignore_errors=True)


# =====================================================================
# Main: EVERYTHING under __main__ (spawn on Windows).
# =====================================================================

if __name__ == '__main__':

    # -----------------------------------------------------------------
    # Command line arguments
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        type=str,
        nargs="+",
        default=["iid"],
        choices=["iid", "non_iid"],
        help="Data distribution(s). Default: iid. Several values allowed.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(configs),
        help="If given, tune only that strategy.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many runs to launch in parallel. Default: 1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for a single run, in seconds. Default: 3600 (1 hour).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        nargs="+",
        default=["quantum"],
        choices=["quantum", "hybrid", "classical"],
        help="Model type(s). Default: quantum. Several values allowed.",
    )
    parser.add_argument(
        "--classical-variant",
        type=str,
        nargs="+",
        default=None,
        choices=[*CLASSICAL_VARIANTS, "all"],
        help="Classical model variant(s): 3h, 9h, 9h_20cli, all. "
             "Only used with --model-type classical.",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default=None,
        help="Circuit from the registry in task.py. Default: the one of the pyproject.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset from the registry in task.py. Default: the one of the pyproject.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=40,
        help="Number of rounds. Default: 40.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="new_tuning",
        help="Root of the destination folders: <save-root>_<distribution>. "
             "Default: new_tuning.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Skip the first N jobs (to resume where it was interrupted).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the jobs and their destination folders without running them.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # What is going to be tuned
    # -----------------------------------------------------------------
    strategies_to_use = [args.strategy] if args.strategy is not None else list(configs)

    classical_variants = []
    if "classical" in args.model_type:
        if args.classical_variant is None:
            print("ERROR: with --model-type classical you must give --classical-variant")
            sys.exit(1)
        if "all" in args.classical_variant:
            classical_variants = list(CLASSICAL_VARIANTS)
        else:
            classical_variants = args.classical_variant

    pool_stagger = 0.0 if args.workers == 1 else 5.0 * args.workers

    print(f">>> Platform: {platform.system()} ({platform.machine()})")
    print(f">>> Distributions: {args.distribution}")
    print(f">>> Strategies: {', '.join(strategies_to_use)}")
    print(f">>> Model types: {args.model_type}")
    if classical_variants:
        print(f">>> Classical variants: {classical_variants}")
    print(f">>> Circuit: {args.circuit or 'pyproject default'}, "
          f"dataset: {args.dataset or 'pyproject default'}")
    print(f">>> Workers: {args.workers}")
    print(f">>> Timeout: {args.timeout}s")
    print(f">>> Seeds: {SEEDS}\n")

    # -----------------------------------------------------------------
    # Building the job list
    # -----------------------------------------------------------------
    jobs = []
    for dist in args.distribution:
        for mt in args.model_type:
            # (hidden units, number of clients, folder label) of every setup of
            # this model type. Only the classical model has variants.
            if mt == "classical":
                model_setups = [CLASSICAL_VARIANTS[v] for v in classical_variants]
            else:
                model_setups = [(None, None, None)]

            for hidden, n_cli, label in model_setups:
                save_path = build_save_path(
                    args.save_root, dist, mt, label, args.circuit, args.dataset
                )
                extra_label = f" hidden={hidden} cli={n_cli}" if hidden else ""
                print(f">>> [{dist} {mt}{extra_label}] Save path: {save_path}")

                for strategy in strategies_to_use:
                    for srv_name, srv_val, cli_name, cli_val in configs[strategy]:
                        for seed in SEEDS:
                            job = {
                                "strategy": strategy,
                                "seed": seed,
                                "srv_name": srv_name,
                                "srv_val": srv_val,
                                "cli_name": cli_name,
                                "cli_val": cli_val,
                                "save_path": save_path,
                                "distribution": dist,
                                "model_type": mt,
                                "num_rounds": args.num_rounds,
                                "circuit": args.circuit,
                                "dataset": args.dataset,
                            }
                            if hidden is not None:
                                job["hidden_classical"] = hidden
                            if n_cli is not None:
                                job["num_clients"] = n_cli
                            jobs.append(job)

    total = len(jobs)

    # Resume: skip the first N jobs
    if args.resume_from > 0:
        jobs = jobs[args.resume_from:]
        print(f"\nTotal jobs: {total} (first {args.resume_from} skipped, {len(jobs)} left)\n")
    else:
        print(f"\nTotal jobs: {total}\n")

    remaining = len(jobs)

    if args.dry_run:
        destinations = {}
        for job in jobs:
            destinations[job["save_path"]] = destinations.get(job["save_path"], 0) + 1
        print("  destination:")
        for path, count in destinations.items():
            print(f"    {path}  ({count} jobs, {'exists' if os.path.isdir(path) else 'to be created'})")
        print("\n  dry run: no job launched.")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Execution: the first job warms up the SuperLink, the others go in the pool
    # -----------------------------------------------------------------
    print(f"{'='*70}")
    print(f"Starting {remaining} jobs ({args.workers} workers after warmup)")
    print(f"{'='*70}\n")

    start_global = time.time()
    results = []
    completed = args.resume_from

    if remaining == 0:
        sys.exit(0)

    try:
        # Warmup: the first job runs alone
        print(">>> WARMUP: first job, sequential")
        warmup_status, warmup_label, warmup_elapsed = run_single_job(
            jobs[0], args.timeout, stagger_max=0.0
        )
        results.append((warmup_status, warmup_label, warmup_elapsed))
        completed += 1
        print(f"  >>> Warmup done ({warmup_status}) in {warmup_elapsed:.0f}s")
        print(f"  >>> Progress: {completed}/{total}\n")

        # Pool
        pool_jobs = jobs[1:]

        if pool_jobs and args.workers > 1:
            print(f">>> POOL: {len(pool_jobs)} jobs on {args.workers} workers\n")
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
                        print(f"  >>> Progress: {completed}/{total}")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        results.append(("error", str(e), 0))
        elif pool_jobs:
            print(f">>> SEQUENTIAL: {len(pool_jobs)} jobs\n")
            for job in pool_jobs:
                status, label, elapsed = run_single_job(job, args.timeout, stagger_max=0.0)
                results.append((status, label, elapsed))
                completed += 1
                print(f"  >>> Progress: {completed}/{total}")

        elapsed_global = time.time() - start_global

        # Summary
        ok = sum(1 for r in results if r[0] == "ok")
        fail = sum(1 for r in results if r[0] == "fail")
        timeout_count = sum(1 for r in results if r[0] == "timeout")
        error = sum(1 for r in results if r[0] == "error")

        print(f"\n{'='*70}")
        print(f"COMPLETED in {elapsed_global:.0f}s ({elapsed_global/3600:.2f} hours)")
        print(f"  OK:      {ok}/{total}")
        print(f"  FAIL:    {fail}/{total}")
        print(f"  TIMEOUT: {timeout_count}/{total}")
        print(f"  ERROR:   {error}/{total}")
        print(f"{'='*70}\n")

        if fail > 0 or timeout_count > 0 or error > 0:
            print("Failed/timed-out jobs:")
            for status, label, elapsed in results:
                if status in ("fail", "timeout", "error"):
                    print(f"  [{status.upper()}] {label}")

    finally:
        print("\n>>> Final cleanup (Ray + orphan processes)...")
        final_cleanup()
        print(">>> Done.")