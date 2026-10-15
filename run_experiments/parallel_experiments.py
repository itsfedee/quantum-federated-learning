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
# Platform detection (done once, at top level)
# =====================================================================

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# =====================================================================
# Command line arguments
# =====================================================================
# Usage examples:
#   python parallel_experiments.py --workers 5
#       runs the campaign exactly as written in the "runs" block below
#   python parallel_experiments.py --strategy FedProx --rounds 40 --workers 4
#   python parallel_experiments.py --strategy FedAdam --mode noisy --noise 0.005 --nshots 1000 --workers 4
#   python parallel_experiments.py --mode mitigated --noise 0.005 --nshots 1000 --dry-run
#
# Every option left out keeps the value written in the "runs" block.


# =====================================================================
# Strategies and hyperparameters
# Top level: the workers spawned on Windows import them.
# =====================================================================

# The hyperparameters chosen by the tuning: (server param, value, client
# param, value). --strategy takes them from here. FedAdam is the only one
# that differs between IID and non-IID.
STRATEGY_HP = {
    "FedAvg":     (None,  None, "eta_l", 0.3),
    "FedProx":    ("mu",  0.03, "eta_l", 0.3),
    "FedAdagrad": ("eta", 0.3,  "eta_l", 0.2),
    "FedAdam":    ("eta", 0.2,  "eta_l", 0.15),
    "FedYogi":    ("eta", 0.1,  "eta_l", 0.1),
}
STRATEGY_HP_NON_IID = {
    "FedAdam":    ("eta", 0.1,  "eta_l", 0.1),
}


# =====================================================================
# Cross-platform helpers to kill processes and clean up
# =====================================================================

def kill_proc_tree(pid):
    """Kill a process and all its descendants.

    Cross-platform: taskkill /T on Windows, the process group on Unix."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    else:
        # On Unix we created a new session group with start_new_session=True,
        # so every descendant shares the process group ID = pid.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            # If it is still alive after SIGTERM, plain SIGKILL
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # gia' morto, ottimo
        except (ProcessLookupError, PermissionError):
            pass


def final_cleanup():
    """Final cleanup: shut ray down and kill any orphans left behind.

    Cross-platform."""
    # Soft attempt through the ray command
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
    except Exception:
        pass

    # OS-specific hard kill for surviving orphans
    if IS_WINDOWS:
        for image in ("ray.exe", "raylet.exe", "gcs_server.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        # pkill returns != 0 when it finds nothing, which is fine: capture_output swallows it
        for pattern in ("raylet", "gcs_server", "ray::"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def cleanup_orphan_ray_dirs(label="startup"):
    """Remove the orphan Ray temp dirs left behind by interrupted runs.

    The per-job cleanup (the rmtree in the finally of run_single_job) never
    runs if the process dies of SIGKILL, if the ssh session drops or if the
    machine reboots: the dirs stay in the temp dir and fill the disk over
    time. On a shared machine that hurts everyone, so we clean up at the
    start and at the end of every sweep.

    Safety: it touches nothing while flwr/ray processes are still alive (it
    could be a parallel sweep), and it skips other users' dirs.
    """
    tmp_root = tempfile.gettempdir()

    if not IS_WINDOWS:
        try:
            alive = subprocess.run(
                ["pgrep", "-f", "flwr|raylet|gcs_server"],
                capture_output=True, timeout=10,
            )
            if alive.returncode == 0:
                print(f">>> Cleanup {label}: flwr/ray processes still running, "
                      f"skipping the cleanup of {tmp_root} to be safe.")
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
            # Leave the temp dirs of other users on this machine alone.
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
        print(f">>> Cleanup {label}: removed {removed} orphan Ray temp dirs from "
              f"{tmp_root} ({freed / 1e9:.1f} GB freed)")
    try:
        usage = shutil.disk_usage(tmp_root)
        print(f">>> Free space on {tmp_root}: {usage.free / 1e9:.1f} GB "
              f"/ {usage.total / 1e9:.1f} GB")
    except Exception:
        pass


def raise_fd_limit_if_possible():
    """On Unix with many workers the file descriptor limit must be raised.

    Not needed on Windows, which has no such limit."""
    if IS_WINDOWS:
        return
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f">>> File descriptor limit raised: {soft} -> {target}")
    except Exception as e:
        print(f">>> WARN: could not raise the fd limit ({e}). With workers>16, "
              f"run 'ulimit -n 65536' before launching the script.")


# =====================================================================
# Function that runs a single job
# Top level: it must be importable by the workers (spawn on Windows).
# =====================================================================

def run_single_job(job, timeout_sec, stagger_max=0.0):
    # Random stagger so the workers do not all hit the SuperLink at the same
    # instant. The local SuperLink takes ~15s to become operational from cold,
    # and the flwr CLI client has a 15s timeout on the initial connection:
    # with N simultaneous clients at least one of them overruns it.
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
    if job.get("circuit"):
        parts.append(f'circuit="{job["circuit"]}"')
    if job.get("dataset"):
        parts.append(f'dataset="{job["dataset"]}"')
    if job.get("save_scatter"):
        parts.append('save-scatter=true')

    config_string = " ".join(parts)

    # Custom environment for this job.
    # We do NOT set a per-job FLWR_HOME: Flower's managed local SuperLink is
    # one per machine, shared by all the parallel flwr runs, and it is
    # designed to handle concurrent runs.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['RAY_DEDUP_LOGS'] = '0'
    env['RAY_memory_monitor_refresh_ms'] = '0'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'

    # A unique RAY_TMPDIR so the Ray work dirs of different jobs stay apart.
    ray_tmp = tempfile.mkdtemp(prefix=f"ray_{job['strategy']}_seed{seed}_")
    env['RAY_TMPDIR'] = ray_tmp

    job_label = (
        f"{job['strategy']} seed={seed} {job['mode']} "
        f"{job['noise_label']} nshots={job['nshots']}"
    )
    start_time = time.time()
    print(f"  [START] {job_label}")

    # Cross-platform Popen arguments.
    # On Unix start_new_session=True creates a new process group, so we can
    # kill every descendant through os.killpg.
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
                # Print only the useful lines: rounds, metrics, errors
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
        # Safety net: if the process is somehow still alive, kill it
        if proc and proc.poll() is None:
            kill_proc_tree(proc.pid)
        # Pause to let the file handles be released (indispensable on
        # Windows, harmless on Unix)
        time.sleep(2)
        shutil.rmtree(ray_tmp, ignore_errors=True)


# =====================================================================
# Setup and execution: EVERYTHING under __main__ (spawn on Windows).
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=sorted(STRATEGY_HP),
        help="Run this strategy, with the tuned hyperparameters of STRATEGY_HP. "
             "Default: the strategies written in the campaign.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["noiseless", "noisy", "mitigated"],
        help="Override the mode of the campaign.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the number of server rounds.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Base noise level: sets base_pauli and base_readout to this value.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override the per-client spread of the noise around the base level.",
    )
    parser.add_argument(
        "--nshots",
        type=str,
        default=None,
        help="Override the number of shots ('none' = exact expectation values).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Destination folder. Default: the one of the campaign, or the one "
             "built by ExperimentPath when mode/noise/nshots are overridden.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many runs to launch in parallel. Default: 1."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Timeout for a single run, in seconds. Default: 21600 (6 hours)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the jobs and their destination folder without running them."
    )
    parser.add_argument("--save-scatter", action="store_true",
                    help="Save the CDR training points (noisy vs noise-free) of every client.")
    
    args = parser.parse_args()

    print(f">>> Platform: {platform.system()} ({platform.machine()})")

    # Clear the leftovers of previously interrupted runs before this sweep
    # takes up more space.
    cleanup_orphan_ray_dirs("startup")

    # On Linux with many workers it pays to raise the fd limit
    if args.workers >= 8:
        raise_fd_limit_if_possible()

    # Maximum stagger when starting the jobs in the pool. Not needed for
    # workers=1. For workers>=2 we spread the start over a few seconds so the
    # clients do not hit the SuperLink all at once.
    pool_stagger = 0.0 if args.workers == 1 else 5.0 * args.workers

    print(f">>> Parallel workers: {args.workers}")
    print(f">>> Timeout per run: {args.timeout}s")
    print(f">>> Max pool stagger: {pool_stagger:.1f}s\n")

    # =================================================================
    # Configuration: the campaign to run
    # =================================================================
    # One campaign at a time. The finished ones no longer live here: their
    # results are in results/ and in archive/, and this file with all the
    # past blocks is in the initial commit and in ../qibo-qfl-torch.

    SEEDS = [1, 2, 3, 4, 5, 6, 7]

    # Strategy comparison, IID arm: noiseless, 60 rounds. It lands in
    # results/strategy_comparison/federated/iid/, next to the runs of the
    # other four strategies already produced.
    runs = [
        {
            "distribution": "iid",
            "mode": "noiseless",
            "model_type": "quantum",
            "base_pauli": 0.0,
            "base_readout": 0.0,
            "scale": 0.0,
            "nshots": None,
            "num_rounds": 60,
            "num_clients": 5,
            "fraction_fit": 0.75,
            "fraction_evaluate": 0.75,
            "save_path_override": "results/strategy_comparison/federated/iid",
            "strategies": {
                # name: (server param, value, client param, value, seeds)
                "FedAvg": (None, None, "eta_l", 0.3, SEEDS),
            },
            "circuit": "2q",
            "dataset": "circle",
        },
    ]

    # -----------------------------------------------------------------
    # Command-line overrides, applied to every block of the campaign.
    # Whatever is not given on the command line keeps the value above.
    # -----------------------------------------------------------------
    for run in runs:
        if args.strategy is not None:
            table = STRATEGY_HP_NON_IID if run["distribution"] == "non_iid" else {}
            hp = table.get(args.strategy, STRATEGY_HP[args.strategy])
            run["strategies"] = {args.strategy: (*hp, SEEDS)}
        if args.rounds is not None:
            run["num_rounds"] = args.rounds
        if args.mode is not None:
            run["mode"] = args.mode
        if args.noise is not None:
            run["base_pauli"] = run["base_readout"] = args.noise
        if args.scale is not None:
            run["scale"] = args.scale
        if args.nshots is not None:
            run["nshots"] = None if args.nshots.lower() == "none" else int(args.nshots)
        if args.save_scatter:
            run["save_scatter"] = True

        # Mode and noise level must agree.
        if run["mode"] == "noiseless":
            if args.noise or args.scale:
                print("ERROR: a noiseless run cannot have --noise or --scale > 0.")
                sys.exit(1)
            run["base_pauli"] = run["base_readout"] = run["scale"] = 0.0
        elif run["base_pauli"] == 0 and run["base_readout"] == 0:
            print(f"ERROR: mode '{run['mode']}' with zero noise: pass --noise "
                  f"(or set base_pauli/base_readout in the campaign).")
            sys.exit(1)

        # The folder written in the campaign describes the campaign as written:
        # once mode, noise or shots change, the runs must not land there.
        noise_overridden = any(v is not None for v in (args.mode, args.noise, args.scale, args.nshots))
        if args.save_path is not None:
            run["save_path_override"] = args.save_path
        elif noise_overridden:
            run.pop("save_path_override", None)

    # The strategies actually defined above: --strategy picks among these.
    available = sorted({s for run in runs for s in run["strategies"]})
    if args.strategy is not None:
        if args.strategy not in available:
            print(f"ERROR: strategy '{args.strategy}' is not in this campaign.")
            print(f"Available strategies: {available}")
            sys.exit(1)
        strategies_to_use = [args.strategy]
        print(f">>> Only the {args.strategy} strategy\n")
    else:
        strategies_to_use = available
        print(f">>> Strategies: {', '.join(available)}\n")

    # =================================================================
    # Building the job list (flatten)
    # =================================================================

    jobs = []
    for run in runs:
        noise_label = ExperimentPath.noise_label(
            run["base_pauli"], run["base_readout"], run["scale"]
        )
        # Accepts both a dict and a list of tuples for the strategies
        strat_items = run["strategies"].items() if isinstance(run["strategies"], dict) else run["strategies"]
        for strategy, (srv_name, srv_val, cli_name, cli_val, seeds) in strat_items:
            if strategy not in strategies_to_use:
                continue
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
                    "circuit": run.get("circuit"),
                    "dataset": run.get("dataset"),
                    "save_scatter": run.get("save_scatter")
                }

                # Seed isolation: the seeds in overrides are held fixed,
                # those NOT in overrides vary with seed.
                if seed_overrides:
                    job["init_seed"] = seed_overrides.get("init_seed", seed)
                    job["data_seed"] = seed_overrides.get("data_seed", seed)
                    job["sampling_seed"] = seed_overrides.get("sampling_seed", seed)

                if seed_label_prefix:
                    # Work out the value of the varying seed for the label
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
    print(f"Total jobs to run: {total}\n")

    if args.dry_run:
        for job in jobs:
            print(f"  {job['strategy']:<11} seed {job['seed']}  "
                  f"{job['mode']}, {job.get('num_rounds')} round, "
                  f"pauli {job['base_pauli']}, readout {job['base_readout']}, "
                  f"scale {job['scale']}, nshots {job['nshots']}, "
                  f"{job['srv_name']}={job['srv_val']} {job['cli_name']}={job['cli_val']}")
        destinazioni = sorted({job["save_path"] for job in jobs})
        print("\n  destination:")
        for d in destinazioni:
            print(f"    {d}  ({'exists' if os.path.isdir(d) else 'to be created'})")
        print("\n  dry run: no job launched.")
        sys.exit(0)

    # =================================================================
    # Execution: dummy warmup + parallel pool
    # =================================================================

    print(f"{'='*70}")
    print(f"Starting {total} jobs ({args.workers} parallel workers after warmup)")
    print(f"{'='*70}\n")

    start_global = time.time()
    results = []
    completed = 0

    if total == 0:
        sys.exit(0)

    try:
        # ---- Dummy warmup ----
        # A 1-round noiseless job that warms up Flower's SuperLink (~15s from
        # cold) without wasting a real job. The result is not saved
        # (save_path points at a temporary dir deleted right afterwards).
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
        print(">>> WARMUP: 1-round noiseless dummy job to warm up the SuperLink")
        warmup_status, warmup_label, warmup_elapsed = run_single_job(
            warmup_job, timeout_sec=300, stagger_max=0.0
        )
        print(f"  >>> Warmup done ({warmup_status}) in {warmup_elapsed:.0f}s")
        # Cleanup: the warmup must leave no trace
        shutil.rmtree(warmup_tmp, ignore_errors=True)

        # ---- Pool: ALL the jobs in parallel ----
        pool_jobs = jobs

        if pool_jobs and args.workers > 1:
            print(f">>> POOL: {len(pool_jobs)} jobs left on {args.workers} workers\n")
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
            # workers=1: everything in sequence, no pool
            print(f">>> SEQUENTIAL: {len(pool_jobs)} jobs left on 1 worker\n")
            for job in pool_jobs:
                status, label, elapsed = run_single_job(job, args.timeout, stagger_max=0.0)
                results.append((status, label, elapsed))
                completed += 1
                print(f"  >>> Progress: {completed}/{total} done")

        elapsed_global = time.time() - start_global

        # =================================================================
        # Final summary
        # =================================================================

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
            print("Jobs falliti/timeout:")
            for status, label, elapsed in results:
                if status in ("fail", "timeout", "error"):
                    print(f"  [{status.upper()}] {label}")

    finally:
        # Final cleanup, ALWAYS executed - even if the user hits Ctrl+C,
        # even if something blows up inside the pool.
        print("\n>>> Pulizia finale (Ray + processi orfani)...")
        final_cleanup()
        # The Ray processes are gone: every temp dir left behind is now an
        # orphan and can be removed safely.
        cleanup_orphan_ray_dirs("finale")
        print(">>> Done.")