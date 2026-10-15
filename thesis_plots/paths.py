"""A single place for the data and figure paths.

The scripts use paths relative to the repository root: the runner
(`python -m thesis_plots`) moves there before calling any figure, so the
commands work from any folder.

Three roots, with the defaults below: the run data in results/, the figures
in thesis_images/, the campaigns outside the thesis in archive/. The runner
can move them with --results, --out and --archive, which call configure()
before the figure modules are imported: some of them copy these paths into
their own constants at import time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- roots
# The campaigns needed to produce the thesis live in results/: a copy of the
# branches that are actually read, with the same internal tree as the
# original folders, which stay where they are.
FINAL = "results"
# Everything else among the results, not in the document: moved here, not
# deleted. Only the exploratory figures in thesis_images/extra/ read it.
ARCHIVE = "archive"
IMAGES = "thesis_images"


def configure(results=None, images=None, archive=None):
    """Set the three roots and rebuild every path derived from them.

    Called once at import with the defaults. The runner calls it again with
    the command-line options; a None leaves that root as it is.
    """
    g = globals()
    if results is not None:
        g["FINAL"] = str(results)
    if images is not None:
        g["IMAGES"] = str(images)
    if archive is not None:
        g["ARCHIVE"] = str(archive)
    FINAL, ARCHIVE, IMAGES = g["FINAL"], g["ARCHIVE"], g["IMAGES"]

    # ------------------------------------------------------------ data
    # Noiseless / noisy / mitigated campaign with a fixed training set.
    NOISE_FED = f"{FINAL}/noise_and_mitigation/federated"
    IID = f"{NOISE_FED}/iid"
    NON_IID = f"{NOISE_FED}/non_iid"
    # Hyperparameter tuning.
    TUNING_IID = f"{FINAL}/tuning/iid"
    # Noise drift: Ornstein-Uhlenbeck in the thesis, random walk only in
    # extra. Same shape as the static-noise campaign:
    # <dist>/<strategy>/<regime>.
    DRIFT_OU = f"{FINAL}/noise_and_mitigation_drift"
    # Campaigns that only feed the figures in thesis_images/extra/: they live
    # in archive/ with the rest of the results that are not in the document.
    SWEEP_RW = f"{ARCHIVE}/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk"

    # --------------------------------------------------------- figures
    # The delivered folders: one per chapter or section of the thesis. The
    # names are those of the files already in the document, so regenerating
    # a figure overwrites it in place instead of creating a copy elsewhere.
    FINAL_043 = f"{IMAGES}/4_3_FINAL"       # mitigation and memory
    # .tex and .csv table bodies: outside the figure folders, which hold
    # PDFs only.
    TABLES = f"{IMAGES}/tables"
    # Exploratory figures, outside the thesis.
    EXTRA = f"{IMAGES}/extra"

    g.update(
        NOISE_FED=NOISE_FED,
        IID=IID,
        NON_IID=NON_IID,
        FEDAVG_IID=f"{IID}/fedavg",
        FEDAVG_NON_IID=f"{NON_IID}/fedavg",

        # Centralized reference used by almost every chapter 4 figure.
        CENTRALIZED=f"{FINAL}/noise_and_mitigation/centralized",

        TUNING_IID=TUNING_IID,
        TUNING_NON_IID=f"{FINAL}/tuning/non_iid",

        # Strategy comparison at 40 rounds.
        STRATEGY_FED=f"{FINAL}/strategy_comparison/federated",
        STRATEGY_CEN=f"{FINAL}/strategy_comparison/centralized",
        # The eta_l sweep of FedAvg: the old campaign inside
        # strategy_comparison had the same eta_l values but seeds 14/71/130,
        # a triple that no other figure uses. The new one is read instead,
        # so that the tuning table and figures talk about the same runs.
        STRATEGY_TUNING=TUNING_IID,

        DRIFT_OU=DRIFT_OU,
        DRIFT_RW=f"{ARCHIVE}/strategies_drift_p005",

        # CDR threshold sweep with memory.
        SWEEP_OU=f"{FINAL}/memory_sweep_drift",
        # FedAvg IID under drift, the two arms. They sit with the other
        # strategies: the noisy one came from the unmitigated arm of the
        # sweep, the mitigated one is a copy of the 0.015 threshold without
        # memory, which also stays in the sweep because the 4.3 figures read
        # it as one of the eighteen combinations. Note: it is the only
        # 40-round arm, the others have 10, so it must be truncated.
        NOISY_OU_10R=f"{DRIFT_OU}/iid/fedavg/noisy",
        MITIGATED_OU_10R=f"{DRIFT_OU}/iid/fedavg/mitigated",
        CEN_MITIGATED=f"{FINAL}/tables_only/centralized_mitigated",
        # Centralized sweep with memory: only the tables of the OU campaign.
        CEN_MEMORY=f"{FINAL}/tables_only/memory_centralized",
        CEN_MEMORY_OU_SWEEP=f"{FINAL}/tables_only/memory_centralized_ou_sweep",

        SWEEP_RW=SWEEP_RW,
        STATIC_CALIBRATIONS=f"{ARCHIVE}/MEMORY_RESULTS_NEW/static_calibrations_federated",
        FEDAVG_TH_003=f"{ARCHIVE}/FEDAVG_TH_0.03",
        THRESH_SWEEP_VARWALK=SWEEP_RW,
        NOISE_SWEEP_TH015=f"{ARCHIVE}/MEMORY_RESULTS_OLD/noise_sweep_th015",
        TUNING_NON_IID_OLD=f"{ARCHIVE}/tuning_non_iid",

        SEED_ISOLATION=f"{FINAL}/seed_isolation",

        FINAL_03=f"{IMAGES}/3_FINAL",           # methodology
        FINAL_041=f"{IMAGES}/4_1_FINAL",        # strategy comparison
        FINAL_042=f"{IMAGES}/4_2_FINAL",        # noise and drift
        FINAL_043=FINAL_043,
        FINAL_06=f"{IMAGES}/6_FINAL",           # appendix: tuning and mitigated strategies

        CAP03=f"{IMAGES}/3_FINAL",
        FIG_STRATEGY=f"{IMAGES}/4_1_FINAL",
        FIG_NOISE=f"{IMAGES}/4_2_FINAL",
        # In 4.3 all the figures sit in the same folder: the distinction
        # between the section and its subsection 4.3.1 only existed in the
        # old tree.
        FIG_MITIGATION=FINAL_043,
        FIG_MITIGATION_431=FINAL_043,

        TABLES=TABLES,

        # Memory campaigns. Of the Ornstein-Uhlenbeck one, three figures
        # enter 4.3 and the others keep them company; the random walk stays
        # out of the thesis.
        MEMORY_OUT_OU=FINAL_043,
        MEMORY_TABLES_OU=TABLES,
        MEMORY_OUT_RW=f"{EXTRA}/rw",
        MEMORY_TABLES_RW=f"{EXTRA}/rw",

        EXTRA=EXTRA,
        # CDR sweep of the pre-OU campaigns: same file names as the OU
        # sweep, hence a separate folder so as not to overwrite the 4.3
        # figures.
        CAP04=f"{EXTRA}/cdr_sweep_old",
    )


configure()


def static_noise(base, regime, p):
    """Folder of a static-noise run: .../<regime>/scaled_pP_rP_s0.002."""
    return f"{base}/{regime}/scaled_p{p}_r{p}_s0.002"


def ensure(path):
    """Create the destination folder and return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
