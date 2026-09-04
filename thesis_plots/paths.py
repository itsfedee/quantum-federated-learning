"""Un solo posto per i percorsi dei dati e delle figure.

Gli script usano percorsi relativi alla radice del repository: il runner
(`python -m thesis_plots`) ci si sposta prima di chiamare qualunque figura,
cosi' i comandi funzionano da qualsiasi cartella.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- dati
RESULTS = "results/fixed_training_set_results"

# Campagna noiseless / noisy / mitigated a training set fisso.
NOISE_FED = f"{RESULTS}/noiseless_noisy_mit_fixed_results/federated"
IID = f"{NOISE_FED}/iid"
NON_IID = f"{NOISE_FED}/non_iid"
FEDAVG_IID = f"{IID}/fedavg"
FEDAVG_NON_IID = f"{NON_IID}/fedavg"

# Riferimento centralizzato usato da quasi tutte le figure del capitolo 4.
CENTRALIZED = "parameters_centralized/lr_0.3"

# Confronto fra strategie a 40 round.
STRATEGY_FED = f"{RESULTS}/strategy_comparison_fixed/quantum/3_layers"
STRATEGY_CEN = "strategy_comp_centr_tmp/lr_0.3"
STRATEGY_TUNING = f"{STRATEGY_FED}/iid/tuning/tuning_experiments"

# Deriva del rumore: random walk (campagna vecchia) e Ornstein-Uhlenbeck.
DRIFT_RW = "strategies_drift_p005"
DRIFT_OU = "strategies_OU_drift_10r"

# Sweep di soglia CDR con memoria.
SWEEP_OU = "MEMORY_RESULTS_NEW/THRESH_SWEEP_FEDERATED_OU"
SWEEP_RW = "MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk"
NOISY_OU_10R = f"{SWEEP_OU}/noisy_10r"
MITIGATED_OU_10R = f"{SWEEP_OU}/thresh_0015/no_memory"
STATIC_CALIBRATIONS = "MEMORY_RESULTS_NEW/static_calibrations_federated"
FEDAVG_TH_003 = "FEDAVG_TH_0.03"
CEN_MITIGATED = "MEMORY_RESULTS_NEW/centralized_mitigated"

# Sweep CDR piu' vecchi, ancora usati da due figure del capitolo 4.
THRESH_SWEEP_VARWALK = "threshold_sweep_p005_varwalk"
NOISE_SWEEP_TH015 = "noise_sweep_th015"

SEED_ISOLATION = "seed_isolation"

# Tuning degli iperparametri.
TUNING_IID = "NEW_TUNING/new_tuning_iid"
TUNING_NON_IID = "NEW_TUNING/new_tuning_non_iid"
TUNING_NON_IID_OLD = "tuning_non_iid"

# ------------------------------------------------------------- figure
IMAGES = "thesis_images"
CAP03 = f"{IMAGES}/cap_03_Methodology"
CAP04 = f"{IMAGES}/cap_04_Results"
FIG_STRATEGY = f"{CAP04}/4.1_strategy_comparison"
FIG_NOISE = f"{CAP04}/4.2_noise"
FIG_MITIGATION = f"{CAP04}/4.3_mitigation"
FIG_MITIGATION_431 = f"{FIG_MITIGATION}/4.3.1"
TABLES = f"{CAP04}/tables"

# Campagne sulla memoria: una cartella per regime di deriva.
MEMORY_OUT_OU = "NEW_THESIS_IMAGES"
MEMORY_OUT_RW = "NEW_THESIS_IMAGES_RW"

# Figure esplorative, fuori dalla tesi.
EXTRA = f"{IMAGES}/extra"


def static_noise(base, regime, p):
    """Cartella di una run a rumore statico: .../<regime>/scaled_pP_rP_s0.002."""
    return f"{base}/{regime}/scaled_p{p}_r{p}_s0.002"


def ensure(path):
    """Crea la cartella di destinazione e restituisce il percorso."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
