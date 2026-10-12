"""Un solo posto per i percorsi dei dati e delle figure.

Gli script usano percorsi relativi alla radice del repository: il runner
(`python -m thesis_plots`) ci si sposta prima di chiamare qualunque figura,
cosi' i comandi funzionano da qualsiasi cartella.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- dati
# Le campagne che servono a produrre la tesi stanno in results/: e' una
# copia dei soli rami letti, fatta con lo stesso albero interno delle cartelle
# originali, che restano al loro posto. Le voci che puntano ancora all'albero
# vecchio servono solo alle figure esplorative di thesis_images/extra/.
FINAL = "results"
# Tutto il resto dei risultati, non impaginato: spostato qui, non cancellato.
ARCHIVE = "archive"

# Campagna noiseless / noisy / mitigated a training set fisso.
NOISE_FED = f"{FINAL}/noise_and_mitigation/federated"
IID = f"{NOISE_FED}/iid"
NON_IID = f"{NOISE_FED}/non_iid"
FEDAVG_IID = f"{IID}/fedavg"
FEDAVG_NON_IID = f"{NON_IID}/fedavg"

# Riferimento centralizzato usato da quasi tutte le figure del capitolo 4.
CENTRALIZED = f"{FINAL}/noise_and_mitigation/centralized"

# Tuning degli iperparametri.
TUNING_IID = f"{FINAL}/tuning/iid"
TUNING_NON_IID = f"{FINAL}/tuning/non_iid"

# Confronto fra strategie a 40 round.
STRATEGY_FED = f"{FINAL}/strategy_comparison/federated"
STRATEGY_CEN = f"{FINAL}/strategy_comparison/centralized"
# Lo sweep di eta_l di FedAvg: la campagna vecchia dentro
# strategy_comparison aveva gli stessi valori di eta_l ma seed 14/71/130, una
# terna che nessun'altra figura usa. Si legge quella nuova, cosi' tabella e
# figure del tuning parlano degli stessi run.
STRATEGY_TUNING = TUNING_IID

# Deriva del rumore: Ornstein-Uhlenbeck in tesi, random walk solo in extra.
# Stessa forma della campagna a rumore statico: <dist>/<strategia>/<regime>.
DRIFT_OU = f"{FINAL}/noise_and_mitigation_drift"
DRIFT_RW = f"{ARCHIVE}/strategies_drift_p005"

# Sweep di soglia CDR con memoria.
SWEEP_OU = f"{FINAL}/memory_sweep_drift"
# FedAvg IID sotto deriva, i due bracci. Stanno con le altre strategie: il
# noisy veniva dal braccio senza mitigazione dello sweep, il mitigato e' una
# copia della soglia 0.015 senza memoria, che resta anche nello sweep perche'
# le figure di 4.3 la leggono come una delle diciotto combinazioni. Attenzione:
# e' l'unico braccio da 40 round, gli altri ne hanno 10, quindi va troncato.
NOISY_OU_10R = f"{DRIFT_OU}/iid/fedavg/noisy"
MITIGATED_OU_10R = f"{DRIFT_OU}/iid/fedavg/mitigated"
CEN_MITIGATED = f"{FINAL}/tables_only/centralized_mitigated"
# Sweep centralizzato con memoria: solo le tabelle della campagna OU.
CEN_MEMORY = f"{FINAL}/tables_only/memory_centralized"
CEN_MEMORY_OU_SWEEP = f"{FINAL}/tables_only/memory_centralized_ou_sweep"

# Campagne che alimentano soltanto le figure di thesis_images/extra/: stanno
# in archive/ con tutto il resto dei risultati non impaginati.
SWEEP_RW = f"{ARCHIVE}/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk"
STATIC_CALIBRATIONS = f"{ARCHIVE}/MEMORY_RESULTS_NEW/static_calibrations_federated"
FEDAVG_TH_003 = f"{ARCHIVE}/FEDAVG_TH_0.03"
THRESH_SWEEP_VARWALK = SWEEP_RW
NOISE_SWEEP_TH015 = f"{ARCHIVE}/MEMORY_RESULTS_OLD/noise_sweep_th015"
TUNING_NON_IID_OLD = f"{ARCHIVE}/tuning_non_iid"

SEED_ISOLATION = f"{FINAL}/seed_isolation"


# ------------------------------------------------------------- figure
IMAGES = "thesis_images"

# Le cartelle consegnate: una per capitolo o sezione della tesi. I nomi sono
# quelli dei file gia' impaginati, quindi rigenerare una figura la sovrascrive
# al suo posto invece di crearne una copia altrove.
FINAL_03 = f"{IMAGES}/3_FINAL"          # metodologia
FINAL_041 = f"{IMAGES}/4_1_FINAL"       # confronto fra strategie
FINAL_042 = f"{IMAGES}/4_2_FINAL"       # rumore e deriva
FINAL_043 = f"{IMAGES}/4_3_FINAL"       # mitigazione e memoria
FINAL_06 = f"{IMAGES}/6_FINAL"          # appendice: tuning e strategie mitigate

CAP03 = FINAL_03
FIG_STRATEGY = FINAL_041
FIG_NOISE = FINAL_042
# In 4.3 le figure stanno tutte nella stessa cartella: la distinzione fra la
# sezione e la sua sottosezione 4.3.1 esisteva solo nell'albero vecchio.
FIG_MITIGATION = FINAL_043
FIG_MITIGATION_431 = FINAL_043

# Corpi .tex e .csv delle tabelle: fuori dalle cartelle delle figure, che
# contengono solo PDF.
TABLES = f"{IMAGES}/tables"

# Campagne sulla memoria. Dell'Ornstein-Uhlenbeck tre figure entrano in 4.3 e
# le altre le fanno compagnia; la random walk resta fuori dalla tesi.
MEMORY_OUT_OU = FINAL_043
MEMORY_TABLES_OU = TABLES
MEMORY_OUT_RW = f"{IMAGES}/extra/rw"
MEMORY_TABLES_RW = MEMORY_OUT_RW

# Figure esplorative, fuori dalla tesi.
EXTRA = f"{IMAGES}/extra"
# Sweep CDR delle campagne pre-OU: stessi nomi di file dello sweep OU, quindi
# una cartella a parte per non sovrascrivere le figure di 4.3.
CAP04 = f"{EXTRA}/cdr_sweep_old"


def static_noise(base, regime, p):
    """Cartella di una run a rumore statico: .../<regime>/scaled_pP_rP_s0.002."""
    return f"{base}/{regime}/scaled_p{p}_r{p}_s0.002"


def ensure(path):
    """Crea la cartella di destinazione e restituisce il percorso."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
