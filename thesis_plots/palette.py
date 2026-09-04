"""Colori, marcatori ed etichette condivisi fra le figure.

Le stesse cinque strategie e le stesse tre metriche compaiono in una decina di
figure: tenerne qui la resa evita che una figura usi un colore diverso dalle
altre per la stessa curva.
"""

SEEDS = [1, 2, 3, 4, 5, 6, 7]

# --------------------------------------------------------------- strategie
STRATEGY_COLORS = {"FedAvg": "C0", "FedProx": "C1", "FedAdagrad": "C2",
                   "FedAdam": "C3", "FedYogi": "C4"}
STRATEGY_MARKERS = {"FedAvg": "o", "FedProx": "s", "FedAdagrad": "^",
                    "FedAdam": "D", "FedYogi": "v"}
# Sfasamento dei marcatori: cinque curve che si intrecciano sul plateau
# restano distinguibili solo se i simboli non cadono sulle stesse ascisse.
STRATEGY_OFFSETS = {"FedAvg": 0, "FedProx": 1, "FedAdagrad": 2,
                    "FedAdam": 3, "FedYogi": 4}

STRATEGY_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx",
                  "fedadagrad": "FedAdagrad", "fedadam": "FedAdam",
                  "fedyogi": "FedYogi"}

# ----------------------------------------------------------------- metriche
# Le run federate chiamano la metrica "f1" lato client e "f1_score" lato
# server: da qui le due tabelle di etichette.
METRICS_CLIENT = ["loss", "accuracy", "f1"]
METRICS_SERVER = ["loss", "accuracy", "f1_score"]

METRIC_INFO = {
    "loss":     {"ylabel": "Loss",         "pct": False},
    "accuracy": {"ylabel": "Accuracy (%)", "pct": True},
    "f1":       {"ylabel": "F1 score",     "pct": True},
}
METRIC_LABELS = {"loss": "Loss", "accuracy": "Accuracy", "f1_score": "F1 score"}
METRIC_LABELS_PCT = {"loss": "Loss", "accuracy": "Accuracy (%)",
                     "f1": "F1 score"}

# ------------------------------------------------------------ regimi/rumore
# Scala di rossi: piu' scuro, piu' rumore.
NOISE_COLORS = {0.005: "#e88a8a", 0.007: "#c43c3c", 0.014: "#67000d"}
NOISY = "#c43c3c"
MITIGATED = "C2"
NOISELESS = "C0"
CENTRALIZED = "0.3"

# ------------------------------------------------------- memoria della CDR
# La terza variante cambia col regime di deriva (ema_full sotto OU, warm_start
# sotto random walk) ma occupa lo stesso posto, quindi condivide colore e
# marcatore.
VAR_COLOR = {"no_memory": "C0", "ema": "C1", "ema_full": "C2",
             "warm_start": "C2"}
VAR_MARKER = {"no_memory": "o", "ema": "s", "ema_full": "^",
              "warm_start": "^"}

# Marcatori per le curve sui round, diradati e sfasati fra loro.
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]
CLIENT_MARKERS = ["o", "s", "^", "D", "v"]
MARKER_OFFSETS = [0, 1, 2, 3]
