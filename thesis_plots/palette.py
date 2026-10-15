"""Colours, markers and labels shared between the figures.

The same five strategies and the same three metrics appear in a dozen
figures: keeping their rendering here prevents one figure from using a
different colour than the others for the same curve.
"""

SEEDS = [1, 2, 3, 4, 5, 6, 7]

# --------------------------------------------------------------- strategies
STRATEGY_COLORS = {"FedAvg": "C0", "FedProx": "C1", "FedAdagrad": "C2",
                   "FedAdam": "C3", "FedYogi": "C4"}
STRATEGY_MARKERS = {"FedAvg": "o", "FedProx": "s", "FedAdagrad": "^",
                    "FedAdam": "D", "FedYogi": "v"}
# Marker offsets: five curves that intertwine on the plateau stay
# distinguishable only if the symbols do not fall on the same abscissae.
STRATEGY_OFFSETS = {"FedAvg": 0, "FedProx": 1, "FedAdagrad": 2,
                    "FedAdam": 3, "FedYogi": 4}

STRATEGY_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx",
                  "fedadagrad": "FedAdagrad", "fedadam": "FedAdam",
                  "fedyogi": "FedYogi"}

# ------------------------------------------------------------------ metrics
# The federated runs call the metric "f1" on the client side and "f1_score"
# on the server side: hence the two label tables.
METRICS_CLIENT = ["loss", "accuracy", "f1"]
METRICS_SERVER = ["loss", "accuracy", "f1_score"]

METRIC_INFO = {
    "loss":     {"ylabel": "Loss",           "pct": False},
    "accuracy": {"ylabel": "Accuracy (%)",   "pct": True},
    "f1":       {"ylabel": "F1 score (%)",   "pct": True},
    # Same metrics read on the server side, where F1 is called f1_score.
    "f1_score": {"ylabel": "F1 score (%)",   "pct": True},
}
METRIC_LABELS = {"loss": "Loss", "accuracy": "Accuracy", "f1_score": "F1 score"}
METRIC_LABELS_PCT = {"loss": "Loss", "accuracy": "Accuracy (%)",
                     "f1": "F1 score"}

# ------------------------------------------------------------ regimes/noise
# Scale of reds: the darker, the noisier.
NOISE_COLORS = {0.005: "#e88a8a", 0.007: "#c43c3c", 0.014: "#67000d"}
NOISY = "#c43c3c"
MITIGATED = "C2"
NOISELESS = "C0"
CENTRALIZED = "0.3"

# ------------------------------------------------------------- CDR memory
# The third variant changes with the drift regime (ema_full under OU,
# warm_start under random walk) but takes the same slot, so it shares colour
# and marker.
VAR_COLOR = {"no_memory": "C0", "ema": "C1", "ema_full": "C2",
             "warm_start": "C2"}
VAR_MARKER = {"no_memory": "o", "ema": "s", "ema_full": "^",
              "warm_start": "^"}

# Markers for the curves over rounds, thinned and staggered.
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]
CLIENT_MARKERS = ["o", "s", "^", "D", "v"]
MARKER_OFFSETS = [0, 1, 2, 3]
