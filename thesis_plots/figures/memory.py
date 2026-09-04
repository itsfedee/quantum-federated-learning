"""Sweep sulla soglia CDR e memoria della mappa.

Lo sweep esiste in due campagne, che differiscono per il processo che muove il
rumore: Ornstein-Uhlenbeck a 40 round e random walk a 10. Cambiano le soglie
disponibili, la terza variante di memoria (ema_full contro warm_start), la
finestra di coda per le statistiche di plateau e la spaziatura dei marcatori.
Tutto il resto e' identico, quindi il regime e' un parametro e non una copia
dello script.

Le figure e le tabelle di questo modulo finiscono in NEW_THESIS_IMAGES/ (OU) o
NEW_THESIS_IMAGES_RW/ (random walk).
"""
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import draw, loaders, noise_model, palette, paths
from .. import style as ts

SEEDS = palette.SEEDS
NOISE_LEVELS = [0.003, 0.006, 0.01, 0.014, 0.018, 0.022]


@dataclass
class Regime:
    """Profilo di una campagna di sweep."""
    name: str
    sweep: str
    thresholds: list
    thresh_dir: dict
    variants: list
    labels: dict
    quality_thresholds: tuple
    tail: int           # round di coda su cui si leggono qualita' e stabilita'
    mark_every: int
    duration: int       # round della campagna, per i tassi in tabella
    out: str
    seeds: list = None  # None: nessun filtro, le cartelle hanno gia' 7 seed
    centralized: bool = False   # esiste il blocco centralizzato in tabella


REGIMES = {
    "OU": Regime(
        name="OU",
        sweep=paths.SWEEP_OU,
        thresholds=[0.005, 0.015, 0.03, 0.05, 0.1, 0.2],
        thresh_dir={0.005: "thresh_0005", 0.015: "thresh_0015",
                    0.03: "thresh_003", 0.05: "thresh_005",
                    0.1: "thresh_01", 0.2: "thresh_02"},
        variants=["no_memory", "ema", "ema_full"],
        labels={"no_memory": "No memory", "ema": "EMA inter-round",
                "ema_full": "EMA full"},
        quality_thresholds=(0.015, 0.05, 0.1, 0.2),
        tail=10,
        mark_every=8,
        duration=40,
        out=paths.MEMORY_OUT_OU,
        centralized=True,
    ),
    "RW": Regime(
        name="RW",
        sweep=paths.SWEEP_RW,
        # 0.0 e' esclusa: soglia nulla significa ricalibrare a ogni check.
        thresholds=[0.005, 0.015, 0.03, 0.05, 0.2],
        thresh_dir={0.005: "thresh_0005", 0.015: "thresh_0015",
                    0.03: "thresh_003", 0.05: "thresh_005",
                    0.2: "thresh_02"},
        variants=["no_memory", "ema", "warm_start"],
        labels={"no_memory": "No memory", "ema": "EMA inter-round",
                "warm_start": "Warm start"},
        quality_thresholds=(0.015, 0.03, 0.05, 0.2),
        tail=5,             # su 10 round la coda non puo' essere di 10
        mark_every=2,
        duration=10,
        out=paths.MEMORY_OUT_RW,
        seeds=list(range(1, 8)),
    ),
}

# Sweep di soglia centralizzato con memoria, sotto la stessa deriva OU del
# federato (sigma = 0.001, theta = 0.3, beta = 0.2) ma su 30 epoche invece di
# 40 round: e' una campagna precedente. Le soglie 0.03 e 0.05 compaiono anche
# in memory_centralized_ou_sweep con gli stessi identici sette seed, e 0.015 e'
# duplicata in memory_centralized/{ema,no_memory} senza sottocartella: qui si
# legge una sola copia di ciascuna, e dall'ou_sweep solo 0.08 e 0.12, che
# stanno unicamente li'.
CEN_MEMORY_DIR = {
    0.005: "MEMORY_RESULTS_OLD/memory_centralized/thresh_0005",
    0.015: "MEMORY_RESULTS_OLD/memory_centralized/thresh_0015",
    0.03: "MEMORY_RESULTS_OLD/memory_centralized/thresh_003",
    0.05: "MEMORY_RESULTS_OLD/memory_centralized/thresh_005",
    0.08: "MEMORY_RESULTS_OLD/memory_centralized_ou_sweep/thresh_008",
    0.12: "MEMORY_RESULTS_OLD/memory_centralized_ou_sweep/thresh_012",
    0.2: "MEMORY_RESULTS_OLD/memory_centralized/thresh_02",
}
CEN_DURATION = 30


def sweep_files(reg, threshold, variant):
    """JSON dello sweep per (soglia, variante), filtrati sui seed del profilo."""
    files = loaders.run_files(f"{reg.sweep}/{reg.thresh_dir[threshold]}/{variant}")
    if reg.seeds is None:
        return files
    keep = set(reg.seeds)
    return [f for f in files if loaders.seed_of(f) in keep]


def _out(reg, name):
    return f"{paths.ensure(reg.out)}/{name}"


# =====================================================================
# Costo: ricalibrazioni contro soglia e contro round
# =====================================================================

def threshold_sweep(reg, logscale=False):
    """Ricalibrazioni totali contro soglia, una curva per variante di memoria.

    Asse x categorico: le soglie sono equispaziate a prescindere dal valore,
    come nelle figure gia' in tesi. Su asse numerico le prime tre si
    accavallerebbero, e il logaritmo renderebbe la spaziatura poco leggibile.
    L'asse y resta lineare, come le altre figure del capitolo: il prezzo e' che
    alle soglie larghe i conteggi si schiacciano, ma il messaggio e' il calo,
    non il valore assoluto, che sta in tabella.
    """
    fig, ax = ts.single()
    x = np.arange(len(reg.thresholds))
    for v in reg.variants:
        totals = [loaders.total_calibrations(sweep_files(reg, t, v))
                  for t in reg.thresholds]
        med = np.array([np.median(c) for c in totals])
        dev = np.array([loaders.mad(c) for c in totals])
        ax.plot(x, med, color=palette.VAR_COLOR[v], lw=ts.LINEWIDTH,
                label=reg.labels[v], marker=palette.VAR_MARKER[v], ms=5)
        draw.band(ax, x, med, dev, palette.VAR_COLOR[v],
                  floor=0.5 if logscale else 0.0)
    if logscale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in reg.thresholds])
    ax.set_xlabel(r"CDR threshold $\epsilon_{\mathrm{th}}$")
    ax.set_ylabel("Total calibrations")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ts.save(fig, _out(reg, "04_threshold_sweep_median_mad"))


def cumulative(reg):
    """Ricalibrazioni cumulate contro round, una curva per soglia (EMA)."""
    fig, ax = ts.single()
    colors = plt.cm.viridis(np.linspace(0, .88, len(reg.thresholds)))
    for i, (t, c) in enumerate(zip(reg.thresholds, colors)):
        rounds, M = loaders.cumulative_calibrations(sweep_files(reg, t, "ema"))
        if not len(M):
            continue
        med = np.median(M, axis=0)
        dev = np.array([loaders.mad(M[:, j]) for j in range(M.shape[1])])
        ax.plot(rounds, med, color=c, lw=ts.LINEWIDTH,
                marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                ms=3.2, markevery=(i, reg.mark_every),
                label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
        draw.band(ax, rounds, med, dev, c, floor=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative calibrations")
    ax.set_xlim(1, None)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ts.save(fig, _out(reg, "04_cumulative_calibrations"))


def savings(reg):
    """Calibrazioni risparmiate dalla memoria, cumulate, per soglia.

    Si plotta la differenza no_memory - EMA appaiata per seed, non i due
    totali: essendo una differenza sta nello stesso intervallo per tutte le
    soglie, mentre i totali coprono tre ordini di grandezza e schiacciano le
    soglie larghe. L'asse y e' cosi' direttamente il beneficio della memoria.
    """
    fig, ax = ts.single()
    for i, t in enumerate(reg.thresholds):
        _, A = loaders.cumulative_calibrations(sweep_files(reg, t, "no_memory"))
        rounds, B = loaders.cumulative_calibrations(sweep_files(reg, t, "ema"))
        if not len(A) or not len(B):
            continue
        n = min(A.shape[0], B.shape[0])
        m = min(A.shape[1], B.shape[1])
        D = A[:n, :m] - B[:n, :m]          # appaiata seed per seed
        med = np.median(D, axis=0)
        dev = np.array([loaders.mad(D[:, j]) for j in range(m)])
        c = f"C{i}"
        ax.plot(rounds[:m], med, color=c, lw=ts.LINEWIDTH,
                marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                ms=3.2, markevery=(i, reg.mark_every),
                label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
        draw.band(ax, rounds[:m], med, dev, c)
    ax.axhline(0, color="0.6", lw=.6, zorder=0)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Saved calibrations")
    ax.set_xlim(1, None)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ts.save(fig, _out(reg, "04_calibration_savings_vs_round"))


# =====================================================================
# Qualita': metriche client contro round, una curva per soglia
# =====================================================================

def quality_panels(reg, variant="ema"):
    """Loss, accuracy e F1 lato client contro il round, una curva per soglia.

    Tre pannelli affiancati, stessa scala dei round ma ordinate indipendenti:
    loss e metriche di classificazione vivono su intervalli diversi.
    """
    keys = [("loss", "Client evaluation loss"),
            ("accuracy", "Client accuracy"),
            ("f1", "Client $F_1$")]
    fig, axes = ts.panels(3, sharex=True)
    for ax, (key, ylabel) in zip(axes, keys):
        for i, t in enumerate(reg.quality_thresholds):
            rounds, M = loaders.client_curves(sweep_files(reg, t, variant), key)
            if not len(M):
                continue
            med = np.median(M, axis=0)
            dev = np.array([loaders.mad(M[:, j]) for j in range(M.shape[1])])
            c = f"C{i}"
            ax.plot(rounds, med, color=c, lw=ts.LINEWIDTH,
                    marker=palette.LINE_MARKERS[i % len(palette.LINE_MARKERS)],
                    ms=3.2, markevery=(i, reg.mark_every),
                    label=rf"$\epsilon_{{\mathrm{{th}}}} = {t}$")
            draw.band(ax, rounds, med, dev, c)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xlabel("Communication round $t$", fontsize=8.5)
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 10, 20, 30, 40])
        # Pochi tick ed etichette piccole: in un pannello largo poco piu' di
        # due pollici la spaziatura di default sbava su quello accanto.
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=8)
    # Legenda unica sopra i pannelli: dentro un pannello stretto quattro voci
    # coprirebbero le curve, e ripeterla tre volte sarebbe ridondante.
    fig.legend(handles=axes[0].get_lines(), frameon=False, fontsize=8.5,
               ncol=4, loc="outside upper center", columnspacing=1.6,
               handlelength=1.8)
    ts.save(fig, _out(reg, "04_thresh_vs_quality"))


def quality_stats(reg, threshold, variant):
    """Costo, qualita' e stabilita' di un punto dello sweep, un valore per seed.

    Le due misure di stabilita' non sono ridondanti: sigma e' la dispersione
    tipica attorno al plateau, spike la coda, cioe' la peggiore regressione
    osservata nella finestra rispetto alla migliore loss raggiunta fino a quel
    round dall'inizio della run - quanto costa il singolo episodio in cui la
    mappa CDR resta stantia troppo a lungo. Il minimo corrente si accumula su
    tutta la traiettoria, ma il massimo si legge solo nella finestra, come
    tutto il resto della tabella.
    """
    files = sweep_files(reg, threshold, variant)
    _, L = loaders.client_curves(files, "loss")
    _, A = loaders.client_curves(files, "accuracy")
    _, F = loaders.client_curves(files, "f1")
    regression = L - np.minimum.accumulate(L, axis=1)
    tail = reg.tail
    return {
        "cal": loaders.total_calibrations(files),
        "loss": np.median(L[:, -tail:], axis=1),
        "acc": np.median(A[:, -tail:], axis=1),
        "f1": np.median(F[:, -tail:], axis=1),
        "sd": L[:, -tail:].std(axis=1, ddof=1),
        "spike": regression[:, -tail:].max(axis=1),
    }


# =====================================================================
# Distanza nei pesi e deriva del rumore (solo OU)
# =====================================================================

# Serie della figura sulle distanze: pipeline, regime, etichetta, colore,
# marcatore.
WEIGHT_SERIES = [
    ("fed", "noisy", "Federated noisy", "C0", "o"),
    ("fed", "mitigated", "Federated mitigated", "C1", "s"),
    ("cen", "noisy", "Centralized noisy", "C4", "D"),
    ("cen", "mitigated", "Centralized mitigated", "C2", "^"),
]


def parameter_paths(kind, regime, p, seed):
    """Pesi di riferimento (noiseless) e della run, per una configurazione."""
    if kind == "fed":
        return (f"{paths.FEDAVG_IID}/noiseless/weights/"
                f"FedAvg_etal0.3_seed{seed}.npz",
                f"{paths.static_noise(paths.FEDAVG_IID, regime, p)}/weights/"
                f"FedAvg_etal0.3_seed{seed}.npz")
    return (f"{paths.CENTRALIZED}/weights/lr_0.3_seed{seed}.npz",
            f"{paths.CENTRALIZED}/{regime}/p{p}/nshots_1000/weights/"
            f"lr_0.3_seed{seed}.npz")


def weight_distances(kind, regime, p):
    """Distanze dalla soluzione noiseless, un valore per seed disponibile."""
    import os
    out = []
    for seed in SEEDS:
        ref, run = parameter_paths(kind, regime, p, seed)
        if os.path.exists(ref) and os.path.exists(run):
            out.append(np.linalg.norm(
                loaders.wrap(loaders.load_weights(ref)
                             - loaders.load_weights(run))))
    return out


def weight_distance(reg=None):
    """Distanza dei pesi dalla soluzione noiseless, quattro pipeline.

    Stessa quantita' della prima colonna di tab_parameters: differenza per
    parametro avvolta sul toro e aggregata in norma euclidea, mediana sui sette
    seed con banda MAD. Asse x categorico perche' i livelli di rumore non sono
    equispaziati (0.003 dista 0.003 dal successivo, gli altri 0.004).
    """
    reg = reg or REGIMES["OU"]
    fig, ax = ts.single()
    x = np.arange(len(NOISE_LEVELS))
    for kind, regime, label, color, marker in WEIGHT_SERIES:
        med, dev = [], []
        for p in NOISE_LEVELS:
            d = weight_distances(kind, regime, p)
            med.append(np.median(d) if d else np.nan)
            dev.append(loaders.mad(d) if d else np.nan)
        med, dev = np.array(med), np.array(dev)
        ax.plot(x, med, color=color, lw=ts.LINEWIDTH, marker=marker, ms=5,
                label=label)
        draw.band(ax, x, med, dev, color, floor=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in NOISE_LEVELS])
    ax.set_xlabel(r"Noise level $p$")
    ax.set_ylabel(r"Weight distance $\|\boldsymbol{\theta} - "
                  r"\boldsymbol{\theta}_{\mathrm{noiseless}}\|$")
    ax.set_ylim(bottom=0)
    # Legenda interna: le curve salgono da sinistra a destra e l'angolo in alto
    # a sinistra resta libero anche con la banda piu' larga.
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ts.save(fig, _out(reg, "04_weight_distance"))


# Parametri della campagna federata OU, letti dall'info delle run.
DRIFT = dict(base=0.005, scale=0.002, sigma=0.001, theta=0.3)


def noise_drift(reg=None, seed=1, n_rounds=10, n_clients=5):
    """Deriva OU del rumore, una curva per client, Pauli e readout affiancati."""
    reg = reg or REGIMES["OU"]
    pauli, readout = noise_model.trajectories(
        noise_model.WALK_SEED_MAP[seed], n_rounds=n_rounds,
        n_clients=n_clients, **DRIFT)
    rounds = np.arange(n_rounds + 1)

    fig, axes = ts.panels(2, sharex=True, sharey=True)
    for ax, data, label in ((axes[0], pauli, r"Pauli $p(t)$"),
                            (axes[1], readout, r"Readout $r(t)$")):
        for pid in range(n_clients):
            ax.plot(rounds, data[pid], color=f"C{pid}", lw=1.3,
                    marker=palette.LINE_MARKERS[pid % len(palette.LINE_MARKERS)],
                    ms=3.5, label=f"Client {pid}")
        # Il valore base e' il centro verso cui il processo OU richiama la
        # deriva, non il punto di partenza dei client: la dispersione statica
        # li sparpaglia gia' al round 0.
        ax.axhline(DRIFT["base"], color="0.6", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_xlim(0, n_rounds)
        ax.tick_params(labelsize=8)
    fig.legend(handles=axes[0].get_lines()[:n_clients], frameon=False,
               fontsize=8.5, ncol=n_clients, loc="outside upper center",
               columnspacing=1.6, handlelength=1.8)
    ts.save(fig, _out(reg, "04_noise_drift_clients"))


# =====================================================================
# Strategie sotto deriva OU, una figura per strategia
# =====================================================================

REGIME_ARMS = (("noiseless", "Noiseless", "tab:blue", "s"),
               ("noisy", "Noisy ($p=0.005$)", "tab:red", "o"),
               ("mitigated", "Mitigated ($p=0.005$)", "tab:green", "^"))


def strategy_sources(reg, strat, regime):
    """Cartella dei JSON per (strategia, regime) sotto deriva OU.

    FedAvg non e' in strategies_OU_drift_10r: le sue run stanno altrove. Il
    mitigato viene dallo sweep a 40 round con soglia 0.015 e no_memory, che ha
    configurazione identica alle altre strategie e di cui si usano i primi 10
    round; il confronto resta quindi controllato.
    """
    if regime == "noiseless":
        return f"{paths.IID}/{strat}/noiseless"
    if strat != "fedavg":
        return f"{paths.DRIFT_OU}/iid/{strat}/{regime}"
    if regime == "noisy":
        return f"{reg.sweep}/noisy_10r"
    return f"{reg.sweep}/thresh_0015/no_memory"


def strategy_drift(reg=None):
    """Un pannello 1x3 per strategia: noiseless, noisy e mitigato sotto deriva.

    Legenda ancorata a mano invece che dal layout constrained, e PNG a 200 dpi:
    e' la resa con cui questa serie e' stata prodotta la prima volta, tenuta
    per non cambiare file che stanno gia' nella cartella della campagna.
    """
    reg = reg or REGIMES["OU"]
    outdir = Path(paths.ensure(f"{reg.out}/strategies_drift_plots"))
    for strat in palette.STRATEGY_NAMES:
        arms = {label: (loaders.client_metrics(
                            loaders.run_files(strategy_sources(reg, strat, regime)),
                            n_rounds=11), color, marker)
                for regime, label, color, marker in REGIME_ARMS}
        fig, axes = ts.panels(3)
        for arm_idx, (label, (data, color, marker)) in enumerate(arms.items()):
            mevery = (palette.MARKER_OFFSETS[arm_idx % len(palette.MARKER_OFFSETS)], 3)
            for ax, m in zip(axes, palette.METRICS_CLIENT):
                if not data[m]:
                    continue
                n = min(len(v) for v in data[m])
                mat = np.array([v[:n] for v in data[m]])
                if m == "accuracy":
                    mat = mat * 100
                med, dev = loaders.median_mad(mat)
                ax.plot(np.arange(n), med, color=color, ls="-", lw=1.8,
                        label=label, marker=marker, markersize=ts.MARKERSIZE,
                        markevery=mevery)
                ax.fill_between(np.arange(n), med - dev, med + dev,
                                alpha=0.15, color=color)
        for ax, m in zip(axes, palette.METRICS_CLIENT):
            ax.set_xlabel("Communication round $t$")
            ax.set_ylabel(palette.METRIC_LABELS_PCT[m])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8,
                   ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        out = outdir / f"{strat}_ou_drift"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  strategies_drift_plots/{strat}_ou_drift.pdf / .png")


# =====================================================================
# Sweep CDR delle campagne precedenti (cartelle allo stesso livello del repo)
# =====================================================================

CDR_MODES = ["no_memory", "ema", "warm_start"]
CDR_LABELS = {"no_memory": "No memory", "ema": "EMA", "warm_start": "Warm start"}
CDR_THRESHOLDS = [0.005, 0.015, 0.03, 0.05, 0.2]
CDR_THRESH_DIR = {0.005: "thresh_0005", 0.015: "thresh_0015",
                  0.03: "thresh_003", 0.05: "thresh_005", 0.2: "thresh_02"}
CDR_NOISE_LEVELS = [0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]


# Queste quattro figure sono nate prima dello stile condiviso, con i soli tre
# rcParams qui sotto sul resto dei default di matplotlib. Riprodurle dentro
# rc_context le tiene identiche ai PDF gia' inclusi nel documento.
CDR_RC = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.labelsize": "medium", "axes.titlesize": "large",
    "xtick.labelsize": "medium", "ytick.labelsize": "medium",
    "legend.fontsize": "medium", "lines.linewidth": 1.5,
    "lines.markersize": 6.0, "axes.grid": False,
}


def _mean_std(values):
    return np.mean(values), np.std(values)


def _median_mad(values):
    return np.median(values), loaders.mad(values)


def _cdr_curve(ax, x_values, base_dir, x_map, stat):
    for mode in CDR_MODES:
        centers, spreads, valid = [], [], []
        for xv in x_values:
            d = Path(base_dir) / x_map[xv] / mode
            if not d.exists():
                continue
            vals = loaders.declared_calibrations(d, SEEDS)
            if len(vals) == 0:
                continue
            c, s = stat(vals)
            centers.append(c)
            spreads.append(s)
            valid.append(xv)
        x, mu, sigma = np.array(valid), np.array(centers), np.array(spreads)
        ax.plot(x, mu, color=palette.VAR_COLOR[mode], lw=1.2,
                label=CDR_LABELS[mode], marker=palette.VAR_MARKER[mode], ms=5)
        ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.15,
                        color=palette.VAR_COLOR[mode])


def cdr_sweep():
    """Ricalibrazioni contro soglia e contro rumore, campagne pre-OU.

    Due statistiche per ciascuno sweep: media con deviazione standard e
    mediana con MAD. La prima e' quella storica, la seconda quella usata dal
    resto del capitolo.
    """
    out = Path(paths.ensure(paths.CAP04))
    stats = [(_mean_std, "mean_std"), (_median_mad, "median_mad")]

    with plt.rc_context(CDR_RC):
        _cdr_sweep_figures(out, stats)


def _cdr_sweep_figures(out, stats):
    for stat, suffix in stats:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        _cdr_curve(ax, CDR_THRESHOLDS, paths.THRESH_SWEEP_VARWALK,
                   CDR_THRESH_DIR, stat)
        ax.set_xlabel(r"CDR threshold $\tau$")
        ax.set_ylabel("Total calibrations")
        ax.set_xscale("log")
        ax.set_xticks(CDR_THRESHOLDS)
        ax.set_xticklabels([str(t) for t in CDR_THRESHOLDS], fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(bottom=-5)
        _cdr_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(out / f"04_threshold_sweep_{suffix}.pdf", bbox_inches="tight")
        fig.savefig(out / f"threshold_sweep_{suffix}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved threshold sweep ({suffix})")

    noise_dir = {p: f"p{str(p).replace('.', '')}" for p in CDR_NOISE_LEVELS}
    for stat, suffix in stats:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        _cdr_curve(ax, CDR_NOISE_LEVELS, paths.NOISE_SWEEP_TH015, noise_dir, stat)
        ax.set_xlabel(r"Base noise level $p$")
        ax.set_ylabel("Total calibrations")
        _cdr_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(out / f"04_noise_sweep_{suffix}.pdf", bbox_inches="tight")
        fig.savefig(out / f"noise_sweep_{suffix}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved noise sweep ({suffix})")


def _cdr_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.15))


# =====================================================================

def all_figures(regime="OU"):
    """Le figure di una campagna, nell'ordine in cui venivano prodotte."""
    reg = REGIMES[regime]
    print(f">>> Regime {reg.name}, output in {reg.out}/")
    threshold_sweep(reg)
    savings(reg)
    cumulative(reg)
    quality_panels(reg)
    if reg.name == "OU":
        # Distanze nei pesi e traiettorie di rumore non dipendono dallo sweep
        # sulla memoria: si producono una volta sola, nel profilo OU.
        weight_distance(reg)
        noise_drift(reg)


FIGURES = {
    "memory.ou": lambda: all_figures("OU"),
    "memory.rw": lambda: all_figures("RW"),
    "memory.strategy_drift": strategy_drift,
    "memory.cdr_sweep": cdr_sweep,
}
