"""Tabelle LaTeX della tesi.

  strategy   corpi delle tabelle di 4.1 (confronto fra strategie e tuning)
  quality    soglia CDR contro costo, qualita' e stabilita' della loss
  savings    calibrazioni risparmiate dalla memoria, federato vs centralizzato
  drift      centralizzato, statico contro deriva OU
  parameters distanza nei pesi e Delta loss sotto valutazione pulita

Le prime finiscono in thesis_images/cap_04_Results/tables/, le altre nella
cartella della campagna (NEW_THESIS_IMAGES/ o NEW_THESIS_IMAGES_RW/), ciascuna
accompagnata dal CSV con le colonne che nel .tex non entrano.
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from .. import loaders, palette, paths
from . import memory

SEEDS = palette.SEEDS
METRICS = palette.METRICS_SERVER


# =====================================================================
# 4.1: confronto fra strategie e tuning di eta_l
# =====================================================================

FINAL_ROUND = 40
STRATEGY_FILES_IID = [
    ("FedAvg",     "fedavg_etal0.3_seed{seed}.json"),
    ("FedProx",    "fedprox_mu0.03_etal0.3_seed{seed}.json"),
    ("FedAdagrad", "fedadagrad_eta0.3_etal0.2_seed{seed}.json"),
    ("FedAdam",    "fedadam_eta0.2_etal0.15_seed{seed}.json"),
    ("FedYogi",    "fedyogi_eta0.1_etal0.1_seed{seed}.json"),
]
STRATEGY_FILES_NON_IID = [
    (name, "fedadam_eta0.1_etal0.1_seed{seed}.json" if name == "FedAdam"
     else pattern)
    for name, pattern in STRATEGY_FILES_IID
]
TUNING_SEEDS = [14, 71, 130]
ETA_L_VALUES = [0.001, 0.005, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35]

EXAMPLE_TABLE = r"""
% === Esempio ambiente table da usare su Overleaf ===
\begin{table}[ht]
\centering
\caption{Strategy comparison — IID, round 40.}
\label{tab:strategy_iid}
\begin{tabular}{lccc}
\toprule
Strategy & Loss & Accuracy & F1 score \\
\midrule
\input{tables/strategy_comparison_iid}
\bottomrule
\end{tabular}
\end{table}
"""


def _cell(vals, decimals=3):
    med, dev = loaders.median_mad(vals)
    return f"{med:.{decimals}f} $\\pm$ {dev:.{decimals}f}"


def _row(name, vals):
    return (f"{name} & {_cell(vals['loss'])} & {_cell(vals['accuracy'])}"
            f" & {_cell(vals['f1_score'])} \\\\")


def strategy():
    """Corpi .tex delle tabelle di 4.1, mediana +/- MAD sui sette seed."""
    out = Path(paths.ensure(paths.TABLES))
    # L'epoca 80 del centralizzato corrisponde al round 40 del federato.
    centr = loaders.centralized_final(paths.STRATEGY_CEN, SEEDS, 80)

    for dist, strategies, name in (
            ("iid", STRATEGY_FILES_IID, "strategy_comparison_iid"),
            ("non_iid", STRATEGY_FILES_NON_IID, "strategy_comparison_non_iid")):
        sim_dir = (Path(paths.STRATEGY_FED) / dist / "simulations"
                   / "simulation_experiments")
        lines = []
        for label, pattern in strategies:
            vals = loaders.server_final(sim_dir, pattern, SEEDS, FINAL_ROUND)
            if not vals["loss"].size:
                continue
            lines.append(_row(label, vals))
        if centr["loss"].size:
            lines.append("\\midrule")
            lines.append(_row("Centralized", centr))
        (out / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved {name}.tex ({len(lines)} lines)")

    lines = []
    for eta_l in ETA_L_VALUES:
        vals = loaders.server_last(
            paths.STRATEGY_TUNING, f"fedavg_etal{eta_l}_seed{{seed}}.json",
            TUNING_SEEDS)
        if not vals["loss"].size:
            continue
        lines.append(_row(str(eta_l), vals))
    (out / "fedavg_tuning_iid.tex").write_text("\n".join(lines),
                                               encoding="utf-8")
    print(f"Saved fedavg_tuning_iid.tex ({len(lines)} lines)")
    print(EXAMPLE_TABLE)


# =====================================================================
# Soglia contro costo, qualita' e stabilita'
# =====================================================================

QUALITY_CAPTION = [
    r"  \caption{Cost, quality and stability of the client-side metrics as a function",
    r"    of the CDR threshold, over seven seeds and 40 rounds under Ornstein--",
    r"    Uhlenbeck drift. $N_{\mathrm{cal}}$ is the total number of recalibrations per",
    r"    run. Every other column is read on the same window, the last ten rounds:",
    r"    $L$, accuracy and $F_1$ are per-seed medians over that window, $\sigma$ is",
    r"    the within-run standard deviation of the loss over it, and $\Delta_{\max}$",
    r"    the largest regression observed within it with respect to the best loss",
    r"    reached up to that round since the start of the run. All entries are medians",
    r"    across the seven seeds, with the MAD in parentheses. An asterisk marks a",
    r"    $\sigma$ larger than at $\epsilon_{\mathrm{th}} = 0.005$ in all seven seeds:",
    r"    the sign test is paired seed by seed, so it does not follow the ordering of",
    r"    the medians, and with seven seeds unanimity is the only significant outcome,",
    r"    since $6/7$ already gives $p = 0.125$. Two remarks. First, accuracy and $F_1$",
    r"    never degrade, not even at $\epsilon_{\mathrm{th}} = 0.2$ without memory,",
    r"    where the loss is worst: what a loose threshold degrades is the calibration",
    r"    of the predicted probabilities, while the discrete decisions are already",
    r"    saturated on this task, the same effect seen for FedAdam. Second, at equal",
    r"    threshold the EMA always has the larger $\sigma$, as expected from the",
    r"    mechanism: the averaged map is older, so the residual it leaves drifts. The",
    r"    memory buys cost and plateau loss at the price of some stability.}",
]


def quality(regime="OU", variants=("no_memory", "ema")):
    """Tabella soglia x (costo, qualita', stabilita'), un blocco per variante.

    Accompagna la figura loss-contro-round: le curve mediane delle soglie si
    sovrappongono quasi ovunque e nascondono proprio la parte che conta, cioe'
    che allentando la soglia con la memoria la loss di plateau non peggiora ma
    la sua varianza si'.
    """
    reg = memory.REGIMES[regime]
    out = Path(paths.ensure(reg.out))
    stats = {(v, t): memory.quality_stats(reg, t, v)
             for v in variants for t in reg.thresholds}
    ref = reg.thresholds[0]

    rows = []
    for v in variants:
        for t in reg.thresholds:
            e = stats[(v, t)]
            d = e["sd"] - stats[(v, ref)]["sd"]      # appaiata seed per seed
            k, n = int((d > 0).sum()), len(d)
            rows.append({
                "variant": v, "thresh": t, "n": len(e["cal"]),
                "cal": np.median(e["cal"]), "cal_mad": loaders.mad(e["cal"]),
                "loss": np.median(e["loss"]), "loss_mad": loaders.mad(e["loss"]),
                "acc": np.median(e["acc"]), "acc_mad": loaders.mad(e["acc"]),
                "f1": np.median(e["f1"]), "f1_mad": loaders.mad(e["f1"]),
                "sd": np.median(e["sd"]), "spike": np.median(e["spike"]),
                "k": k, "p": 1.0 if t == ref else loaders.sign_test(k, n),
            })

    with open(out / "tab_thresh_vs_quality.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "threshold", "n_seeds", "calib_median",
                    "calib_mad", "loss_tail_median", "loss_tail_mad",
                    "accuracy_tail", "accuracy_tail_mad", "f1_tail",
                    "f1_tail_mad", "sd_tail", "max_regression",
                    "seeds_sd_above_ref", "sign_p_vs_ref"])
        for r in rows:
            w.writerow([r["variant"], r["thresh"], r["n"], f"{r['cal']:.0f}",
                        f"{r['cal_mad']:.0f}", f"{r['loss']:.4f}",
                        f"{r['loss_mad']:.4f}", f"{r['acc']:.4f}",
                        f"{r['acc_mad']:.4f}", f"{r['f1']:.4f}",
                        f"{r['f1_mad']:.4f}", f"{r['sd']:.4f}",
                        f"{r['spike']:.4f}", f"{r['k']}/{r['n']}",
                        f"{r['p']:.3f}"])

    L = ([r"\begin{table}[!htbp]", r"  \centering", r"  \small",
          r"  \setlength{\tabcolsep}{4pt}"]
         + QUALITY_CAPTION
         + [r"  \label{tab:thresh-vs-quality}",
            r"  \begin{tabular}{lcccccc}", r"    \toprule",
            r"    $\epsilon_{\mathrm{th}}$ & $N_{\mathrm{cal}}$ & $L$ & Acc. & $F_1$"
            r" & $\sigma$ & $\Delta_{\max}$ \\",
            r"    \midrule"])
    for i, v in enumerate(variants):
        if i:
            L.append(r"    \midrule")
        L.append(rf"    \multicolumn{{7}}{{l}}{{\textbf{{{reg.labels[v]}}}}} \\")
        for r in (r for r in rows if r["variant"] == v):
            star = r"^{*}" if r["p"] <= 0.05 else ""
            L.append(
                f"    {r['thresh']} & ${r['cal']:.0f}$~({r['cal_mad']:.0f})"
                f" & ${r['loss']:.4f}$~({r['loss_mad']:.4f})"
                f" & ${r['acc']:.3f}$~({r['acc_mad']:.3f})"
                f" & ${r['f1']:.3f}$~({r['f1_mad']:.3f})"
                f" & ${r['sd']:.4f}{star}$"
                f" & ${r['spike']:.4f}$ \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "tab_thresh_vs_quality.tex").write_text("\n".join(L) + "\n",
                                                   encoding="utf-8")
    print("  tab_thresh_vs_quality.tex e .csv")
    return rows


# =====================================================================
# Calibrazioni risparmiate dalla memoria
# =====================================================================

SAVINGS_CAPTION = [
    r"  \caption{Calibrations saved by the EMA memory, under Ornstein--Uhlenbeck",
    r"    drift with $\sigma = 0.001$, $\theta = 0.3$ and $\beta = 0.2$ in both",
    r"    regimes. $N_{\mathrm{cal}}$ is the median total number of recalibrations",
    r"    over seven seeds, and the saving is the relative drop between the two",
    r"    medians, while $N_{\mathrm{cal}}/t$ is the same count per unit of protocol",
    r"    time, that is per round in the federated regime and per epoch in the",
    r"    centralized one. Neither of the first four columns should be read across the",
    r"    two blocks: the federated runs sum the three clients sampled per round, out",
    r"    of five, over 40 rounds, while the centralized ones follow a single model",
    r"    over 30 epochs, so opportunities to recalibrate and duration both differ, and",
    r"    the factor of three from the clients stays inside the federated rate. The",
    r"    comparison that matters is column by column within each regime, and it is",
    r"    carried by the saving, which is a ratio between two runs of identical",
    r"    duration, client count and configuration, differing only in the memory:",
    r"    duration and clients cancel out. Truncating the federated runs to 30 rounds,",
    r"    to match the centralized duration, leaves it unchanged, at $+9\%$, $+42\%$",
    r"    and $+85\%$ for the three thresholds. In the",
    r"    federated regime it grows with the threshold and is systematic: the EMA saves",
    r"    calibrations in every single seed at $\epsilon_{\mathrm{th}} = 0.015$ and",
    r"    $0.05$ (paired sign test, $p = 0.016$ and $p = 0.031$), and although at $0.2$",
    r"    one seed goes the other way ($5/6$, $p = 0.219$), the median cost drops by",
    r"    four fifths. In the centralized regime the saving is absent: at the tight",
    r"    threshold the memory even costs slightly more, and nowhere do more than $3/7$",
    r"    seeds improve. At $\epsilon_{\mathrm{th}} = 0.2$ the centralized model",
    r"    recalibrates once per run, so that ratio carries no information. The",
    r"    memory pays off where recalibrations are many and asynchronous across",
    r"    clients, not where a single model recalibrates a handful of times.}",
]


def savings_rows(thresholds, getter, duration=1):
    """Righe (soglia, mediane, tasso, risparmio) per un regime.

    Il risparmio in tabella e' quello fra le due mediane, cosi' che il lettore
    lo ritrovi dalle due colonne accanto. Il rapporto appaiato seed per seed,
    piu' solido ma non ricostruibile a occhio, finisce nel CSV insieme al test
    dei segni; i seed che non calibrano mai senza memoria sono esclusi dal
    rapporto perche' lo renderebbero indefinito.

    Il tasso divide per la durata del protocollo - round nel federato, epoche
    nel centralizzato - e resta quindi una frequenza interna a ciascun regime.
    Non e' una quantita' da leggere trasversalmente ai due blocchi.
    """
    rows = []
    for t in thresholds:
        a, b = getter(t, "no_memory"), getter(t, "ema")
        seeds = sorted(set(a) & set(b))
        A = np.array([a[k][0] for k in seeds], dtype=float)
        B = np.array([b[k][0] for k in seeds], dtype=float)
        ok = A > 0
        paired = 100 * (1 - B[ok] / A[ok])
        k = int((paired > 0).sum())
        rows.append({
            "thresh": t, "n": len(seeds), "nomem": np.median(A),
            "ema": np.median(B),
            "rate_nomem": np.median(A) / duration,
            "rate_ema": np.median(B) / duration,
            "saving": 100 * (1 - np.median(B) / np.median(A)),
            "paired": np.median(paired) if len(paired) else float("nan"),
            "k": k, "n_paired": len(paired),
            "p": loaders.sign_test(k, len(paired)),
        })
    return rows


def savings(regime="OU", thresholds=(0.015, 0.05, 0.2)):
    """Quanto costo toglie la memoria, federato contro centralizzato.

    Le soglie in tabella sono tre fra quelle presenti in entrambe le campagne,
    distribuite sul range. Nel centralizzato non esiste 0.1, quindi la riga
    larga e' 0.2, che c'e' da entrambe le parti.
    """
    reg = memory.REGIMES[regime]
    out = Path(paths.ensure(reg.out))

    def federated(t, v):
        return loaders.calibration_map(memory.sweep_files(reg, t, v))

    def centralized(t, v):
        return loaders.centralized_calibrations(
            f"{memory.CEN_MEMORY_DIR[t]}/{v}")

    regimes = [("Federated", federated, reg.thresholds, reg.duration)]
    if reg.centralized:
        regimes.append(("Centralized", centralized,
                        sorted(memory.CEN_MEMORY_DIR), memory.CEN_DURATION))

    with open(out / "tab_memory_savings.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regime", "threshold", "n_seeds", "calib_no_memory_median",
                    "calib_ema_median", "calib_per_unit_no_memory",
                    "calib_per_unit_ema", "saving_pct_from_medians",
                    "saving_pct_paired_median", "seeds_with_saving", "sign_p"])
        for label, getter, full, dur in regimes:
            for r in savings_rows(full, getter, dur):
                w.writerow([label.lower(), r["thresh"], r["n"],
                            f"{r['nomem']:.0f}", f"{r['ema']:.0f}",
                            f"{r['rate_nomem']:.1f}", f"{r['rate_ema']:.1f}",
                            f"{r['saving']:.1f}", f"{r['paired']:.1f}",
                            f"{r['k']}/{r['n_paired']}", f"{r['p']:.3f}"])

    L = ([r"\begin{table}[!htbp]", r"  \centering", r"  \small"]
         + SAVINGS_CAPTION
         + [r"  \label{tab:memory-savings}",
            r"  \begin{tabular}{lccccc}", r"    \toprule",
            r"    & \multicolumn{2}{c}{$N_{\mathrm{cal}}$}"
            r" & \multicolumn{2}{c}{$N_{\mathrm{cal}}/t$} & \\",
            r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"    $\epsilon_{\mathrm{th}}$ & No memory & EMA & No memory & EMA"
            r" & Saving \\",
            r"    \midrule"])
    for i, (label, getter, _, dur) in enumerate(regimes):
        if i:
            L.append(r"    \midrule")
        L.append(rf"    \multicolumn{{6}}{{l}}{{\textbf{{{label}}}}} \\")
        for r in savings_rows(thresholds, getter, dur):
            L.append(f"    {r['thresh']} & ${r['nomem']:.0f}$ & ${r['ema']:.0f}$"
                     f" & ${r['rate_nomem']:.1f}$ & ${r['rate_ema']:.1f}$"
                     f" & ${r['saving']:+.0f}\\%$ \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "tab_memory_savings.tex").write_text("\n".join(L) + "\n",
                                                encoding="utf-8")
    print("  tab_memory_savings.tex e .csv")


# =====================================================================
# Centralizzato: statico contro deriva OU
# =====================================================================

def drift(regime="OU"):
    """Ricalibrazioni del centralizzato con e senza deriva, appaiate per seed."""
    reg = memory.REGIMES[regime]
    out = Path(paths.ensure(reg.out))

    def total(sub, seed):
        f = glob.glob(f"{paths.CEN_MITIGATED}/{sub}/*seed{seed}.json")
        if not f:
            return None
        with open(f[0], encoding="utf-8") as fh:
            return json.load(fh)["info"].get("total_calibrations")

    rows, ratios = [], []
    for s in SEEDS:
        a, b = total("static", s), total("ou_drift", s)
        if a and b:
            rows.append((s, a, b, b / a))
            ratios.append(b / a)
    ratios = np.array(ratios)
    k = int((ratios > 1).sum())

    L = [
        r"\begin{table}[!htbp]", r"  \centering", r"  \small",
        r"  \caption{Centralized pipeline: total recalibrations over 40 epochs without",
        r"    drift and under Ornstein--Uhlenbeck drift at $\sigma = 0.001$, $\theta = 0.3$,",
        r"    at threshold $\tau = 0.015$. All other settings are identical, and the",
        r"    comparison is paired by seed. Unlike the federated case, where the same",
        r"    contrast yields a ratio of $1.02$ ($4/7$ seeds, $p = 1.000$), here the drift",
        r"    more than doubles the calibration cost.}",
        r"  \label{tab:centralized-drift}",
        r"  \begin{tabular}{lccc}", r"    \toprule",
        r"    Seed & Static & OU drift & Ratio \\", r"    \midrule",
    ]
    for s, a, b, r in rows:
        L.append(f"    {s} & {a} & {b} & ${r:.2f}$ \\\\")
    L += [
        r"    \midrule",
        f"    Median & ${np.median([r[1] for r in rows]):.0f}$ &"
        f" ${np.median([r[2] for r in rows]):.0f}$ &"
        f" $\\mathbf{{{np.median(ratios):.2f}}}$ \\\\",
        r"    \multicolumn{3}{l}{Seeds with drift $>$ static}"
        f" & ${k}/{len(rows)}$ \\\\",
        r"    \multicolumn{3}{l}{Two-sided sign test}"
        f" & $p = {loaders.sign_test(k, len(rows)):.3f}$ \\\\",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ]
    (out / "tab_centralized_drift.tex").write_text("\n".join(L) + "\n",
                                                   encoding="utf-8")
    print(f"  tab_centralized_drift.tex  (rapporto mediano {np.median(ratios):.2f},"
          f" {k}/{len(rows)}, p={loaders.sign_test(k, len(rows)):.3f})")


# =====================================================================
# Parametri: distanza nei pesi e Delta loss sotto valutazione pulita
# =====================================================================

PARAMETERS_CAPTION = [
    r"  \caption{Parameter-level summary over seven seeds. For each configuration,",
    r"    $d$ is the distance in parameter space from the noiseless solution of the",
    r"    same pipeline and seed (median, MAD in parentheses), computed per parameter",
    r"    on the torus and aggregated in Euclidean norm; in practice no wrapping",
    r"    occurs, as per-parameter displacements never exceed $0.2\pi$. $\Delta L$ is",
    r"    the paired median of $L(\theta_{\text{noise}}) - L(\theta_{\text{noiseless}})$",
    r"    with both sets of weights evaluated through the \emph{noiseless} channel, so",
    r"    that it isolates the quality of the learned parameters from the penalty of a",
    r"    noisy readout. An asterisk marks significance under a two-sided sign test",
    r"    ($p \leq 0.05$; with seven seeds the attainable minimum is $0.016$). Only the",
    r"    unmitigated federated runs show a degradation of the learned solution.}",
]


def parameters(regime="OU"):
    """Distanza nei pesi e Delta loss valutata attraverso il canale pulito.

    Richiede qibo: ogni set di pesi va rivalutato senza rumore (~200
    valutazioni, un paio di minuti). Gli import stanno dentro la funzione
    perche' tutte le altre tabelle si producono senza.
    """
    reg = memory.REGIMES[regime]
    out = Path(paths.ensure(reg.out))

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.nn as nn
    import qibo
    qibo.config.log.setLevel("ERROR")
    from qibo import set_backend
    set_backend("numpy")
    import qibo_qfl_pt.patches  # noqa: F401
    from qibo_qfl_pt.task import set_seed, create_model, set_weights

    rng = np.random.default_rng(40)
    x = rng.uniform(-1, 1, size=(200, 2))
    xe = torch.tensor(x, dtype=torch.float64)
    ye = torch.tensor((np.linalg.norm(x, axis=1) <= 1.0).astype(np.float64))

    set_seed(1)
    model = create_model(model_type="quantum", noise_model=None, nshots=None)
    loss_fn = nn.BCELoss()
    cache = {}

    def clean_loss(path):
        if path not in cache:
            w = list(np.load(path).values())[0]
            set_weights(model, [w] if np.ndim(w) == 1 else list(w))
            with torch.no_grad():
                cache[path] = loss_fn(model(xe).squeeze(), ye).item()
        return cache[path]

    configs = [("fed", "noisy"), ("fed", "mitigated"),
               ("cen", "noisy"), ("cen", "mitigated")]
    res = {}
    for kind, noise_regime in configs:
        for p in memory.NOISE_LEVELS:
            dists, dloss = [], []
            for s in SEEDS:
                ref, run = memory.parameter_paths(kind, noise_regime, p, s)
                if not (os.path.exists(ref) and os.path.exists(run)):
                    continue
                dists.append(np.linalg.norm(
                    loaders.wrap(loaders.load_weights(ref)
                                 - loaders.load_weights(run))))
                dloss.append(clean_loss(run) - clean_loss(ref))
            dists, dloss = np.array(dists), np.array(dloss)
            k = int((dloss > 0).sum())
            res[(kind, noise_regime, p)] = {
                "n": len(dists), "d": np.median(dists),
                "mad": loaders.mad(dists), "dl": np.median(dloss), "k": k,
                "p": loaders.sign_test(k, len(dists)),
            }

    with open(out / "tab_parameters.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "regime", "noise", "n_seeds", "dist_median_rad",
                    "dist_mad_rad", "delta_loss_median", "seeds_positive",
                    "sign_p"])
        for (kind, noise_regime), p in ((c, p) for c in configs
                                        for p in memory.NOISE_LEVELS):
            e = res[(kind, noise_regime, p)]
            w.writerow([kind, noise_regime, p, e["n"], f"{e['d']:.4f}",
                        f"{e['mad']:.4f}", f"{e['dl']:.5f}",
                        f"{e['k']}/{e['n']}", f"{e['p']:.3f}"])

    L = ([r"\begin{table}[!htbp]", r"  \centering", r"  \small",
          r"  \setlength{\tabcolsep}{4pt}"]
         + PARAMETERS_CAPTION
         + [r"  \label{tab:new-parameters}",
            r"  \begin{tabular}{lcccccccc}", r"    \toprule",
            r"    & \multicolumn{2}{c}{Fed.\ noisy} & \multicolumn{2}{c}{Fed.\ mitigated}"
            r" & \multicolumn{2}{c}{Cen.\ noisy} & \multicolumn{2}{c}{Cen.\ mitigated} \\",
            r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"    $p$ & $d$ [rad] & $\Delta L$ & $d$ [rad] & $\Delta L$"
            r" & $d$ [rad] & $\Delta L$ & $d$ [rad] & $\Delta L$ \\",
            r"    \midrule"])
    for p in memory.NOISE_LEVELS:
        cells = []
        for kind, noise_regime in configs:
            e = res[(kind, noise_regime, p)]
            star = r"$^{*}$" if e["p"] <= 0.05 else ""
            cells.append(f"${e['d']:.3f}$~({e['mad']:.3f})")
            cells.append(f"${e['dl']:+.4f}${star}")
        L.append(f"    {p:.3f} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "tab_parameters.tex").write_text("\n".join(L) + "\n",
                                            encoding="utf-8")
    print(f"  scritti tab_parameters.tex e .csv ({len(cache)} valutazioni)")
    return res


def memory_tables(regime="OU"):
    """Le tabelle della campagna sulla memoria.

    tab_parameters resta fuori: e' l'unica che rivaluta il modello e va chiesta
    a parte (tables.parameters), altrimenti un giro completo delle figure si
    ferma qui dove qibo non c'e'.
    """
    quality(regime)
    savings(regime)
    if regime == "OU":
        drift(regime)


FIGURES = {
    "tables.strategy": strategy,
    "tables.memory_ou": lambda: memory_tables("OU"),
    "tables.memory_rw": lambda: memory_tables("RW"),
    "tables.parameters": parameters,
}
