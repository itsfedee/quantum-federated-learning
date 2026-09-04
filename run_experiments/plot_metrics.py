import json
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from collections import defaultdict
from scipy.stats import median_abs_deviation
import glob

import torch
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from qibo_qfl_pt.task import create_model, set_weights, build_noise_model

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
})

# =====================================================================
# Core
# =====================================================================

METRIC_INFO = {
    "loss":     {"ylabel": "Loss",         "pct": False, "title": "Loss"},
    "accuracy": {"ylabel": "Accuracy (%)", "pct": True,  "title": "Accuracy"},
    "f1":       {"ylabel": "F1",           "pct": True,  "title": "F1 score"},
}

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan',
          'darkblue', 'darkred', 'darkgreen']


def _read_folder(folder, source="eval_metrics_client", save_json=False, seeds=None):
    """Legge tutti i JSON in una cartella, raggruppa per config, calcola mediana+MAD.

    seeds: lista di seed da includere (es. [1,2,3,5,6,7]). Se None, usa tutti.
    """
    files = glob.glob(f"{folder}/*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return {}

    if seeds is not None:
        seed_strs = {f"_seed{s}.json" for s in seeds}
        files = [f for f in files if any(f.endswith(s) for s in seed_strs)]
        if not files:
            print(f"  No files matching seeds={seeds} in {folder}/")
            return {}

    # raggruppa per config (togliendo _seed/_dataseed/_initseed/_samplingseed + N.json)
    import re
    groups = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        key = re.sub(r"_((?:data|init|sampling)?seed|run)\d+\.json$", "", name)
        groups[key].append(f)

    results = {}
    for key, file_list in groups.items():
        merged = defaultdict(lambda: {"loss": [], "accuracy": [], "f1": []})
        for filename in file_list:
            with open(filename) as f:
                data = json.load(f)
            for entry in data["rounds"]:
                r = entry["round"]
                m = entry.get(source, {})
                if not m:
                    continue
                merged[r]["loss"].append(m["loss"])
                merged[r]["accuracy"].append(m["accuracy"])
                merged[r]["f1"].append(m.get("f1_score", m.get("f1", 0)))

        rounds = sorted(merged.keys())
        result = {"rounds": rounds, "n_seeds": len(file_list)}
        for metric in ("loss", "accuracy", "f1"):
            arr_per_round = [np.array(merged[r][metric]) for r in rounds]
            result[f"{metric}_median"] = [float(np.median(a)) for a in arr_per_round]
            result[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr_per_round]
        results[key] = result
        print(f"  {key}: {len(file_list)} seeds")

        if save_json:
            json_dir = os.path.join(folder, "aggregated")
            os.makedirs(json_dir, exist_ok=True)
            out = {}
            for i, r in enumerate(rounds):
                out[str(r)] = {}
                for m in ("loss", "accuracy", "f1"):
                    out[str(r)][f"{m}_median"] = result[f"{m}_median"][i]
                    out[str(r)][f"{m}_mad"] = result[f"{m}_mad"][i]
            with open(os.path.join(json_dir, f"{key}_aggregated.json"), "w") as f:
                json.dump(out, f, indent=4)

    return results


# =====================================================================
# Metriche
# =====================================================================

def _pretty_label(raw_key):
    """Trasforma 'fedavg_etal0.35' in 'FedAvg (η_l=0.35)' ecc."""
    import re
    parts = raw_key.split("_")
    strat = parts[0].capitalize()
    # FedAvg, FedAdam, ecc.
    strat_map = {"Fedavg": "FedAvg", "Fedadam": "FedAdam", "Fedadagrad": "FedAdagrad",
                 "Fedprox": "FedProx", "Fedyogi": "FedYogi"}
    strat = strat_map.get(strat, strat)

    params = []
    rest = "_".join(parts[1:])
    for m in re.finditer(r'(etal|eta|mu)([\d.]+)', rest):
        name, val = m.group(1), m.group(2)
        if name == "etal":
            params.append(f"$\\eta_l$={val}")
        elif name == "eta":
            params.append(f"$\\eta$={val}")
        elif name == "mu":
            params.append(f"$\\mu$={val}")

    if params:
        return f"{strat} ({', '.join(params)})"
    return strat


def plot(scenarios, save_path=None, source="eval_metrics_client", metrics=("loss",), prefix="", title=None, save_json=False, seeds=None, xlim=None, dashed_labels=None):
    """
    Plotta confronto tra scenari.

    scenarios: dict {label: cartella} oppure dict {label: data_dict} (output di _read_folder)
    seeds: lista di seed da includere (es. [1,2,3,5,6,7]). Se None, usa tutti.
    """
    all_data = {}
    for label, val in scenarios.items():
        if isinstance(val, str):
            folder_data = _read_folder(val, source=source, save_json=save_json, seeds=seeds)
            if len(folder_data) == 1:
                all_data[label] = list(folder_data.values())[0]
            else:
                for key, data in folder_data.items():
                    all_data[_pretty_label(key)] = data
        else:
            all_data[label] = val

    if not all_data:
        print("No data to plot.")
        return

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    n_series = len(all_data)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n_series, 10)))

    if dashed_labels is None:
        dashed_labels = []

    for idx, (label, data) in enumerate(all_data.items()):
        color = cmap[idx]
        ls = "--" if label in dashed_labels else "-"
        for ax_idx, metric in enumerate(metrics):
            info = METRIC_INFO[metric]
            vals = list(data[f"{metric}_median"])
            mads = list(data[f"{metric}_mad"])
            if info["pct"]:
                vals = [v * 100 for v in vals]
                mads = [m * 100 for m in mads]

            axes[ax_idx].plot(data["rounds"], vals, color=color, label=label, linewidth=1.5, linestyle=ls)
            if any(m > 0 for m in mads):
                axes[ax_idx].fill_between(
                    data["rounds"],
                    [v - m for v, m in zip(vals, mads)],
                    [v + m for v, m in zip(vals, mads)],
                    color=color, alpha=0.2,
                )

    for ax_idx, metric in enumerate(metrics):
        info = METRIC_INFO[metric]
        axes[ax_idx].set_xlabel("Communication round $t$")
        axes[ax_idx].set_ylabel(info["ylabel"])
        if xlim is not None:
            axes[ax_idx].set_xlim(xlim)

    # Legenda condivisa sopra
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(len(handles), 5)
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=ncol,
               loc='upper center', bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        # Salva anche PDF
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to {save_path} and {pdf_path}")
    plt.show()


def plot_tuning(folder, save_dir=None, source="eval_metrics_server", metrics=("loss",), save_json=False, title_prefix=""):
    """
    Plotta il tuning: un grafico per strategia, una curva per config di iperparametri.

    folder: cartella con tutti i JSON di tuning
    title_prefix: es. "Classical 9h IID" — aggiunto al titolo di ogni strategia
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(folder), "plots")

    all_data = _read_folder(folder, source=source, save_json=save_json)
    if not all_data:
        return

    # raggruppa per strategia con pretty labels
    strat_map = {"fedavg": "FedAvg", "fedadam": "FedAdam", "fedadagrad": "FedAdagrad",
                 "fedprox": "FedProx", "fedyogi": "FedYogi"}
    by_strategy = defaultdict(dict)
    for key, data in all_data.items():
        strategy = key.split("_")[0]
        pretty = _pretty_label(key)
        # Rimuovi il nome strategia dal label, tieni solo i parametri
        strat_pretty = strat_map.get(strategy, strategy)
        param_label = pretty.replace(f"{strat_pretty} ", "").strip()
        by_strategy[strategy][param_label] = data

    # calcola xlim dai dati (dopo rimozione round 0)
    all_rounds = [r for data in all_data.values() for r in data["rounds"]]
    tuning_xlim = (min(all_rounds), max(all_rounds)) if all_rounds else None

    for strategy, configs in by_strategy.items():
        strat_pretty = strat_map.get(strategy, strategy)
        title = f"{title_prefix} — {strat_pretty} Tuning" if title_prefix else f"{strat_pretty} Tuning"
        print(f"\n--- {title} ---")
        plot(
            configs,
            save_path=os.path.join(save_dir, f"{strategy}_tuning.png"),
            source=source,
            metrics=metrics,
            title=title,
            xlim=tuning_xlim,
        )


def plot_tuning_centralized(folder, save_path=None, metrics=("loss",), title="Centralized Tuning"):
    """Plotta tuning centralizzato: una curva per ogni lr.

    Legge file con formato lr_{val}_seed{N}.json (epochs-based).
    """
    files = glob.glob(f"{folder}/lr_*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return

    groups = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        parts = name.rsplit("_seed", 1)
        key = parts[0] if len(parts) == 2 else name.replace(".json", "")
        groups[key].append(f)

    all_data = {}
    for key, file_list in groups.items():
        merged = defaultdict(lambda: {"loss": [], "accuracy": [], "f1": []})
        for filename in file_list:
            with open(filename) as fh:
                data = json.load(fh)
            for entry in data["epochs"]:
                e = entry["epoch"]
                if entry.get("eval_loss") is not None:
                    merged[e]["loss"].append(entry["eval_loss"])
                    merged[e]["accuracy"].append(entry.get("eval_accuracy", 0))
                    merged[e]["f1"].append(entry.get("eval_f1", 0))

        epochs = sorted(merged.keys())
        result = {"rounds": epochs, "n_seeds": len(file_list)}
        for metric in ("loss", "accuracy", "f1"):
            arr = [np.array(merged[e][metric]) for e in epochs]
            result[f"{metric}_median"] = [float(np.median(a)) for a in arr]
            result[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr]
        label = key.replace("lr_", "lr=")
        all_data[label] = result
        print(f"  {label}: {len(file_list)} seeds")

    if not all_data:
        return

    plot(
        scenarios=all_data,
        save_path=save_path,
        metrics=metrics,
        title=title,
    )


# =====================================================================
# Partizioni
# =====================================================================

def _generate_data(n_data, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n_data, 2))
    y = (np.linalg.norm(x, axis=1) <= 1.0).astype(np.int32)
    return {"point": x.astype(np.float64), "label": y}


def plot_partitions(seed, alpha=1.0, n_data=500, num_clients=5, save_path=None):
    """Plotta la distribuzione dei dati per ogni client."""
    data = _generate_data(n_data, seed)
    dataset = Dataset.from_dict(data)
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, alpha=alpha, partition_by="label", seed=seed
    )
    partitioner.dataset = dataset

    fig, axes = plt.subplots(1, num_clients + 1, figsize=(4 * (num_clients + 1), 4))

    # plot globale
    x_all = np.array(data["point"])
    y_all = np.array(data["label"])
    colors_all = ["red" if l == 1 else "blue" for l in y_all]
    axes[0].scatter(x_all[:, 0], x_all[:, 1], c=colors_all, s=10, alpha=0.6)
    axes[0].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
    axes[0].set_title(f"Global ({len(y_all)})\nc0={np.sum(y_all==0)}, c1={np.sum(y_all==1)}", fontsize=11)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for pid in range(num_clients):
        part = partitioner.load_partition(pid)
        part.set_format("numpy")
        x_p = np.array(part["point"])
        y_p = np.array(part["label"])
        n0 = int(np.sum(y_p == 0))
        n1 = int(np.sum(y_p == 1))
        colors = ["red" if l == 1 else "blue" for l in y_p]

        ax = axes[pid + 1]
        ax.scatter(x_p[:, 0], x_p[:, 1], c=colors, s=10, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
        ax.set_title(f"Client {pid} ({len(y_p)})\nc0={n0}, c1={n1}", fontsize=11)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Dirichlet partition (seed={seed}, alpha={alpha})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_partitions_grid(seeds, alpha=1.0, n_data=500, num_clients=5, save_path="partitions"):
    """Plotta la distribuzione per più seed in una griglia."""
    n_seeds = len(seeds)
    n_cols = num_clients + 1
    fig, axes = plt.subplots(n_seeds, n_cols, figsize=(3.5 * n_cols, 3.5 * n_seeds))
    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    for row, seed in enumerate(seeds):
        data = _generate_data(n_data, seed)
        dataset = Dataset.from_dict(data)
        partitioner = DirichletPartitioner(
            num_partitions=num_clients, alpha=alpha, partition_by="label", seed=seed
        )
        partitioner.dataset = dataset

        x_all = np.array(data["point"])
        y_all = np.array(data["label"])
        colors_all = ["red" if l == 1 else "blue" for l in y_all]
        axes[row, 0].scatter(x_all[:, 0], x_all[:, 1], c=colors_all, s=8, alpha=0.5)
        axes[row, 0].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1))
        axes[row, 0].set_title(f"Global\n{np.sum(y_all==0)}/{np.sum(y_all==1)}", fontsize=9)
        axes[row, 0].set_xlim(-1.1, 1.1)
        axes[row, 0].set_ylim(-1.1, 1.1)
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"seed={seed}", fontsize=11, fontweight="bold")

        for pid in range(num_clients):
            part = partitioner.load_partition(pid)
            part.set_format("numpy")
            x_p = np.array(part["point"])
            y_p = np.array(part["label"])
            n0 = int(np.sum(y_p == 0))
            n1 = int(np.sum(y_p == 1))
            colors = ["red" if l == 1 else "blue" for l in y_p]

            ax = axes[row, pid + 1]
            ax.scatter(x_p[:, 0], x_p[:, 1], c=colors, s=8, alpha=0.5)
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1))
            ax.set_title(f"C{pid} ({len(y_p)})\n{n0}/{n1}", fontsize=9)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"Dirichlet partitions (alpha={alpha})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================
# Predizioni
# =====================================================================

def _predict(weights_path, noise_model=None, nshots=None):
    model = create_model(noise_model=noise_model, nshots=nshots)
    weights = list(np.load(weights_path).values())
    set_weights(model, weights)

    rng = np.random.default_rng(0)
    x_grid = torch.tensor(rng.uniform(-1, 1, size=(2500, 2)), dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        y_prob = model(x_grid).squeeze(-1).numpy()

    return x_grid.numpy(), y_prob


def _draw(ax, x_grid, y_prob, title, colorbar=True):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
    scatter = ax.scatter(x_grid[:, 0], x_grid[:, 1], c=y_prob, cmap="RdBu",
                         vmin=0, vmax=1, s=15, alpha=0.8)
    if colorbar:
        plt.colorbar(scatter, ax=ax, label="P(inside)", shrink=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_predictions(weights_path, title="Predictions", save_path=None,
                     noise_model=None, nshots=None):
    x_grid, y_prob = _predict(weights_path, noise_model=noise_model, nshots=nshots)
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw(ax, x_grid, y_prob, title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_predictions_grid(scenarios, save_path=None, ncols=None):
    """
    Plotta più predizioni in griglia.

    scenarios: lista di (weights_path, title) oppure (weights_path, title, noise_kwargs)
    noise_kwargs: dict con chiavi opzionali 'pauli_base', 'readout_base', 'nshots'
    """
    n = len(scenarios)
    if ncols is None:
        ncols = n if n <= 3 else 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):
        if len(scenario) == 3:
            weights_path, title, noise_kwargs = scenario
            nm, _ = build_noise_model(
                pauli_base=noise_kwargs.get("pauli_base", 0),
                readout_base=noise_kwargs.get("readout_base", 0),
                partition_id=0, scale=0,
            )
            nshots = noise_kwargs.get("nshots", 1000)
            x_grid, y_prob = _predict(weights_path, noise_model=nm, nshots=nshots)
        else:
            weights_path, title = scenario
            x_grid, y_prob = _predict(weights_path)

        _draw(axes[i], x_grid, y_prob, title)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================
# Centralizzato
# =====================================================================

def _read_centralized(folder, seeds=None, save_json=False, epochs_per_round=2):
    """Legge i JSON centralizzati e restituisce un data_dict compatibile con plot().

    I JSON hanno struttura: {"epochs": [{"epoch": N, "eval_loss": ..., "eval_accuracy": ..., "eval_f1": ...}, ...]}
    epochs_per_round: quante epoche centralizzate corrispondono a 1 round federato (default: 2).
        L'asse x viene convertito in round = epoch / epochs_per_round.
    """
    files = glob.glob(f"{folder}/seed*.json")
    if not files:
        # Prova anche il pattern lr_*_seed*.json (centralizzato classico)
        files = glob.glob(f"{folder}/lr_*_seed*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return None

    if seeds is not None:
        seed_strs = {f"seed{s}.json" for s in seeds}
        files = [f for f in files if any(f.endswith(s) for s in seed_strs)]

    merged = defaultdict(lambda: {"loss": [], "accuracy": [], "f1": []})
    for filename in files:
        with open(filename) as f:
            data = json.load(f)
        for entry in data["epochs"]:
            e = entry["epoch"]
            if entry.get("eval_loss") is not None:
                merged[e]["loss"].append(entry["eval_loss"])
                merged[e]["accuracy"].append(entry.get("eval_accuracy", 0))
                merged[e]["f1"].append(entry.get("eval_f1", 0))

    epochs = sorted(merged.keys())
    rounds = [e / epochs_per_round for e in epochs]
    result = {"rounds": rounds, "n_seeds": len(files)}
    for metric in ("loss", "accuracy", "f1"):
        arr = [np.array(merged[e][metric]) for e in epochs]
        result[f"{metric}_median"] = [float(np.median(a)) for a in arr]
        result[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr]

    if save_json:
        json_dir = os.path.join(folder, "aggregated")
        os.makedirs(json_dir, exist_ok=True)
        out = {}
        for i, e in enumerate(epochs):
            out[str(e)] = {}
            for m in ("loss", "accuracy", "f1"):
                out[str(e)][f"{m}_median"] = result[f"{m}_median"][i]
                out[str(e)][f"{m}_mad"] = result[f"{m}_mad"][i]
        out_path = os.path.join(json_dir, "centralized_aggregated.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"  Aggregated JSON saved to {out_path}")

    return result


def plot_centralized_weight_distances(base_dir, noise_levels, seeds=range(1, 8),
                                       save_path=None, title="Centralized: weight distance",
                                       mode="2pi", regimes=("noisy", "mitigated")):
    """Plotta distanza pesi centralizzati noiseless vs noisy/mitigated."""
    noiseless_dir = f"{base_dir}/noiseless/nshots_1000/weights"
    diff_noisy = []
    diff_mitigated = []

    for p in noise_levels:
        noisy_dir = f"{base_dir}/noisy/p{p}/nshots_1000/weights"
        mitigated_dir = f"{base_dir}/mitigated/p{p}/nshots_1000/weights"

        diffs_n, diffs_m = [], []
        for seed in seeds:
            w_nl_path = f"{noiseless_dir}/seed{seed}.npz"
            try:
                w_nl = np.concatenate(list(np.load(w_nl_path).values()))
            except FileNotFoundError:
                continue
            if "noisy" in regimes:
                try:
                    w_ny = np.concatenate(list(np.load(f"{noisy_dir}/seed{seed}.npz").values()))
                    diffs_n.append(weight_distance(w_nl, w_ny, mode=mode))
                except FileNotFoundError:
                    pass
            if "mitigated" in regimes:
                try:
                    w_mt = np.concatenate(list(np.load(f"{mitigated_dir}/seed{seed}.npz").values()))
                    diffs_m.append(weight_distance(w_nl, w_mt, mode=mode))
                except FileNotFoundError:
                    pass

        if diffs_n:
            diff_noisy.append((p, np.median(diffs_n), median_abs_deviation(diffs_n)))
        if diffs_m:
            diff_mitigated.append((p, np.median(diffs_m), median_abs_deviation(diffs_m)))

    ylabel = "Angular distance mod 2π (rad)" if mode == "2pi" else "Euclidean distance"
    fig, ax = plt.subplots(figsize=(8, 5))

    if diff_noisy:
        x_n, y_n, e_n = zip(*diff_noisy)
        ax.plot(x_n, y_n, marker='o', label='|noiseless - noisy|', linewidth=2)
        ax.fill_between(x_n, [y - e for y, e in zip(y_n, e_n)],
                        [y + e for y, e in zip(y_n, e_n)], alpha=0.3)
    if diff_mitigated:
        x_m, y_m, e_m = zip(*diff_mitigated)
        ax.plot(x_m, y_m, marker='s', label='|noiseless - mitigated|', linewidth=2)
        ax.fill_between(x_m, [y - e for y, e in zip(y_m, e_m)],
                        [y + e for y, e in zip(y_m, e_m)], alpha=0.3)

    ax.set_xlabel("Noise level (p)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(noise_levels)
    ax.set_xticklabels([str(p) for p in noise_levels], rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================

def plot_cdr_noise_map(pauli_base, readout_base, scale=0.002, client_id=0, weights_path=None,
                       n_training_samples=120, nshots_cdr=30000, cdr_seed=40,
                       nshots_model=1000, seed=None, save_path=None, title=None, ax=None, x_input=None, plot=True):
    """Scatterplot dei training points CDR (noisy vs noise-free) con retta di regressione.

    Se ax è passato, disegna su quell'asse (per subplot). Altrimenti crea una figura nuova.
    Se seed è passato, inizializza i pesi del modello con quel seed.
    """
    from qibo_qfl_pt.task import create_model, build_noise_model, set_weights, set_seed
    if seed is not None:
        set_seed(seed)

    noise_model, readout_prob = build_noise_model(
        pauli_base=pauli_base, readout_base=readout_base, partition_id=client_id, scale=scale,
    )
    single = np.array([[1 - readout_prob, readout_prob],
                        [readout_prob, 1 - readout_prob]])
    response_matrix = np.kron(single, single)
    mitigation_config = {
        "threshold": 0.1, "min_iterations": 500, "method": "CDR",
        "method_kwargs": {
            "n_training_samples": n_training_samples,
            "nshots": nshots_cdr, "seed": cdr_seed,
            "readout": {"response_matrix": response_matrix},
        },
    }

    model = create_model(model_type="quantum", nshots=nshots_model,
                          noise_model=noise_model, mitigation_config=mitigation_config)
    if weights_path is not None:
        w = list(np.load(weights_path).values())
        set_weights(model, w)

    import torch
    if x_input is None:
        x_input = [[0.5, 0.3]]
    x = torch.tensor(x_input, dtype=torch.float64)         # <-- input parametrico
    with torch.no_grad():
        model(x)

    mitigator = model.q_model.decoding.mitigator
    train_data = mitigator._training_data
    popt = mitigator._mitigation_map_popt
    noisy = np.array(train_data["noisy"])
    noisefree = np.array(train_data["noise-free"])
    a, b = float(popt[0]), float(popt[1])

    if not plot:                                           # <-- solo numeri, niente figura
        return noisy, noisefree, a, b

    if title is None:
        title = f"CDR noise map (p={pauli_base})"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    is_subplot = not own_fig
    fs_label = 10 if is_subplot else 14
    fs_title = 12 if is_subplot else 16
    fs_legend = 8 if is_subplot else 12

    ax.scatter(noisy, noisefree, color="blue", alpha=0.5, s=20, label="Training points")
    x_fit = np.linspace(noisy.min(), noisy.max(), 100)
    ax.plot(x_fit, a * x_fit + b, color="red", linewidth=2,
            label=f"Fit: y = {a:.3f}x + {b:.3f}")
    ax.set_xlabel("Noisy exp. value", fontsize=fs_label)
    ax.set_ylabel("Noise-free exp. value", fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title, fontweight="bold")
    ax.legend(fontsize=fs_legend)
    ax.grid(True, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()

    return noisy, noisefree, a, b




def weight_distance(theta_a, theta_b, mode="2pi"):
    """Distanza tra due vettori di pesi.

    mode: '2pi' = distanza angolare modulo 2pi, 'euclidean' = distanza euclidea.
    """
    diff = np.asarray(theta_a) - np.asarray(theta_b)
    if mode == "2pi":
        diff = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.linalg.norm(diff))


def print_weight_distances(base_dir, noise_levels, etal="0.3", seeds=range(1, 4), mode="2pi"):
    """Stampa distanza pesi noiseless-noisy e noiseless-mitigated."""
    noiseless_dir = f"{base_dir}/noiseless/uniform_p0.0_r0.0/nshots_1000/weights"

    medians_n, medians_m = [], []

    for p in noise_levels:
        noise_tag = f"scaled_p{p}_r{p}_s0.002"
        noisy_dir = f"{base_dir}/noisy/{noise_tag}/nshots_1000/weights"
        mitigated_dir = f"{base_dir}/mitigated/{noise_tag}/nshots_1000/weights"

        dists_n, dists_m = [], []
        for seed in seeds:
            try:
                w_nl = np.concatenate(list(np.load(f"{noiseless_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                w_ny = np.concatenate(list(np.load(f"{noisy_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                dists_n.append(weight_distance(w_nl, w_ny, mode=mode))
            except FileNotFoundError:
                pass
            try:
                w_nl = np.concatenate(list(np.load(f"{noiseless_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                w_mt = np.concatenate(list(np.load(f"{mitigated_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                dists_m.append(weight_distance(w_nl, w_mt, mode=mode))
            except FileNotFoundError:
                pass

        medians_n.append(float(np.median(dists_n)) if dists_n else None)
        medians_m.append(float(np.median(dists_m)) if dists_m else None)

    print(f"mode = {mode}")
    print(f"noise_levels = {noise_levels}")
    print(f"noisy:     {[f'{v:.3f}' if v is not None else '—' for v in medians_n]}")
    print(f"mitigated: {[f'{v:.3f}' if v is not None else '—' for v in medians_m]}")


def plot_weight_distances(base_dir, noise_levels, etal="0.3", seeds=range(1, 3),
                          save_path=None, title="Weight distance vs noise level",
                          mode="2pi", regimes=("noisy", "mitigated")):
    """Plotta distanza pesi noiseless-noisy e/o noiseless-mitigated.

    mode: '2pi' = distanza angolare modulo 2pi, 'euclidean' = distanza euclidea.
    regimes: tupla con 'noisy' e/o 'mitigated'.
    """
    diff_noisy = []
    diff_mitigated = []

    noiseless_dir = f"{base_dir}/noiseless/uniform_p0.0_r0.0/nshots_1000/weights"

    for p in noise_levels:
        noise_tag = f"scaled_p{p}_r{p}_s0.002"
        noisy_dir = f"{base_dir}/noisy/{noise_tag}/nshots_1000/weights"
        mitigated_dir = f"{base_dir}/mitigated/{noise_tag}/nshots_1000/weights"

        diffs_n = []
        diffs_m = []
        for seed in seeds:
            w_nl = None
            try:
                w_nl = np.concatenate(list(np.load(f"{noiseless_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
            except FileNotFoundError:
                continue
            if "noisy" in regimes:
                try:
                    w_ny = np.concatenate(list(np.load(f"{noisy_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                    diffs_n.append(weight_distance(w_nl, w_ny, mode=mode))
                except FileNotFoundError:
                    pass
            if "mitigated" in regimes:
                try:
                    w_mt = np.concatenate(list(np.load(f"{mitigated_dir}/FedAvg_etal{etal}_seed{seed}.npz").values()))
                    diffs_m.append(weight_distance(w_nl, w_mt, mode=mode))
                except FileNotFoundError:
                    pass

        if diffs_n:
            diff_noisy.append((p, np.median(diffs_n), median_abs_deviation(diffs_n)))
        if diffs_m:
            diff_mitigated.append((p, np.median(diffs_m), median_abs_deviation(diffs_m)))

    ylabel = "Angular distance mod 2π (rad)" if mode == "2pi" else "Euclidean distance"

    fig, ax = plt.subplots(figsize=(8, 5))

    if diff_noisy:
        x_n, y_n, e_n = zip(*diff_noisy)
        ax.plot(x_n, y_n, marker='o', label='|noiseless - noisy|', linewidth=2)
        ax.fill_between(x_n, [y - e for y, e in zip(y_n, e_n)],
                        [y + e for y, e in zip(y_n, e_n)], alpha=0.3)
    if diff_mitigated:
        x_m, y_m, e_m = zip(*diff_mitigated)
        ax.plot(x_m, y_m, marker='s', label='|noiseless - mitigated|', linewidth=2)
        ax.fill_between(x_m, [y - e for y, e in zip(y_m, e_m)],
                        [y + e for y, e in zip(y_m, e_m)], alpha=0.3)

    ax.set_xlabel("Noise level (p)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(noise_levels)
    ax.set_xticklabels([str(p) for p in noise_levels], rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()

def _read_drift(folder, drift_key="drift_median", seeds=None):
    """Legge il drift dai JSON, raggruppa per config, calcola mediana+MAD tra seed."""
    files = glob.glob(f"{folder}/*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return {}

    if seeds is not None:
        seed_strs = {f"_seed{s}.json" for s in seeds}
        files = [f for f in files if any(f.endswith(s) for s in seed_strs)]

    import re
    groups = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        key = re.sub(r"_((?:data|init|sampling)?seed|run)\d+\.json$", "", name)
        groups[key].append(f)

    results = {}
    for key, file_list in groups.items():
        merged = defaultdict(list)
        for filename in file_list:
            with open(filename) as f:
                data = json.load(f)
            for entry in data["rounds"]:
                dm = entry.get("drift_metrics")
                if dm and dm.get(drift_key) is not None:
                    merged[entry["round"]].append(dm[drift_key])

        rounds = sorted(merged.keys())
        if not rounds:
            continue
        arr_per_round = [np.array(merged[r]) for r in rounds]
        results[key] = {
            "rounds": rounds,
            "n_seeds": len(file_list),
            "drift_median": [float(np.median(a)) for a in arr_per_round],
            "drift_mad": [float(median_abs_deviation(a)) for a in arr_per_round],
        }
        print(f"  {key}: {len(file_list)} seeds (drift)")

    return results


def plot_drift(scenarios, save_path=None, drift_key="drift_median", title=None, seeds=None, xlim=None):
    """Plotta il drift mediano (con MAD) per più scenari.

    scenarios: dict {label: cartella} oppure dict {label: data_dict}
    """
    all_data = {}
    for label, val in scenarios.items():
        if isinstance(val, str):
            folder_data = _read_drift(val, drift_key=drift_key, seeds=seeds)
            if len(folder_data) == 1:
                all_data[label] = list(folder_data.values())[0]
            else:
                for key, data in folder_data.items():
                    all_data[_pretty_label(key)] = data
        else:
            all_data[label] = val

    if not all_data:
        print("No drift data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, (label, data) in enumerate(all_data.items()):
        color = COLORS[idx % len(COLORS)]
        vals = data["drift_median"]
        mads = data["drift_mad"]
        ax.plot(data["rounds"], vals, color=color, label=label, linewidth=2)
        if any(m > 0 for m in mads):
            ax.fill_between(
                data["rounds"],
                [v - m for v, m in zip(vals, mads)],
                [v + m for v, m in zip(vals, mads)],
                color=color, alpha=0.3,
            )

    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel("Client drift (L2)", fontsize=14)
    ax.set_title(title or "Client drift", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_comparison(distribution="iid", with_centralized=True):
    # Confronto strategie federate + centralizzato (non-IID)
    sim_folder = f"strategies_comparison/{distribution}/noiseless/simulations/simulation_experiments"
    fed_data = _read_folder(sim_folder, source="eval_metrics_server")

    centr_data = _read_centralized("centralized_model_results/noiseless/nshots_None")
    # Tieni solo le prime 30 epoche centralizzate
    if centr_data is not None:
        mask = [i for i, r in enumerate(centr_data["rounds"]) if r <= 30]
        centr_data["rounds"] = [centr_data["rounds"][i] for i in mask]
        for metric in ("loss", "accuracy", "f1"):
            for suffix in ("_median", "_mad"):
                key = f"{metric}{suffix}"
                centr_data[key] = [centr_data[key][i] for i in mask]

    scenarios = dict(fed_data)
    save_path = f"plots/{distribution}/strategies_comparison.pdf"

    if centr_data is not None and with_centralized:
        scenarios["Centralized"] = centr_data
        save_path = f"plots/{distribution}/strategies_comparison_with_centralized.pdf"


    plot(
        scenarios=scenarios,
        save_path=save_path,
        source="eval_metrics_server",
        metrics=("loss",),
        title=f"Strategies comparison {distribution} (noiseless)",
    )

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
 
 
# ---------- lettura ----------
 
def load_rounds(json_path):
    return json.load(open(json_path))["rounds"]
 
 
def collect_last_per_client(rounds):
    """{client: (round, a, b, noisy, noisefree)} = ultimo round in cui il client appare."""
    last = {}
    for r in rounds:
        rn = r.get("round")
        for e in r.get("cdr_per_client", []) or []:
            cid = e.get("client")
            if cid is None or cid < 0:
                continue
            last[cid] = (
                rn, float(e["a"]), float(e["b"]),
                np.array(e.get("noisy", []), dtype=float),
                np.array(e.get("noisefree", []), dtype=float),
            )
    return last
 
 
def collect_series_per_client(rounds):
    """{client: {'round':[...], 'a':[...], 'b':[...]}} ordinato per round."""
    series = {}
    for r in rounds:
        rn = r.get("round")
        for e in r.get("cdr_per_client", []) or []:
            cid = e.get("client")
            if cid is None or cid < 0:
                continue
            s = series.setdefault(cid, {"round": [], "a": [], "b": []})
            s["round"].append(rn)
            s["a"].append(float(e["a"]))
            s["b"].append(float(e["b"]))
    for cid in series:  # ordina per round
        order = np.argsort(series[cid]["round"])
        for k in ("round", "a", "b"):
            series[cid][k] = list(np.array(series[cid][k])[order])
    return series
 
 
# ---------- figura 1: scatter + retta ----------
 
def plot_maps_grid(rounds, save_path=None, title=""):
    last = collect_last_per_client(rounds)
    clients = sorted(last)
    n = len(clients); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
 
    for ax, cid in zip(axes, clients):
        _, a, b, noisy, noisefree = last[cid]
        if noisy.size:
            ax.scatter(noisy, noisefree, s=18, alpha=0.6, label="training points")
            x0, x1 = noisy.min(), noisy.max()
            # R^2 della retta a*x+b sui punti reali
            y_pred = a * noisy + b
            ss_res = np.sum((noisefree - y_pred) ** 2)
            ss_tot = np.sum((noisefree - noisefree.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            x0, x1 = -1.0, 1.0
            r2 = float("nan")
        xs = np.linspace(x0, x1, 100)
        sign = "+" if b >= 0 else "-"
        fit_label = f"y = {a:.3f}x {sign} {abs(b):.3f}\n$R^2$ = {r2:.4f}"
        ax.plot(xs, a * xs + b, "r-", lw=2, label=fit_label)
        ax.set_title(f"Client {cid}", fontweight="bold")
        ax.set_xlabel("noisy expectation"); ax.set_ylabel("noise-free expectation")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"CDR noise map per client{title}", fontsize=16, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"salvato: {save_path}")
 
 
# ---------- figura 2: a e b vs round ----------
 
def plot_ab_vs_round(rounds, save_path=None, title=""):
    series = collect_series_per_client(rounds)
    clients = sorted(series)
    n = len(clients); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
 
    for ax, cid in zip(axes, clients):
        s = series[cid]
        l1, = ax.plot(s["round"], s["a"], "o-", color="tab:blue", label="a (slope)")
        ax.set_xlabel("round"); ax.set_ylabel("a (slope)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        l2, = ax2.plot(s["round"], s["b"], "s--", color="tab:orange", label="b (offset)")
        ax2.set_ylabel("b (offset)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title(f"Client {cid}", fontweight="bold")
        ax.legend(handles=[l1, l2], fontsize=8, loc="best")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Andamento (a, b) per client vs round {title}", fontsize=16, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"salvato: {save_path}")
 
def collect_train_series_per_client(rounds):
    """{client: {'round':[...], 'a':[...], 'b':[...], 'n_maps':[...]}} dalla mappa
    realmente usata in training (cdr_train_per_client)."""
    series = {}
    for r in rounds:
        rn = r.get("round")
        for e in r.get("cdr_train_per_client", []) or []:
            cid = e.get("client")
            if cid is None or cid < 0:
                continue
            s = series.setdefault(cid, {"round": [], "a": [], "b": [], "n_maps": []})
            s["round"].append(rn)
            s["a"].append(float(e["a"]))
            s["b"].append(float(e["b"]))
            s["n_maps"].append(int(e.get("n_maps", -1)))
    for cid in series:
        order = np.argsort(series[cid]["round"])
        for k in ("round", "a", "b", "n_maps"):
            series[cid][k] = list(np.array(series[cid][k])[order])
    return series
 

 
def plot_train_vs_eval_a(rounds, save_path=None, title=""):
    """Per ogni client: slope 'a' usata in TRAINING vs 'a' ricostruita in EVAL.
    Il gap che si allarga = mappa stale (problema). Curve vicine = mappa allineata."""
    eval_s = collect_series_per_client(rounds)        # cdr_per_client (eval)
    train_s = collect_train_series_per_client(rounds)  # cdr_train_per_client (training)
    clients = sorted(set(eval_s) | set(train_s))
    n = len(clients); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
 
    for ax, cid in zip(axes, clients):
        if cid in train_s:
            ax.plot(train_s[cid]["round"], train_s[cid]["a"], "o-",
                    color="tab:red", label="a TRAINING (usata)")
        if cid in eval_s:
            ax.plot(eval_s[cid]["round"], eval_s[cid]["a"], "s--",
                    color="tab:blue", label="a EVAL (ideale)")
        ax.set_title(f"Client {cid}", fontweight="bold")
        ax.set_xlabel("round"); ax.set_ylabel("a (slope)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Slope CDR: training (usata) vs eval (ideale) {title}",
                 fontsize=16, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"salvato: {save_path}")
 
 
def plot_n_maps(rounds, save_path=None, title=""):
    """Numero cumulato di ricalcoli della mappa (n_maps) vs round, per client."""
    train_s = collect_train_series_per_client(rounds)
    clients = sorted(train_s)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cid in clients:
        ax.plot(train_s[cid]["round"], train_s[cid]["n_maps"], "o-", label=f"client {cid}")
    ax.set_xlabel("round"); ax.set_ylabel("n_maps (ricalcoli cumulati)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Ricalcoli della noise map vs round {title}", fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"salvato: {save_path}")

# =====================================================================

def plot_drift_by_client(folder, strategy_names, save_path=None, title=None, num_clients=5):
    """Plotta drift per client ID: un subplot per client, una curva per strategia.

    folder: cartella con i JSON
    strategy_names: lista di nomi strategia da confrontare (es. ["fedavg", "fedadam"])
    """
    from collections import defaultdict

    # Per ogni strategia, raccogli drift per (client_id, round) aggregato su seed
    strategy_data = {}  # label -> {client_id -> {round -> [drift values]}}

    for strat_name in strategy_names:
        files = glob.glob(f"{folder}/{strat_name}_*.json")
        client_rounds = defaultdict(lambda: defaultdict(list))

        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            for entry in data.get("rounds", []):
                dm = entry.get("drift_metrics", {})
                by_id = dm.get("drift_by_client_id", {})
                for cid, drift in by_id.items():
                    client_rounds[int(cid)][entry["round"]].append(drift)

        strategy_data[strat_name] = client_rounds

    nrows, ncols = (2, 3) if num_clients == 5 else (1, num_clients)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes_grid.flatten() if nrows > 1 else (axes_grid if num_clients > 1 else [axes_grid])

    for cid in range(num_clients):
        ax = axes[cid]
        for idx, label in enumerate(strategy_names):
            cr = strategy_data[label].get(cid, {})
            if not cr:
                continue
            rounds = sorted(cr.keys())
            medians = [float(np.median(cr[r])) for r in rounds]
            mads = [float(median_abs_deviation(cr[r])) if len(cr[r]) > 1 else 0.0 for r in rounds]
            color = COLORS[idx % len(COLORS)]
            ax.plot(rounds, medians, color=color, label=label, linewidth=2)
            if any(m > 0 for m in mads):
                ax.fill_between(rounds,
                    [v - m for v, m in zip(medians, mads)],
                    [v + m for v, m in zip(medians, mads)],
                    color=color, alpha=0.3)
        ax.set_title(f"Client {cid}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Round", fontsize=12)
        if cid == 0:
            ax.set_ylabel("Drift (L2)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Nascondi subplot vuoti
    for i in range(num_clients, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.93] if title else None)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":

    BASE_DIR = "results/fixed_training_set_results/noiseless_noisy_mit_fixed_results/federated/iid/fedavg/"


    centr_data = _read_centralized("parameters_centralized/lr_0.3")

    scenarios={
        "Centralized": centr_data,
        "Noiseless": f"{BASE_DIR}/noiseless",
        #"p=0.003": f"{BASE_DIR}/noisy/scaled_p0.003_r0.003_s0.002",
        "$p=0.005$": f"{BASE_DIR}/noisy/scaled_p0.005_r0.005_s0.002",
        "$p=0.007$": f"{BASE_DIR}/noisy/scaled_p0.007_r0.007_s0.002",
        "$p=0.014$": f"{BASE_DIR}/noisy/scaled_p0.014_r0.014_s0.002",
    }

    plot(
        scenarios=scenarios,
        metrics = ("loss", "accuracy", "f1",),
        save_path="thesis_images/cap_04_Results/4.2_noise/04_fedavg_noiselevels.png",
        dashed_labels=["Centralized"],
    )