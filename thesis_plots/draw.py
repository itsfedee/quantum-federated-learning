"""Primitive di disegno comuni alle figure del capitolo 4.

Quasi tutte le figure hanno la stessa forma: tre pannelli affiancati (loss,
accuracy, F1), una curva mediana per braccio con banda MAD, marcatori sfasati
e legenda unica sopra. Quella forma sta qui una volta sola.
"""
import numpy as np

from . import palette, style as ts


def band(ax, x, med, dev, color, alpha=0.15, zorder=1, floor=None):
    """Banda MAD attorno alla mediana."""
    lo = med - dev if floor is None else np.maximum(med - dev, floor)
    ax.fill_between(x, lo, med + dev, color=color, alpha=alpha, lw=0,
                    zorder=zorder)


def curve(ax, x, med, dev, label, color, marker=None,
          marker_every=None, marker_offset=0, lw=None, alpha=0.15):
    """Curva mediana con banda, marcatori diradati e sfasati."""
    lw = ts.LINEWIDTH if lw is None else lw
    every = ts.MARKEVERY if marker_every is None else marker_every
    ax.plot(x, med, color=color, lw=lw, label=label, marker=marker,
            ms=ts.MARKERSIZE, markevery=(marker_offset, every))
    ax.fill_between(x, med - dev, med + dev, alpha=alpha, color=color)


def scenario_panels(scenarios, out_path, metrics=("loss", "accuracy", "f1")):
    """Figura a piu' pannelli da curve gia' aggregate (read_federated & co.).

    scenarios e' una lista di tuple (etichetta, dati, colore, stile di linea,
    marcatore, sfasamento). Il tratteggio identifica i riferimenti - il
    centralizzato e il FedAvg noiseless - che vanno dietro alle altre curve e
    con banda piu' tenue.
    """
    fig, axes = ts.panels(len(metrics))
    if len(metrics) == 1:
        axes = [axes]
    for label, data, color, ls, marker, moffset in scenarios:
        for ax, metric in zip(axes, metrics):
            info = palette.METRIC_INFO[metric]
            vals = np.array(data[f"{metric}_median"])
            mads = np.array(data[f"{metric}_mad"])
            if info["pct"]:
                vals, mads = vals * 100, mads * 100
            mkwargs = (dict(marker=marker, ms=ts.MARKERSIZE,
                            markevery=(moffset, ts.MARKEVERY))
                       if marker else {})
            ax.plot(data["rounds"], vals, color=color,
                    lw=ts.LINEWIDTH if ls == "-" else 1.5, ls=ls, label=label,
                    zorder=0 if ls == "--" else 1, **mkwargs)
            if any(mads > 0):
                ax.fill_between(data["rounds"], vals - mads, vals + mads,
                                alpha=0.08 if ls == "--" else 0.15,
                                color=color, zorder=0 if ls == "--" else 1)
    for ax, metric in zip(axes, metrics):
        ax.set_xlabel(r"Round $t$")
        ax.set_ylabel(palette.METRIC_INFO[metric]["ylabel"])
        ax.tick_params(direction="in")
    ts.legend_above(fig, axes[0], ncol=min(len(scenarios), 6))
    ts.save(fig, out_path)


def arm_panels(arms, out_path, metrics=None):
    """Figura a tre pannelli da matrici grezze (seed, round).

    arms e' un dizionario etichetta -> {"data", "color", "ls", "marker",
    "lw"}, con data nel formato di loaders.client_arms. L'accuratezza si
    riporta in percentuale.
    """
    metrics = metrics or palette.METRICS_CLIENT
    fig, axes = ts.panels(len(metrics))

    for arm_idx, (label, cfg) in enumerate(arms.items()):
        color = cfg["color"]
        ls = cfg.get("ls", "-")
        marker = cfg.get("marker")
        lw = cfg.get("lw", 1.8)
        if marker:
            offset = palette.MARKER_OFFSETS[arm_idx % len(palette.MARKER_OFFSETS)]
            mevery = (offset, cfg.get("markevery", ts.MARKEVERY))
        else:
            mevery = None

        for ax, m in zip(axes, metrics):
            mat_list = cfg["data"][m]
            if not mat_list:
                continue
            min_len = min(len(v) for v in mat_list)
            mat = np.array([v[:min_len] for v in mat_list])
            if m == "accuracy":
                mat = mat * 100
            rounds = np.arange(min_len)
            med = np.median(mat, axis=0)
            dev = np.median(np.abs(mat - med), axis=0)
            ax.plot(rounds, med, color=color, ls=ls, lw=lw, label=label,
                    marker=marker, markersize=ts.MARKERSIZE, markevery=mevery)
            ax.fill_between(rounds, med - dev, med + dev, alpha=0.15,
                            color=color)

    for ax, m in zip(axes, metrics):
        ax.set_xlabel(r"Round $t$")
        ax.set_ylabel(palette.METRIC_LABELS_PCT[m])
        ax.tick_params(direction="in")

    ts.legend_above(fig, axes[0])
    ts.save(fig, str(out_path))
