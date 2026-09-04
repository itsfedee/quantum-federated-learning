"""Stile condiviso per le figure della tesi.

Il principio è disegnare alla dimensione finale: una figura prodotta a 15
pollici e inclusa a \\textwidth viene rimpicciolita di un fattore 0.42, e un
font da 9 pt finisce a 3.8 pt sulla pagina. Qui la larghezza nativa coincide
con la larghezza di inclusione, così il fattore di scala è 1 e i font restano
quelli impostati.

Due larghezze:
    FULL    figure a più pannelli, da includere con width=\\textwidth
    SINGLE  figure a un pannello, da includere con width=0.7\\textwidth

Gli rcParams sono gli stessi per entrambe, quindi il testo ha la stessa
dimensione ovunque nel documento.

Uso tipico:

    from thesis_plots import style as ts
    ts.apply()
    fig, axes = ts.panels(3)
    ...
    ts.legend_above(fig, axes[0], ncol=5)
    ts.save(fig, "thesis_images/.../nome_figura")
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Geometria. TEXTWIDTH va verificato nel documento con \the\textwidth: 6.27 in
# corrisponde ad A4 con margini da un pollice.
# --------------------------------------------------------------------------
FULL = 6.27
SINGLE = 0.7 * FULL          # ~4.39 in

# Altezza di default per numero di pannelli affiancati. Nasce dal voler tenere
# ogni pannello vicino al rapporto 4:3 una volta tolti assi ed etichette.
_HEIGHT = {1: 3.0, 2: 2.6, 3: 2.1, 4: 1.9}

# --------------------------------------------------------------------------
# Tipografia. Un solo posto per tutti gli script.
# --------------------------------------------------------------------------
FONT_BASE = 9                # etichette degli assi e testo generico
FONT_TICK = 8
FONT_LEGEND = 8
LINEWIDTH = 0.8
MARKERSIZE = 2.5
MARKEVERY = 5               # un marcatore ogni 5 punti
BAND_ALPHA = 0.15


def apply():
    """Applica gli rcParams comuni. Da chiamare una volta per script."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_BASE,
        "axes.titlesize": FONT_BASE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "axes.grid": False,
    })


def panels(n=3, height=None, width=None, **kwargs):
    """Figura a n pannelli affiancati, larga quanto il testo.

    kwargs passa a plt.subplots (sharex, sharey, ...). L'altezza segue il
    numero di pannelli se non specificata.
    """
    w = FULL if width is None else width
    h = height if height is not None else _HEIGHT.get(n, 2.15)
    fig, axes = plt.subplots(1, n, figsize=(w, h),
                             layout="constrained", **kwargs)
    _tick_in(axes)
    return fig, axes


def single(height=None, **kwargs):
    """Figura a un pannello, da includere con width=0.7\\textwidth."""
    h = height if height is not None else 3.0
    fig, ax = plt.subplots(figsize=(SINGLE, h),
                           layout="constrained", **kwargs)
    _tick_in(ax)
    return fig, ax


def grid(rows, cols, height=None, **kwargs):
    """Griglia di pannelli larga quanto il testo, per i layout 2x2."""
    h = height if height is not None else 2.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, h),
                             layout="constrained", **kwargs)
    _tick_in(axes)
    return fig, axes


def _tick_in(axes):
    for ax in (axes.ravel() if hasattr(axes, "ravel") else
               (axes if isinstance(axes, (list, tuple)) else [axes])):
        ax.tick_params(direction="in")


def legend_above(fig, source, ncol=None, handles=None, labels=None):
    """Legenda unica sopra la figura, con le maniglie del primo pannello."""
    if handles is None:
        handles, labels = source.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT_LEGEND,
               ncol=ncol or len(handles), loc="outside upper center",
               columnspacing=1.2, handlelength=1.6)


def save(fig, path, tight_top=None, dpi=300):
    """Salva PDF e PNG con lo stesso nome.

    Lo spazio per la legenda lo gestisce il layout constrained impostato alla
    creazione della figura, quindi qui non serve toccare i margini. Il
    parametro tight_top resta accettato per compatibilita' con le chiamate
    esistenti, ma viene ignorato.
    """
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}.pdf / .png")
