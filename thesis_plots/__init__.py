"""Figure e tabelle della tesi.

Un modulo per capitolo sotto thesis_plots/figures/, e sotto thesis_plots/ i
pezzi condivisi: stile, percorsi, colori, lettura dei JSON, primitive di
disegno. Le figure si producono dal runner:

    python -m thesis_plots --list        elenco delle figure
    python -m thesis_plots noise         un gruppo
    python -m thesis_plots all           tutto
"""
from . import draw, loaders, noise_model, palette, paths, style

__all__ = ["draw", "loaders", "noise_model", "palette", "paths", "style"]
