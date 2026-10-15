"""Thesis figures and tables.

One module per chapter under thesis_plots/figures/, and under thesis_plots/
the shared pieces: style, paths, colours, JSON loading, drawing primitives.
Figures are produced by the runner:

    python -m thesis_plots --list        list of figures
    python -m thesis_plots noise         one group
    python -m thesis_plots all           everything

For a quick comparison between run folders, outside the registered figures:

    python -m thesis_plots.compare confronto FedAvg=<folder> FedProx=<folder>

or, from a script:

    from thesis_plots.compare import compare
"""
from . import draw, loaders, noise_model, palette, paths, style

__all__ = ["draw", "loaders", "noise_model", "palette", "paths",
           "style"]
