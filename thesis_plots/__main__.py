"""Runner delle figure.

    python -m thesis_plots --list          elenco di gruppi e figure
    python -m thesis_plots all             tutte, tranne quelle che usano qibo
    python -m thesis_plots noise           un gruppo intero
    python -m thesis_plots noise.circle    una figura sola
    python -m thesis_plots all --with-qibo tutto, comprese quelle lente

Gli script leggono percorsi relativi alla radice del repository, quindi il
runner ci si sposta prima di produrre qualsiasi cosa: i comandi funzionano da
qualunque cartella.
"""
import argparse
import os
import sys
import traceback

from . import paths
from .figures import FIGURES, NEEDS_QIBO, groups, resolve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m thesis_plots")
    parser.add_argument("names", nargs="*", default=[],
                        help="figure, gruppi o 'all'")
    parser.add_argument("--list", action="store_true",
                        help="elenca gruppi e figure disponibili")
    parser.add_argument("--with-qibo", action="store_true",
                        help="includi anche le figure che valutano il modello")
    parser.add_argument("--keep-going", action="store_true",
                        help="prosegui anche se una figura fallisce")
    args = parser.parse_args(argv)

    if args.list or not args.names:
        for group, names in groups().items():
            print(f"{group}:")
            for name in names:
                flag = "  [qibo]" if name in NEEDS_QIBO else ""
                print(f"    {name}{flag}")
        return 0

    try:
        selected = resolve(args.names, args.with_qibo)
    except KeyError as exc:
        print(f"figura sconosciuta: {exc.args[0]}", file=sys.stderr)
        return 2

    os.chdir(paths.ROOT)
    failed = []
    for name in selected:
        print(f"\n=== {name}")
        try:
            FIGURES[name]()
        except Exception:
            failed.append(name)
            traceback.print_exc()
            if not args.keep_going:
                return 1

    if failed:
        print(f"\nfallite: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n{len(selected)} voci prodotte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
