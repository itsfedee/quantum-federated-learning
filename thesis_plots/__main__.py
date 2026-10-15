"""Figure runner.

    python -m thesis_plots --list          list of groups and figures
    python -m thesis_plots all             everything except the figures that use qibo
    python -m thesis_plots noise           a whole group
    python -m thesis_plots noise.circle    a single figure
    python -m thesis_plots all --with-qibo everything, including the slow ones

    python -m thesis_plots noise --results D:/data --out D:/figures
                                           same figures, data and output elsewhere

The scripts read paths relative to the repository root, so the runner moves
there before producing anything: the commands work from any folder. The
roots given with --results, --out and --archive are resolved against the
folder the command is launched from, before the move.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

from . import paths


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m thesis_plots")
    parser.add_argument("names", nargs="*", default=[],
                        help="figures, groups or 'all'")
    parser.add_argument("--list", action="store_true",
                        help="list the available groups and figures")
    parser.add_argument("--with-qibo", action="store_true",
                        help="also include the figures that evaluate the model")
    parser.add_argument("--keep-going", action="store_true",
                        help="keep going when a figure fails")
    parser.add_argument("--results", metavar="DIR",
                        help="root of the run data (default: results/ in the repository)")
    parser.add_argument("--out", metavar="DIR",
                        help="root of the figures and tables (default: thesis_images/)")
    parser.add_argument("--archive", metavar="DIR",
                        help="root of the campaigns outside the thesis (default: archive/)")
    args = parser.parse_args(argv)

    # The roots are resolved from the caller's folder, before moving to the
    # repository root, so that a relative --results means what the user sees.
    roots = {}
    for key in ("results", "out", "archive"):
        value = getattr(args, key)
        if value is None:
            continue
        p = Path(value).resolve()
        if key != "out" and not p.is_dir():
            print(f"--{key}: folder not found: {p}", file=sys.stderr)
            return 2
        roots[key] = p
    paths.configure(results=roots.get("results"), images=roots.get("out"),
                    archive=roots.get("archive"))

    # Imported only now: the figure modules copy some paths at import time.
    from .figures import FIGURES, NEEDS_QIBO, groups, resolve  # noqa: E402

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
        print(f"unknown figure: {exc.args[0]}", file=sys.stderr)
        return 2

    os.chdir(paths.ROOT)
    print(f"data: {paths.FINAL}\nfigures: {paths.IMAGES}")
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
        print(f"\nfailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n{len(selected)} entries produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
