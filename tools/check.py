"""Controlla che le run salvate abbiano il numero di round atteso e 3 client per round.

Uso:
    python check_runs.py <cartella>                      # 30 round, 3 client, sottocartelle incluse
    python check_runs.py <cartella> --rounds 40
    python check_runs.py <cartella> --rounds 40 --clients 3
"""

import argparse
import glob
import json
import os


def check_file(path, expected_rounds, expected_clients):
    with open(path) as f:
        j = json.load(f)
    n_rounds = len(j["rounds"]) - 1  # rounds[0] è la valutazione iniziale
    bad = [
        r["round"]
        for r in j["rounds"][1:]
        if len(r.get("drift_metrics", {}).get("drift_by_client_id", {})) != expected_clients
    ]
    return n_rounds, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()

    pattern = os.path.join(args.folder, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"Nessun JSON trovato in {args.folder}")
        return

    n_bad = 0
    for path in files:
        try:
            n_rounds, bad = check_file(path, args.rounds, args.clients)
        except Exception as e:
            print(f"ERR  {path}: {e}")
            n_bad += 1
            continue
        ok = n_rounds == args.rounds and not bad
        if ok:
            print(f"OK   {path}")
        else:
            n_bad += 1
            print(f"BAD  {path}: rounds={n_rounds}" + (f", rounds with !={args.clients} clients: {bad}" if bad else ""))

    print(f"\n{len(files) - n_bad}/{len(files)} OK")


if __name__ == "__main__":
    main()