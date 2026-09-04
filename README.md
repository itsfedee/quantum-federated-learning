# Quantum Federated Learning with Qiboml and Flower

Federated learning on a variational quantum classifier, with simulated noise
(Pauli + readout), noise drift across rounds, and Clifford Data Regression (CDR)
mitigation with a memory of the map.

This repository holds **code only**: data, runs and figures stay local. The
`.gitignore` works as a whitelist — it ignores everything at the root and
re-admits the code folders by hand — so a new results folder is ignored by
default, with nothing to add here each time.

## Layout

| folder             | contents                                                     |
| ------------------ | ------------------------------------------------------------ |
| `qibo_qfl_pt/`     | Flower app: client, server, custom strategy, model, noise    |
| `run_experiments/` | campaign launchers (federated, centralized, tuning)          |
| `thesis_plots/`    | figures and tables for the thesis                            |
| `notebooks/`       | `example.ipynb`, a toy run — start here                      |
| `tools/`           | housekeeping utilities (run checks, result download)         |

## Start here

`notebooks/example.ipynb` walks through the mechanism on a tiny run — 2 clients,
5 rounds, a 2-qubit circuit — that finishes in half a minute on a laptop: data,
circuit, federated loop, and a comparison between two strategies. It is not a
result: the thesis campaigns run on a server, and they are what the figures
below come from.

## Experiments

```bash
python run_experiments/parallel_experiments.py --strategy FedAvg --workers 4
python run_experiments/centralized_experiments.py --mode mitigated --epochs 30
python run_experiments/parallel_tuning.py
```

Defaults for the Flower app live in `pyproject.toml`, under
`[tool.flwr.app.config]`.

## Figures and tables

Every figure goes through the same runner, to be launched from the repository
root (or from anywhere, if the package is installed with `pip install -e .`); it
resolves the data paths on its own.

```bash
python -m thesis_plots --list            # available groups and figures
python -m thesis_plots all               # everything except the figures that use qibo
python -m thesis_plots noise             # a whole group
python -m thesis_plots noise.circle      # a single figure
python -m thesis_plots all --with-qibo   # including the ones that re-evaluate the model
```

The groups follow the chapters:

| group        | contents                                                        |
| ------------ | --------------------------------------------------------------- |
| `cap03`      | Dirichlet partitions, noise drift                               |
| `strategies` | strategy comparison, at 40 rounds and under noise               |
| `tuning`     | hyperparameter sweeps, IID and non-IID                          |
| `noise`      | effect of noise on FedAvg, seed isolation                       |
| `mitigation` | CDR: recovery, dependence on `p`, threshold, regression         |
| `memory`     | threshold sweep and memory of the CDR map                       |
| `tables`     | `.tex` (and `.csv`) table bodies                                |
| `extra`      | exploratory figures, outside the document                       |

Four entries are marked `[qibo]`: they re-evaluate the quantum model, so they
need the full run environment and take minutes rather than seconds.

Under `thesis_plots/` sit the pieces every figure shares: `style.py` (one
typography, drawn at the final inclusion size), `paths.py` (a single place for
data paths), `palette.py` (colours and markers), `loaders.py` (reading the run
JSONs) and `draw.py` (the recurring figure shapes).
