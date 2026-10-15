# thesis_plots

Every figure and table of the thesis is produced from here. One command per
figure, the data read from `results/`, the PDFs written to `thesis_images/`.

```
python -m thesis_plots --list              list of groups and figures
python -m thesis_plots noise.circle        a single figure
python -m thesis_plots noise               a whole group
python -m thesis_plots all                 everything, except the figures that use qibo
python -m thesis_plots all --with-qibo     really everything (slow)
python -m thesis_plots all --keep-going    carry on when a figure fails
python -m thesis_plots noise --results D:/data --out D:/figures
                                           data and figures under another root
```

The runner moves to the repository root before producing anything, so the
commands work from any folder. The roots given with `--results`, `--out` and
`--archive` are resolved against the folder the command is launched from;
without options they are `results/`, `thesis_images/` and `archive/` inside
the repository. The tree below the root does not change: `--results` is for
whoever keeps the data elsewhere, `--out` for regenerating a trial figure
without overwriting the one in the document.

## Quick comparisons

To look at a result that has no registered figure there is `compare`: give
it the run folders, it aggregates them (median and MAD over the seeds) and
draws the three panels into `thesis_images/extra/`.

```
python -m thesis_plots.compare name FedAvg=folder FedProx=folder --ref Centralized=folder
```

A source can be a folder or a path with a wildcard (`.../fedavg_*.json`) when
several arms share the same folder. With `--ref` the reference curves come
out dashed and behind the others; `--metrics`, `--max-round` and `--out` pick
the panels, the truncation and the destination. At the bottom of `compare.py`
there is also an `esempio()` block to edit, for those who prefer working on
the file.

## Layout

| file | role |
|---|---|
| `paths.py` | **the only place with paths**: data in `results/`, figures in `thesis_images/`, campaigns outside the thesis in `archive/` |
| `loaders.py` | reading of the run JSONs, median and MAD over the seeds |
| `draw.py` | drawing primitives shared by chapter 4 (scenario panels, bands) |
| `style.py` | panel sizes, legends, `save()` which writes PDF + PNG |
| `palette.py` | colours, markers, labels and the list of the seven seeds |
| `noise_model.py` | rebuilds the noise trajectories used in the runs |
| `figures/__init__.py` | the registry: figure name -> function that produces it |
| `figures/*.py` | one module per chapter |

A new figure is added by writing its function in the module of its chapter
and registering it in the `FIGURES` dictionary at the bottom of the file.

## The figures, one per line

`[qibo]` marks the ones that re-evaluate the model: they are slow and `all`
skips them unless you pass `--with-qibo`. The data column lists the folders
under `results/`, except where `archive/` is stated.

### Chapter 3 — methodology

| command | produces | reads |
|---|---|---|
| `cap03.partitions` [qibo] | `3_FINAL/03_dirichlet_partitions` | generates the data on the fly |
| `cap03.drift` | `3_FINAL/03_noise_drift_ou_sigma001_seed5` | `noise_model.py`, no file |

### 4.1 — strategy comparison

| command | produces | reads |
|---|---|---|
| `strategies.comparison` | `4_1_FINAL/04_strategy_comparison_{iid,non_iid}` + `extra/04_fedavg_iid_tuning` | `strategy_comparison` |
| `tuning.per_strategy_iid` | `4_1_FINAL/04_<strategy>_tuning_iid` (5 figures) | `tuning/iid` |
| `tuning.grid_iid` | `6_FINAL/06_full_tuning_iid` | `tuning/iid` |
| `tuning.grid_non_iid` | `6_FINAL/06_tuning_non_iid` | `tuning/non_iid` |
| `tuning.fedavg` | `6_FINAL/06_tuning_non_iid_fedavg` | `tuning/non_iid` |
| `tuning.per_strategy_non_iid_old` | old campaign, not in the document | `archive/tuning_non_iid` |

### 4.2 — noise and drift

| command | produces | reads |
|---|---|---|
| `noise.levels` | `4_2_FINAL/04_fedavg_noiselevels` | `noise_and_mitigation` |
| `noise.seed_isolation` | `4_2_FINAL/seed_isolation_median_mad` | `seed_isolation` |
| `noise.drift` | `4_2_FINAL/04_noiseless_vs_noisy_vs_drift` + the `_RW` variant in `extra/` | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |
| `noise.circle` [qibo] | `4_2_FINAL/04_circle_predictions` | `noise_and_mitigation` (weights) |
| `strategies.under_noise` | `4_2_FINAL/04_strategy_comparison_{noisy,drift}_{iid,noniid}_p005`, `04_fedavg_noiseless_vs_noisy_vs_drift_noniid`, `6_FINAL/06_strategy_comparison_mitigated_*` + 3 `_RW` variants in `extra/` | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |

### 4.3 — mitigation and memory

| command | produces | reads |
|---|---|---|
| `mitigation.fedavg` | `4_3_FINAL/04_fedavg_noiseless_noisy_mitigated_p005`, `04_fedavg_noisy_vs_mitigated_drift_p005` + `_RW` variant | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |
| `mitigation.metrics_vs_p` | `4_3_FINAL/04_fedavg_metrics_vs_noise_level` | `noise_and_mitigation` |
| `mitigation.loss_vs_p` | `4_3_FINAL/04_mitigation_vs_p` | `noise_and_mitigation` |
| `mitigation.regression` [qibo] | `4_3_FINAL/04_cdr_regression_panels` | builds the model, no file |
| `memory.ou` | `4_3_FINAL/04_threshold_sweep`, `04_calibration_savings_vs_round`, `04_weight_distance` + 4 diagnostics in `extra/` | `memory_sweep_drift`, `noise_and_mitigation`, `noise_and_mitigation_drift` |
| `mitigation.panels` | the same curves from the raw client values, in `extra/` | as above |
| `mitigation.static_threshold` | `extra/04_fedavg_static_loss_thresh_0015_vs_003` | `archive/FEDAVG_TH_0.03`, `archive/MEMORY_RESULTS_NEW/static_calibrations_federated` |
| `mitigation.recovery` | on-screen table, no file | `noise_and_mitigation` |
| `memory.strategy_drift` | 5 panels in `extra/strategies_drift_plots/` | `memory_sweep_drift`, `noise_and_mitigation`, `noise_and_mitigation_drift` |
| `memory.rw` | the random walk campaign, in `extra/rw/` | `archive/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk` |
| `memory.cdr_sweep` | pre-OU CDR sweep, in `extra/cdr_sweep_old/` | `archive/MEMORY_RESULTS_OLD/{noise_sweep_th015, threshold_sweep_p005_varwalk}` |

### Tables

| command | produces | reads |
|---|---|---|
| `tables.strategy` | `tables/strategy_comparison_{iid,non_iid}.tex`, `fedavg_tuning_iid.tex` | `strategy_comparison`, `tuning/iid` |
| `tables.memory_ou` | `tables/tab_{memory_savings, thresh_vs_quality, centralized_drift}.tex` and `.csv` | `memory_sweep_drift`, `noise_and_mitigation_drift`, `tables_only` |
| `tables.parameters` [qibo] | `tables/tab_parameters.tex` and `.csv` | `noise_and_mitigation` (weights) |
| `tables.memory_rw` | tables of the random walk campaign | `archive/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk` |

### Exploratory

`extra.loss_thresholds`, `extra.pareto`, `extra.threshold_comparison`,
`extra.noiseless_noisy_mitigated_ou` and `extra.drift_random_walk` all write
to `thesis_images/extra/` and read `memory_sweep_drift`,
`noise_and_mitigation_drift` and `archive/FEDAVG_TH_0.03`.

## Four things to know

**One command produces several figures.** `memory.ou` saves seven,
`strategies.under_noise` eight. So to regenerate `04_weight_distance` you run
`memory.ou`, not a command that carries that name.

**`ts.save` always writes the PDF + PNG pair.** The delivered folders hold
only the PDFs: the PNGs appear as soon as you regenerate.

**Variants outside the document go to `extra/`.** Where a figure exists in
two versions — Ornstein-Uhlenbeck drift and random walk — the one in the
thesis is the OU and takes the name without suffix; the random walk comes
out with `_RW` in `extra/`. This holds for `noise.drift`,
`strategies.under_noise` and `mitigation.fedavg`.

**`memory.thresh_inset` is redundant.** It produces the same
`4_3_FINAL/04_threshold_sweep` that `memory.ou` already produces: it is a
leftover from when that figure was not included in `all_figures`. It can be
removed from the registry.

## How this table was checked

The folders read are not inferred from the code but recorded: a Python audit
hook logged every file opened by each of the 34 figures. The command -> files
mapping comes from the log of a full regeneration starting from an empty
`thesis_images/`, compared figure by figure with the delivered PDFs.
