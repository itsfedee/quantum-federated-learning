# thesis_plots

Tutte le figure e le tabelle della tesi si producono da qui. Un comando per
figura, i dati letti da `results/`, i PDF scritti in `thesis_images/`.

```
python -m thesis_plots --list              elenco di gruppi e figure
python -m thesis_plots noise.circle        una figura sola
python -m thesis_plots noise               un gruppo intero
python -m thesis_plots all                 tutte, tranne quelle che usano qibo
python -m thesis_plots all --with-qibo     proprio tutte (lento)
python -m thesis_plots all --keep-going    prosegue anche se una fallisce
python -m thesis_plots noise --results D:/dati --out D:/figure
                                           dati e figure in un'altra radice
```

Il runner si sposta nella radice del repository prima di produrre qualsiasi
cosa, quindi i comandi funzionano da qualunque cartella. Le radici passate
con `--results`, `--out` e `--archive` si risolvono rispetto alla cartella da
cui lanci il comando; senza opzioni valgono `results/`, `thesis_images/` e
`archive/` dentro il repository. La struttura sotto la radice non cambia:
`--results` serve a chi tiene i dati altrove, `--out` a rigenerare una figura
di prova senza sovrascrivere quella impaginata.

## Confronti al volo

Per guardare un risultato che non ha una figura registrata c'è `compare`: si
danno le cartelle delle run, aggrega con mediana e MAD sui seed e disegna i
tre pannelli in `thesis_images/extra/`.

```
python -m thesis_plots.compare nome FedAvg=cartella FedProx=cartella --ref Centralized=cartella
```

Le sorgenti possono essere cartelle o percorsi con jolly (`.../fedavg_*.json`)
quando più bracci condividono la stessa cartella. Con `--ref` le curve di
riferimento escono tratteggiate e dietro alle altre; `--metrics`, `--max-round`
e `--out` scelgono pannelli, troncamento e destinazione. In fondo a
`compare.py` c'è anche un blocco `esempio()` da editare, per chi preferisce
lavorare sul file.

## Com'è fatto

| file | ruolo |
|---|---|
| `paths.py` | **l'unico posto con i percorsi**: dati in `results/`, figure in `thesis_images/`, campagne fuori tesi in `archive/` |
| `loaders.py` | lettura dei JSON delle run, mediana e MAD sui seed |
| `draw.py` | primitive di disegno comuni al capitolo 4 (pannelli di scenario, bande) |
| `style.py` | dimensioni dei pannelli, legende, `save()` che scrive PDF + PNG |
| `palette.py` | colori, marcatori, etichette e la lista dei sette seed |
| `noise_model.py` | ricostruisce le traiettorie di rumore usate nelle run |
| `figures/__init__.py` | il registro: nome della figura -> funzione che la produce |
| `figures/*.py` | un modulo per capitolo |

Una figura nuova si aggiunge scrivendo la funzione nel modulo del suo capitolo
e registrandola nel dizionario `FIGURES` in fondo al file.

## Le figure, una per riga

`[qibo]` segna quelle che rivalutano il modello: sono lente e `all` le salta se
non passi `--with-qibo`. La colonna dei dati elenca le cartelle sotto
`results/`, salvo dove indicato `archive/`.

### Capitolo 3 — metodologia

| comando | produce | legge |
|---|---|---|
| `cap03.partitions` [qibo] | `3_FINAL/03_dirichlet_partitions` | genera i dati al volo |
| `cap03.drift` | `3_FINAL/03_noise_drift_ou_sigma001_seed5` | `noise_model.py`, nessun file |

### 4.1 — confronto fra strategie

| comando | produce | legge |
|---|---|---|
| `strategies.comparison` | `4_1_FINAL/04_strategy_comparison_{iid,non_iid}` + `extra/04_fedavg_iid_tuning` | `strategy_comparison` |
| `tuning.per_strategy_iid` | `4_1_FINAL/04_<strategia>_tuning_iid` (5 figure) | `tuning/iid` |
| `tuning.grid_iid` | `6_FINAL/06_full_tuning_iid` | `tuning/iid` |
| `tuning.grid_non_iid` | `6_FINAL/06_tuning_non_iid` | `tuning/non_iid` |
| `tuning.fedavg` | `6_FINAL/06_tuning_non_iid_fedavg` | `tuning/non_iid` |
| `tuning.per_strategy_non_iid_old` | campagna vecchia, non impaginata | `archive/tuning_non_iid` |

### 4.2 — rumore e deriva

| comando | produce | legge |
|---|---|---|
| `noise.levels` | `4_2_FINAL/04_fedavg_noiselevels` | `noise_and_mitigation` |
| `noise.seed_isolation` | `4_2_FINAL/seed_isolation_median_mad` | `seed_isolation` |
| `noise.drift` | `4_2_FINAL/04_noiseless_vs_noisy_vs_drift` + la variante `_RW` in `extra/` | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |
| `noise.circle` [qibo] | `4_2_FINAL/04_circle_predictions` | `noise_and_mitigation` (pesi) |
| `strategies.under_noise` | `4_2_FINAL/04_strategy_comparison_{noisy,drift}_{iid,noniid}_p005`, `04_fedavg_noiseless_vs_noisy_vs_drift_noniid`, `6_FINAL/06_strategy_comparison_mitigated_*` + 3 varianti `_RW` in `extra/` | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |

### 4.3 — mitigazione e memoria

| comando | produce | legge |
|---|---|---|
| `mitigation.fedavg` | `4_3_FINAL/04_fedavg_noiseless_noisy_mitigated_p005`, `04_fedavg_noisy_vs_mitigated_drift_p005` + variante `_RW` | `noise_and_mitigation`, `noise_and_mitigation_drift`, `archive/strategies_drift_p005` |
| `mitigation.metrics_vs_p` | `4_3_FINAL/04_fedavg_metrics_vs_noise_level` | `noise_and_mitigation` |
| `mitigation.loss_vs_p` | `4_3_FINAL/04_mitigation_vs_p` | `noise_and_mitigation` |
| `mitigation.regression` [qibo] | `4_3_FINAL/04_cdr_regression_panels` | costruisce il modello, nessun file |
| `memory.ou` | `4_3_FINAL/04_threshold_sweep`, `04_calibration_savings_vs_round`, `04_weight_distance` + 4 diagnostiche in `extra/` | `memory_sweep_drift`, `noise_and_mitigation`, `noise_and_mitigation_drift` |
| `mitigation.panels` | le stesse curve dai valori client grezzi, in `extra/` | come sopra |
| `mitigation.static_threshold` | `extra/04_fedavg_static_loss_thresh_0015_vs_003` | `archive/FEDAVG_TH_0.03`, `archive/MEMORY_RESULTS_NEW/static_calibrations_federated` |
| `mitigation.recovery` | tabella a schermo, nessun file | `noise_and_mitigation` |
| `memory.strategy_drift` | 5 pannelli in `extra/strategies_drift_plots/` | `memory_sweep_drift`, `noise_and_mitigation`, `noise_and_mitigation_drift` |
| `memory.rw` | la campagna random walk, in `extra/rw/` | `archive/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk` |
| `memory.cdr_sweep` | sweep CDR pre-OU, in `extra/cdr_sweep_old/` | `archive/MEMORY_RESULTS_OLD/{noise_sweep_th015, threshold_sweep_p005_varwalk}` |

### Tabelle

| comando | produce | legge |
|---|---|---|
| `tables.strategy` | `tables/strategy_comparison_{iid,non_iid}.tex`, `fedavg_tuning_iid.tex` | `strategy_comparison`, `tuning/iid` |
| `tables.memory_ou` | `tables/tab_{memory_savings, thresh_vs_quality, centralized_drift}.tex` e `.csv` | `memory_sweep_drift`, `noise_and_mitigation_drift`, `tables_only` |
| `tables.parameters` [qibo] | `tables/tab_parameters.tex` e `.csv` | `noise_and_mitigation` (pesi) |
| `tables.memory_rw` | tabelle della campagna random walk | `archive/MEMORY_RESULTS_OLD/threshold_sweep_p005_varwalk` |

### Esplorative

`extra.loss_thresholds`, `extra.pareto`, `extra.threshold_comparison`,
`extra.noiseless_noisy_mitigated_ou`, `extra.drift_random_walk` scrivono tutte
in `thesis_images/extra/` e leggono `memory_sweep_drift`,
`noise_and_mitigation_drift` e `archive/FEDAVG_TH_0.03`.

## Quattro cose da sapere

**Un comando produce più figure.** `memory.ou` ne salva sette, `strategies.under_noise`
otto. Quindi per rigenerare `04_weight_distance` si lancia `memory.ou`, non un
comando che porta quel nome.

**`ts.save` scrive sempre la coppia PDF + PNG.** Nelle cartelle consegnate ci
sono solo i PDF: i PNG compaiono appena rigeneri.

**Le varianti non impaginate finiscono in `extra/`.** Dove una figura esiste in
due versioni — deriva Ornstein-Uhlenbeck e random walk — quella in tesi è la OU
e prende il nome senza suffisso; la random walk esce con `_RW` in `extra/`.
Vale per `noise.drift`, `strategies.under_noise` e `mitigation.fedavg`.

**`memory.thresh_inset` è ridondante.** Produce lo stesso
`4_3_FINAL/04_threshold_sweep` che produce già `memory.ou`: è il residuo di
quando quella figura non era inclusa in `all_figures`. Si può togliere dal
registro.

## Com'è stata verificata questa tabella

Le cartelle lette non sono dedotte dal codice ma registrate: un audit hook di
Python ha annotato ogni file aperto da ciascuna delle 34 figure. La
corrispondenza comando -> file prodotti viene dal log di una rigenerazione
completa a partire da `thesis_images/` vuota, confrontata figura per figura con
i PDF consegnati.
