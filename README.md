# Quantum Federated Learning con Qiboml e Flower

Federated learning su un classificatore quantistico variazionale, con rumore
simulato (Pauli + readout), deriva del rumore fra i round e mitigazione
Clifford Data Regression (CDR) con memoria della mappa.

Il repository contiene **solo il codice**: dati, run e figure restano in
locale. Il `.gitignore` funziona per lista bianca — ignora tutto quello che sta
nella radice e riammette a mano le cartelle di codice — quindi una nuova
cartella di risultati e' ignorata di default, senza doverla aggiungere ogni
volta.

## Struttura

| cartella          | contenuto                                                 |
| ----------------- | --------------------------------------------------------- |
| `qibo_qfl_pt/`    | app Flower: client, server, strategia custom, modello, rumore |
| `run_experiments/`| lancio delle campagne (federate, centralizzate, tuning)   |
| `thesis_plots/`   | figure e tabelle della tesi                               |
| `notebooks/`      | `example.ipynb`, run giocattolo da leggere per prima cosa |
| `tools/`          | utility di servizio (controllo run, scarico risultati)    |

## Da dove partire

`notebooks/example.ipynb` mostra il meccanismo su una run minuscola — 2 client,
5 round, un circuito a 2 qubit — che gira in mezzo minuto su un portatile: dati,
circuito, ciclo federato e confronto fra due strategie. Non e' un risultato: le
campagne della tesi girano su server, e stanno nelle figure qui sotto.

## Esperimenti

```bash
python run_experiments/parallel_experiments.py --strategy FedAvg --workers 4
python run_experiments/centralized_experiments.py --mode mitigated --epochs 30
python run_experiments/parallel_tuning.py
```

I parametri di default della app Flower stanno in `pyproject.toml`, sotto
`[tool.flwr.app.config]`.

## Figure e tabelle

Tutte le figure passano dallo stesso runner, da lanciare dalla radice del
repository (o da qualunque cartella, se il pacchetto e' installato con
`pip install -e .`): i percorsi dei dati li risolve da solo.

```bash
python -m thesis_plots --list            # gruppi e figure disponibili
python -m thesis_plots all               # tutto tranne le figure che usano qibo
python -m thesis_plots noise             # un gruppo intero
python -m thesis_plots noise.circle      # una figura sola
python -m thesis_plots all --with-qibo   # comprese quelle che rivalutano il modello
```

I gruppi seguono i capitoli:

| gruppo       | contenuto                                                      |
| ------------ | -------------------------------------------------------------- |
| `cap03`      | partizioni Dirichlet, deriva del rumore                        |
| `strategies` | confronto fra strategie, a 40 round e sotto rumore             |
| `tuning`     | sweep degli iperparametri, IID e non-IID                       |
| `noise`      | effetto del rumore su FedAvg, isolamento dei seed              |
| `mitigation` | CDR: recupero, dipendenza da `p`, soglia, regressione          |
| `memory`     | sweep di soglia e memoria della mappa CDR                      |
| `tables`     | corpi `.tex` (e `.csv`) delle tabelle                          |
| `extra`      | figure esplorative, fuori dal documento                        |

Quattro voci sono marcate `[qibo]`: rivalutano il modello quantistico, quindi
richiedono l'ambiente completo delle run e qualche minuto invece di qualche
secondo.

Sotto `thesis_plots/` i pezzi condivisi da tutte le figure: `style.py` (una sola
tipografia, disegno alla dimensione finale di inclusione), `paths.py` (un solo
posto per i percorsi dei dati), `palette.py` (colori e marcatori),
`loaders.py` (lettura dei JSON delle run) e `draw.py` (le forme di figura
ricorrenti).
