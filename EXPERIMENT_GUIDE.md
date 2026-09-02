# Fixed 40-instance experiment

## Design

- Benchmark: Cicirello WTSDS, 60 jobs per instance.
- Keep only `tau={0.3,0.9}`, `R={0.25,0.75}`, `eta={0.25,0.75}`.
- 8 classes × first 5 source-numbered instances in each class = 40 fixed instances.
- Baselines: EDD and deterministic ATCS.
- Stochastic algorithms: SA, GA, TS, ACO, DPSO, DDE, GVNS.
- All stochastic algorithms receive the same seed set and the same wall-clock `time_limit=10` seconds.
- `improvement_vs_ATCS = (ATCS_twt - best_twt) / ATCS_twt * 100`.
- Search code records every strict improvement of the best objective encountered. The runner samples this history at 0,1,...,10 seconds with forward fill.

## Files

- `build_reduced_dataset.py` -> `data/reduced_dataset.json`
- `run_baselines.py` -> `results/baseline_results.csv`
- `run_experiments.py` -> `results/results_raw.csv`, `results/convergence.csv`
- `dataset_io.py` -> loader shared by the runners

## Run order

```bash
python build_reduced_dataset.py
python run_baselines.py
python run_experiments.py --time-limit 10 --seeds 0 1 2 3 4
```

If the 120 raw `.instance` files are already stored locally:

```bash
python build_reduced_dataset.py --source-dir PATH_TO_WTSDS_FILES
```

The experiment runner resumes completed `(algorithm, instance, seed)` runs by default. Use `--overwrite` only when you intentionally want to restart from scratch.

## Expected scale with default seeds

40 instances × 7 algorithms × 5 seeds = 1,400 stochastic runs. At a nominal 10 seconds each, the search-time component alone is about 3.9 hours, plus Python overhead. If you want a quick validation first, use one seed and/or a subset of algorithms, e.g.:

```bash
python run_experiments.py --time-limit 0.2 --seeds 0 --algorithms SA GA GVNS --overwrite
```
