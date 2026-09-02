# SMTWT-SDST Metaheuristics Project

Target problem:

    1 | s_ij | sum(w_j T_j)

All algorithms use the same `WTSDSInstance`, Cicirello loader, objective function,
and ATCS initialization code from `gvns_smtwt_sdst.py`.

## Files

- `gvns_smtwt_sdst.py` — original GVNS/data-loading module, with one ATCS k2 formula correction.
- `metaheuristic_common.py` — shared result/budget/permutation operators.
- `sa_smtwt_sdst.py` — Simulated Annealing.
- `ga_smtwt_sdst.py` — Genetic Algorithm (OX + swap mutation).
- `ts_smtwt_sdst.py` — Tabu Search (swap neighborhood + aspiration).
- `aco_smtwt_sdst.py` — Ant Colony Optimization (arc pheromone + ATCS desirability).
- `dpso_smtwt_sdst.py` — Discrete PSO (swap-sequence velocity).
- `dde_smtwt_sdst.py` — Discrete Differential Evolution (swap-operator difference + permutation-safe crossover).
- `run_sa.py`, `run_ga.py`, `run_ts.py`, `run_aco.py`, `run_dpso.py`, `run_dde.py` — individual runners.
- `run_all.py` — run all algorithms on the same instance.

## Important ATCS correction

The uploaded GVNS file had:

    k2 = tau / sqrt(2 * eta)

The papers define:

    k2 = tau / (2 * sqrt(eta))

Only this ATCS formula was corrected. The instance-reading functions
`_read_source`, `parse_cicirello_instance`, and `load_cicirello_instance`
are otherwise unchanged.

## Basic execution

From this folder:

    python run_sa.py 1 --time-limit 10 --seed 42
    python run_ga.py 1 --time-limit 10 --seed 42
    python run_ts.py 1 --time-limit 10 --seed 42
    python run_aco.py 1 --time-limit 10 --seed 42
    python run_dpso.py 1 --time-limit 10 --seed 42
    python run_dde.py 1 --time-limit 10 --seed 42

`1` can be replaced by:
- a Cicirello instance number from 1 to 120,
- a local `.instance` file path,
- or a raw/GitHub URL supported by the original loader.

Run all six:

    python run_all.py 1 --time-limit 10 --seed 42

Run all and save a summary:

    python run_all.py 1 --time-limit 10 --seed 42 --csv results.csv

## Fair comparison

For algorithm comparisons, use the same:
- instance,
- wall-clock limit or objective-evaluation limit,
- random seeds,
- number of independent replications.

A single seed is not enough for stochastic algorithms. For research-style comparison,
run multiple seeds and compare best/mean/worst/std as well as runtime.

## Algorithm notes

These implementations are logically consistent permutation variants adapted to SMTWT-SDST.
They are not claimed to be exact reproductions of every parameter/operator detail of the
specific ACO, DPSO, or DDE papers cited by Kirlik & Oguz (2012). Those papers use
problem-specific variants whose details should be implemented separately if strict paper
replication is the goal.
