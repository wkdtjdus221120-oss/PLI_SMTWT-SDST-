# SMTWT-SDST Metaheuristics Project

## 1. Problem

Target scheduling problem:

```text
1 | s_ij | sum(w_j T_j)
```

- **Single machine scheduling**
- **Sequence-dependent setup times (SDST)**
- **Total weighted tardiness (TWT)** minimization

For a job sequence, tardiness is

```text
T_j = max(C_j - d_j, 0)
```

and the objective is

```text
min sum(w_j * T_j)
```

All algorithms use the same instance representation, objective evaluation, and scheduling data so that the comparison is performed on an identical problem definition.

---

## 2. Project Goal

This project compares simple dispatching rules, a problem-specific heuristic, multiple metaheuristics, and a general-purpose MILP exact approach for SMTWT-SDST.

The main questions are:

1. How much can simple scheduling rules solve the problem?
2. How much do metaheuristics improve solution quality over heuristic baselines?
3. Are the stochastic algorithms robust across random seeds?
4. Does algorithm performance change with instance characteristics (`tau`, `R`, `eta`)?
5. How quickly does each algorithm find a good solution under a short computational budget?
6. How does a general-purpose MILP/Gurobi approach scale compared with heuristic search?

---

## 3. Current Project Structure

```text
PLI_SMTWT-SDST/
├─ algorithms/
│  ├─ __init__.py
│  ├─ aco_smtwt_sdst.py
│  ├─ dde_smtwt_sdst.py
│  ├─ dpso_smtwt_sdst.py
│  ├─ ga_smtwt_sdst.py
│  ├─ gvns_smtwt_sdst.py
│  ├─ metaheuristic_common.py
│  ├─ sa_smtwt_sdst.py
│  └─ ts_smtwt_sdst.py
├─ data/
│  ├─ reduced_dataset.json
│  └─ tuning_dataset.json
├─ results/
│  ├─ baseline_results.csv
│  ├─ best_params.json
│  ├─ results_raw.csv
│  ├─ convergence.csv
│  └─ analysis/
├─ build_reduced_dataset.py
├─ dataset_io.py
├─ run_tuning.py
├─ run_experiments.py
├─ analyze_results.py
├─ run_exact_gurobi.py
└─ README.md
```

The exact set of result files depends on which experiments have already been executed.

---

## 4. Dataset Design

### Original benchmark

The original Cicirello benchmark contains **120 instances**, each with **60 jobs**.

The problem classes are generated from combinations of:

```text
tau ∈ {0.3, 0.6, 0.9}
R   ∈ {0.25, 0.75}
eta ∈ {0.25, 0.75}
```

which gives 12 classes × 10 instances = 120 instances.

### Reduced final evaluation dataset

To keep the project computationally manageable, the final experiment uses:

```text
tau ∈ {0.3, 0.9}
R   ∈ {0.25, 0.75}
eta ∈ {0.25, 0.75}
```

This gives **8 classes**.

For each selected class, the first 5 source instances are used:

```text
8 classes × 5 instances = 40 final instances
```

Saved as:

```text
data/reduced_dataset.json
```

### Tuning dataset

The 6th source instance from each selected class is used for hyperparameter tuning:

```text
8 classes × 1 representative instance = 8 tuning instances
```

Saved as:

```text
data/tuning_dataset.json
```

The tuning instances and final evaluation instances are separated to avoid tuning directly on the final test set.

---

## 5. Baselines

The project currently uses three deterministic baselines.

### EDD — Earliest Due Date

```text
Sort jobs by increasing due date d_j.
```

Purpose: simple due-date-oriented dispatching baseline.

### WSPT — Weighted Shortest Processing Time

```text
Sort jobs by decreasing w_j / p_j.
```

Purpose: simple baseline that reflects job weight and processing time.

### ATCS — Apparent Tardiness Cost with Setups

ATCS considers:

- job weight,
- processing time,
- slack / due-date urgency,
- sequence-dependent setup time.

ATCS is the main problem-specific heuristic baseline and is also used as the initial solution for the metaheuristic implementations where applicable.

Expected baseline output:

```text
results/baseline_results.csv
```

Recommended columns:

```text
instance, tau, R, eta, EDD_twt, WSPT_twt, ATCS_twt
```

---

## 6. Metaheuristics

The final experiment includes seven stochastic metaheuristics.

| Algorithm | Main implementation idea |
|---|---|
| SA | Simulated Annealing with permutation neighborhood |
| GA | Order Crossover (OX) + swap mutation |
| TS | Swap neighborhood + tabu tenure + aspiration |
| ACO | Arc pheromone + scheduling heuristic desirability |
| DPSO | Discrete PSO using swap-sequence velocity |
| DDE | Discrete DE using swap-operator difference + permutation-safe crossover |
| GVNS | Shaking + VND with multiple neighborhoods |

These implementations are logically consistent permutation variants adapted to SMTWT-SDST. They are **not claimed to be exact reproductions of every parameter/operator detail** of the original ACO, DPSO, DDE, GA, or other algorithms in the literature.

---

## 7. Important ATCS Correction

The original uploaded GVNS code contained:

```python
k2 = tau / sqrt(2 * eta)
```

The corrected formula is:

```python
k2 = tau / (2 * sqrt(eta))
```

Only this ATCS formula was corrected. The benchmark instance interpretation and core problem definition remain unchanged.

---

## 8. Hyperparameter Tuning

### Common tuning protocol

- Dataset: `data/tuning_dataset.json`
- Instances: 8
- Seeds: 3
- Wall-clock time: 10 seconds per run
- Selection criterion: mean improvement relative to ATCS

For one run:

```text
improvement_vs_ATCS
= (ATCS_twt - best_twt) / ATCS_twt × 100
```

A larger value is better.

The mean is first evaluated across the tuning runs for each configuration. When two configurations have very similar mean performance, lower variability can be used as a secondary criterion.

### Tuning grids and selected parameters

| Algorithm | Tuned parameters | Candidate values | Selected |
|---|---|---|---|
| SA | cooling rate | {0.90, 0.95, 0.99} | 0.99 |
| GA | population size | {30, 50} | 50 |
|  | mutation rate | {0.10, 0.20} | 0.20 |
|  | tournament size | {2, 4} | 2 |
| TS | tabu tenure | {5, 10, 20} | 5 |
|  | candidate / neighborhood sample | {150, 300} | 300 |
| DDE | population size | {30, 50} | 50 |
|  | differential weight `F` | {0.40, 0.60, 0.80} | 0.60 |
| DPSO | swarm size | {30, 50} | 50 |
|  | inertia probability / weight | {0.30, 0.50, 0.70} | 0.70 |
| ACO | pheromone importance `alpha` | {1, 2} | 1 |
|  | heuristic importance `beta` | {2, 4} | 2 |
|  | evaporation rate `rho` | {0.1, 0.3} | 0.1 |

ACO therefore uses 8 configurations:

```text
2 alpha × 2 beta × 2 rho = 8 configurations
```

### Tuning execution

Examples:

```powershell
python run_tuning.py --algorithm SA
python run_tuning.py --algorithm GA
python run_tuning.py --algorithm TS
python run_tuning.py --algorithm ACO
python run_tuning.py --algorithm DPSO
python run_tuning.py --algorithm DDE
```

Tuning results are accumulated in:

```text
results/best_params.json
```

The tuning runner should use resume logic so completed configuration-instance-seed combinations do not need to be repeated.

---

## 9. Final Metaheuristic Experiment

### Final protocol

- Dataset: `data/reduced_dataset.json`
- Instances: 40
- Algorithms: SA, GA, TS, ACO, DPSO, DDE, GVNS
- Seeds: 5 per stochastic algorithm
- Wall-clock limit: 10 seconds per run
- Common problem data and objective evaluation

Total runs when all seven algorithms are executed:

```text
40 instances × 7 algorithms × 5 seeds = 1,400 runs
```

### Run one algorithm

Example — ACO only:

```powershell
python run_experiments.py --algorithms ACO
```

Example — GVNS only:

```powershell
python run_experiments.py --algorithms GVNS
```

### Run all algorithms

```powershell
python run_experiments.py --algorithms SA GA TS ACO DPSO DDE GVNS
```

### Important resume warning

When adding a missing algorithm later, **do not use `--overwrite`** if existing results should be preserved.

The runner appends/resumes into the common final files:

```text
results/results_raw.csv
results/convergence.csv
```

Expected raw-result columns include:

```text
algorithm
instance
tau
R
eta
seed
runtime_sec
best_twt
improvement_vs_ATCS
```

---

## 10. Random Seeds and Robustness

The benchmark instance itself is unchanged across seeds. The seed only controls stochastic decisions inside each algorithm.

Examples:

- **SA**: random neighbor generation and probabilistic acceptance
- **GA**: initial population, parent selection, crossover points, mutation
- **TS**: sampled neighborhood candidates in the simplified implementation
- **ACO**: probabilistic job selection based on pheromone / heuristic desirability
- **DPSO**: initial swarm and stochastic discrete movement
- **DDE**: initial population, selected individuals for mutation, discrete variation
- **GVNS**: random shaking moves; VND is largely deterministic

EDD, WSPT, and ATCS are deterministic when tie-breaking is fixed.

---

## 11. Analysis

Run:

```powershell
python analyze_results.py
```

The current analysis is divided into four main parts.

### 11.1 Instance-level solution quality

Main output:

```text
instance_algorithm_mean_table.csv
```

Each cell is the mean final TWT across seeds for an algorithm-instance pair.

### 11.2 Robustness

Main outputs:

```text
instance_seed_statistics.csv
seed_count_check.csv
robustness_<ALGORITHM>.png
```

For each algorithm-instance pair, the analysis reports:

- minimum TWT,
- mean TWT,
- maximum TWT,
- standard deviation,
- number of completed seeds.

The current experiment showed relatively large run-to-run variation for **GA, DPSO, and DDE** in some classes. This should be interpreted as sensitivity under the **current simplified implementations and fixed 10-second computational budget**, not as an inherent weakness of those algorithms.

Possible reasons include:

1. Population-based methods must maintain multiple solutions under a short wall-clock budget.
2. DPSO and DDE require a discrete reinterpretation of mechanisms originally developed for continuous search spaces.
3. GA crossover can preserve permutation validity while disrupting favorable SDST job adjacencies.

### 11.3 Instance-characteristic analysis

The current analysis uses **GVNS as the reference algorithm**, following the idea of the 2012 GVNS study.

For comparator algorithm `A` and instance `i`:

```text
Delta(A, i)
= (mean_TWT_GVNS,i - mean_TWT_A,i) / mean_TWT_A,i × 100
```

Interpretation:

```text
Delta < 0 : GVNS is better than comparator A
Delta = 0 : same performance
Delta > 0 : comparator A is better than GVNS
```

Seeds are averaged first for each algorithm-instance pair, then the relative differences are aggregated by instance class.

Main outputs:

```text
gvns_relative_instance_performance.csv
class_gvns_relative_performance_table.csv
class_gvns_relative_performance_table.png
instance_characteristic_gvns_relative_summary.csv
gvns_relative_by_tau.png
gvns_relative_by_R.png
gvns_relative_by_eta.png
```

### 11.4 Convergence analysis

The project records incumbent solution quality over elapsed time.

Main outputs:

```text
convergence_instance_level.csv
convergence_summary.csv
convergence_mean_twt.png
convergence_relative_twt.png
```

The normalized convergence metric is:

```text
relative_twt_pct = best_twt / ATCS_twt × 100
```

Interpretation:

```text
100% = ATCS level
<100% = better than ATCS
lower = better
```

Seeds are averaged within each algorithm-instance-time point first, then results are averaged across the 40 final instances.

---

## 12. Exact Method / Gurobi Experiment

The project also tests a **general-purpose MILP formulation solved by Gurobi**.

This is different from claiming that all exact scheduling algorithms have the same scalability. Problem-specific exact algorithms can use stronger lower bounds, dominance rules, dynamic programming, decomposition, or specialized branching.

### Scalability mode

Example:

```powershell
python run_exact_gurobi.py --mode scalability --class-id C01 --job-sizes 10 20 30 40 50 60 --time-limit 300
```

Typical outputs include:

```text
n_jobs
status
optimal_proven
runtime_sec
best_twt
best_bound
mip_gap
node_count
```

### Optimal mode

Example:

```powershell
python run_exact_gurobi.py --mode optimal --class-id C01 --job-sizes 60
```

No time limit is imposed in this mode.

### Current 60-job observation

For one 60-job instance, the Gurobi run was manually stopped after approximately:

```text
67,180 seconds ≈ 18.7 hours
```

At that point:

```text
Best feasible incumbent TWT = 2,412
Best lower bound            = 0
MIP gap                     = 100%
```

Interpretation:

- Gurobi found a feasible schedule.
- The current MILP relaxation did not produce a meaningful lower bound.
- Therefore, the solver could not verify how close the incumbent was to the optimum.
- This result demonstrates a scalability limitation of the **current general-purpose MILP formulation**, not of exact methods in general.

---

## 13. Current Main Findings

The current 40-instance experiment suggests:

1. **GVNS provides the strongest overall solution quality** in this implementation and experimental setting.
2. GVNS produced the best result on **39 of the 40 final instances** in the current result table.
3. Among the simple dispatching rules, **WSPT performed better than EDD** for the weighted tardiness objective in the current benchmark subset.
4. **TS and ACO** showed good short-time performance with relatively stable behavior.
5. **SA** showed a useful balance of solution quality and robustness under the 10-second budget.
6. **GA, DPSO, and DDE** showed larger seed sensitivity in some instance classes under the current simplified implementation.
7. The Gurobi experiment illustrates that finding a feasible solution and proving optimality can be very different computational tasks.

These findings are specific to the current implementation, parameter tuning range, selected 40-instance benchmark subset, and fixed computational budget.

---

## 14. Fair Comparison Rules

For algorithm comparisons, keep the following consistent:

- same instances,
- same final test set,
- same wall-clock budget,
- same seed set,
- same number of independent replications,
- same objective function,
- same hardware/runtime environment where possible.

A single stochastic run is not enough for robust comparison.

For final interpretation, compare both:

- solution quality,
- variability across seeds,
- convergence speed,
- runtime / computational budget.

---

## 15. Limitations

- Only 40 of the original 120 benchmark instances are used for the final experiment.
- The metaheuristic implementations are simplified educational / comparative implementations rather than strict reproductions of every source paper.
- A fixed 10-second wall-clock budget does not imply equal numbers of iterations or objective evaluations across algorithms.
- Hyperparameter tuning uses only 8 tuning instances and 3 seeds, so the selected parameters are not guaranteed to be globally optimal.
- Five final seeds are sufficient for a small project robustness comparison but do not fully characterize the stochastic distribution.
- The Gurobi result reflects the current MILP formulation and should not be generalized to all exact algorithms.
- The experiments use benchmark data rather than direct industrial production data.

---

## 16. Future Work

Possible extensions:

1. Compare exact and heuristic scalability as the number of jobs increases.
2. Strengthen the MILP formulation or compare with problem-specific exact algorithms.
3. Improve permutation-specific operators for GA, DDE, and DPSO.
4. Test hybrid approaches such as metaheuristic warm starts for exact solvers.
5. Extend the problem to unrelated parallel machines, batching, release dates, operator availability, machine breakdowns, or dynamic job arrivals.
6. Consider multi-objective scheduling including setup cost, makespan, energy, or production cost.
7. Validate the algorithms with real semiconductor manufacturing data such as dicing or final-test scheduling environments.

---

## 17. Reproducibility Notes

- Do not mix final evaluation instances with tuning instances.
- Do not use `--overwrite` when appending one missing algorithm to an existing final experiment.
- Keep the same hardware/runtime environment for fixed wall-clock comparisons whenever possible.
- Record random seeds for every stochastic run.
- When interpreting Gurobi results, distinguish between:
  - best feasible incumbent,
  - best lower bound,
  - MIP gap,
  - optimality proof.

