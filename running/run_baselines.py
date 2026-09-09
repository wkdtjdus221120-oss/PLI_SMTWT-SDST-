from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_io import load_dataset
from algorithms.gvns_smtwt_sdst import (
    WTSDSInstance,
    generate_atcs_sequence,
    total_weighted_tardiness,
)


def edd_sequence(due_dates: tuple[int, ...]) -> list[int]:
    """Earliest Due Date; job id is only a deterministic tie-breaker."""
    return sorted(range(len(due_dates)), key=lambda j: (due_dates[j], j))


def wspt_sequence(instance: WTSDSInstance) -> list[int]:
    """Descending weight / processing time; ties retain ascending job id."""
    return sorted(
        range(instance.n_jobs),
        key=lambda j: instance.weights[j] / instance.processing_times[j],
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDD, ATCS and WSPT on the dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/reduced_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_results.csv"))
    args = parser.parse_args()

    instances = load_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["instance", "class_id", "tau", "R", "eta", "EDD_twt", "ATCS_twt", "WSPT_twt"],
        )
        writer.writeheader()

        for item in instances:
            problem = item.problem
            edd = edd_sequence(problem.due_dates)
            atcs, _ = generate_atcs_sequence(problem)
            wspt = wspt_sequence(problem)
            writer.writerow(
                {
                    "instance": item.instance,
                    "class_id": item.class_id,
                    "tau": item.tau,
                    "R": item.R,
                    "eta": item.eta,
                    "EDD_twt": total_weighted_tardiness(edd, problem),
                    "ATCS_twt": total_weighted_tardiness(atcs, problem),
                    "WSPT_twt": total_weighted_tardiness(wspt, problem),
                }
            )

    print(f"Saved baseline results to {args.output}")


if __name__ == "__main__":
    main()
