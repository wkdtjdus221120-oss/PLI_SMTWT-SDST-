from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset_io import load_reduced_dataset
from gvns_smtwt_sdst import generate_atcs_sequence, total_weighted_tardiness


def edd_sequence(due_dates: tuple[int, ...]) -> list[int]:
    """Earliest Due Date; job id is only a deterministic tie-breaker."""
    return sorted(range(len(due_dates)), key=lambda j: (due_dates[j], j))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDD and ATCS on all 40 reduced instances.")
    parser.add_argument("--dataset", type=Path, default=Path("data/reduced_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_results.csv"))
    args = parser.parse_args()

    instances = load_reduced_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["instance", "class_id", "tau", "R", "eta", "EDD_twt", "ATCS_twt"],
        )
        writer.writeheader()

        for item in instances:
            problem = item.problem
            edd = edd_sequence(problem.due_dates)
            atcs, _ = generate_atcs_sequence(problem)
            writer.writerow(
                {
                    "instance": item.instance,
                    "class_id": item.class_id,
                    "tau": item.tau,
                    "R": item.R,
                    "eta": item.eta,
                    "EDD_twt": total_weighted_tardiness(edd, problem),
                    "ATCS_twt": total_weighted_tardiness(atcs, problem),
                }
            )

    print(f"Saved baseline results to {args.output}")


if __name__ == "__main__":
    main()
