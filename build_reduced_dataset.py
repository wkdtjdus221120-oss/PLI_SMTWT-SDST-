from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from algorithms.gvns_smtwt_sdst import WTSDSInstance, load_cicirello_instance


# --------------------------------------------------
# Dataset settings
# --------------------------------------------------

TAUS = (0.3, 0.9)
RS = (0.25, 0.75)
ETAS = (0.25, 0.75)

# Final evaluation set
TEST_PER_CLASS = 5

# Hyperparameter tuning set
TUNING_PER_CLASS = 1

# Therefore, for each class:
# first 5  -> test
# 6th      -> tuning
TOTAL_NEEDED_PER_CLASS = TEST_PER_CLASS + TUNING_PER_CLASS


TARGETS = [
    (tau, R, eta)
    for tau in TAUS
    for R in RS
    for eta in ETAS
]

CLASS_IDS = {
    combo: f"C{idx:02d}"
    for idx, combo in enumerate(TARGETS, start=1)
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def _rounded_key(
    instance: WTSDSInstance
) -> tuple[float, float, float]:

    p = instance.generator_parameters

    try:
        return (
            round(float(p["tau"]), 6),
            round(float(p["r"]), 6),
            round(float(p["eta"]), 6),
        )

    except KeyError as exc:
        raise ValueError(
            f"{instance.name} is missing Tau/R/Eta generator metadata."
        ) from exc


def _load(
    number: int,
    source_dir: Path | None,
) -> WTSDSInstance:

    if source_dir is None:
        return load_cicirello_instance(str(number))

    path = source_dir / f"wt_sds_{number}.instance"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {path}"
        )

    return load_cicirello_instance(path)


def _serialize(
    instance: WTSDSInstance,
    number: int,
    combo: tuple[float, float, float],
) -> dict:

    tau, R, eta = combo

    return {
        "instance": f"wt_sds_{number}",
        "source_number": number,
        "class_id": CLASS_IDS[combo],
        "tau": tau,
        "R": R,
        "eta": eta,
        "n_jobs": instance.n_jobs,
        "processing_times": list(instance.processing_times),
        "weights": list(instance.weights),
        "due_dates": list(instance.due_dates),
        "setup_times": [
            list(row)
            for row in instance.setup_times
        ],
        "generator_parameters": instance.generator_parameters,
    }


def _save_dataset(
    rows: list[dict],
    output_path: Path,
    dataset_type: str,
    instances_per_class: int,
    selection_rule: str,
) -> None:

    payload = {
        "benchmark":
            "Cicirello weighted tardiness "
            "with sequence-dependent setups",

        "dataset_type": dataset_type,

        "selection_rule": selection_rule,

        "n_classes": 8,

        "instances_per_class": instances_per_class,

        "n_instances": len(rows),

        "class_order": [
            {
                "class_id": CLASS_IDS[c],
                "tau": c[0],
                "R": c[1],
                "eta": c[2],
            }
            for c in TARGETS
        ],

        "instances": rows,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create fixed Cicirello datasets for "
            "final evaluation and parameter tuning. "
            "tau={0.3,0.9}, "
            "R={0.25,0.75}, "
            "eta={0.25,0.75}. "
            "For each class, the first five instances "
            "are used for final evaluation and the "
            "sixth instance is used for tuning."
        )
    )

    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path(
            "data/reduced_dataset.json"
        ),
        help=(
            "Final evaluation dataset output path. "
            "(default: data/reduced_dataset.json)"
        ),
    )

    parser.add_argument(
        "--tuning-output",
        type=Path,
        default=Path(
            "data/tuning_dataset.json"
        ),
        help=(
            "Tuning dataset output path. "
            "(default: data/tuning_dataset.json)"
        ),
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing "
            "wt_sds_1.instance ... "
            "wt_sds_120.instance. "
            "If omitted, files are loaded using "
            "the existing Cicirello loader."
        ),
    )

    args = parser.parse_args()

    target_set = set(TARGETS)

    grouped: dict[
        tuple[float, float, float],
        list[tuple[int, WTSDSInstance]],
    ] = defaultdict(list)

    # --------------------------------------------------
    # Collect first SIX instances for each target class
    # --------------------------------------------------

    # Scan benchmark in source-number order.
    # Therefore selection is deterministic.
    for number in range(1, 121):

        instance = _load(
            number,
            args.source_dir,
        )

        key = _rounded_key(instance)

        if (
            key in target_set
            and len(grouped[key])
            < TOTAL_NEEDED_PER_CLASS
        ):
            grouped[key].append(
                (number, instance)
            )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    missing = {
        combo:
            TOTAL_NEEDED_PER_CLASS
            - len(grouped[combo])

        for combo in TARGETS

        if (
            len(grouped[combo])
            != TOTAL_NEEDED_PER_CLASS
        )
    }

    if missing:
        raise RuntimeError(
            "Could not collect six instances "
            "for every target class: "
            f"{missing}"
        )

    # --------------------------------------------------
    # Split into TEST and TUNING datasets
    # --------------------------------------------------

    test_rows = []
    tuning_rows = []

    for combo in TARGETS:

        class_instances = grouped[combo]

        # First five instances
        test_instances = (
            class_instances[:TEST_PER_CLASS]
        )

        # Sixth instance
        tuning_instances = (
            class_instances[
                TEST_PER_CLASS:
                TEST_PER_CLASS
                + TUNING_PER_CLASS
            ]
        )

        for number, instance in test_instances:

            test_rows.append(
                _serialize(
                    instance,
                    number,
                    combo,
                )
            )

        for number, instance in tuning_instances:

            tuning_rows.append(
                _serialize(
                    instance,
                    number,
                    combo,
                )
            )

    # --------------------------------------------------
    # Save final evaluation dataset
    # --------------------------------------------------

    _save_dataset(
        rows=test_rows,

        output_path=args.test_output,

        dataset_type="final_evaluation",

        instances_per_class=TEST_PER_CLASS,

        selection_rule=(
            "tau in {0.3,0.9}; "
            "R in {0.25,0.75}; "
            "eta in {0.25,0.75}; "
            "first 5 source-numbered "
            "instances per class"
        ),
    )

    # --------------------------------------------------
    # Save tuning dataset
    # --------------------------------------------------

    _save_dataset(
        rows=tuning_rows,

        output_path=args.tuning_output,

        dataset_type="hyperparameter_tuning",

        instances_per_class=TUNING_PER_CLASS,

        selection_rule=(
            "tau in {0.3,0.9}; "
            "R in {0.25,0.75}; "
            "eta in {0.25,0.75}; "
            "6th source-numbered instance "
            "per class"
        ),
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print()
    print("Dataset generation completed.")
    print()

    print(
        f"Final evaluation dataset: "
        f"{len(test_rows)} instances"
    )

    print(
        f"Saved to: "
        f"{args.test_output}"
    )

    print()

    print(
        f"Tuning dataset: "
        f"{len(tuning_rows)} instances"
    )

    print(
        f"Saved to: "
        f"{args.tuning_output}"
    )

    print()

    print(
        "Selected instances by class"
    )

    print("-" * 70)

    for combo in TARGETS:

        selected = grouped[combo]

        test_nums = [
            number
            for number, _
            in selected[:TEST_PER_CLASS]
        ]

        tuning_nums = [
            number
            for number, _
            in selected[
                TEST_PER_CLASS:
                TEST_PER_CLASS
                + TUNING_PER_CLASS
            ]
        ]

        print(
            f"{CLASS_IDS[combo]} | "
            f"tau={combo[0]}, "
            f"R={combo[1]}, "
            f"eta={combo[2]} | "
            f"test={test_nums} | "
            f"tuning={tuning_nums}"
        )


if __name__ == "__main__":
    main()