from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gvns_smtwt_sdst import WTSDSInstance, load_cicirello_instance

TAUS = (0.3, 0.9)
RS = (0.25, 0.75)
ETAS = (0.25, 0.75)
PER_CLASS = 5
TARGETS = [(tau, R, eta) for tau in TAUS for R in RS for eta in ETAS]
CLASS_IDS = {combo: f"C{idx:02d}" for idx, combo in enumerate(TARGETS, start=1)}


def _rounded_key(instance: WTSDSInstance) -> tuple[float, float, float]:
    p = instance.generator_parameters
    try:
        return (round(float(p["tau"]), 6), round(float(p["r"]), 6), round(float(p["eta"]), 6))
    except KeyError as exc:
        raise ValueError(
            f"{instance.name} is missing Tau/R/Eta generator metadata."
        ) from exc


def _load(number: int, source_dir: Path | None) -> WTSDSInstance:
    if source_dir is None:
        return load_cicirello_instance(str(number))
    path = source_dir / f"wt_sds_{number}.instance"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")
    return load_cicirello_instance(path)


def _serialize(instance: WTSDSInstance, number: int, combo: tuple[float, float, float]) -> dict:
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
        "setup_times": [list(row) for row in instance.setup_times],
        "generator_parameters": instance.generator_parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed 40-instance Cicirello subset: tau={0.3,0.9}, "
            "R={0.25,0.75}, eta={0.25,0.75}, first five instances per class."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reduced_dataset.json"),
        help="Output JSON path (default: data/reduced_dataset.json).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing wt_sds_1.instance ... wt_sds_120.instance. "
            "If omitted, files are loaded from the Cicirello GitHub source."
        ),
    )
    args = parser.parse_args()

    target_set = set(TARGETS)
    grouped: dict[tuple[float, float, float], list[tuple[int, WTSDSInstance]]] = defaultdict(list)

    # Scan in source-number order. Taking the first five in each class is therefore deterministic.
    for number in range(1, 121):
        instance = _load(number, args.source_dir)
        key = _rounded_key(instance)
        if key in target_set and len(grouped[key]) < PER_CLASS:
            grouped[key].append((number, instance))

    missing = {combo: PER_CLASS - len(grouped[combo]) for combo in TARGETS if len(grouped[combo]) != PER_CLASS}
    if missing:
        raise RuntimeError(f"Could not collect five instances for every target class: {missing}")

    rows = []
    for combo in TARGETS:
        for number, instance in grouped[combo]:
            rows.append(_serialize(instance, number, combo))

    payload = {
        "benchmark": "Cicirello weighted tardiness with sequence-dependent setups",
        "selection_rule": "tau in {0.3,0.9}; R in {0.25,0.75}; eta in {0.25,0.75}; first 5 source-numbered instances per class",
        "n_classes": 8,
        "instances_per_class": 5,
        "n_instances": 40,
        "class_order": [
            {"class_id": CLASS_IDS[c], "tau": c[0], "R": c[1], "eta": c[2]}
            for c in TARGETS
        ],
        "instances": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} instances to {args.output}")
    for combo in TARGETS:
        nums = [number for number, _ in grouped[combo]]
        print(f"{CLASS_IDS[combo]} tau={combo[0]} R={combo[1]} eta={combo[2]} -> {nums}")


if __name__ == "__main__":
    main()
