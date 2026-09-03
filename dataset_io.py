from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from algorithms.gvns_smtwt_sdst import WTSDSInstance


@dataclass(frozen=True)
class ReducedInstance:
    instance: str
    source_number: int
    class_id: str
    tau: float
    R: float
    eta: float
    problem: WTSDSInstance


def _to_problem(row: dict) -> WTSDSInstance:
    return WTSDSInstance(
        name=row["instance"],
        n_jobs=int(row["n_jobs"]),
        processing_times=tuple(int(x) for x in row["processing_times"]),
        weights=tuple(int(x) for x in row["weights"]),
        due_dates=tuple(int(x) for x in row["due_dates"]),
        setup_times=tuple(tuple(int(x) for x in r) for r in row["setup_times"]),
        generator_parameters={
            str(k): float(v) for k, v in row.get("generator_parameters", {}).items()
        },
    )


def load_dataset(path: str | Path) -> list[ReducedInstance]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("instances")
    if not isinstance(rows, list):
        raise ValueError("Dataset JSON must contain an 'instances' list.")

    result: list[ReducedInstance] = []
    for row in rows:
        result.append(
            ReducedInstance(
                instance=str(row["instance"]),
                source_number=int(row["source_number"]),
                class_id=str(row["class_id"]),
                tau=float(row["tau"]),
                R=float(row["R"]),
                eta=float(row["eta"]),
                problem=_to_problem(row),
            )
        )

    
    return result
