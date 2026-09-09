from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# Settings
# ============================================================================

PREFERRED_ALGORITHM_ORDER = [
    "SA",
    "GA",
    "TS",
    "ACO",
    "DPSO",
    "DDE",
    "GVNS",
]

REFERENCE_ALGORITHM = "GVNS"

REQUIRED_RESULT_COLUMNS = {
    "algorithm",
    "instance",
    "tau",
    "R",
    "eta",
    "seed",
    "best_twt",
}

REQUIRED_CONVERGENCE_COLUMNS = {
    "algorithm",
    "instance",
    "tau",
    "R",
    "eta",
    "seed",
    "elapsed_sec",
    "best_twt",
}


# ============================================================================
# Basic helpers
# ============================================================================

def _instance_number(value: str) -> int:
    """Extract the last integer from an instance name."""
    numbers = re.findall(r"\d+", str(value))
    if not numbers:
        return 10**9
    return int(numbers[-1])


def _ordered_algorithms(df: pd.DataFrame) -> list[str]:
    present = list(
        df["algorithm"]
        .dropna()
        .astype(str)
        .unique()
    )

    ordered = [
        algorithm
        for algorithm in PREFERRED_ALGORITHM_ORDER
        if algorithm in present
    ]

    extras = sorted(
        algorithm
        for algorithm in present
        if algorithm not in ordered
    )

    return ordered + extras


def _add_class_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct class_id from (tau, R, eta).

    For the reduced design:
      tau = {0.3, 0.9}
      R   = {0.25, 0.75}
      eta = {0.25, 0.75}

    the lexicographically sorted combinations become C01 ... C08.
    """
    if "class_id" in df.columns:
        return df

    class_meta = (
        df[["tau", "R", "eta"]]
        .drop_duplicates()
        .sort_values(
            ["tau", "R", "eta"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    class_meta["class_id"] = [
        f"C{i:02d}"
        for i in range(1, len(class_meta) + 1)
    ]

    return df.merge(
        class_meta,
        on=["tau", "R", "eta"],
        how="left",
        validate="many_to_one",
    )


def _save_figure(
    fig: plt.Figure,
    path: Path,
) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[Saved] Figure: {path}")


def _delta_reference_vs_comparator(
    reference_twt: float,
    comparator_twt: float,
) -> float:
    """
    Paper-style relative difference:

        Delta(A)
        = (TWT_GVNS - TWT_A) / TWT_A * 100

    Interpretation
    --------------
    Delta < 0 : GVNS is better than comparator A
    Delta = 0 : same
    Delta > 0 : comparator A is better than GVNS

    Zero denominator handling
    -------------------------
    comparator = 0 and GVNS = 0 -> 0
    comparator = 0 and GVNS > 0 -> NaN
    """
    comparator_twt = float(comparator_twt)
    reference_twt = float(reference_twt)

    if comparator_twt == 0:
        if reference_twt == 0:
            return 0.0
        return float("nan")

    return (
        (reference_twt - comparator_twt)
        / comparator_twt
        * 100.0
    )


# ============================================================================
# Load results_raw.csv
# ============================================================================

def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"results_raw.csv not found: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "results_raw.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        "tau",
        "R",
        "eta",
        "seed",
        "best_twt",
    ]

    if "runtime_sec" in df.columns:
        numeric_columns.append("runtime_sec")

    if "improvement_vs_ATCS" in df.columns:
        numeric_columns.append(
            "improvement_vs_ATCS"
        )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["algorithm"] = (
        df["algorithm"].astype(str)
    )
    df["instance"] = (
        df["instance"].astype(str)
    )

    # Resume execution can occasionally leave duplicated raw rows.
    run_key = [
        "algorithm",
        "instance",
        "seed",
    ]

    duplicated = df.duplicated(
        run_key,
        keep=False,
    )

    if duplicated.any():
        n_rows = int(duplicated.sum())

        print(
            "[Warning] Duplicate "
            "(algorithm, instance, seed) rows found: "
            f"{n_rows}. "
            "The last row for each key will be used."
        )

        df = df.drop_duplicates(
            run_key,
            keep="last",
        )

    df = _add_class_id(df)

    df["instance_no"] = (
        df["instance"]
        .map(_instance_number)
    )

    return df


# ============================================================================
# Load baseline_results.csv
# ============================================================================

def _load_baselines(path: Path) -> pd.DataFrame:
    """
    Load ATCS baseline values.

    Used only for normalized convergence analysis.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"baseline_results.csv not found: {path}"
        )

    df = pd.read_csv(path)

    if "instance" not in df.columns:
        raise ValueError(
            "baseline_results.csv must contain "
            "an 'instance' column."
        )

    atcs_candidates = [
        column
        for column in df.columns
        if column.lower() == "atcs_twt"
    ]

    if not atcs_candidates:
        raise ValueError(
            "baseline_results.csv must contain "
            "an 'ATCS_twt' column."
        )

    atcs_column = atcs_candidates[0]

    df = df.rename(
        columns={
            atcs_column: "ATCS_twt"
        }
    )

    df["instance"] = (
        df["instance"].astype(str)
    )

    df["ATCS_twt"] = pd.to_numeric(
        df["ATCS_twt"],
        errors="raise",
    )

    duplicated = df.duplicated(
        ["instance"],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "Duplicate instance rows found in "
            "baseline_results.csv."
        )

    return df[
        [
            "instance",
            "ATCS_twt",
        ]
    ].copy()


# ============================================================================
# Analysis 1:
# Paper-like instance x algorithm mean TWT table
# ============================================================================

def make_mean_table(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_path: Path = Path("results/baseline_results.csv"),
) -> pd.DataFrame:
    """
    Each cell = mean best_twt across seeds
    for one algorithm on one instance.
    """
    algorithms = _ordered_algorithms(df)

    means = (
        df.groupby(
            [
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
                "algorithm",
            ],
            as_index=False,
        )["best_twt"]
        .mean()
        .rename(
            columns={
                "best_twt": "mean_best_twt"
            }
        )
    )

    wide = (
        means.pivot(
            index=[
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
            ],
            columns="algorithm",
            values="mean_best_twt",
        )
        .reset_index()
    )

    meta_columns = [
        "class_id",
        "tau",
        "R",
        "eta",
        "instance",
        "instance_no",
    ]

    algorithm_columns = [
        algorithm
        for algorithm in algorithms
        if algorithm in wide.columns
    ]

    wide = wide[
        meta_columns
        + algorithm_columns
    ].sort_values(
        ["class_id", "instance_no"],
        kind="stable",
    )

    for algorithm in algorithm_columns:
        wide[algorithm] = (
            wide[algorithm].round(2)
        )

    path = (
        output_dir
        / "instance_algorithm_mean_table.csv"
    )

    wide = _add_baseline_columns(wide, baseline_path)
    wide.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[Saved] Paper-like mean table: {path}"
    )

    save_mean_table_image(path, output_dir / "instance_algorithm_mean_table.png")

    return wide


def _add_baseline_columns(table: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    """Join deterministic baseline TWTs by instance, refreshing existing columns."""
    names = ["EDD", "ATCS", "WSPT"]
    source_columns = [f"{name}_twt" for name in names]
    baseline = pd.read_csv(baseline_path)
    missing = {"instance", *source_columns} - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline CSV is missing columns: {sorted(missing)}")
    baseline = baseline[["instance", *source_columns]].rename(
        columns=dict(zip(source_columns, names))
    )
    for name in names:
        baseline[name] = pd.to_numeric(baseline[name], errors="raise")
    merged = table.drop(columns=names, errors="ignore").merge(
        baseline, on="instance", how="left", validate="many_to_one", sort=False
    )
    if merged[names].isna().any(axis=1).any():
        instances = merged.loc[merged[names].isna().any(axis=1), "instance"].tolist()
        raise ValueError(f"Missing baseline TWT values for instances: {instances}")
    return merged


def save_mean_table_image(csv_path: Path, image_path: Path) -> None:
    """Render the CSV with each instance's minimum TWT in bold (all ties)."""
    table_df = pd.read_csv(csv_path)
    meta_columns = {"class_id", "tau", "R", "eta", "instance", "instance_no"}
    algorithms = [column for column in table_df if column not in meta_columns]
    if table_df.empty or not algorithms:
        raise ValueError("Mean table must contain rows and algorithm TWT columns.")

    values = table_df[algorithms].apply(pd.to_numeric, errors="raise")
    winners = values.eq(values.min(axis=1), axis=0) & values.notna()
    display = table_df.copy()
    for algorithm in algorithms:
        display[algorithm] = values[algorithm].map(
            lambda value: "-" if pd.isna(value) else f"{value:,.2f}"
        )
    display = display.fillna("-")

    fig, ax = plt.subplots(
        figsize=(max(12, len(display.columns) * 1.15),
                 max(2, (len(display) + 1) * 0.30 + 0.8))
    )
    try:
        ax.axis("off")
        ax.set_title("TWT by instance (metaheuristics: seed mean; bold = best; ties included)", pad=12)
        table = ax.table(
            cellText=display.astype(str).values,
            colLabels=display.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(list(range(len(display.columns))))
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor("#dce6f1")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
                name = display.columns[column]
                if name in algorithms and winners.iloc[row - 1][name]:
                    cell.get_text().set_weight("bold")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    print(f"[Saved] Mean table image: {image_path}")


# ============================================================================
# Analysis 2:
# Min / mean / max / standard deviation across seeds
# ============================================================================

def make_instance_stats(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "algorithm",
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
            ]
        )["best_twt"]
        .agg(
            n_seeds="count",
            min_twt="min",
            mean_twt="mean",
            max_twt="max",
            std_twt="std",
        )
        .reset_index()
    )

    grouped["std_twt"] = (
        grouped["std_twt"]
        .fillna(0.0)
    )

    for column in [
        "min_twt",
        "mean_twt",
        "max_twt",
        "std_twt",
    ]:
        grouped[column] = (
            grouped[column].round(2)
        )

    grouped = grouped.sort_values(
        [
            "algorithm",
            "class_id",
            "instance_no",
        ],
        kind="stable",
    )

    stats_path = (
        output_dir
        / "instance_seed_statistics.csv"
    )

    grouped.to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[Saved] Instance statistics: {stats_path}"
    )

    seed_count_path = (
        output_dir
        / "seed_count_check.csv"
    )

    grouped[
        [
            "algorithm",
            "class_id",
            "instance",
            "n_seeds",
        ]
    ].to_csv(
        seed_count_path,
        index=False,
        encoding="utf-8-sig",
    )

    return grouped


def plot_algorithm_robustness(
    stats: pd.DataFrame,
    output_dir: Path,
    show_std: bool = True,
) -> None:
    """
    One figure per algorithm.

    Each class is a subplot.
    For each instance:
      v : min
      o : mean
      ^ : max
      vertical range : min to max
      errorbar       : mean +/- 1 SD
    """
    algorithms = _ordered_algorithms(
        stats
    )

    for algorithm in algorithms:
        algorithm_df = stats[
            stats["algorithm"]
            == algorithm
        ].copy()

        classes = (
            algorithm_df[
                [
                    "class_id",
                    "tau",
                    "R",
                    "eta",
                ]
            ]
            .drop_duplicates()
            .sort_values("class_id")
        )

        n_classes = len(classes)

        if n_classes == 0:
            continue

        n_columns = (
            4
            if n_classes > 4
            else n_classes
        )

        n_rows = math.ceil(
            n_classes / n_columns
        )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(
                4.2 * n_columns,
                3.4 * n_rows,
            ),
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for axis_index, (_, class_row) in enumerate(
            classes.iterrows()
        ):
            ax = axes_flat[axis_index]

            class_id = class_row[
                "class_id"
            ]

            class_df = (
                algorithm_df[
                    algorithm_df["class_id"]
                    == class_id
                ]
                .sort_values(
                    "instance_no",
                    kind="stable",
                )
            )

            x = list(
                range(
                    1,
                    len(class_df) + 1,
                )
            )

            min_values = (
                class_df["min_twt"]
                .astype(float)
                .to_numpy()
            )

            mean_values = (
                class_df["mean_twt"]
                .astype(float)
                .to_numpy()
            )

            max_values = (
                class_df["max_twt"]
                .astype(float)
                .to_numpy()
            )

            std_values = (
                class_df["std_twt"]
                .astype(float)
                .to_numpy()
            )

            ax.vlines(
                x,
                min_values,
                max_values,
                linewidth=1.0,
                alpha=0.65,
                label=(
                    "Min-Max range"
                    if axis_index == 0
                    else None
                ),
            )

            ax.scatter(
                x,
                min_values,
                marker="v",
                s=30,
                label=(
                    "Min"
                    if axis_index == 0
                    else None
                ),
            )

            ax.scatter(
                x,
                mean_values,
                marker="o",
                s=34,
                label=(
                    "Mean"
                    if axis_index == 0
                    else None
                ),
                zorder=3,
            )

            ax.scatter(
                x,
                max_values,
                marker="^",
                s=30,
                label=(
                    "Max"
                    if axis_index == 0
                    else None
                ),
            )

            if show_std:
                ax.errorbar(
                    x,
                    mean_values,
                    yerr=std_values,
                    fmt="none",
                    capsize=3,
                    linewidth=1.0,
                    label=(
                        "Mean +/- 1 SD"
                        if axis_index == 0
                        else None
                    ),
                    zorder=2,
                )

            labels = list(
                class_df["instance"]
                .astype(str)
            )

            ax.set_xticks(
                x,
                labels,
                rotation=35,
                ha="right",
            )

            ax.set_xlabel(
                "Instance"
            )

            ax.set_ylabel(
                "TWT"
            )

            ax.set_title(
                f"{class_id}  "
                f"(tau={class_row['tau']}, "
                f"R={class_row['R']}, "
                f"eta={class_row['eta']})"
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

        for index in range(
            n_classes,
            len(axes_flat),
        ):
            axes_flat[index].axis("off")

        handles, labels = (
            axes_flat[0]
            .get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=(
                    5
                    if show_std
                    else 4
                ),
                frameon=True,
            )

        fig.suptitle(
            f"{algorithm}: seed-to-seed robustness by instance",
            y=0.995,
            fontsize=14,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.94,
            ]
        )

        path = (
            output_dir
            / f"robustness_{algorithm}.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[Saved] Robustness figure: {path}"
        )


# ============================================================================
# Analysis 3:
# GVNS-centered instance-characteristic analysis
# ============================================================================

def make_gvns_relative_instance_analysis(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the logic of the paper's class-level delta analysis.

    Step 1
    ------
    Average stochastic seeds for each algorithm-instance:

        mean_TWT(A, i)

    Step 2
    ------
    For every comparator A != GVNS:

        Delta(A, i)
        = [mean_TWT(GVNS, i) - mean_TWT(A, i)]
          / mean_TWT(A, i) * 100

    Interpretation
    --------------
    Delta < 0 : GVNS is better
    Delta = 0 : tie
    Delta > 0 : comparator is better

    Step 3
    ------
    Average the instance-level deltas within each
    (tau, R, eta) class.

    Step 4
    ------
    Aggregate again by tau, R, and eta to study
    how the relative strength of GVNS changes with
    problem characteristics.
    """

    if REFERENCE_ALGORITHM not in set(
        df["algorithm"]
    ):
        raise ValueError(
            f"{REFERENCE_ALGORITHM} results are required "
            "for GVNS-centered characteristic analysis."
        )

    algorithms = _ordered_algorithms(df)

    comparators = [
        algorithm
        for algorithm in algorithms
        if algorithm != REFERENCE_ALGORITHM
    ]

    if not comparators:
        raise ValueError(
            "At least one comparator algorithm is required."
        )

    # ----------------------------------------------------------------------
    # 1. Mean TWT across seeds for every algorithm-instance
    # ----------------------------------------------------------------------

    seed_means = (
        df.groupby(
            [
                "algorithm",
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
            ],
            as_index=False,
        )
        .agg(
            mean_twt=(
                "best_twt",
                "mean",
            ),
            n_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    wide = (
        seed_means.pivot(
            index=[
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
            ],
            columns="algorithm",
            values="mean_twt",
        )
        .reset_index()
    )

    # Validate that GVNS exists for every instance.
    if wide[REFERENCE_ALGORITHM].isna().any():
        missing_instances = sorted(
            wide.loc[
                wide[REFERENCE_ALGORITHM].isna(),
                "instance",
            ].unique()
        )

        raise ValueError(
            "GVNS is missing for the following instances: "
            f"{missing_instances}"
        )

    # ----------------------------------------------------------------------
    # 2. Instance-level Delta relative to each comparator
    # ----------------------------------------------------------------------

    instance_delta_rows = []

    for _, row in wide.iterrows():
        gvns_twt = row[
            REFERENCE_ALGORITHM
        ]

        for comparator in comparators:
            comparator_twt = row.get(
                comparator,
                float("nan"),
            )

            if pd.isna(comparator_twt):
                delta = float("nan")
            else:
                delta = _delta_reference_vs_comparator(
                    gvns_twt,
                    comparator_twt,
                )

            instance_delta_rows.append(
                {
                    "class_id":
                        row["class_id"],

                    "tau":
                        row["tau"],

                    "R":
                        row["R"],

                    "eta":
                        row["eta"],

                    "instance":
                        row["instance"],

                    "instance_no":
                        row["instance_no"],

                    "comparator":
                        comparator,

                    "gvns_mean_twt":
                        gvns_twt,

                    "comparator_mean_twt":
                        comparator_twt,

                    "delta_gvns_vs_comparator_pct":
                        delta,
                }
            )

    instance_delta = pd.DataFrame(
        instance_delta_rows
    )

    instance_delta = (
        instance_delta.sort_values(
            [
                "class_id",
                "instance_no",
                "comparator",
            ],
            kind="stable",
        )
    )

    instance_delta_path = (
        output_dir
        / "gvns_relative_instance_performance.csv"
    )

    instance_delta.to_csv(
        instance_delta_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[Saved] GVNS relative instance performance: "
        f"{instance_delta_path}"
    )

    # ----------------------------------------------------------------------
    # 3. Paper-style class table
    # ----------------------------------------------------------------------

    class_long = (
        instance_delta.groupby(
            [
                "class_id",
                "tau",
                "R",
                "eta",
                "comparator",
            ],
            as_index=False,
        )
        .agg(
            n_instances=(
                "instance",
                "nunique",
            ),
            mean_delta_pct=(
                "delta_gvns_vs_comparator_pct",
                "mean",
            ),
        )
    )

    class_wide = (
        class_long.pivot(
            index=[
                "class_id",
                "tau",
                "R",
                "eta",
                "n_instances",
            ],
            columns="comparator",
            values="mean_delta_pct",
        )
        .reset_index()
    )

    class_wide = class_wide.sort_values(
        "class_id",
        kind="stable",
    )

    # Add source-number range to resemble the paper.
    class_ranges = (
        df[
            [
                "class_id",
                "instance_no",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "class_id"
        )["instance_no"]
        .agg(
            first_instance="min",
            last_instance="max",
        )
        .reset_index()
    )

    class_wide = class_wide.merge(
        class_ranges,
        on="class_id",
        how="left",
        validate="one_to_one",
    )

    class_wide["Instance"] = (
        "tau="
        + class_wide["tau"].astype(str)
        + ", R="
        + class_wide["R"].astype(str)
        + ", eta="
        + class_wide["eta"].astype(str)
        + " ("
        + class_wide[
            "first_instance"
        ].astype(int).astype(str)
        + "-"
        + class_wide[
            "last_instance"
        ].astype(int).astype(str)
        + ")"
    )

    delta_columns = []

    for comparator in comparators:
        if comparator in class_wide.columns:
            new_name = f"Delta_{comparator}"
            class_wide = class_wide.rename(
                columns={
                    comparator: new_name
                }
            )
            delta_columns.append(
                new_name
            )

    paper_table = class_wide[
        [
            "Instance",
            "class_id",
            "tau",
            "R",
            "eta",
            "n_instances",
        ]
        + delta_columns
    ].copy()

    # Average row: mean of all instance-level deltas.
    # Because every class currently has five instances,
    # this is also equal to the unweighted mean of class means.
    average_row = {
        "Instance": "Average",
        "class_id": "",
        "tau": "",
        "R": "",
        "eta": "",
        "n_instances":
            instance_delta[
                "instance"
            ].nunique(),
    }

    for comparator in comparators:
        column = f"Delta_{comparator}"

        if column in paper_table.columns:
            values = (
                instance_delta.loc[
                    instance_delta["comparator"]
                    == comparator,
                    "delta_gvns_vs_comparator_pct",
                ]
            )

            average_row[column] = (
                values.mean()
            )

    paper_table = pd.concat(
        [
            paper_table,
            pd.DataFrame(
                [average_row]
            ),
        ],
        ignore_index=True,
    )

    for column in delta_columns:
        paper_table[column] = (
            pd.to_numeric(
                paper_table[column],
                errors="coerce",
            )
            .round(2)
        )

    paper_table_path = (
        output_dir
        / "class_gvns_relative_performance_table.csv"
    )

    paper_table.to_csv(
        paper_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[Saved] Paper-style GVNS class table: "
        f"{paper_table_path}"
    )

    # ----------------------------------------------------------------------
    # 4. Render a paper-like PNG table
    # ----------------------------------------------------------------------

    display_columns = (
        ["Instance"]
        + delta_columns
    )

    display_df = paper_table[
        display_columns
    ].copy()

    display_labels = {
        "Instance": "Instance"
    }

    for comparator in comparators:
        display_labels[
            f"Delta_{comparator}"
        ] = f"Delta {comparator}"

    display_df = display_df.rename(
        columns=display_labels
    )

    for column in display_df.columns[1:]:
        display_df[column] = display_df[
            column
        ].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{value:.2f}"
            )
        )

    figure_height = (
        1.6
        + 0.42 * len(display_df)
    )

    figure_width = (
        6.8
        + 1.0
        * max(
            0,
            len(display_df.columns) - 4,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(
        1.0,
        1.35,
    )

    # Make first column wider.
    for (row_idx, col_idx), cell in table.get_celld().items():
        if col_idx == 0:
            cell.set_width(0.42)
            cell.get_text().set_ha(
                "left"
            )
        else:
            cell.set_width(
                0.11
            )

        if row_idx == 0:
            cell.get_text().set_weight(
                "bold"
            )

        if row_idx == len(display_df):
            cell.get_text().set_weight(
                "bold"
            )

    ax.set_title(
        "GVNS relative performance by instance class\n"
        "Delta(A) = (GVNS - A) / A x 100; negative values favor GVNS",
        fontsize=11,
        pad=14,
    )

    table_png_path = (
        output_dir
        / "class_gvns_relative_performance_table.png"
    )

    _save_figure(
        fig,
        table_png_path,
    )

    # ----------------------------------------------------------------------
    # 5. Factor-level characteristic analysis
    # ----------------------------------------------------------------------

    factor_frames = []

    for factor in [
        "tau",
        "R",
        "eta",
    ]:
        summary = (
            instance_delta.groupby(
                [
                    "comparator",
                    factor,
                ],
                as_index=False,
            )
            .agg(
                n_instances=(
                    "instance",
                    "nunique",
                ),
                mean_delta_pct=(
                    "delta_gvns_vs_comparator_pct",
                    "mean",
                ),
                std_delta_pct=(
                    "delta_gvns_vs_comparator_pct",
                    "std",
                ),
                min_delta_pct=(
                    "delta_gvns_vs_comparator_pct",
                    "min",
                ),
                max_delta_pct=(
                    "delta_gvns_vs_comparator_pct",
                    "max",
                ),
            )
            .rename(
                columns={
                    factor: "level"
                }
            )
        )

        summary.insert(
            1,
            "factor",
            factor,
        )

        summary[
            "std_delta_pct"
        ] = (
            summary[
                "std_delta_pct"
            ]
            .fillna(0.0)
        )

        factor_frames.append(
            summary
        )

    factor_summary = pd.concat(
        factor_frames,
        ignore_index=True,
    )

    for column in [
        "mean_delta_pct",
        "std_delta_pct",
        "min_delta_pct",
        "max_delta_pct",
    ]:
        factor_summary[column] = (
            factor_summary[column]
            .round(4)
        )

    factor_summary_path = (
        output_dir
        / "instance_characteristic_gvns_relative_summary.csv"
    )

    factor_summary.to_csv(
        factor_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[Saved] GVNS characteristic summary: "
        f"{factor_summary_path}"
    )

    # ----------------------------------------------------------------------
    # 6. tau / R / eta plots
    # ----------------------------------------------------------------------

    for factor in [
        "tau",
        "R",
        "eta",
    ]:
        factor_df = factor_summary[
            factor_summary["factor"]
            == factor
        ].copy()

        fig, ax = plt.subplots(
            figsize=(8.5, 5.2)
        )

        for comparator in comparators:
            comparator_df = (
                factor_df[
                    factor_df["comparator"]
                    == comparator
                ]
                .sort_values("level")
            )

            if comparator_df.empty:
                continue

            ax.plot(
                comparator_df["level"],
                comparator_df[
                    "mean_delta_pct"
                ],
                marker="o",
                linewidth=1.8,
                label=f"vs {comparator}",
            )

        ax.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )

        ax.set_xlabel(
            factor
        )

        ax.set_ylabel(
            "Mean Delta vs comparator (%)"
        )

        ax.set_title(
            "GVNS relative performance by "
            f"{factor}"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.legend(
            title="Comparison"
        )

        # Explicitly explain the sign on the graph.
        ax.text(
            0.01,
            0.02,
            "Negative Delta: GVNS better    |    Positive Delta: comparator better",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

        path = (
            output_dir
            / f"gvns_relative_by_{factor}.png"
        )

        _save_figure(
            fig,
            path,
        )

    return paper_table, factor_summary


# ============================================================================
# Analysis 4:
# Convergence analysis
# ============================================================================

def _load_convergence(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"convergence.csv not found: {path}"
        )

    df = pd.read_csv(path)

    missing = (
        REQUIRED_CONVERGENCE_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "convergence.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    for column in [
        "tau",
        "R",
        "eta",
        "seed",
        "elapsed_sec",
        "best_twt",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["algorithm"] = (
        df["algorithm"].astype(str)
    )
    df["instance"] = (
        df["instance"].astype(str)
    )

    convergence_key = [
        "algorithm",
        "instance",
        "seed",
        "elapsed_sec",
    ]

    duplicated = df.duplicated(
        convergence_key,
        keep=False,
    )

    if duplicated.any():
        n_rows = int(
            duplicated.sum()
        )

        print(
            "[Warning] Duplicate convergence rows found: "
            f"{n_rows}. "
            "The last row for each "
            "(algorithm, instance, seed, elapsed_sec) "
            "will be used."
        )

        df = df.drop_duplicates(
            convergence_key,
            keep="last",
        )

    df = _add_class_id(df)

    df["instance_no"] = (
        df["instance"]
        .map(_instance_number)
    )

    return df


def make_convergence_analysis(
    convergence: pd.DataFrame,
    baselines: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Analyze how quickly each algorithm finds good solutions.

    Aggregation:
        1) Average seeds within each algorithm-instance-time.
        2) Average across instances.

    Output:
        A. Mean raw TWT vs time
        B. Mean relative TWT vs ATCS vs time

    Relative TWT:
        best_twt / ATCS_twt * 100

        100% = ATCS
        <100% = better than ATCS
        lower = better
    """
    merged = convergence.merge(
        baselines,
        on="instance",
        how="left",
        validate="many_to_one",
    )

    if merged["ATCS_twt"].isna().any():
        missing = sorted(
            merged.loc[
                merged["ATCS_twt"].isna(),
                "instance",
            ].unique()
        )

        raise ValueError(
            "Missing ATCS baseline for convergence "
            f"instances: {missing}"
        )

    merged["relative_twt_pct"] = (
        merged["best_twt"]
        / merged["ATCS_twt"]
        * 100.0
    )

    merged.loc[
        merged["ATCS_twt"] == 0,
        "relative_twt_pct",
    ] = 100.0

    merged[
        "improvement_vs_ATCS_pct"
    ] = (
        100.0
        - merged["relative_twt_pct"]
    )

    # 1. Average seeds within each instance.
    instance_time = (
        merged.groupby(
            [
                "algorithm",
                "class_id",
                "tau",
                "R",
                "eta",
                "instance",
                "instance_no",
                "elapsed_sec",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            mean_twt_across_seeds=(
                "best_twt",
                "mean",
            ),
            mean_relative_twt_pct=(
                "relative_twt_pct",
                "mean",
            ),
            mean_improvement_pct=(
                "improvement_vs_ATCS_pct",
                "mean",
            ),
        )
    )

    instance_time_path = (
        output_dir
        / "convergence_instance_level.csv"
    )

    instance_time.to_csv(
        instance_time_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[Saved] Instance-level convergence: "
        f"{instance_time_path}"
    )

    # 2. Average across instances.
    summary = (
        instance_time.groupby(
            [
                "algorithm",
                "elapsed_sec",
            ]
        )
        .agg(
            n_instances=(
                "instance",
                "nunique",
            ),
            mean_twt=(
                "mean_twt_across_seeds",
                "mean",
            ),
            std_twt=(
                "mean_twt_across_seeds",
                "std",
            ),
            mean_relative_twt_pct=(
                "mean_relative_twt_pct",
                "mean",
            ),
            std_relative_twt_pct=(
                "mean_relative_twt_pct",
                "std",
            ),
            mean_improvement_vs_ATCS_pct=(
                "mean_improvement_pct",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["std_twt"] = (
        summary["std_twt"]
        .fillna(0.0)
    )

    summary[
        "std_relative_twt_pct"
    ] = (
        summary[
            "std_relative_twt_pct"
        ]
        .fillna(0.0)
    )

    for column in [
        "mean_twt",
        "std_twt",
        "mean_relative_twt_pct",
        "std_relative_twt_pct",
        "mean_improvement_vs_ATCS_pct",
    ]:
        summary[column] = (
            summary[column].round(4)
        )

    summary_path = (
        output_dir
        / "convergence_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[Saved] Convergence summary: {summary_path}"
    )

    algorithms = _ordered_algorithms(
        instance_time
    )

    # Plot A: raw TWT vs time
    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    for algorithm in algorithms:
        algorithm_df = (
            summary[
                summary["algorithm"]
                == algorithm
            ]
            .sort_values("elapsed_sec")
        )

        if algorithm_df.empty:
            continue

        ax.plot(
            algorithm_df["elapsed_sec"],
            algorithm_df["mean_twt"],
            marker="o",
            linewidth=1.8,
            label=algorithm,
        )

    ax.set_xlabel(
        "Elapsed time (sec)"
    )

    ax.set_ylabel(
        "Mean best TWT"
    )

    ax.set_title(
        "Convergence: TWT vs Time"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        title="Algorithm"
    )

    raw_twt_path = (
        output_dir
        / "convergence_mean_twt.png"
    )

    _save_figure(
        fig,
        raw_twt_path,
    )

    # Plot B: normalized TWT vs time
    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    for algorithm in algorithms:
        algorithm_df = (
            summary[
                summary["algorithm"]
                == algorithm
            ]
            .sort_values("elapsed_sec")
        )

        if algorithm_df.empty:
            continue

        ax.plot(
            algorithm_df["elapsed_sec"],
            algorithm_df[
                "mean_relative_twt_pct"
            ],
            marker="o",
            linewidth=1.8,
            label=algorithm,
        )

    ax.axhline(
        100.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        label="ATCS baseline",
    )

    ax.set_xlabel(
        "Elapsed time (sec)"
    )

    ax.set_ylabel(
        "Mean relative TWT (% of ATCS)"
    )

    ax.set_title(
        "Convergence: Relative TWT vs Time"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        title="Algorithm"
    )

    relative_twt_path = (
        output_dir
        / "convergence_relative_twt.png"
    )

    _save_figure(
        fig,
        relative_twt_path,
    )

    return summary


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SMTWT-SDST experiment results: "
            "instance-level comparison, stochastic robustness, "
            "GVNS-centered instance characteristics, "
            "and convergence."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/results_raw.csv"
        ),
        help=(
            "Input results_raw.csv "
            "(default: results/results_raw.csv)"
        ),
    )

    parser.add_argument(
        "--convergence",
        type=Path,
        default=Path(
            "results/convergence.csv"
        ),
        help=(
            "Input convergence.csv "
            "(default: results/convergence.csv)"
        ),
    )

    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path(
            "results/baseline_results.csv"
        ),
        help=(
            "Input baseline_results.csv "
            "(default: results/baseline_results.csv)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/analysis"
        ),
        help=(
            "Directory for tables and figures "
            "(default: results/analysis)"
        ),
    )

    parser.add_argument(
        "--no-std",
        action="store_true",
        help=(
            "Do not draw mean +/- SD error bars "
            "in the robustness figure."
        ),
    )

    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help=(
            "Skip convergence.csv analysis."
        ),
    )

    parser.add_argument(
        "--mean-table-csv",
        type=Path,
        help="Add baseline columns to an existing mean table and save CSV and PNG.",
    )

    args = parser.parse_args()

    if args.mean_table_csv is not None:
        table = _add_baseline_columns(pd.read_csv(args.mean_table_csv), args.baselines)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table_path = args.output_dir / "instance_algorithm_mean_table.csv"
        table.to_csv(table_path, index=False, encoding="utf-8-sig")
        save_mean_table_image(
            table_path,
            args.output_dir / "instance_algorithm_mean_table.png",
        )
        return

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Load final result data
    # ----------------------------------------------------------------------

    results = _load_results(
        args.input
    )

    algorithms = _ordered_algorithms(
        results
    )

    print()
    print("=" * 82)
    print("SMTWT-SDST RESULT ANALYSIS")
    print("=" * 82)
    print(f"Results input : {args.input}")
    print(
        "Algorithms   : "
        + ", ".join(algorithms)
    )
    print(
        f"Instances    : "
        f"{results['instance'].nunique()}"
    )
    print(
        f"Classes      : "
        f"{results['class_id'].nunique()}"
    )
    print()
    print(
        "GVNS characteristic-analysis delta:"
    )
    print(
        "  Delta(A) = "
        "(mean TWT_GVNS - mean TWT_A) "
        "/ mean TWT_A x 100"
    )
    print(
        "  Delta < 0 : GVNS better"
    )
    print(
        "  Delta > 0 : comparator better"
    )
    print("=" * 82)
    print()

    # ----------------------------------------------------------------------
    # 1. Instance x algorithm mean TWT table
    # ----------------------------------------------------------------------

    make_mean_table(
        results,
        args.output_dir,
        args.baselines,
    )

    # ----------------------------------------------------------------------
    # 2. Robustness across stochastic seeds
    # ----------------------------------------------------------------------

    instance_stats = make_instance_stats(
        results,
        args.output_dir,
    )

    plot_algorithm_robustness(
        instance_stats,
        args.output_dir,
        show_std=not args.no_std,
    )

    # ----------------------------------------------------------------------
    # 3. GVNS-centered instance-characteristic analysis
    # ----------------------------------------------------------------------

    make_gvns_relative_instance_analysis(
        results,
        args.output_dir,
    )

    # ----------------------------------------------------------------------
    # 4. Convergence analysis
    # ----------------------------------------------------------------------

    if args.skip_convergence:
        print(
            "[Skipped] Convergence analysis "
            "because --skip-convergence was used."
        )

    elif not args.convergence.exists():
        print(
            "[Warning] convergence.csv was not found. "
            "Convergence analysis was skipped."
        )

    elif not args.baselines.exists():
        print(
            "[Warning] baseline_results.csv was not found. "
            "Normalized convergence requires ATCS values, "
            "so convergence analysis was skipped."
        )

    else:
        baselines = _load_baselines(
            args.baselines
        )

        convergence = _load_convergence(
            args.convergence
        )

        make_convergence_analysis(
            convergence,
            baselines,
            args.output_dir,
        )

    print()
    print("=" * 82)
    print("ANALYSIS COMPLETED")
    print("=" * 82)
    print(
        "Main outputs:\n"
        "\n"
        "[Instance / robustness]\n"
        "  - instance_algorithm_mean_table.csv\n"
        "  - instance_seed_statistics.csv\n"
        "  - seed_count_check.csv\n"
        "  - robustness_<ALGORITHM>.png\n"
        "\n"
        "[GVNS-centered instance characteristics]\n"
        "  - gvns_relative_instance_performance.csv\n"
        "  - class_gvns_relative_performance_table.csv\n"
        "  - class_gvns_relative_performance_table.png\n"
        "  - instance_characteristic_gvns_relative_summary.csv\n"
        "  - gvns_relative_by_tau.png\n"
        "  - gvns_relative_by_R.png\n"
        "  - gvns_relative_by_eta.png\n"
        "\n"
        "[Convergence]\n"
        "  - convergence_instance_level.csv\n"
        "  - convergence_summary.csv\n"
        "  - convergence_mean_twt.png\n"
        "  - convergence_relative_twt.png"
    )


if __name__ == "__main__":
    main()
