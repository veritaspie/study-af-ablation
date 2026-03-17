#!/usr/bin/env python3
"""Generate markdown statistics report for RFCA zarr finetune manifest."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/data/projects/study-af-ablation/manifests/finetune_lvr05_high_rfca.parquet",
        help="Path to RFCA zarr finetune manifest parquet.",
    )
    parser.add_argument(
        "--metadata-full",
        default="/data/ecg/zarr/rfca.log/metadata_full.parquet",
        help="Path to RFCA ingest metadata_full.parquet.",
    )
    parser.add_argument(
        "--ablation-table",
        default="/data/projects/af_ablation/test/assets/V_ABLATION(2025-11-30).csv",
        help="Path to RFCA ablation table (xlsx/xls/csv/tsv/parquet).",
    )
    parser.add_argument(
        "--builder-script",
        default=None,
        help="Optional path to build_finetune_manifest_rfca_zarr.py. Defaults to sibling script.",
    )
    parser.add_argument(
        "--output",
        default="/data/projects/study-af-ablation/data/rfca_zarr_manifest_stats.md",
        help="Output markdown path.",
    )
    parser.add_argument("--label-column", default="LVR05_high", help="Binary label column in manifest.")
    parser.add_argument("--crf-pid-column", default="No", help="PID column in ablation table.")
    parser.add_argument("--crf-date-column", default="DateofProcedure", help="Procedure date column in ablation table.")
    parser.add_argument("--crf-label-column", default="LVR05_TotalLB", help="Continuous target column in ablation table.")
    return parser.parse_args()


def _load_builder_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("rfca_manifest_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load builder module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalize_sex(series: pd.Series) -> pd.Series:
    sex = series.astype("string").str.upper()
    sex = sex.replace({"M": "MALE", "F": "FEMALE"})
    sex = sex.fillna("UNKNOWN").replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
    return sex.astype(str)


def _extract_age_sex_from_xml(xml_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    xml = xml_series.astype("string").fillna("")
    age_raw = xml.str.extract(r"(?is)<PatientAge>\s*([^<\s]+)\s*</PatientAge>", expand=False)
    age_units = (
        xml.str.extract(r"(?is)<AgeUnits>\s*([^<\s]+)\s*</AgeUnits>", expand=False)
        .astype("string")
        .str.upper()
        .fillna("YEARS")
    )
    sex = xml.str.extract(r"(?is)<Gender>\s*([^<\s]+)\s*</Gender>", expand=False)
    sex = _normalize_sex(sex)

    age_val = pd.to_numeric(age_raw, errors="coerce")
    unit_scale = age_units.map(
        {
            "YEARS": 1.0,
            "MONTHS": 1.0 / 12.0,
            "WEEKS": 1.0 / 52.1775,
            "DAYS": 1.0 / 365.25,
        }
    ).fillna(1.0)
    age_years = age_val * unit_scale
    return age_years.astype(float), sex


def _first_non_na(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[0]) if not s.empty else float("nan")


def _first_non_unknown(series: pd.Series) -> str:
    for value in series.astype("string").fillna("").astype(str):
        v = value.strip().upper()
        if v and v not in {"UNKNOWN", "NAN", "NONE"}:
            return v
    return "UNKNOWN"


def _summary_numeric(df: pd.DataFrame, group_col: str, value_col: str, group_order: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = group_order if group_order is not None else sorted(df[group_col].dropna().astype(str).unique().tolist())
    for group in groups:
        part = df[df[group_col].astype(str) == str(group)]
        s = pd.to_numeric(part[value_col], errors="coerce").dropna()
        if s.empty:
            rows.append(
                {
                    group_col: group,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                }
            )
            continue
        rows.append(
            {
                group_col: group,
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA"
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return str(value)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(_format_value(row[c]) for c in df.columns) + " |")
    return "\n".join([header, sep, *body])


def _add_percent_column(df: pd.DataFrame, count_col: str = "count", out_col: str = "pct") -> pd.DataFrame:
    out = df.copy()
    total = float(out[count_col].sum())
    out[out_col] = (out[count_col] / total * 100.0) if total > 0 else 0.0
    return out


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    metadata_path = Path(args.metadata_full).expanduser().resolve()
    ablation_path = Path(args.ablation_table).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    builder_path = (
        Path(args.builder_script).expanduser().resolve()
        if args.builder_script
        else Path(__file__).resolve().parent / "build_finetune_manifest_rfca_zarr.py"
    )

    for p, name in [
        (manifest_path, "manifest"),
        (metadata_path, "metadata_full"),
        (ablation_path, "ablation_table"),
        (builder_path, "builder_script"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    builder = _load_builder_module(builder_path)

    manifest = pd.read_parquet(manifest_path)
    if args.label_column not in manifest.columns:
        raise ValueError(f"Manifest missing label column: {args.label_column}")
    required_manifest_cols = {"split", "sample_id", "pid", "sample_rate", "ds_idx", args.label_column}
    missing_manifest = sorted(c for c in required_manifest_cols if c not in manifest.columns)
    if missing_manifest:
        raise ValueError(f"Manifest missing required columns: {missing_manifest}")

    # Build cohort-level continuous LVR05 labels using the same matching logic as manifest builder.
    meta_cols = [
        "FileID",
        "ds_idx",
        "ds_row_idx",
        "hea_base_date",
        "hea_base_time",
        "hea_raw_text",
        "sample_rate",
        "zarr_group",
        "zarr_store",
    ]
    # Read only needed columns directly; pyarrow will ignore unrequested columns.
    metadata_full = pd.read_parquet(metadata_path, columns=meta_cols)
    build_args = type(
        "BuildArgs",
        (),
        {
            "meta_id_column": "FileID",
            "meta_pid_column": None,
            "meta_raw_text_column": "hea_raw_text",
            "meta_acq_date_column": "hea_base_date",
            "meta_acq_time_column": "hea_base_time",
            "sample_id_column": "FileID",
        },
    )
    index_df, _ = builder._build_index_from_meta(metadata_full, build_args)

    ablation_df = builder._load_ablation_table(ablation_path)
    for col in [args.crf_pid_column, args.crf_date_column, args.crf_label_column]:
        if col not in ablation_df.columns:
            raise ValueError(f"Ablation table missing required column: {col}")
    ablation_df = ablation_df.copy()
    ablation_df[args.crf_pid_column] = builder._normalize_pid(ablation_df[args.crf_pid_column])

    cohort = builder._pick_anchor_rows(
        index_df=index_df,
        crf_df=ablation_df,
        pid_col=builder.PID_COLUMN,
        acq_date_col=builder.INDEX_ACQ_DATE_COLUMN,
        acq_time_col=builder.INDEX_ACQ_TIME_COLUMN,
        crf_pid_col=args.crf_pid_column,
        proc_date_col=args.crf_date_column,
        crf_label_col=args.crf_label_column,
        label_col=args.label_column,
        label_threshold=10.0,
    )
    cohort_info = cohort[[builder.PID_COLUMN, args.crf_label_column, args.crf_date_column]].copy()
    cohort_info.rename(columns={builder.PID_COLUMN: "pid"}, inplace=True)
    cohort_info[args.crf_label_column] = pd.to_numeric(cohort_info[args.crf_label_column], errors="coerce")

    # Pull only manifest rows from metadata to extract age/sex.
    sample_ids = manifest["sample_id"].astype(str).tolist()
    demo_df = pd.read_parquet(
        metadata_path,
        columns=["FileID", "hea_raw_text"],
        filters=[("FileID", "in", sample_ids)],
    )
    age_years, sex = _extract_age_sex_from_xml(demo_df["hea_raw_text"])
    demo_df = pd.DataFrame(
        {
            "sample_id": demo_df["FileID"].astype(str),
            "age_years": age_years,
            "sex": sex,
        }
    )

    rows = manifest.merge(demo_df, on="sample_id", how="left")
    rows = rows.merge(cohort_info, on="pid", how="left")
    rows = rows.rename(columns={args.crf_label_column: "LVR05_TotalLB"})
    rows["LVR05_TotalLB"] = pd.to_numeric(rows["LVR05_TotalLB"], errors="coerce")
    rows["split"] = rows["split"].astype(str)
    rows["sex"] = _normalize_sex(rows["sex"])

    pid_df = rows.groupby("pid", as_index=False).agg(
        split=("split", "first"),
        n_ecg=("sample_id", "size"),
        LVR05_high=(args.label_column, "first"),
        LVR05_TotalLB=("LVR05_TotalLB", "first"),
        age_years=("age_years", _first_non_na),
        sex=("sex", _first_non_unknown),
    )

    split_order = [s for s in ["train", "valid", "test"] if s in rows["split"].unique().tolist()]

    split_summary = (
        rows.groupby("split", as_index=False)
        .agg(
            n_ecg=("sample_id", "size"),
            n_pid=("pid", "nunique"),
            positive_rate=(args.label_column, "mean"),
            lvr05_mean_ecg=("LVR05_TotalLB", "mean"),
            age_mean_ecg=("age_years", "mean"),
        )
        .sort_values(by="split", key=lambda s: s.map({k: i for i, k in enumerate(split_order)}))
    )
    overall_summary = pd.DataFrame(
        [
            {
                "split": "overall",
                "n_ecg": int(len(rows)),
                "n_pid": int(rows["pid"].nunique()),
                "positive_rate": float(pd.to_numeric(rows[args.label_column], errors="coerce").mean()),
                "lvr05_mean_ecg": float(rows["LVR05_TotalLB"].mean()),
                "age_mean_ecg": float(rows["age_years"].mean()),
            }
        ]
    )
    split_summary = pd.concat([split_summary, overall_summary], ignore_index=True)

    ecg_per_pid = _summary_numeric(
        pd.concat(
            [
                pid_df[["split", "n_ecg"]],
                pd.DataFrame({"split": ["overall"] * len(pid_df), "n_ecg": pid_df["n_ecg"]}),
            ],
            ignore_index=True,
        ),
        group_col="split",
        value_col="n_ecg",
        group_order=split_order + ["overall"],
    )

    lvr_split_ecg = _summary_numeric(
        pd.concat(
            [
                rows[["split", "LVR05_TotalLB"]],
                pd.DataFrame({"split": ["overall"] * len(rows), "LVR05_TotalLB": rows["LVR05_TotalLB"]}),
            ],
            ignore_index=True,
        ),
        group_col="split",
        value_col="LVR05_TotalLB",
        group_order=split_order + ["overall"],
    )

    lvr_split_pid = _summary_numeric(
        pd.concat(
            [
                pid_df[["split", "LVR05_TotalLB"]],
                pd.DataFrame({"split": ["overall"] * len(pid_df), "LVR05_TotalLB": pid_df["LVR05_TotalLB"]}),
            ],
            ignore_index=True,
        ),
        group_col="split",
        value_col="LVR05_TotalLB",
        group_order=split_order + ["overall"],
    )

    lvr_by_group = (
        rows.groupby(["split", args.label_column], as_index=False)
        .agg(
            count=("sample_id", "size"),
            lvr05_mean=("LVR05_TotalLB", "mean"),
            lvr05_median=("LVR05_TotalLB", "median"),
        )
        .sort_values(by=["split", args.label_column], key=lambda s: s.map({k: i for i, k in enumerate(split_order)}))
    )
    lvr_by_group[args.label_column] = lvr_by_group[args.label_column].astype(int)

    age_split_ecg = _summary_numeric(
        pd.concat(
            [
                rows[["split", "age_years"]],
                pd.DataFrame({"split": ["overall"] * len(rows), "age_years": rows["age_years"]}),
            ],
            ignore_index=True,
        ),
        group_col="split",
        value_col="age_years",
        group_order=split_order + ["overall"],
    )

    age_split_pid = _summary_numeric(
        pd.concat(
            [
                pid_df[["split", "age_years"]],
                pd.DataFrame({"split": ["overall"] * len(pid_df), "age_years": pid_df["age_years"]}),
            ],
            ignore_index=True,
        ),
        group_col="split",
        value_col="age_years",
        group_order=split_order + ["overall"],
    )

    age_bins = [0, 49, 59, 69, 79, np.inf]
    age_labels = ["<=49", "50-59", "60-69", "70-79", "80+"]
    pid_df["age_bin"] = pd.cut(pid_df["age_years"], bins=age_bins, labels=age_labels, include_lowest=True)
    age_bin_counts = pid_df["age_bin"].value_counts(dropna=False).rename_axis("age_bin").reset_index(name="count")
    age_bin_counts["age_bin"] = age_bin_counts["age_bin"].astype("string").fillna("missing")
    age_bin_counts = _add_percent_column(age_bin_counts, count_col="count", out_col="pct")

    sex_pid = pid_df["sex"].value_counts().rename_axis("sex").reset_index(name="count")
    sex_pid = _add_percent_column(sex_pid, count_col="count", out_col="pct")

    sex_split_pid = (
        pid_df.groupby(["split", "sex"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(by=["split", "sex"], key=lambda s: s.map({k: i for i, k in enumerate(split_order)}).fillna(999))
    )

    source_stats = pd.DataFrame(
        [
            {"metric": "sample_rate_count_500", "value": int((rows["sample_rate"] == 500).sum())},
            {"metric": "sample_rate_count_250", "value": int((rows["sample_rate"] == 250).sum())},
            {"metric": "sample_rate_unique", "value": int(rows["sample_rate"].nunique())},
            {"metric": "ds_idx_0_count", "value": int((rows["ds_idx"] == 0).sum())},
            {"metric": "ds_idx_1_count", "value": int((rows["ds_idx"] == 1).sum())},
            {"metric": "ds_idx_unique", "value": int(rows["ds_idx"].nunique())},
            {"metric": "zarr_store_unique", "value": int(rows["zarr_store"].nunique())},
            {"metric": "age_available_ecg", "value": int(rows["age_years"].notna().sum())},
            {"metric": "age_available_pid", "value": int(pid_df["age_years"].notna().sum())},
            {"metric": "lvr05_available_ecg", "value": int(rows["LVR05_TotalLB"].notna().sum())},
            {"metric": "lvr05_available_pid", "value": int(pid_df["LVR05_TotalLB"].notna().sum())},
        ]
    )

    split_map = {k: i for i, k in enumerate(split_order)}

    def _sort_split(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_ord"] = out["split"].map(split_map).fillna(999)
        out = out.sort_values(by="_ord").drop(columns="_ord")
        return out

    md_lines = [
        "# RFCA Zarr Manifest Statistics",
        "",
        f"- Generated at: `{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- Manifest: `{manifest_path}`",
        f"- Metadata: `{metadata_path}`",
        f"- Ablation table: `{ablation_path}`",
        "",
        "## 1) Split Snapshot",
        _to_markdown_table(split_summary),
        "",
        "## 2) ECG Per Patient",
        _to_markdown_table(_sort_split(ecg_per_pid)),
        "",
        "## 3) LVR05 TotalLB (ECG-level)",
        _to_markdown_table(_sort_split(lvr_split_ecg)),
        "",
        "## 4) LVR05 TotalLB (Patient-level)",
        _to_markdown_table(_sort_split(lvr_split_pid)),
        "",
        "## 5) LVR05 Group Means by Split",
        _to_markdown_table(_sort_split(lvr_by_group)),
        "",
        "## 6) Age Summary (ECG-level)",
        _to_markdown_table(_sort_split(age_split_ecg)),
        "",
        "## 7) Age Summary (Patient-level)",
        _to_markdown_table(_sort_split(age_split_pid)),
        "",
        "## 8) Age Bins (Patient-level)",
        _to_markdown_table(age_bin_counts),
        "",
        "## 9) Sex Distribution (Patient-level)",
        _to_markdown_table(sex_pid),
        "",
        "## 10) Sex Distribution by Split (Patient-level)",
        _to_markdown_table(_sort_split(sex_split_pid)),
        "",
        "## 11) Source/Pointer Coverage",
        _to_markdown_table(source_stats),
        "",
        "## Notes",
        "- `LVR05_TotalLB` is reconstructed with the same cohort matching logic used in `build_finetune_manifest_rfca_zarr.py` (anchor window + per-PID earliest procedure).",
        "- Train split contains repeated ECGs per PID by design; therefore ECG-level and patient-level summaries are both provided.",
        "- Age/Sex are parsed from `hea_raw_text` XML fields (`PatientAge`, `AgeUnits`, `Gender`).",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote stats report: {output_path}")
    print(f"[ok] rows={len(rows)} pid={rows['pid'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
