#!/usr/bin/env python3
"""Build RFCA zarr finetune manifest from metadata_full + ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANCHOR_WINDOW_DAYS = 90
TRAIN_WINDOW_DAYS = 180
SPLIT_SEED = 0
SPLIT_MAP = {0: "train", 1: "train", 2: "train", 3: "valid", 4: "test"}

PATIENT_ID_REGEX = r"(?is)<PatientID>\s*([^<\s]+)\s*</PatientID>"

PID_COLUMN = "PID"
INDEX_FILE_COLUMN = "FILE_NAME"
INDEX_ACQ_DATE_COLUMN = "ACQUISITION_DATE"
INDEX_ACQ_TIME_COLUMN = "ACQUISITION_TIME"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-parquet",
        default="/data/ecg/zarr/rfca.log/metadata_full.parquet",
        help="Path to RFCA ingest metadata_full.parquet.",
    )
    parser.add_argument(
        "--ablation-excel",
        default="/data/projects/af_ablation/test/assets/V_ABLATION(2025-11-30).csv",
        help="Path to RFCA ablation table (xlsx/xls/csv/tsv/parquet).",
    )
    parser.add_argument("--meta-id-column", default="FileID", help="Sample key column in metadata parquet.")
    parser.add_argument(
        "--meta-pid-column",
        default=None,
        help="Optional PID column in metadata parquet. If omitted, PID is parsed from --meta-raw-text-column XML.",
    )
    parser.add_argument(
        "--meta-raw-text-column",
        default="hea_raw_text",
        help="Metadata XML/raw-text column used to parse PatientID when --meta-pid-column is not provided.",
    )
    parser.add_argument(
        "--meta-acq-date-column",
        default="hea_base_date",
        help="Acquisition date column in metadata parquet.",
    )
    parser.add_argument(
        "--meta-acq-time-column",
        default="hea_base_time",
        help="Acquisition time column in metadata parquet.",
    )
    parser.add_argument("--sample-id-column", default="FileID", help="Preferred sample id column in joined rows.")
    parser.add_argument("--crf-pid-column", default="No", help="PID column name in ablation excel.")
    parser.add_argument("--crf-procedure-date-column", default="DateofProcedure", help="Procedure date column in ablation excel.")
    parser.add_argument("--crf-label-column", default="LVR05_TotalLB", help="Continuous target column in ablation excel.")
    parser.add_argument("--label-column", default="LVR05_high", help="Output binary label column name.")
    parser.add_argument("--label-threshold", type=float, default=10.0, help="Threshold for positive label.")
    parser.add_argument(
        "--label-comparison",
        default="ge",
        choices=["ge", "gt"],
        help="Label comparison for positive label: ge (>=) or gt (>).",
    )
    parser.add_argument("--zarr-store", default=None, help="Override zarr_store in output manifest.")
    parser.add_argument(
        "--default-sample-rate",
        type=int,
        default=500,
        help="Fallback sample rate when metadata sample_rate is missing/invalid.",
    )
    parser.add_argument(
        "--output",
        default="/data/projects/study-af-ablation/manifests/finetune_lvr05_high_rfca.parquet",
        help="Output parquet path.",
    )
    return parser.parse_args()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format: {path}")


def _load_ablation_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Excel requires openpyxl in the current environment. "
                "Install openpyxl, or convert the file to csv/tsv/parquet and pass that path."
            ) from exc
    return _load_table(path)


def _extract_pid_from_xml(series: pd.Series) -> pd.Series:
    pid = series.astype("string").fillna("").str.extract(PATIENT_ID_REGEX, expand=False)
    pid = _normalize_pid(pid)
    return pid


def _normalize_pid(series: pd.Series) -> pd.Series:
    out = series.astype("string").fillna("").str.strip()
    # CSV/Excel numeric coercion can produce ids like "47435504.0".
    out = out.str.replace(r"\.0+$", "", regex=True)
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})
    return out


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _combine_datetime(date_series: pd.Series, time_series: pd.Series | None) -> pd.Series:
    date_ts = pd.to_datetime(date_series, errors="coerce")
    if time_series is None:
        return date_ts
    time_str = time_series.astype("string").fillna("00:00:00")
    time_str = time_str.where(time_str.str.len() > 0, "00:00:00")
    date_part = date_ts.dt.strftime("%Y-%m-%d")
    full = date_part + " " + time_str.astype(str)
    full = full.where(date_part.notna(), pd.NA)
    return pd.to_datetime(full, errors="coerce")


def _resolve_zarr_store(meta: pd.DataFrame, override: str | None) -> str:
    if override:
        return str(Path(override))
    if "zarr_store" not in meta.columns:
        raise ValueError("metadata parquet is missing zarr_store; pass --zarr-store.")
    uniq = meta["zarr_store"].dropna().astype(str).str.strip()
    uniq = uniq[uniq.ne("")].unique()
    if len(uniq) != 1:
        raise ValueError(f"metadata zarr_store must be unique, got {uniq[:5]}")
    return str(uniq[0])


def _assign_pid_split(cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty:
        return cohort
    work = cohort.copy()
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(work))
    work["_shuffle"] = shuffled
    work["split"] = pd.Series(shuffled, index=work.index).mod(5).map(SPLIT_MAP).astype(str)
    return work.drop(columns=["_shuffle"])


def _build_index_from_meta(meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    required = {
        args.meta_id_column,
        "ds_idx",
        "ds_row_idx",
        args.meta_acq_date_column,
    }
    if args.meta_acq_time_column:
        required.add(args.meta_acq_time_column)
    if args.meta_pid_column:
        required.add(args.meta_pid_column)
    else:
        required.add(args.meta_raw_text_column)

    missing = sorted(c for c in required if c not in meta.columns)
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")

    work = meta.copy()
    if args.meta_pid_column:
        pid = _normalize_pid(work[args.meta_pid_column])
    else:
        pid = _extract_pid_from_xml(work[args.meta_raw_text_column])

    work[PID_COLUMN] = pid
    work[INDEX_FILE_COLUMN] = work[args.meta_id_column].astype("string").fillna("").str.strip()
    work[INDEX_FILE_COLUMN] = work[INDEX_FILE_COLUMN].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})
    work[INDEX_ACQ_DATE_COLUMN] = work[args.meta_acq_date_column]
    if args.meta_acq_time_column in work.columns:
        work[INDEX_ACQ_TIME_COLUMN] = work[args.meta_acq_time_column]
    else:
        work[INDEX_ACQ_TIME_COLUMN] = "00:00:00"

    base_cols = [
        PID_COLUMN,
        INDEX_FILE_COLUMN,
        INDEX_ACQ_DATE_COLUMN,
        INDEX_ACQ_TIME_COLUMN,
        args.meta_id_column,
        "ds_idx",
        "ds_row_idx",
    ]
    for optional in ("sample_rate", "zarr_group", "zarr_store"):
        if optional in work.columns:
            base_cols.append(optional)
    if args.sample_id_column in work.columns and args.sample_id_column not in base_cols:
        base_cols.append(args.sample_id_column)

    out = work[base_cols].copy()
    before = len(out)
    out = out[out[PID_COLUMN].notna() & out[INDEX_FILE_COLUMN].notna()].copy()
    dropped = int(before - len(out))
    if out.empty:
        raise ValueError("No metadata rows left after PID/FileID extraction.")

    dup_file = int(out[INDEX_FILE_COLUMN].duplicated(keep=False).sum())
    if dup_file > 0:
        raise ValueError(f"metadata has duplicate {INDEX_FILE_COLUMN} values: {dup_file} rows")
    return out, dropped


def _pick_anchor_rows(
    *,
    index_df: pd.DataFrame,
    crf_df: pd.DataFrame,
    pid_col: str,
    acq_date_col: str,
    acq_time_col: str,
    crf_pid_col: str,
    proc_date_col: str,
    crf_label_col: str,
    label_col: str,
    label_threshold: float,
    label_comparison: str,
) -> pd.DataFrame:
    work_index = index_df.copy()
    work_index["_acq_date"] = _to_date(work_index[acq_date_col])
    work_index["_acq_dt"] = _combine_datetime(work_index[acq_date_col], work_index[acq_time_col] if acq_time_col in work_index.columns else None)

    work_crf = crf_df[[crf_pid_col, proc_date_col, crf_label_col]].copy()
    work_crf[crf_pid_col] = work_crf[crf_pid_col].astype("string").fillna("").str.strip()
    work_crf[proc_date_col] = _to_date(work_crf[proc_date_col])
    work_crf[crf_label_col] = pd.to_numeric(work_crf[crf_label_col], errors="coerce")

    joined = work_index.merge(work_crf, left_on=pid_col, right_on=crf_pid_col, how="inner")
    joined["date_diff"] = joined["_acq_date"] - joined[proc_date_col]

    anchor_mask = joined["date_diff"].ge(pd.Timedelta(days=-ANCHOR_WINDOW_DAYS)) & joined["date_diff"].lt(pd.Timedelta(days=0))
    anchor_mask &= joined[crf_label_col].notna()
    joined = joined[anchor_mask].copy()
    if joined.empty:
        raise ValueError("No rows left after anchor-window filtering. Check source tables and date columns.")

    # Keep the closest pre-procedure ECG for each PID + procedure date.
    joined = joined.sort_values(
        by=[pid_col, proc_date_col, "date_diff", "_acq_dt"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    by_proc = joined.groupby([pid_col, proc_date_col], as_index=False, sort=False).tail(1)

    # Keep earliest procedure per PID.
    by_proc = by_proc.sort_values(by=[pid_col, proc_date_col], ascending=[True, True], kind="mergesort")
    cohort = by_proc.groupby(pid_col, as_index=False, sort=False).head(1).copy()
    if label_comparison == "gt":
        cohort[label_col] = cohort[crf_label_col].gt(label_threshold)
    else:
        cohort[label_col] = cohort[crf_label_col].ge(label_threshold)
    cohort = _assign_pid_split(cohort)
    return cohort


def _dedup_single_ecg_per_pid(
    df: pd.DataFrame,
    *,
    split: str,
    pid_col: str,
    proc_date_col: str,
    acq_date_col: str,
    acq_time_col: str,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    if pid_col not in df.columns:
        raise ValueError(f"{split} split requires pid column for dedup: {pid_col}")

    work = df.copy()
    proc_dt = _to_date(work[proc_date_col]) if proc_date_col in work.columns else pd.Series(pd.NaT, index=work.index)
    acq_dt = _combine_datetime(work[acq_date_col], work[acq_time_col] if acq_time_col in work.columns else None)

    work["_dt_diff_sec"] = (proc_dt - acq_dt).dt.total_seconds().abs().fillna(np.inf)
    work["_acq_dt_sort"] = acq_dt.fillna(pd.Timestamp.min)
    before = int((work[pid_col].value_counts() > 1).sum())
    dedup = (
        work.sort_values(by=[pid_col, "_dt_diff_sec", "_acq_dt_sort"], ascending=[True, True, False], kind="mergesort")
        .drop_duplicates(subset=[pid_col], keep="first")
        .drop(columns=["_dt_diff_sec", "_acq_dt_sort"], errors="ignore")
    )
    dedup = dedup.sort_index()
    return dedup, before


def _build_base_rows(
    *,
    index_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    pid_col: str,
    acq_date_col: str,
    acq_time_col: str,
    proc_date_col: str,
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    idx = index_df.copy()
    idx["_acq_date"] = _to_date(idx[acq_date_col])
    idx["_acq_dt"] = _combine_datetime(idx[acq_date_col], idx[acq_time_col] if acq_time_col in idx.columns else None)

    train_anchor = cohort_df[cohort_df["split"] == "train"][[pid_col, proc_date_col, label_col, "split"]].copy()
    valid_anchor = cohort_df[cohort_df["split"] == "valid"][[pid_col, acq_date_col, proc_date_col, label_col, "split"]].copy()
    test_anchor = cohort_df[cohort_df["split"] == "test"][[pid_col, acq_date_col, proc_date_col, label_col, "split"]].copy()

    train = train_anchor.merge(idx, on=pid_col, how="inner", suffixes=("_anchor", ""))
    train["date_diff"] = train["_acq_date"] - _to_date(train[proc_date_col])
    train = train[train["date_diff"].ge(pd.Timedelta(days=-TRAIN_WINDOW_DAYS)) & train["date_diff"].lt(pd.Timedelta(days=0))].copy()
    train["split"] = "train"

    valid = valid_anchor.merge(idx, on=[pid_col, acq_date_col], how="inner", suffixes=("_anchor", ""))
    valid["split"] = "valid"
    valid, valid_dup_groups = _dedup_single_ecg_per_pid(
        valid,
        split="valid",
        pid_col=pid_col,
        proc_date_col=proc_date_col,
        acq_date_col=acq_date_col,
        acq_time_col=acq_time_col,
    )

    test = test_anchor.merge(idx, on=[pid_col, acq_date_col], how="inner", suffixes=("_anchor", ""))
    test["split"] = "test"
    test, test_dup_groups = _dedup_single_ecg_per_pid(
        test,
        split="test",
        pid_col=pid_col,
        proc_date_col=proc_date_col,
        acq_date_col=acq_date_col,
        acq_time_col=acq_time_col,
    )

    out = pd.concat([train, valid, test], axis=0, ignore_index=True)
    if out.empty:
        raise ValueError("No rows left after split expansion.")

    split_rows = {
        "train": int((out["split"] == "train").sum()),
        "valid": int((out["split"] == "valid").sum()),
        "test": int((out["split"] == "test").sum()),
        "valid_dup_pid_groups_removed": int(valid_dup_groups),
        "test_dup_pid_groups_removed": int(test_dup_groups),
    }
    return out, split_rows


def _coerce_sample_id(merged: pd.DataFrame, preferred: str, fallback_cols: list[str]) -> pd.Series:
    choices = [preferred] + fallback_cols
    for col in choices:
        if col in merged.columns:
            s = merged[col].astype("string").fillna("").str.strip()
            s = s.where(s.str.len() > 0, pd.NA)
            if s.notna().any():
                return s.fillna("").astype(str)
    return pd.Series([f"row_{i}" for i in range(len(merged))], index=merged.index, dtype="string")


def main() -> int:
    args = parse_args()

    meta_path = Path(args.meta_parquet)
    crf_path = Path(args.ablation_excel)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")
    if not crf_path.exists():
        raise FileNotFoundError(f"Ablation file not found: {crf_path}")

    meta = pd.read_parquet(meta_path)
    index_df, dropped_meta_rows = _build_index_from_meta(meta, args)
    crf_df = _load_ablation_table(crf_path)

    index_required = {PID_COLUMN, INDEX_FILE_COLUMN, INDEX_ACQ_DATE_COLUMN}
    crf_required = {args.crf_pid_column, args.crf_procedure_date_column, args.crf_label_column}
    for name, df, required in (
        ("metadata_index", index_df, index_required),
        ("ablation", crf_df, crf_required),
    ):
        missing = sorted(c for c in required if c not in df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    index_df = index_df.copy()
    index_df[PID_COLUMN] = _normalize_pid(index_df[PID_COLUMN])
    crf_df = crf_df.copy()
    crf_df[args.crf_pid_column] = _normalize_pid(crf_df[args.crf_pid_column])

    cohort = _pick_anchor_rows(
        index_df=index_df,
        crf_df=crf_df,
        pid_col=PID_COLUMN,
        acq_date_col=INDEX_ACQ_DATE_COLUMN,
        acq_time_col=INDEX_ACQ_TIME_COLUMN,
        crf_pid_col=args.crf_pid_column,
        proc_date_col=args.crf_procedure_date_column,
        crf_label_col=args.crf_label_column,
        label_col=args.label_column,
        label_threshold=args.label_threshold,
        label_comparison=args.label_comparison,
    )

    base_rows, split_rows = _build_base_rows(
        index_df=index_df,
        cohort_df=cohort,
        pid_col=PID_COLUMN,
        acq_date_col=INDEX_ACQ_DATE_COLUMN,
        acq_time_col=INDEX_ACQ_TIME_COLUMN,
        proc_date_col=args.crf_procedure_date_column,
        label_col=args.label_column,
    )

    merged = base_rows.copy()

    if "sample_rate" in merged.columns:
        sample_rate = pd.to_numeric(merged["sample_rate"], errors="coerce").fillna(float(args.default_sample_rate)).astype("int64")
    else:
        sample_rate = pd.Series(np.full(len(merged), args.default_sample_rate, dtype=np.int64), index=merged.index)

    zarr_store = _resolve_zarr_store(meta, args.zarr_store)
    if "zarr_group" in merged.columns:
        zarr_group = merged["zarr_group"].fillna("").astype(str)
    else:
        zarr_group = pd.Series([""] * len(merged), index=merged.index, dtype="string")

    sample_id = _coerce_sample_id(
        merged,
        preferred=args.sample_id_column,
        fallback_cols=[args.meta_id_column, INDEX_FILE_COLUMN],
    )
    file_name = sample_id.astype("string").str.strip()
    file_name = file_name.where(file_name.str.lower().str.endswith(".xml"), file_name + ".xml")

    manifest = pd.DataFrame(
        {
            "split": merged["split"].astype(str).to_numpy(),
            "sample_id": sample_id.astype(str).to_numpy(),
            "file_name": file_name.astype(str).to_numpy(),
            "sample_rate": sample_rate.to_numpy(),
            args.label_column: pd.to_numeric(merged[args.label_column], errors="coerce").astype(np.float32).to_numpy(),
            "zarr_store": zarr_store,
            "zarr_group": zarr_group.to_numpy(),
            "ds_idx": pd.to_numeric(merged["ds_idx"], errors="raise").astype("int64").to_numpy(),
            "ds_row_idx": pd.to_numeric(merged["ds_row_idx"], errors="raise").astype("int64").to_numpy(),
            "pid": merged[PID_COLUMN].astype("string").fillna("").astype(str).to_numpy(),
        }
    )
    manifest = manifest[manifest[args.label_column].notna()].copy()
    if manifest.empty:
        raise ValueError("Manifest is empty after label normalization.")

    for split in ["valid", "test"]:
        part = manifest[manifest["split"] == split]
        if part.empty:
            raise ValueError(f"{split} split is empty after merge.")
        dup_groups = int((part["pid"].value_counts() > 1).sum())
        if dup_groups != 0:
            raise ValueError(f"{split} split still has duplicate pid rows: duplicate_pid_groups={dup_groups}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out, index=False)

    split_counts = manifest["split"].value_counts().to_dict()
    pos_rate = manifest.groupby("split")[args.label_column].mean().to_dict()
    pid_stats = {
        split: {
            "rows": int(len(part)),
            "unique_pid": int(part["pid"].nunique()),
            "max_ecg_per_pid": int(part["pid"].value_counts().max()),
            "duplicate_pid_groups": int((part["pid"].value_counts() > 1).sum()),
        }
        for split in ["train", "valid", "test"]
        for part in [manifest[manifest["split"] == split]]
        if not part.empty
    }

    print(f"[ok] wrote rfca finetune manifest: {out}")
    print(f"[ok] rows={len(manifest)} split_counts={split_counts} positive_rate={pos_rate}")
    print(f"[ok] anchor_window_days={ANCHOR_WINDOW_DAYS} train_window_days={TRAIN_WINDOW_DAYS} split_seed={SPLIT_SEED}")
    print(f"[ok] metadata_rows={len(meta)} metadata_rows_dropped={dropped_meta_rows}")
    print(f"[ok] anchor_rows={len(cohort)} expanded_rows={len(base_rows)} joined_rows={len(merged)}")
    print(f"[ok] expanded_split_rows={split_rows}")
    print(f"[ok] pid_stats={pid_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
