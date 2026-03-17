#!/usr/bin/env python3
"""Build af_ablation finetune manifest for ai-ecg."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="/data/projects/af_ablation/test", help="Directory with split pickle files.")
    parser.add_argument("--train-fname", default="train.pkl", help="Train split pickle filename.")
    parser.add_argument("--valid-fname", default="valid.pkl", help="Validation split pickle filename.")
    parser.add_argument("--test-fname", default="test.pkl", help="Test split pickle filename.")
    parser.add_argument("--label-column", default="LVR05_high", help="Target label column.")
    parser.add_argument("--path-column", default="FILE_PATH", help="Waveform pickle path column.")
    parser.add_argument("--pid-column", default="PID", help="Patient ID column.")
    parser.add_argument("--sample-rate-column", default="SAMPLE_RATE", help="Sample rate column.")
    parser.add_argument("--sample-id-column", default="FILE_NAME", help="Preferred sample id column.")
    parser.add_argument("--procedure-date-column", default="DateofProcedure", help="Procedure date column.")
    parser.add_argument("--acq-date-column", default="ACQUISITION_DATE", help="ECG acquisition date column.")
    parser.add_argument("--acq-time-column", default="ACQUISITION_TIME", help="ECG acquisition time column.")
    parser.add_argument("--default-sample-rate", type=int, default=500, help="Fallback sample rate if column is missing.")
    parser.add_argument(
        "--output",
        default="/data/projects/study-af-ablation/manifests/finetune_lvr05_high.parquet",
        help="Output parquet path.",
    )
    return parser.parse_args()


def _load_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split} split file not found: {path}")
    return pd.read_pickle(path)


def _build_sample_id(df: pd.DataFrame, split: str, preferred_col: str) -> pd.Series:
    if preferred_col in df.columns:
        out = df[preferred_col].astype("string").fillna("")
        out = out.replace({"": pd.NA, "nan": pd.NA})
        if out.notna().all():
            return out.astype(str)

    idx = pd.Series(np.arange(len(df)), index=df.index, dtype="int64")
    return split + "_" + idx.astype(str)


def _combine_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_ts = pd.to_datetime(date_series, errors="coerce")
    if time_series is None:
        return date_ts

    time_str = time_series.astype("string").fillna("00:00:00")
    time_str = time_str.where(time_str.str.len() > 0, "00:00:00")
    full = date_ts.dt.strftime("%Y-%m-%d") + " " + time_str.astype(str)
    return pd.to_datetime(full, errors="coerce")


def _dedup_single_ecg_per_pid(
    df: pd.DataFrame,
    split: str,
    pid_col: str,
    procedure_date_col: str,
    acq_date_col: str,
    acq_time_col: str,
) -> tuple[pd.DataFrame, int]:
    if pid_col not in df.columns:
        raise ValueError(f"{split} split requires pid column for dedup: {pid_col}")

    if df.empty:
        return df, 0

    work = df.copy()
    proc_dt = pd.to_datetime(work[procedure_date_col], errors="coerce") if procedure_date_col in work.columns else pd.Series(pd.NaT, index=work.index)
    acq_date = work[acq_date_col] if acq_date_col in work.columns else pd.Series(pd.NaT, index=work.index)
    acq_time = work[acq_time_col] if acq_time_col in work.columns else pd.Series(pd.NA, index=work.index)
    acq_dt = _combine_datetime(acq_date, acq_time)

    # ranking: nearest to procedure first, then latest acquisition first
    dt_diff_sec = (proc_dt - acq_dt).dt.total_seconds().abs()
    work["_dt_diff_sec"] = dt_diff_sec.fillna(np.inf)
    work["_acq_dt"] = acq_dt
    work["_acq_dt_sort"] = acq_dt.fillna(pd.Timestamp.min)

    before = int((work[pid_col].value_counts() > 1).sum())
    dedup = (
        work.sort_values(
            by=[pid_col, "_dt_diff_sec", "_acq_dt_sort"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=[pid_col], keep="first")
        .drop(columns=["_dt_diff_sec", "_acq_dt", "_acq_dt_sort"], errors="ignore")
    )
    dedup = dedup.sort_index()
    after = int((dedup[pid_col].value_counts() > 1).sum())
    if after != 0:
        raise RuntimeError(f"{split} split dedup failed; still has duplicate pid rows")
    return dedup, before


def _normalize_split(
    df: pd.DataFrame,
    split: str,
    label_col: str,
    path_col: str,
    pid_col: str,
    sample_rate_col: str,
    sample_id_col: str,
    default_sample_rate: int,
) -> pd.DataFrame:
    required = [label_col, path_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split} split missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    out["split"] = split
    out["sample_id"] = _build_sample_id(df, split, sample_id_col)
    out["pkl_path"] = df[path_col].astype("string")
    out[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    if pid_col in df.columns:
        out["pid"] = df[pid_col].astype("string")

    if sample_rate_col in df.columns:
        out["sample_rate"] = pd.to_numeric(df[sample_rate_col], errors="coerce")
    else:
        out["sample_rate"] = float(default_sample_rate)
    out["sample_rate"] = out["sample_rate"].fillna(float(default_sample_rate)).astype("int64")

    valid_path = out["pkl_path"].notna() & out["pkl_path"].ne("") & out["pkl_path"].ne("nan")
    valid_label = out[label_col].notna()
    out = out.loc[valid_path & valid_label].copy()

    out["pkl_path"] = out["pkl_path"].astype(str)
    out[label_col] = out[label_col].astype(np.float32)
    out["sample_id"] = out["sample_id"].astype(str)
    if "pid" in out.columns:
        out["pid"] = out["pid"].astype(str)
    return out


def main() -> int:
    args = parse_args()
    base = Path(args.index_dir)

    frames = []
    for split, fname in (
        ("train", args.train_fname),
        ("valid", args.valid_fname),
        ("test", args.test_fname),
    ):
        raw = _load_split(base / fname, split)
        duplicated_pid_before = 0
        if split in {"valid", "test"}:
            raw, duplicated_pid_before = _dedup_single_ecg_per_pid(
                raw,
                split=split,
                pid_col=args.pid_column,
                procedure_date_col=args.procedure_date_column,
                acq_date_col=args.acq_date_column,
                acq_time_col=args.acq_time_column,
            )
        norm = _normalize_split(
            raw,
            split=split,
            label_col=args.label_column,
            path_col=args.path_column,
            pid_col=args.pid_column,
            sample_rate_col=args.sample_rate_column,
            sample_id_col=args.sample_id_column,
            default_sample_rate=args.default_sample_rate,
        )
        if split in {"valid", "test"}:
            print(f"[info] {split}: duplicate PID groups removed={duplicated_pid_before}")
        frames.append(norm)

    manifest = pd.concat(frames, axis=0, ignore_index=True)
    if manifest.empty:
        raise ValueError("Manifest is empty after normalization/filtering")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out, index=False)

    split_counts = manifest["split"].value_counts().to_dict()
    pos_rate = manifest.groupby("split")[args.label_column].mean().to_dict()
    pid_stats = {}
    if "pid" in manifest.columns:
        for split in ["train", "valid", "test"]:
            part = manifest[manifest["split"] == split]
            if part.empty:
                continue
            vc = part["pid"].value_counts()
            pid_stats[split] = {
                "rows": int(len(part)),
                "unique_pid": int(part["pid"].nunique()),
                "max_ecg_per_pid": int(vc.max()),
                "duplicate_pid_groups": int((vc > 1).sum()),
            }
    print(f"[ok] wrote finetune manifest: {out}")
    print(f"[ok] rows={len(manifest)} split_counts={split_counts} positive_rate={pos_rate}")
    if pid_stats:
        print(f"[ok] pid_stats={pid_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
