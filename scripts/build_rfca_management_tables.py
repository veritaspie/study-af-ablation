#!/usr/bin/env python3
"""Build RFCA ECG/PID/training management tables from raw XML, CRF, and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_ECG_SRC = Path("/data/projects/ai-ecg/src")
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(AI_ECG_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ECG_SRC))

from build_finetune_manifest_rfca_zarr import (  # noqa: E402
    ANCHOR_WINDOW_DAYS,
    INDEX_ACQ_DATE_COLUMN,
    INDEX_ACQ_TIME_COLUMN,
    INDEX_FILE_COLUMN,
    PID_COLUMN,
    TRAIN_WINDOW_DAYS,
    _load_ablation_table,
    _normalize_pid,
    _pick_anchor_rows,
)
from preprocessing.ingest.xml_diagnosis import extract_xml_labeling_fields  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", default="/data/raw/AF_ABLATION", help="Raw XML directory.")
    parser.add_argument(
        "--crf-path",
        default="/data/projects/af_ablation/test/assets/V_ABLATION(2025-11-30).csv",
        help="RFCA CRF/ablation table.",
    )
    parser.add_argument(
        "--manifest-path",
        default="/data/projects/study-af-ablation/manifests/finetune_lvr05_high_rfca_th5.parquet",
        help="Active RFCA training manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/projects/study-af-ablation/manifests/rfca_management",
        help="Output directory for generated management tables.",
    )
    parser.add_argument("--crf-pid-column", default="No", help="PID column name in the CRF table.")
    parser.add_argument(
        "--crf-procedure-date-column",
        default="DateofProcedure",
        help="Procedure date column name in the CRF table.",
    )
    parser.add_argument("--crf-label-column", default="LVR05_TotalLB", help="Continuous LVR05 label column in CRF.")
    parser.add_argument("--label-column", default="LVR05_high", help="Binary training label column name.")
    parser.add_argument("--label-threshold", type=float, default=5.0, help="Binary threshold for LVR05_high.")
    parser.add_argument(
        "--label-comparison",
        default="ge",
        choices=["ge", "gt"],
        help="Binary label comparison rule: ge (>=) or gt (>).",
    )
    return parser.parse_args()


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_iter_nodes_by_name(root: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in root.iter() if _xml_local_name(node.tag) == name]


def _xml_find_first_text(root: ET.Element, node_name: str) -> str:
    for node in root.iter():
        if _xml_local_name(node.tag) == node_name:
            text = (node.text or "").strip()
            if text:
                return text
    return ""


def _resolve_resting_ecg(root: ET.Element) -> ET.Element:
    if _xml_local_name(root.tag) == "RestingECG":
        return root
    matches = _xml_iter_nodes_by_name(root, "RestingECG")
    if not matches:
        raise ValueError("Missing RestingECG node in XML")
    return matches[0]


def _normalize_pid_value(value: str) -> str | None:
    text = re.sub(r"\.0+$", "", str(value or "").strip())
    return text or None


def _to_number(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scan_xml_inventory(xml_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under: {xml_dir}")

    for xml_path in xml_files:
        raw_bytes = xml_path.read_bytes()
        root = ET.fromstring(raw_bytes)
        resting = _resolve_resting_ecg(root)
        diagnosis_fields = extract_xml_labeling_fields(resting)

        sample_id = xml_path.stem
        pid = _normalize_pid_value(_xml_find_first_text(resting, "PatientID"))
        age_years = _to_number(_xml_find_first_text(resting, "PatientAge"))

        rows.append(
            {
                "sample_id": sample_id,
                "file_name": xml_path.name,
                "file_path": str(xml_path),
                "file_size_bytes": int(xml_path.stat().st_size),
                "xml_sha256": _sha256_hex(raw_bytes),
                "pid": pid,
                "acquisition_date": _xml_find_first_text(resting, "AcquisitionDate"),
                "acquisition_time": _xml_find_first_text(resting, "AcquisitionTime"),
                "patient_age_years": age_years,
                "patient_gender": _xml_find_first_text(resting, "Gender") or None,
                "DIAGNOSIS": diagnosis_fields.get("DIAGNOSIS"),
                "VENTRICULAR_RATE": diagnosis_fields.get("VENTRICULAR_RATE"),
                "PR_INTERVAL": diagnosis_fields.get("PR_INTERVAL"),
                "QRS_DURATION": diagnosis_fields.get("QRS_DURATION"),
                "QT_CORRECTED": diagnosis_fields.get("QT_CORRECTED"),
                "R_AXIS": diagnosis_fields.get("R_AXIS"),
            }
        )

    inventory = pd.DataFrame(rows)
    inventory["pid"] = _normalize_pid(inventory["pid"])
    hash_counts = inventory["xml_sha256"].value_counts(dropna=False)
    inventory["duplicate_content_count"] = inventory["xml_sha256"].map(hash_counts).astype(int)
    inventory["has_duplicate_content"] = inventory["duplicate_content_count"].gt(1)
    inventory["duplicate_content_rank"] = inventory.groupby("xml_sha256").cumcount().add(1).astype(int)
    return inventory


def _build_pid_crf_table(
    *,
    crf: pd.DataFrame,
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    pid_col: str,
    proc_date_col: str,
    label_col: str,
) -> pd.DataFrame:
    work = crf.copy()
    work[pid_col] = _normalize_pid(work[pid_col])
    work[proc_date_col] = pd.to_datetime(work[proc_date_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[work[pid_col].notna()].copy()
    work["lvr05_is_missing"] = work[label_col].isna()

    pid_table = (
        work.groupby(pid_col, dropna=False)
        .agg(
            crf_row_count=(pid_col, "size"),
            earliest_procedure_date=(proc_date_col, "min"),
            latest_procedure_date=(proc_date_col, "max"),
            lvr05_nonnull_row_count=(label_col, lambda s: int(s.notna().sum())),
            lvr05_missing_row_count=("lvr05_is_missing", lambda s: int(s.sum())),
            lvr05_min=(label_col, "min"),
            lvr05_max=(label_col, "max"),
        )
        .reset_index()
        .rename(columns={pid_col: "pid"})
    )

    xml_counts = inventory.groupby("pid").agg(xml_file_count=("sample_id", "size")).reset_index()
    manifest_counts = manifest.groupby("pid").agg(manifest_ecg_count=("sample_id", "size")).reset_index()
    manifest_pid_flag = manifest[["pid"]].drop_duplicates().assign(in_training_manifest_pid=True)

    pid_table = pid_table.merge(xml_counts, on="pid", how="left")
    pid_table = pid_table.merge(manifest_counts, on="pid", how="left")
    pid_table = pid_table.merge(manifest_pid_flag, on="pid", how="left")
    pid_table["xml_file_count"] = pid_table["xml_file_count"].fillna(0).astype(int)
    pid_table["manifest_ecg_count"] = pid_table["manifest_ecg_count"].fillna(0).astype(int)
    pid_table["in_training_manifest_pid"] = pid_table["in_training_manifest_pid"].fillna(False).astype(bool)
    pid_table["has_nonnull_lvr05"] = pid_table["lvr05_nonnull_row_count"].gt(0)
    return pid_table.sort_values(["in_training_manifest_pid", "pid"], ascending=[False, True], kind="mergesort")


def _selected_pid_label_table(
    *,
    inventory: pd.DataFrame,
    crf: pd.DataFrame,
    pid_col: str,
    proc_date_col: str,
    label_col: str,
    binary_label_col: str,
    label_threshold: float,
    label_comparison: str,
) -> pd.DataFrame:
    index_df = inventory.rename(
        columns={
            "pid": PID_COLUMN,
            "sample_id": INDEX_FILE_COLUMN,
            "acquisition_date": INDEX_ACQ_DATE_COLUMN,
            "acquisition_time": INDEX_ACQ_TIME_COLUMN,
        }
    )[[PID_COLUMN, INDEX_FILE_COLUMN, INDEX_ACQ_DATE_COLUMN, INDEX_ACQ_TIME_COLUMN]].copy()

    cohort = _pick_anchor_rows(
        index_df=index_df,
        crf_df=crf,
        pid_col=PID_COLUMN,
        acq_date_col=INDEX_ACQ_DATE_COLUMN,
        acq_time_col=INDEX_ACQ_TIME_COLUMN,
        crf_pid_col=pid_col,
        proc_date_col=proc_date_col,
        crf_label_col=label_col,
        label_col=binary_label_col,
        label_threshold=label_threshold,
        label_comparison=label_comparison,
    )
    selected = cohort[
        [PID_COLUMN, INDEX_FILE_COLUMN, proc_date_col, label_col, binary_label_col, "split"]
    ].copy()
    selected = selected.rename(
        columns={
            PID_COLUMN: "pid",
            INDEX_FILE_COLUMN: "anchor_sample_id",
            proc_date_col: "selected_procedure_date",
            label_col: "selected_lvr05_total_lb",
            binary_label_col: "selected_lvr05_high",
            "split": "selected_pid_split",
        }
    )
    selected["selected_procedure_date"] = pd.to_datetime(selected["selected_procedure_date"], errors="coerce")
    return selected


def _build_training_ecg_label_table(
    *,
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    pid_table: pd.DataFrame,
    selected_pid: pd.DataFrame,
    binary_label_col: str,
) -> pd.DataFrame:
    inventory_cols = [
        "sample_id",
        "file_name",
        "file_path",
        "xml_sha256",
        "duplicate_content_count",
        "has_duplicate_content",
        "duplicate_content_rank",
        "acquisition_date",
        "acquisition_time",
        "patient_age_years",
        "patient_gender",
        "DIAGNOSIS",
        "VENTRICULAR_RATE",
        "PR_INTERVAL",
        "QRS_DURATION",
        "QT_CORRECTED",
        "R_AXIS",
    ]
    training = manifest.copy()
    training = training.merge(inventory[inventory_cols], on="sample_id", how="left")
    training = training.merge(
        pid_table[["pid", "crf_row_count", "lvr05_nonnull_row_count", "lvr05_missing_row_count"]],
        on="pid",
        how="left",
    )
    training = training.merge(selected_pid, on="pid", how="left")
    training["is_anchor_ecg"] = training["sample_id"].eq(training["anchor_sample_id"])
    if "file_name" in training.columns:
        missing_file_name = training["file_name"].isna() | training["file_name"].astype(str).str.strip().eq("")
        training.loc[missing_file_name, "file_name"] = training.loc[missing_file_name, "sample_id"].astype(str) + ".xml"
    if binary_label_col in training.columns:
        training[binary_label_col] = pd.to_numeric(training[binary_label_col], errors="coerce").astype("float32")
    return training


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> int:
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    crf_path = Path(args.crf_path)
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)

    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    if not crf_path.exists():
        raise FileNotFoundError(f"CRF table not found: {crf_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    inventory = _scan_xml_inventory(xml_dir)
    manifest = pd.read_parquet(manifest_path)
    crf = _load_ablation_table(crf_path)
    crf[args.crf_pid_column] = _normalize_pid(crf[args.crf_pid_column])
    crf[args.crf_label_column] = pd.to_numeric(crf[args.crf_label_column], errors="coerce")

    pid_table = _build_pid_crf_table(
        crf=crf,
        inventory=inventory,
        manifest=manifest,
        pid_col=args.crf_pid_column,
        proc_date_col=args.crf_procedure_date_column,
        label_col=args.crf_label_column,
    )
    selected_pid = _selected_pid_label_table(
        inventory=inventory,
        crf=crf,
        pid_col=args.crf_pid_column,
        proc_date_col=args.crf_procedure_date_column,
        label_col=args.crf_label_column,
        binary_label_col=args.label_column,
        label_threshold=args.label_threshold,
        label_comparison=args.label_comparison,
    )
    training = _build_training_ecg_label_table(
        manifest=manifest,
        inventory=inventory,
        pid_table=pid_table,
        selected_pid=selected_pid,
        binary_label_col=args.label_column,
    )

    inventory_path = output_dir / "rfca_ecg_inventory.parquet"
    pid_table_path = output_dir / "rfca_pid_crf_table.parquet"
    training_path = output_dir / "rfca_training_ecg_labels_th5.parquet"
    summary_path = output_dir / "rfca_management_summary.json"

    _write_table(inventory, inventory_path)
    _write_table(pid_table, pid_table_path)
    _write_table(training, training_path)

    manifest_pids = set(manifest["pid"].astype(str))
    crf_pids = set(pid_table["pid"].astype(str))
    labeled_crf_pids = set(pid_table.loc[pid_table["has_nonnull_lvr05"], "pid"].astype(str))
    duplicate_groups = inventory.loc[inventory["has_duplicate_content"], "xml_sha256"].nunique()
    duplicate_rows = int(inventory["has_duplicate_content"].sum())
    summary = {
        "anchor_window_days": ANCHOR_WINDOW_DAYS,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "xml_total_rows": int(len(inventory)),
        "xml_unique_pid": int(inventory["pid"].dropna().nunique()),
        "xml_unique_content_hash": int(inventory["xml_sha256"].nunique()),
        "xml_duplicate_content_groups": int(duplicate_groups),
        "xml_duplicate_content_rows": duplicate_rows,
        "crf_total_rows": int(len(crf)),
        "crf_unique_pid": int(len(crf_pids)),
        "crf_lvr05_nonnull_rows": int(crf[args.crf_label_column].notna().sum()),
        "crf_lvr05_nonnull_pid": int(len(labeled_crf_pids)),
        "manifest_total_rows": int(len(manifest)),
        "manifest_unique_pid": int(len(manifest_pids)),
        "manifest_pid_not_in_crf": int(len(manifest_pids - crf_pids)),
        "crf_pid_not_in_manifest": int(len(crf_pids - manifest_pids)),
        "labeled_crf_pid_not_in_manifest": int(len(labeled_crf_pids - manifest_pids)),
        "table_paths": {
            "ecg_inventory": str(inventory_path),
            "pid_crf": str(pid_table_path),
            "training_ecg_labels": str(training_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[ok] wrote ECG inventory: {inventory_path}")
    print(f"[ok] wrote PID/CRF table: {pid_table_path}")
    print(f"[ok] wrote training ECG label table: {training_path}")
    print(f"[ok] wrote summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
