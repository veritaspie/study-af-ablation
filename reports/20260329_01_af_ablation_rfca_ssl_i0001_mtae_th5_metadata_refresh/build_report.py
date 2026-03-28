#!/usr/bin/env python3
"""Build a metadata-refreshed RFCA TH5 report using management tables and existing eval outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency in some envs
    plt = None


REPORT_DIR = Path(__file__).resolve().parent
SOURCE_REPORT_DIR = Path(
    "/data/projects/study-af-ablation/reports/20260323_01_af_ablation_rfca_ssl_i0001_mtae_th5"
)
TRAINING_TABLE = Path(
    "/data/projects/study-af-ablation/manifests/rfca_management/rfca_training_ecg_labels_th5.parquet"
)
SUMMARY_JSON = Path(
    "/data/projects/study-af-ablation/manifests/rfca_management/rfca_management_summary.json"
)
AI_ECG_SRC = Path("/data/projects/ai-ecg/src")
if str(AI_ECG_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ECG_SRC))

from preprocessing.ingest.xml_diagnosis import add_xml_diagnosis_labels  # noqa: E402


def _metric_dict(df: pd.DataFrame, prob_col: str, label_col: str) -> dict[str, float | int]:
    n = int(len(df))
    pos = int(df[label_col].sum())
    neg = int(n - pos)
    mean_prob = float(df[prob_col].mean()) if n else float("nan")
    auroc = float("nan")
    auprc = float("nan")
    if pos > 0 and neg > 0:
        auroc = float(roc_auc_score(df[label_col], df[prob_col]))
        auprc = float(average_precision_score(df[label_col], df[prob_col]))
    return {
        "n": n,
        "pos": pos,
        "neg": neg,
        "positive_rate": float(pos / n) if n else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "mean_prob": mean_prob,
    }


def _load_test_metadata() -> pd.DataFrame:
    frame = pl.read_parquet(str(TRAINING_TABLE))
    labeled = add_xml_diagnosis_labels(frame).to_pandas()
    test = labeled[labeled["split"].astype(str) == "test"].copy()
    test["filename"] = test.get("file_name", test["sample_id"]).astype(str)
    missing = test["filename"].str.strip().eq("")
    test.loc[missing, "filename"] = test.loc[missing, "sample_id"].astype(str) + ".xml"

    def rhythm_class(row: pd.Series) -> str:
        if bool(row.get("AFIB", False)):
            return "afib"
        if bool(row.get("AF", False)):
            return "flutter"
        if bool(row.get("SR", False)):
            return "sinus"
        return "other"

    test["rhythm_class"] = test.apply(rhythm_class, axis=1)
    test["age_group"] = pd.cut(
        test["patient_age_years"],
        bins=[-1, 59, 69, 200],
        labels=["0-59", "60-69", "70+"],
        include_lowest=True,
    ).astype("object").fillna("missing")
    return test


def _load_joined_outputs(test_meta: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    runs = json.loads((SOURCE_REPORT_DIR / "best_trial_roc_eval_runs.json").read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for set_index, run in enumerate(runs):
        out = pd.read_csv(run["test_outputs_csv"])
        out["set_index"] = int(set_index)
        out["set_name"] = str(run["set_name"])
        frames.append(out)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.merge(test_meta, on=["sample_id", "filename"], how="left", validate="many_to_one")
    return pooled, runs


def _setwise_group_metrics(pooled: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (set_index, group_value), part in pooled.groupby(["set_index", group_col], dropna=False, sort=True):
        metrics = _metric_dict(part, prob_col="prob_LVR05_high", label_col="label_LVR05_high")
        row = {"set_index": int(set_index), group_col: str(group_value)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "set_index"], kind="mergesort")


def _summary_group_metrics(pooled: pd.DataFrame, test_meta: pd.DataFrame, group_col: str) -> pd.DataFrame:
    setwise = _setwise_group_metrics(pooled, group_col=group_col)
    base_counts = []
    for group_value, part in test_meta.groupby(group_col, dropna=False, sort=True):
        metrics = _metric_dict(part, prob_col="LVR05_high", label_col="LVR05_high")
        base_counts.append(
            {
                group_col: str(group_value),
                "n": int(metrics["n"]),
                "pos": int(metrics["pos"]),
                "neg": int(metrics["neg"]),
                "positive_rate": float(metrics["positive_rate"]),
            }
        )
    summary = pd.DataFrame(base_counts)
    mean_metrics = (
        setwise.groupby(group_col, dropna=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_prob=("mean_prob", "mean"),
        )
        .reset_index()
    )
    pooled_rows = []
    for group_value, part in pooled.groupby(group_col, dropna=False, sort=True):
        metrics = _metric_dict(part, prob_col="prob_LVR05_high", label_col="label_LVR05_high")
        pooled_rows.append(
            {
                group_col: str(group_value),
                "pooled_auroc": metrics["auroc"],
                "pooled_auprc": metrics["auprc"],
            }
        )
    pooled_metrics = pd.DataFrame(pooled_rows)
    return summary.merge(mean_metrics, on=group_col, how="left").merge(pooled_metrics, on=group_col, how="left")


def _write_figure(rhythm_summary: pd.DataFrame, age_summary: pd.DataFrame, out_path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    rhythm_order = [value for value in ["afib", "flutter", "sinus", "other"] if value in rhythm_summary["rhythm_class"].tolist()]
    age_order = [value for value in ["0-59", "60-69", "70+", "missing"] if value in age_summary["age_group"].tolist()]

    rhythm_plot = rhythm_summary.set_index("rhythm_class").loc[rhythm_order]
    age_plot = age_summary.set_index("age_group").loc[age_order]

    axes[0, 0].bar(rhythm_plot.index, rhythm_plot["positive_rate"], color="#3a6ea5")
    axes[0, 0].set_title("Positive Rate By Rhythm")
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].bar(age_plot.index, age_plot["positive_rate"], color="#c97b63")
    axes[0, 1].set_title("Positive Rate By Age")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(
        rhythm_plot.index,
        rhythm_plot["mean_auroc"],
        yerr=rhythm_plot["std_auroc"].fillna(0.0),
        color="#3a6ea5",
        capsize=4,
    )
    axes[1, 0].set_title("Mean AUROC By Rhythm")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(
        age_plot.index,
        age_plot["mean_auroc"],
        yerr=age_plot["std_auroc"].fillna(0.0),
        color="#c97b63",
        capsize=4,
    )
    axes[1, 1].set_title("Mean AUROC By Age")
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_report_md(
    *,
    test_meta: pd.DataFrame,
    rhythm_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    summary_json: dict[str, object],
    figure_path: Path,
) -> str:
    old_roc_summary = json.loads((SOURCE_REPORT_DIR / "best_trial_roc_summary.json").read_text(encoding="utf-8"))

    def _table(df: pd.DataFrame, columns: list[str]) -> str:
        view = df[columns].fillna("").copy()
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = ["| " + " | ".join(str(view.iloc[i][col]) for col in columns) + " |" for i in range(len(view))]
        return "\n".join([header, separator] + rows)

    rhythm_view = rhythm_summary.copy()
    age_view = age_summary.copy()
    for df in (rhythm_view, age_view):
        for col in ["positive_rate", "mean_auroc", "std_auroc", "mean_auprc", "pooled_auroc", "pooled_auprc"]:
            if col in df.columns:
                df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

    lines = [
        "# AF Ablation RFCA SSL i0001 MTAE TH5 Metadata Refresh",
        "",
        "## Executive Summary",
        "",
        (
            "이번 리포트는 기존 best-trial RFCA TH5 eval output을 유지한 채, 새로 만든 management table과 XML-derived metadata를 붙여 "
            "test subgroup 해석을 다시 정리한 버전입니다."
        ),
        "",
        (
            f"전체 archived 성능은 변하지 않았습니다. mean test AUROC는 `{old_roc_summary['archived_mean_auroc']:.4f} +/- "
            f"{old_roc_summary['archived_std_auroc']:.4f}` 입니다."
        ),
        "",
        (
            f"다만 cohort boundary는 더 명확해졌습니다. raw XML은 `{summary_json['xml_total_rows']}` ECG / `{summary_json['xml_unique_pid']}` PID, "
            f"CRF labeled cohort는 `{summary_json['crf_lvr05_nonnull_pid']}` PID, 실제 TH5 manifest는 `{summary_json['manifest_unique_pid']}` PID입니다."
        ),
        "",
        (
            f"또한 raw XML에는 duplicate-content ECG가 존재합니다. unique XML hash는 `{summary_json['xml_unique_content_hash']}`, "
            f"duplicate-content group은 `{summary_json['xml_duplicate_content_groups']}`, affected row는 `{summary_json['xml_duplicate_content_rows']}` 입니다."
        ),
        "",
        "## Test Metadata Snapshot",
        "",
        f"- test ECG rows: `{len(test_meta)}`",
        f"- test unique PIDs: `{test_meta['pid'].astype(str).nunique()}`",
        f"- age available rows: `{int(test_meta['patient_age_years'].notna().sum())}`",
        f"- figure: `{figure_path}`",
        "",
        "## Rhythm Subgroups",
        "",
        _table(
            rhythm_view,
            ["rhythm_class", "n", "pos", "neg", "positive_rate", "mean_auroc", "std_auroc", "mean_auprc", "pooled_auroc"],
        ),
        "",
        "## Age Subgroups",
        "",
        _table(
            age_view,
            ["age_group", "n", "pos", "neg", "positive_rate", "mean_auroc", "std_auroc", "mean_auprc", "pooled_auroc"],
        ),
        "",
        "## Readout",
        "",
        "- Rhythm 쪽에서는 여전히 AFib subgroup의 positive prevalence가 sinus보다 높습니다.",
        "- Age 쪽에서는 `70+` subgroup에서 positive prevalence가 가장 높고, `0-59`는 가장 낮습니다.",
        "- 이 결과는 구조적 burden signal만이 아니라 rhythm/state 및 age-correlated signal이 함께 섞였을 가능성을 더 강하게 시사합니다.",
        "- 따라서 다음 단계 분석은 `filename` 또는 `sample_id`로 joined sample-level table을 기준으로 해야 합니다.",
        "",
        "## Generated Files",
        "",
        "- `test_metadata_snapshot.csv`",
        "- `pooled_test_outputs_with_metadata.csv`",
        "- `test_rhythm_metrics_by_set.csv`",
        "- `test_rhythm_metrics_summary.csv`",
        "- `test_age_metrics_by_set.csv`",
        "- `test_age_metrics_summary.csv`",
        "- `test_rhythm_age_counts.csv`",
        "- `figures/metadata_refresh_subgroup_summary.png`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    test_meta = _load_test_metadata().sort_values("sample_id", kind="mergesort")
    pooled, runs = _load_joined_outputs(test_meta)

    test_meta.to_csv(REPORT_DIR / "test_metadata_snapshot.csv", index=False)
    pooled.to_csv(REPORT_DIR / "pooled_test_outputs_with_metadata.csv", index=False)

    rhythm_by_set = _setwise_group_metrics(pooled, group_col="rhythm_class")
    rhythm_summary = _summary_group_metrics(pooled, test_meta, group_col="rhythm_class")
    age_by_set = _setwise_group_metrics(pooled, group_col="age_group")
    age_summary = _summary_group_metrics(pooled, test_meta, group_col="age_group")
    rhythm_age_counts = (
        test_meta.groupby(["rhythm_class", "age_group"], dropna=False)
        .agg(n=("sample_id", "size"), pos=("LVR05_high", "sum"))
        .reset_index()
        .sort_values(["rhythm_class", "age_group"], kind="mergesort")
    )
    rhythm_age_counts["neg"] = rhythm_age_counts["n"] - rhythm_age_counts["pos"]

    rhythm_by_set.to_csv(REPORT_DIR / "test_rhythm_metrics_by_set.csv", index=False)
    rhythm_summary.to_csv(REPORT_DIR / "test_rhythm_metrics_summary.csv", index=False)
    age_by_set.to_csv(REPORT_DIR / "test_age_metrics_by_set.csv", index=False)
    age_summary.to_csv(REPORT_DIR / "test_age_metrics_summary.csv", index=False)
    rhythm_age_counts.to_csv(REPORT_DIR / "test_rhythm_age_counts.csv", index=False)

    figure_path = REPORT_DIR / "figures" / "metadata_refresh_subgroup_summary.png"
    _write_figure(rhythm_summary, age_summary, figure_path)

    report_md = _build_report_md(
        test_meta=test_meta,
        rhythm_summary=rhythm_summary,
        age_summary=age_summary,
        summary_json=summary_json,
        figure_path=figure_path,
    )
    (REPORT_DIR / "report.md").write_text(report_md, encoding="utf-8")

    readme_lines = [
        "# RFCA TH5 Metadata Refresh Report",
        "",
        "- created_at: `2026-03-29`",
        f"- report_dir: `{REPORT_DIR}`",
        f"- source_report_dir: `{SOURCE_REPORT_DIR}`",
        f"- training_label_table: `{TRAINING_TABLE}`",
        "",
        "## Files",
        "",
        "- `report.md`: refreshed narrative report",
        "- `test_metadata_snapshot.csv`: test cohort with refreshed metadata",
        "- `pooled_test_outputs_with_metadata.csv`: all best-trial test outputs joined with metadata",
        "- `test_rhythm_metrics_summary.csv`: rhythm subgroup summary",
        "- `test_age_metrics_summary.csv`: age subgroup summary",
        "- `test_rhythm_age_counts.csv`: rhythm x age base counts",
    ]
    (REPORT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "source_report_dir": str(SOURCE_REPORT_DIR),
        "n_runs": len(runs),
        "test_rows": int(len(test_meta)),
        "test_unique_pid": int(test_meta["pid"].astype(str).nunique()),
        "figure_path": str(figure_path),
    }
    (REPORT_DIR / "report_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[ok] wrote refreshed report to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
