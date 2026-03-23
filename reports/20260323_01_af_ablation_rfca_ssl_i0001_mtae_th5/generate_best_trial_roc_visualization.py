from __future__ import annotations

from pathlib import Path
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from cli import run_eval

REPORT_DIR = Path("/data/projects/study-af-ablation/reports/20260323_01_af_ablation_rfca_ssl_i0001_mtae_th5")
TRIAL_DIR = Path(
    "/data/projects/ai-ecg/outputs/af_ablation_202602/"
    "af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/"
    "trial_004"
)
FIG_DIR = REPORT_DIR / "figures"
TMP_CFG_DIR = REPORT_DIR / "roc_eval_configs"
EVAL_ROOT = REPORT_DIR / "roc_eval_outputs"
MAPPINGS_JSON = REPORT_DIR / "best_trial_roc_eval_runs.json"


def build_eval_config(run_dir: Path) -> Path:
    cfg_path = run_dir / "run_config.yaml"
    best_ckpt = Path((run_dir / "best_checkpoint.txt").read_text(encoding="utf-8").strip())
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(EVAL_ROOT)
    cfg["experiment_name"] = f"{cfg['experiment_name']}_roc_eval"
    cfg.setdefault("runtime", {})["accelerator"] = "cpu"
    cfg["runtime"]["devices"] = 1
    cfg["runtime"]["strategy"] = "auto"
    cfg["runtime"]["precision"] = 32
    cfg.setdefault("data", {})["num_workers"] = 0
    cfg.setdefault("eval", {})["checkpoint_path"] = str(best_ckpt)
    cfg["eval"]["threshold_rule"] = "max_youden"

    out_path = TMP_CFG_DIR / f"{run_dir.name}_eval.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def load_or_run_evals() -> list[dict[str, str]]:
    if MAPPINGS_JSON.exists():
        mappings = json.loads(MAPPINGS_JSON.read_text(encoding="utf-8"))
        if mappings and all(Path(item["test_outputs_csv"]).exists() for item in mappings):
            print("[roc-viz] using cached eval outputs", flush=True)
            return mappings

    mappings: list[dict[str, str]] = []
    run_dirs = sorted((TRIAL_DIR / "runs").glob("*"))
    for idx, run_dir in enumerate(run_dirs):
        if not run_dir.is_dir():
            continue
        print(f"[roc-viz] eval {idx + 1}/10: {run_dir.name}", flush=True)
        eval_cfg = build_eval_config(run_dir)
        eval_run_dir = run_eval(str(eval_cfg))
        mappings.append(
            {
                "set_name": run_dir.name,
                "source_run_dir": str(run_dir),
                "eval_run_dir": str(eval_run_dir),
                "test_outputs_csv": str(eval_run_dir / "test_outputs.csv"),
            }
        )

    MAPPINGS_JSON.write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    return mappings


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    mappings = load_or_run_evals()

    roc_grid = np.linspace(0.0, 1.0, 401)
    interpolated = []
    roc_rows = []

    for item in mappings:
        df = pd.read_csv(item["test_outputs_csv"])
        prob_col = [c for c in df.columns if c.startswith("prob_")][0]
        label_col = [c for c in df.columns if c.startswith("label_")][0]
        y_true = df[label_col].to_numpy(dtype=float)
        y_prob = df[prob_col].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(roc_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interpolated.append(interp_tpr)
        roc_rows.append({"set_name": item["set_name"], "n_points": int(len(fpr))})

    interp_arr = np.vstack(interpolated)
    tpr_mean = interp_arr.mean(axis=0)
    tpr_min = interp_arr.min(axis=0)
    tpr_max = interp_arr.max(axis=0)

    archived_auc_df = pd.read_csv(TRIAL_DIR / "set_scores.csv")[["set_index", "objective_value"]].rename(
        columns={"objective_value": "auroc"}
    )
    archived_auc_df["set_label"] = archived_auc_df["set_index"].map(lambda x: f"s{int(x):02d}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    ax_roc, ax_auc = axes

    for interp_tpr in interp_arr:
        ax_roc.plot(roc_grid, interp_tpr, color="#94a3b8", alpha=0.30, linewidth=1.0)
    ax_roc.fill_between(roc_grid, tpr_min, tpr_max, color="#93c5fd", alpha=0.35, label="Min-Max band")
    ax_roc.plot(roc_grid, tpr_mean, color="#1d4ed8", linewidth=2.4, label="Mean ROC")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.0)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Band Across 10 Sampled Sets\n(checkpoint-based re-evaluation)")
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.grid(alpha=0.2)

    x = archived_auc_df["set_index"].to_numpy()
    y = archived_auc_df["auroc"].to_numpy()
    mean_auc = float(y.mean())
    min_auc = float(y.min())
    max_auc = float(y.max())
    std_auc = float(y.std())

    ax_auc.axhspan(min_auc, max_auc, color="#bfdbfe", alpha=0.45, label="Min-Max")
    ax_auc.plot(x, y, color="#1d4ed8", linewidth=1.2, alpha=0.85)
    ax_auc.scatter(x, y, color="#1e40af", s=30, zorder=3)
    ax_auc.axhline(mean_auc, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean={mean_auc:.4f}")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(archived_auc_df["set_label"].tolist())
    ax_auc.set_ylim(min_auc - 0.01, max_auc + 0.01)
    ax_auc.set_ylabel("AUROC")
    ax_auc.set_title("Archived AUROC By Sampled Set")
    ax_auc.grid(alpha=0.2)
    ax_auc.legend(frameon=False, loc="lower right")

    fig.suptitle("Trial 004: ROC Band And AUROC Visualization", fontsize=13)
    fig.tight_layout()
    out_path = FIG_DIR / "best_trial_roc_band_and_auroc.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        {
            "fpr": roc_grid,
            "tpr_mean": tpr_mean,
            "tpr_min": tpr_min,
            "tpr_max": tpr_max,
        }
    ).to_csv(REPORT_DIR / "best_trial_roc_band.csv", index=False)
    archived_auc_df.to_csv(REPORT_DIR / "best_trial_auroc_by_set.csv", index=False)

    summary = {
        "archived_mean_auroc": mean_auc,
        "archived_std_auroc": std_auc,
        "archived_min_auroc": min_auc,
        "archived_max_auroc": max_auc,
        "n_sets": int(len(archived_auc_df)),
        "figure_path": str(out_path.resolve()),
        "roc_band_source": "checkpoint-based eval outputs",
        "auroc_points_source": "archived trial_004/set_scores.csv",
    }
    (REPORT_DIR / "best_trial_roc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
