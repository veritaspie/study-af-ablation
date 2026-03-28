# study-af-ablation

Study-local assets for the AF ablation work that runs on top of the shared
`ai-ecg` codebase.

This repo intentionally keeps only AF-ablation-specific assets:
- finetune/eval/hparam configs
- manifest builders and small shell wrappers
- data notes and manifest statistics

This repo intentionally does not keep the older contrastive or pretrain-side
experiment assets.

## Layout

- `configs/`: study-local runnable configs that inherit from `/data/projects/ai-ecg/configs/`
- `scripts/`: manifest builders, reporting helpers, and thin wrappers
- `data/`: durable data notes and manifest statistics
- `manifests/`: local generated manifests only (`.parquet` is gitignored)

Key data note:
- [`data/af_ablation_source_and_preprocessing.md`](./data/af_ablation_source_and_preprocessing.md)
- [`data/rfca_cohort_and_management_tables.md`](./data/rfca_cohort_and_management_tables.md)

## ai-ecg dependency

The shared runtime stays in `/data/projects/ai-ecg`.

Wrappers in this repo use:

```bash
AI_ECG_ROOT=/data/projects/ai-ecg
```

Override `AI_ECG_ROOT` only if the shared repo lives somewhere else.

## Common commands

Build the RFCA zarr finetune manifest:

```bash
python scripts/build_finetune_manifest_rfca_zarr.py
```

Run the RFCA threshold sweep hparam search:

```bash
bash scripts/run_ssl_pilot_rfca_threshold_sweep.sh
```

Run the RFCA feature dump helper config:

```bash
bash scripts/run_extract_features_rfca_ssl_i0001_mtae.sh
```
