# RFCA Cohort And Management Tables

This note separates three different RFCA cohorts that were easy to conflate in earlier notes:

1. Raw ECG XML cohort: `/data/raw/AF_ABLATION`
2. PID/CRF cohort: `/data/projects/af_ablation/test/assets/V_ABLATION(2025-11-30).csv`
3. Active training cohort: `/data/projects/study-af-ablation/manifests/finetune_lvr05_high_rfca_th5.parquet`

Those cohorts are related but not identical.

- The CRF table contains many rows where `LVR05_TotalLB` is blank.
- The active RFCA TH5 training manifest is a subset of the CRF PID cohort.
- The raw XML directory also contains duplicate-content ECGs: different XML filenames can share the same XML body.

Current counts from the generated summary:

- Raw XML ECG cohort: `30199` ECGs, `1242` PIDs
- CRF total cohort: `1321` rows, `1244` PIDs
- CRF with non-null `LVR05_TotalLB`: `738` rows, `722` PIDs
- Active RFCA TH5 manifest: `1961` ECGs, `713` PIDs
- Duplicate-content XML groups: `3642`
- XML rows participating in duplicate content: `7772`
- Labeled CRF PIDs missing from the active TH5 manifest: `9`

## Generated management tables

Run this from `/data/projects/ai-ecg` so the shared environment is available:

```bash
cd /data/projects/ai-ecg

./scripts/with_ai_ecg_env.sh python /data/projects/study-af-ablation/scripts/build_rfca_management_tables.py
```

Generated outputs:

- ECG inventory: `/data/projects/study-af-ablation/manifests/rfca_management/rfca_ecg_inventory.parquet`
- PID/CRF table: `/data/projects/study-af-ablation/manifests/rfca_management/rfca_pid_crf_table.parquet`
- Training ECG label table: `/data/projects/study-af-ablation/manifests/rfca_management/rfca_training_ecg_labels_th5.parquet`
- Summary JSON: `/data/projects/study-af-ablation/manifests/rfca_management/rfca_management_summary.json`

## Table roles

### 1) ECG inventory table

Use this as the raw ECG source-of-truth table.

Key columns:

- `sample_id`: XML stem used by the training manifest and current eval outputs
- `file_name`: original XML filename
- `file_path`: absolute XML path
- `pid`: PatientID parsed from XML
- `acquisition_date`, `acquisition_time`
- `xml_sha256`: full-file content hash
- `duplicate_content_count`, `has_duplicate_content`, `duplicate_content_rank`
- XML-derived fields for future subgrouping: `DIAGNOSIS`, `VENTRICULAR_RATE`, `PR_INTERVAL`, `QRS_DURATION`, `QT_CORRECTED`, `R_AXIS`

### 2) PID/CRF table

Use this as the patient/procedure management table.

Key columns:

- `pid`
- `crf_row_count`
- `earliest_procedure_date`, `latest_procedure_date`
- `lvr05_nonnull_row_count`, `lvr05_missing_row_count`
- `lvr05_min`, `lvr05_max`
- `xml_file_count`
- `manifest_ecg_count`
- `in_training_manifest_pid`

This is the table to inspect when a PID exists in CRF but not in the current training set.

### 3) Training ECG label table

Use this as the model-analysis join table.

Key columns:

- `sample_id`, `file_name`, `pid`, `split`
- `LVR05_high`
- `selected_lvr05_total_lb`
- `selected_procedure_date`
- `anchor_sample_id`, `is_anchor_ecg`
- duplicate-content flags from the ECG inventory table
- XML-derived rhythm and interval columns for subgroup analysis

This is the right table to join with model `test_outputs.csv` after eval outputs include a `filename` column.
