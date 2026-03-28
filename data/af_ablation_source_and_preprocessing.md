# AF_ABLATION Source And Preprocessing

## Raw XML source

- Raw XML directory: `/data/raw/AF_ABLATION`
- Example verified file: `/data/raw/AF_ABLATION/MUSE_20260207_012829_63000.xml`

The AF_ABLATION XML files contain the fields used for rhythm labeling directly in the source XML:

- `Diagnosis/DiagnosisStatement/StmtText`
- `RestingECGMeasurements/VentricularRate`
- `RestingECGMeasurements/PRInterval`
- `RestingECGMeasurements/QRSDuration`
- `RestingECGMeasurements/QTCorrected`
- `RestingECGMeasurements/RAxis`

## Shared ingest path

The shared ingest implementation lives in `/data/projects/ai-ecg/src/preprocessing/ingest/`.

Recommended XML ingest command for this cohort:

```bash
cd /data/projects/ai-ecg

./scripts/with_ai_ecg_env.sh python -m preprocessing.ingest.cli ingest \
  --input /data/raw/AF_ABLATION \
  --input-type xml-dir \
  --zarr-store /data/ecg/zarr/rfca.zarr \
  --out-dir /data/ecg/zarr/rfca.log \
  --num-workers 8
```

Primary outputs:

- Zarr waveform store: `/data/ecg/zarr/rfca.zarr`
- Ingest metadata: `/data/ecg/zarr/rfca.log/metadata_full.parquet`
- Trace links: `/data/ecg/zarr/rfca.log/trace_links.parquet`
- Error table: `/data/ecg/zarr/rfca.log/errors.parquet`

## Metadata expectations

Current XML ingest is expected to persist the following dedicated columns in `metadata_full.parquet`:

- `DIAGNOSIS`
- `VENTRICULAR_RATE`
- `PR_INTERVAL`
- `QRS_DURATION`
- `QT_CORRECTED`
- `R_AXIS`

The same values are also written into `hea_full_json` for XML records.

## Old vs new metadata files

Older RFCA metadata files may predate the dedicated XML columns. In those files:

- diagnosis text may still be recoverable from `hea_comments_json`
- measurement fields such as ventricular rate may be absent from the top-level parquet schema

For notebook-side inspection this means rhythm string labels can often still be reconstructed, but measurement-dependent labels are more reliable after re-ingesting from `/data/raw/AF_ABLATION` with the current shared ingest.

## Study-side usage

Current study assets reference:

- metadata: `/data/ecg/zarr/rfca.log/metadata_full.parquet`
- manifest stats note: [`rfca_zarr_manifest_stats.md`](./rfca_zarr_manifest_stats.md)

The manual inspection notebook at `/data/projects/study-af-ablation/reports/20260323_02_manual_investigation/check_manifest.ipynb` imports the shared `xml_diagnosis.py` helper from `ai-ecg`.
