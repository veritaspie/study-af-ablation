# AF Ablation RFCA SSL i0001 MTAE TH5 Metadata Refresh

## Executive Summary

이번 리포트는 기존 best-trial RFCA TH5 eval output을 유지한 채, 새로 만든 management table과 XML-derived metadata를 붙여 test subgroup 해석을 다시 정리한 버전입니다.

전체 archived 성능은 변하지 않았습니다. mean test AUROC는 `0.8199 +/- 0.0130` 입니다.

다만 cohort boundary는 더 명확해졌습니다. raw XML은 `30199` ECG / `1242` PID, CRF labeled cohort는 `722` PID, 실제 TH5 manifest는 `713` PID입니다.

또한 raw XML에는 duplicate-content ECG가 존재합니다. unique XML hash는 `26069`, duplicate-content group은 `3642`, affected row는 `7772` 입니다.

## Test Metadata Snapshot

- test ECG rows: `142`
- test unique PIDs: `142`
- age available rows: `131`
- figure: `/data/projects/study-af-ablation/reports/20260329_01_af_ablation_rfca_ssl_i0001_mtae_th5_metadata_refresh/figures/metadata_refresh_subgroup_summary.png`

## Rhythm Subgroups

| rhythm_class | n | pos | neg | positive_rate | mean_auroc | std_auroc | mean_auprc | pooled_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afib | 65 | 22 | 43 | 0.338 | 0.755 | 0.015 | 0.546 | 0.739 |
| flutter | 7 | 2 | 5 | 0.286 | 1.000 | 0.000 | 1.000 | 1.000 |
| other | 1 | 0 | 1 | 0.000 |  |  |  |  |
| sinus | 69 | 7 | 62 | 0.101 | 0.623 | 0.033 | 0.290 | 0.617 |

## Age Subgroups

| age_group | n | pos | neg | positive_rate | mean_auroc | std_auroc | mean_auprc | pooled_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-59 | 40 | 4 | 36 | 0.100 | 0.774 | 0.052 | 0.422 | 0.746 |
| 60-69 | 60 | 12 | 48 | 0.200 | 0.676 | 0.022 | 0.372 | 0.670 |
| 70+ | 31 | 15 | 16 | 0.484 | 0.847 | 0.038 | 0.860 | 0.837 |
| missing | 11 | 0 | 11 | 0.000 |  |  |  |  |

## Readout

- Rhythm 쪽에서는 여전히 AFib subgroup의 positive prevalence가 sinus보다 높습니다.
- Age 쪽에서는 `70+` subgroup에서 positive prevalence가 가장 높고, `0-59`는 가장 낮습니다.
- 이 결과는 구조적 burden signal만이 아니라 rhythm/state 및 age-correlated signal이 함께 섞였을 가능성을 더 강하게 시사합니다.
- 따라서 다음 단계 분석은 `filename` 또는 `sample_id`로 joined sample-level table을 기준으로 해야 합니다.

## Generated Files

- `test_metadata_snapshot.csv`
- `pooled_test_outputs_with_metadata.csv`
- `test_rhythm_metrics_by_set.csv`
- `test_rhythm_metrics_summary.csv`
- `test_age_metrics_by_set.csv`
- `test_age_metrics_summary.csv`
- `test_rhythm_age_counts.csv`
- `figures/metadata_refresh_subgroup_summary.png`
