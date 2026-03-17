# RFCA Zarr Manifest Statistics

- Generated at: `2026-03-04 00:52:53 +0900`
- Manifest: `/data/projects/study-af-ablation/manifests/finetune_lvr05_high_rfca.parquet`
- Metadata: `/data/ecg/zarr/rfca.log/metadata_full.parquet`
- Ablation table: `/data/projects/af_ablation/test/assets/V_ABLATION(2025-11-30).csv`

## 1) Split Snapshot
| split | n_ecg | n_pid | positive_rate | lvr05_mean_ecg | age_mean_ecg |
| --- | --- | --- | --- | --- | --- |
| train | 1677 | 429 | 0.243 | 7.640 | 62.569 |
| valid | 142 | 142 | 0.176 | 6.103 | 63.022 |
| test | 142 | 142 | 0.155 | 5.231 | 63.023 |
| overall | 1961 | 713 | 0.232 | 7.354 | 62.635 |

## 2) ECG Per Patient
| split | n | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 429 | 3.909 | 3.651 | 1.000 | 2.000 | 3.000 | 4.000 | 44.000 |
| valid | 142 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| test | 142 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| overall | 713 | 2.750 | 3.169 | 1.000 | 1.000 | 2.000 | 3.000 | 44.000 |

## 3) LVR05 TotalLB (ECG-level)
| split | n | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1677 | 7.640 | 12.174 | 0.000 | 0.510 | 2.180 | 9.770 | 80.490 |
| valid | 142 | 6.103 | 11.780 | 0.000 | 0.520 | 1.555 | 6.068 | 73.700 |
| test | 142 | 5.231 | 10.846 | 0.000 | 0.323 | 1.390 | 4.015 | 71.580 |
| overall | 1961 | 7.354 | 12.070 | 0.000 | 0.490 | 1.920 | 8.640 | 80.490 |

## 4) LVR05 TotalLB (Patient-level)
| split | n | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 429 | 6.090 | 11.044 | 0.000 | 0.370 | 1.660 | 6.210 | 80.490 |
| valid | 142 | 6.103 | 11.780 | 0.000 | 0.520 | 1.555 | 6.068 | 73.700 |
| test | 142 | 5.231 | 10.846 | 0.000 | 0.323 | 1.390 | 4.015 | 71.580 |
| overall | 713 | 5.921 | 11.145 | 0.000 | 0.370 | 1.580 | 5.890 | 80.490 |

## 5) LVR05 Group Means by Split
| split | LVR05_high | count | lvr05_mean | lvr05_median |
| --- | --- | --- | --- | --- |
| train | 0 | 1270 | 2.137 | 1.190 |
| train | 1 | 407 | 24.811 | 22.710 |
| valid | 0 | 117 | 2.030 | 1.030 |
| valid | 1 | 25 | 25.160 | 18.920 |
| test | 0 | 120 | 1.695 | 0.975 |
| test | 1 | 22 | 24.520 | 16.670 |

## 6) Age Summary (ECG-level)
| split | n | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1562 | 62.569 | 9.049 | 30.000 | 58.000 | 64.000 | 69.000 | 81.000 |
| valid | 137 | 63.022 | 8.844 | 33.000 | 59.000 | 64.000 | 69.000 | 80.000 |
| test | 131 | 63.023 | 9.908 | 10.000 | 57.500 | 64.000 | 69.000 | 84.000 |
| overall | 1830 | 62.635 | 9.094 | 10.000 | 58.000 | 64.000 | 69.000 | 84.000 |

## 7) Age Summary (Patient-level)
| split | n | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 429 | 62.385 | 9.010 | 31.000 | 58.000 | 64.000 | 69.000 | 81.000 |
| valid | 137 | 63.022 | 8.844 | 33.000 | 59.000 | 64.000 | 69.000 | 80.000 |
| test | 131 | 63.023 | 9.908 | 10.000 | 57.500 | 64.000 | 69.000 | 84.000 |
| overall | 697 | 62.630 | 9.145 | 10.000 | 58.000 | 64.000 | 69.000 | 84.000 |

## 8) Age Bins (Patient-level)
| age_bin | count | pct |
| --- | --- | --- |
| 60-69 | 330 | 46.283 |
| 50-59 | 154 | 21.599 |
| 70-79 | 149 | 20.898 |
| <=49 | 61 | 8.555 |
| missing | 16 | 2.244 |
| 80+ | 3 | 0.421 |

## 9) Sex Distribution (Patient-level)
| sex | count | pct |
| --- | --- | --- |
| MALE | 539 | 75.596 |
| FEMALE | 173 | 24.264 |
| UNKNOWN | 1 | 0.140 |

## 10) Sex Distribution by Split (Patient-level)
| split | sex | count |
| --- | --- | --- |
| train | FEMALE | 103 |
| train | MALE | 326 |
| valid | FEMALE | 40 |
| valid | MALE | 101 |
| valid | UNKNOWN | 1 |
| test | FEMALE | 30 |
| test | MALE | 112 |

## 11) Source/Pointer Coverage
| metric | value |
| --- | --- |
| sample_rate_count_500 | 1961 |
| sample_rate_count_250 | 0 |
| sample_rate_unique | 1 |
| ds_idx_0_count | 1961 |
| ds_idx_1_count | 0 |
| ds_idx_unique | 1 |
| zarr_store_unique | 1 |
| age_available_ecg | 1830 |
| age_available_pid | 697 |
| lvr05_available_ecg | 1961 |
| lvr05_available_pid | 713 |

## Notes
- `LVR05_TotalLB` is reconstructed with the same cohort matching logic used in `build_finetune_manifest_rfca_zarr.py` (anchor window + per-PID earliest procedure).
- Train split contains repeated ECGs per PID by design; therefore ECG-level and patient-level summaries are both provided.
- Age/Sex are parsed from `hea_raw_text` XML fields (`PatientAge`, `AgeUnits`, `Gender`).
