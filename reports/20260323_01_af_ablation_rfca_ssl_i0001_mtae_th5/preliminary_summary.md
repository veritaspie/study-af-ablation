# Preliminary Summary

## 1. 실행 스냅샷

- 결과 디렉토리 생성 시각: `2026-03-16_001959`
- 복구 메모 작성 시각: `2026-03-17`
- 리포트 작성 시각: `2026-03-23`
- objective metric: `test_macro_auroc`
- sample sets per trial: `10`
- configured trial limit in archived run config: `25`
- recovered/recorded trials in summary: `19`

## 2. 핵심 결과

이번 결과에서 가장 좋은 설정은 `trial_004`였고, 평균 `test_macro_auroc = 0.8199 +/- 0.0130`을 기록했습니다. 동일 trial의 10개 sample set은 모두 정상 완료됐고, set별 AUROC 범위는 `0.7957`부터 `0.8402`까지였습니다. 최고 단일 set에서는 `test_macro_auprc = 0.5602`, `test_loss = 0.4881`이었습니다.

best trial의 조합은 아래와 같습니다.

| hyperparameter | value |
| --- | ---: |
| finetune.lr | 0.001 |
| finetune.weight_decay | 0.01 |
| finetune.layer_decay | 0.8 |
| data.batch_size | 128 |
| finetune.num_freeze_layers | 0 |

top trial들은 아래 순서였습니다.

| rank | trial | mean AUROC | std | lr | wd | layer_decay | batch | freeze_layers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.8199 | 0.0130 | 0.001 | 0.01 | 0.8 | 128 | 0 |
| 2 | 14 | 0.8133 | 0.0169 | 0.0001 | 0.05 | 0.9 | 32 | 0 |
| 3 | 11 | 0.8099 | 0.0151 | 0.03 | 0.05 | 0.9 | 64 | 9 |
| 4 | 16 | 0.7947 | 0.0220 | 0.0003 | 0.01 | 0.9 | 128 | 0 |
| 5 | 12 | 0.7939 | 0.0130 | 0.0003 | 0.01 | 0.8 | 128 | 6 |

요약하면, 이번 탐색 범위 안에서는 `large batch + no freezing + moderate layer decay + nonzero weight decay` 조합이 가장 잘 작동했습니다. 다만 2위와의 차이는 약 `0.0066 AUROC`로 아주 크지는 않습니다.

## 3. preliminary 해석

가장 중요한 신호는, RFCA 코호트의 ECG 기반 downstream 분류에서 mean AUROC가 0.82 수준까지 올라갔다는 점입니다. test split 양성률이 약 15.5%라는 점을 감안하면, random 또는 약한 baseline 대비 의미 있는 분리 성능이 있다고 볼 수 있습니다.

또한 best trial 내부 분산이 `std 0.0130`으로 아주 크지 않아, pid-based sample set을 바꿔도 성능이 완전히 붕괴하지는 않았습니다. 이는 신호가 특정 소수 set에만 우연히 맞아떨어진 결과일 가능성을 다소 낮춰 줍니다.

반대로, 현재 결과는 그대로 논문화 가능한 결론으로 쓰기 어렵습니다. 이유는 다음과 같습니다.

- best trial selection 자체가 `test_macro_auroc` 기준이라 test leakage 성격이 있습니다.
- 검색이 끝까지 완료되지 않았습니다. archived config상 `25` trial 예정이었지만 실제 요약에는 `19` trial만 남아 있습니다.
- `trial_018`은 일부 artifact가 손상되어 recovery가 필요했습니다.
- current study-local config와 archived run config가 다르므로, 그대로 재실행하면 동일 결과가 재현되지 않을 수 있습니다.

## 4. 지금 단계에서 적을 수 있는 문장

현재 preliminary 수준에서는 다음 정도의 서술이 방어 가능합니다.

> RFCA cohort에서 MTAE-pretrained ECG encoder를 이용한 `LVR05_high` 분류는 repeated pid subsampling 기준 평균 test macro-AUROC 약 0.82를 보였다. 다만 본 수치는 test metric 기반의 exploratory hyperparameter search에서 얻어진 결과이므로, 독립된 hold-out 또는 validation-driven 재평가가 필요하다.

## 5. 다음 액션 제안

1. objective를 `valid_macro_auroc` 또는 validation-driven criterion으로 바꿔 재탐색
2. archived run config를 기준으로 same-search 재현본을 `/data/projects/study-af-ablation/outputs` 아래에 다시 생성
3. best setting 고정 후 독립 test 1회만 수행해 최종 수치 분리
4. AUROC 외에 AUPRC, calibration, thresholded sensitivity/specificity를 함께 정리
5. 가능한 경우 non-SSL 또는 다른 pretrained checkpoint 대비 delta를 같은 split에서 비교

## 6. 근거 파일

- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/hparam_search_summary.csv`
- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/best_trial.yaml`
- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/run_config.yaml`
- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/trial_004/set_scores.csv`
- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/trial_004/runs/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_t004_s08_finetune_20260316_004245/test_metrics.yaml`
- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/RECOVERY_NOTES_20260317.txt`
- `/data/projects/study-af-ablation/data/rfca_zarr_manifest_stats.md`
