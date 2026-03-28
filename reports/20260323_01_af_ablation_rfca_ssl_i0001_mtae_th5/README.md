# AF Ablation RFCA SSL i0001 MTAE TH5 Report

- 작성일: `2026-03-23`
- 메인 보고서: `report.md`
- 리포트 디렉토리: `/data/projects/study-af-ablation/reports/20260323_01_af_ablation_rfca_ssl_i0001_mtae_th5`
- 기반 결과 디렉토리: `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959`

## 문서 구성

- `report.md`: 읽기용 메인 보고서. AUROC figure와 rhythm/subgroup 해석을 포함한 메인 버전
- `rhythm_dataset_analysis.md`: mixed-rhythm dataset 구성, split별 label prevalence, AFib/AFL/NSR 분포와 subgroup 성능 정리
- `code_trace.md`: 실제 실행 entrypoint, config inheritance, hparam-search/finetune/data loading 코드 경로
- `study_description.md`: 연구 배경과 설계 설명
- `preliminary_summary.md`: preliminary 메모 버전 요약

## 생성된 시각화 / 요약 파일

- `figures/best_trial_roc_band_and_auroc.png`: best trial 10개 sampled set ROC min-max band + archived AUROC points
- `figures/rhythm_violin_and_subgroup.png`: AFib/AFL/NSR 각각의 negative/positive paired violin plot과 rhythm subgroup 성능 비교
- `best_trial_roc_summary.json`: archived AUROC mean/std/min/max 요약
- `rhythm_split_label_summary.csv`: split x rhythm별 count / positive rate 요약
- `best_trial_test_subgroup_metrics_checkpoint_eval_summary.csv`: 기존 set-wise subgroup AUROC/AUPRC 요약
- `best_trial_test_subgroup_metrics_delong_pooled.csv`: 현재 figure에 사용한 pooled subgroup AUROC + 95% DeLong CI 요약
- `best_trial_test_score_distribution_by_rhythm.csv`: violin plot에 사용한 AFib/AFL/NSR sample-level score table

## 메모

- 현재 `study-af-ablation/configs/`의 최신 YAML은 이번 archived run과 일부 다릅니다.
- 재현과 해석의 기준은 메인 보고서와 archived `run_config.yaml`입니다.
- RFCA TH5 manifest는 sinus-only나 AFib-only가 아니라 mixed-rhythm cohort입니다.
- 현재 best-trial subgroup 성능은 surviving checkpoint artifact 기반 재평가 결과이므로 exploratory view로 해석해야 합니다.
- CI convention: AUROC bar는 95% DeLong CI, sensitivity/specificity bar는 95% Clopper-Pearson CI를 사용합니다.
