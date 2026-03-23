# 연구 설명

## 1. 연구 질문

이번 실험의 목적은 RFCA 코호트의 12-lead ECG만으로 `LVR05_high`를 예측할 수 있는지 확인하는 것입니다. 라벨은 ablation table의 `LVR05_TotalLB` 기반 이진 타깃으로 구성되어 있고, 실험명에 포함된 `th5`는 threshold 5 기준 코호트를 의미합니다.

실험은 단순 supervised finetuning이 아니라, 사전학습된 self-supervised encoder를 초기값으로 사용한 뒤 downstream binary classification 성능을 보는 구조입니다. 이번 결과는 "ECG foundation representation이 AF ablation 관련 structural burden 신호를 어느 정도 포착하는가"를 보는 pilot/preliminary 단계로 해석하는 것이 적절합니다.

## 2. 데이터셋과 코호트

- manifest: `/data/projects/ai-ecg/experiments/af_ablation_202602/manifests/finetune_lvr05_high_rfca_th5.parquet`
- 입력 신호: 12-lead ECG, `signal_length=2250`, `target_sample_rate=250`, 9초 crop
- split: `train/valid/test`
- 샘플링 단위: `pid` 그룹 기준 without replacement

이번 hparam search에서 사용된 각 sample set의 크기는 아래와 같습니다.

| split | rows per set |
| --- | ---: |
| train | 1677 |
| valid | 142 |
| test | 142 |

스터디 저장소의 RFCA manifest 통계 문서를 함께 보면, 전체 RFCA zarr 코호트는 1961 ECG, 713 PID 규모이며 test split의 양성률은 약 15.5%입니다. train split은 반복 ECG를 포함하고, valid/test는 PID당 1개 ECG로 구성되어 있습니다. 따라서 train과 evaluation의 분포 구조가 완전히 동일하지는 않습니다.

## 3. 모델과 학습 설정

- backbone: ViT 계열 encoder (`embed_dim=768`, `depth=12`, `heads=12`)
- SSL method: `mtae`
- SSL pretrain checkpoint: `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_pretrain_emory_i0001_mtae_echonext_longitudinal_gap3_pretrain_20260311_024344/checkpoints/last.ckpt`
- finetune epochs: 10
- optimizer search space 핵심 축:
  - `finetune.lr`
  - `finetune.weight_decay`
  - `finetune.layer_decay`
  - `data.batch_size`
  - `finetune.num_freeze_layers`

전처리는 random crop, standardization, 그리고 `shift/cutout/drop/flip/erase/sine/partial_sine/partial_white_noise`를 포함한 augmentation을 사용했습니다. evaluation은 고정 crop 기준으로 수행됐습니다.

## 4. 이번 실험에서 중요하게 봐야 할 점

첫째, 하이퍼파라미터 탐색의 objective가 `test_macro_auroc`입니다. 즉, test 성능을 기준으로 best trial을 고른 구조이므로 이번 수치는 최종 결론이 아니라 exploratory/preliminary 결과로만 써야 합니다.

둘째, 결과 디렉토리의 `RECOVERY_NOTES_20260317.txt`에 따르면 이 검색은 원래 `trial_limit=25`로 설정됐지만 실제로는 19개 trial만 남아 있고, 그중 `trial_018`은 일부 artifact가 불완전합니다. 다만 best trial인 `trial_004`는 10개 sample set이 모두 성공적으로 평가되었습니다.

셋째, 현재 스터디 저장소의 `configs/hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5.yaml`는 더 작은 탐색 공간과 다른 pretrained checkpoint를 가리킵니다. 따라서 재현 또는 후속 실험은 현재 config 파일이 아니라 본 결과 디렉토리의 `run_config.yaml`을 기준으로 잡아야 합니다.

## 5. 해석 프레임

이번 결과는 다음 질문에 대한 1차 답으로 보는 것이 적절합니다.

1. RFCA 코호트의 ECG 기반 representation만으로 `LVR05_high`를 분리할 수 있는가
2. MTAE pretraining이 downstream AUROC를 0.8 이상으로 끌어올릴 정도의 signal을 주는가
3. 제한된 RFCA evaluation set에서도 성능이 sample set 간 크게 무너지지 않는가

즉, 이 연구는 "최종 clinical claim"보다 "ECG-derived latent representation이 ablation-relevant remodeling burden을 반영할 가능성"을 탐색하는 단계에 가깝습니다.
