# Code Trace

## 1. 실제 실행 entrypoint

현재 study-local wrapper는 아래 스크립트입니다.

- `/data/projects/study-af-ablation/scripts/run_ssl_pilot_rfca_threshold_sweep.sh:19-27`

이 스크립트는 두 단계만 합니다.

1. `build_finetune_manifest_rfca_zarr.py --label-threshold 5`로 `finetune_lvr05_high_rfca_th5.parquet` 생성
2. `python -m cli hparam-search --config .../hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5.yaml` 실행

즉, 실험 orchestration은 `study-af-ablation`에 있지만 실제 hparam-search, finetune, eval 로직은 `/data/projects/ai-ecg/src/cli.py`에 있습니다.

## 2. config가 실제로 합쳐지는 방식

YAML inheritance는 `/data/projects/ai-ecg/src/config/loader.py:11-64`에서 처리됩니다.

- `inherit_from`이 있으면 부모 YAML을 먼저 읽습니다.
- dict는 deep merge 됩니다.
- 자식 YAML 값이 부모 값을 override 합니다.

현재 study-local th5 config 체인은 다음입니다.

1. `/data/projects/study-af-ablation/configs/hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5.yaml`
2. `/data/projects/study-af-ablation/configs/finetune_lvr05_high_rfca_ssl_i0001_mtae.yaml`
3. `/data/projects/study-af-ablation/configs/finetune_lvr05_high_rfca.yaml`
4. `/data/projects/ai-ecg/configs/base.yaml`

하지만 이번 결과는 현재 저장소 YAML과 완전히 일치하지 않습니다. 실제 실행본은 결과 디렉토리의 archived `run_config.yaml`을 봐야 합니다.

## 3. 현재 config와 archived run의 중요한 차이

현재 study-local th5 config:

- `trial_limit: 8`
- `n_sample_sets: 3`
- search space는 `weight_decay=[0, 0.01]`, `batch_size=[128,256]`, `num_freeze_layers=[11,12]`
- pretrained checkpoint는 LEJEPA 계열 checkpoint

근거:

- `/data/projects/study-af-ablation/configs/hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5.yaml:8-31`
- `/data/projects/study-af-ablation/configs/finetune_lvr05_high_rfca_ssl_i0001_mtae.yaml:5-19`

반면 이번 결과의 archived run config:

- `trial_limit: 25`
- `n_sample_sets: 10`
- search space는 `lr=[0.03,0.01,0.001,0.0003,0.0001]`, `weight_decay=[0.01,0.05]`, `layer_decay=[0.5,0.8,0.9,1.0]`, `batch_size=[16,32,64,128]`, `num_freeze_layers=[0,6,9]`
- pretrained checkpoint는 MTAE checkpoint

근거:

- `/data/projects/ai-ecg/outputs/af_ablation_202602/af_ablation_202602_hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5_hparam_search_20260316_001959/run_config.yaml:222-285`

즉 지금 `configs/`를 다시 실행하면 같은 실험이 아닙니다.

## 4. hparam-search가 실제로 하는 일

핵심 로직은 `/data/projects/ai-ecg/src/cli.py:1240-1437`입니다.

실행 순서는 아래와 같습니다.

1. merged config 로드
2. base manifest 로드 후 validation
3. `n_sample_sets` 만큼 sampled manifest 생성
4. grid search이면 search space를 전개하고 shuffle
5. 각 trial마다 `n_sample_sets`개의 finetune run 실행
6. 각 set의 objective를 평균내어 `objective_mean` 계산
7. 최고 mean trial을 `best_trial.yaml`로 저장

중요한 디테일:

- sampled manifest 생성은 `/data/projects/ai-ecg/src/cli.py:820-856`입니다.
- `group_column=pid`, `replacement=false`이면 PID 단위로 샘플링합니다.
- 이번 archived run은 `train_frac=valid_frac=test_frac=1.0`이라 사실상 각 split 전체를 그대로 사용합니다. 그래서 `sampled_set_sizes.csv`가 모든 set에서 동일합니다.

## 5. objective metric이 실제로 어디서 오나

이 부분이 가장 중요합니다.

- trial 실행은 `/data/projects/ai-ecg/src/cli.py:912-1043`
- 각 set에서 `run_finetune()`을 호출한 뒤 `/data/projects/ai-ecg/src/cli.py:866-881`의 `_extract_objective_metric()`으로 점수를 읽습니다.
- 이 함수는 먼저 `test_metrics.yaml`을 보고, 없으면 `csv/metrics.csv`의 마지막 row를 봅니다.

즉 `objective_metric: test_macro_auroc`는 선언만 있는 게 아니라, 실제 ranking이 test 결과 파일을 직접 읽어서 결정됩니다.

## 6. finetune가 실제로 하는 일

진입점은 `/data/projects/ai-ecg/src/cli.py:1090-1133`입니다.

- `ECGDataModule(cfg["data"])` 생성
- `FinetuneBinaryLitModule(...)` 생성
- `trainer.fit(...)`
- `run_test_after_fit=true`면 `trainer.test(...)`
- test 결과를 `test_metrics.yaml`에 저장

즉 hparam-search는 별도 evaluator를 쓰는 게 아니라, finetune 끝난 직후 같은 run에서 test를 돌린 결과를 가져다 ranking합니다.

## 7. model 내부 구현

실제 classification module은 `/data/projects/ai-ecg/src/modeling/binary_classification/finetune_binary_lit.py:18-308`입니다.

핵심 구현은 다음과 같습니다.

- encoder는 `ViT`
- classifier는 `Linear(embed_dim, num_targets)`
- `pretrained_checkpoint`가 있으면 checkpoint의 `encoder.*` key만 로드
- `num_freeze_layers > 0`이면 patch embedding, pos embedding, cls token, 초기 block들을 freeze
- `layer_decay < 1.0`이면 layer별로 learning rate scale을 다르게 줌
- validation epoch end에서 `val_macro_auroc` 계산
- test epoch end에서 `test_macro_auroc`, `test_macro_auprc` 계산

구체 위치:

- checkpoint loading: `:55-65`
- early-layer freezing: `:67-81`
- validation AUROC logging: `:141-158`
- test AUROC/AUPRC logging: `:163-184`
- layer-wise lr decay optimizer grouping: `:259-308`

## 8. 데이터가 실제로 읽히는 방식

데이터 split과 dataloader는 `/data/projects/ai-ecg/src/preprocessing/datamodule.py:167-301`입니다.

- manifest를 읽고 `train/valid/test`로 분리
- train은 shuffle=True, valid/test는 shuffle=False
- 각 row는 `ECGDataset.__getitem__()`에서 reader를 통해 waveform을 읽어 tensor로 변환

zarr reader는 `/data/projects/ai-ecg/src/preprocessing/readers/zarr_reader.py:19-165`입니다.

- manifest row에서 `zarr_store`, `zarr_group`, `ds_idx`, `ds_row_idx`를 읽음
- zarr array에서 waveform 1개를 꺼냄
- shape을 `[12, T]`로 정리
- `value_scale=4.0` 적용
- preprocessor를 통해 crop/standardize/augmentation 처리

즉 문서상의 "12-lead zarr ECG"는 실제로 이 reader 경로를 통해 로드됩니다.

## 9. 이번 결과를 코드 기준으로 한 줄 요약

이번 결과는 "현재 study-local YAML"의 산출물이 아니라, archived `run_config.yaml` 기준으로 `python -m cli hparam-search`가 `test_macro_auroc`를 직접 읽어 ranking한 grid search 결과입니다. 다시 말해, 이 결과를 해석하거나 재현하려면 먼저 문서가 아니라 archived run config와 `ai-ecg/src/cli.py`, `finetune_binary_lit.py`를 기준으로 봐야 합니다.
