# EgoHTR–PromptHMR 평가 파이프라인

EgoHTR 데이터셋의 2nd Aria RGB 영상을 PromptHMR로 복원하고, **SLAM Origin** 좌표계에서 GT와 비교하는 이미지·비디오 평가 도구입니다. PromptHMR 본체는 별도로 설치된 로컬 저장소를 사용하며, 이 저장소는 데이터 로딩, 추론 연결, 좌표 변환, 정량 평가 및 시각화를 담당합니다.


## 1. 저장소 및 입출력 폴더 구조

```text
egohtr-prompthmr-eval/
├── configs/
│   ├── example.yaml                 # 이미지 평가 설정 및 입력 경로
│   └── example_video.yaml           # 비디오 평가 설정 및 입력 경로
├── src/egohtr_phmr_eval/
│   ├── cli.py                       # 설정 로딩, 실행 모드 선택
│   ├── dataset.py                   # 입력 경로 확인, NPZ 프레임 로딩
│   ├── sample.py                    # 이미지용 RGB·GT 추출
│   ├── gt.py                        # GT 관절 및 메시 생성
│   ├── inference.py                 # 이미지 PromptHMR 추론
│   ├── video.py                     # 비디오 추론 및 결과 저장
│   ├── transforms.py                # SLAM Origin 좌표 변환
│   ├── metrics.py                   # MPJPE / PA-MPJPE
│   ├── video_metrics.py             # WA-MPJPE / W-MPJPE
│   ├── gender_fit.py                # 디스플레이용 성별 모델 피팅
│   ├── viewer.py                    # Viser 시각화
│   ├── figure.py                    # 정성 비교 이미지 생성
│   └── crop_figure.py               # 비교 이미지 여백 제거
├── tests/
│   └── test_video_metrics.py
├── runs/                             # 실행 시 생성되는 결과(경로 변경 가능)
├── pyproject.toml
└── README.md

[외부 입력 — 경로 및 파일명 자유]
/data/session_A/capture.npz            # EgoHTR 형식 RGB, GT, 카메라 pose 등
/data/session_A/metadata.yaml          # 시퀀스 설정(SMPL-X gender 등)

[이미지 모드 출력 — output_dir]
├── rgb.png
├── eval_sample.npz
├── gt_body25_mesh.npz
├── output.pt
├── prediction.npz
├── evaluation.npz
├── mpjpe_body25.csv
├── pa_mpjpe_body25.csv
├── mpjpe_body25.txt
├── manifest.json
└── gender_fit/
    ├── fitted_<gender>_prediction.npz
    └── fit_report.json

[비디오 모드 출력 — output_dir]
├── prompthmr_video_work/              # 공식 로더 입력용 중간 MP4 및 작업 파일
├── video_evaluation.npz
├── prediction_vertices_slam_origin.npy
├── video_frame_metrics.csv
├── video_windows.json
├── video_excluded_frames.json
├── video_summary.json
├── prediction_vertices_fitted_<gender>_slam_origin.npy   # 피팅 활성화 시
├── prediction_fitted_<gender>_video.npz                  # 피팅 활성화 시
└── video_gender_fit_<gender>_report.json                # 피팅 활성화 시
```

이 저장소 자체에는 원본 EgoHTR 데이터셋과 PromptHMR 가중치가 포함되지 않습니다. **외부 입력 파일을 읽고, 추론 및 평가 결과를 설정한 `output_dir`에 저장**합니다.

## 2. 입력 파일 지정 방법

이미지와 비디오 모두 같은 입력 경로 설정을 사용합니다.

```yaml
input:
  sequence_npz: /data/session_A/capture.npz
  sequence_config: /data/session_A/metadata.yaml
output_dir: ./runs/experiment_01
```

- `input.sequence_npz`: EgoHTR 형식 배열을 담은 NPZ 파일의 **직접 경로**
- `input.sequence_config`: 시퀀스 설정 YAML의 **직접 경로**. 최소한 `smplx_config.gender`가 필요합니다.
- `output_dir`: 결과를 저장할 경로
- 경로에는 `~` 및 환경 변수를 사용할 수 있습니다.
- `input`의 두 경로는 **함께 지정**해야 합니다. 하나만 지정하거나 파일이 없으면 오류를 반환합니다.
- 지정한 입력 파일은 다운로드하거나 덮어쓰지 않습니다.

### 이전 설정 방식과 호환

예전의 데이터셋 규격 폴더를 사용하는 경우, `input` 항목을 **제거하고** 아래처럼 설정해도 됩니다.

```yaml
dataset_root: ~/Datasets/EgoHTR
sequence: exo-high-object_s4_xhall_01
huggingface_repo_id: null
```

이때만 프로그램이 다음의 **기존 고정 경로**에서 파일을 탐색합니다.

```text
~/Datasets/EgoHTR/sequences/exo-high-object_s4_xhall_01/
├── full_sequence.npz
└── config.yaml
```

구방식에서는 필요한 파일이 없을 때 `huggingface_repo_id`를 지정하면 Hugging Face 데이터셋에서 두 파일만 다운로드하는 기능을 사용할 수 있습니다. 현재 예제 설정은 **직접 입력 경로 방식**을 사용하며 `huggingface_repo_id: null`입니다. 새로운 방식에서는 파일을 미리 준비해야 합니다.

### NPZ 내부에서 읽는 주요 데이터

파일명과 저장 폴더는 자유롭지만, 평가에 필요한 주요 키는 유지되어야 합니다.

- RGB: `aria_obs_rgb_imgs`
- 시간: `aria_timestamp`, `exomc_timestamp`
- 카메라 포즈: `aria_obs_T_origin_device`, `aria_obs_T_device_camera`
- 카메라 내부 파라미터: `aria_obs_rgb_focal_length`, `aria_obs_rgb_img_width`, `aria_obs_rgb_img_height`
- GT: `smplx_exomc_joints`, `smplx_exomc_body_pose`, `smplx_exomc_global_orient`, `smplx_exomc_betas`, `smplx_exomc_transl`

## 3. 설치

로컬 PromptHMR 체크아웃과 CUDA/Python 환경이 먼저 준비되어 있어야 합니다. PromptHMR 체크포인트, YOLO 가중치, SMPL-X 모델 파일도 해당 위치에 있어야 합니다.

```bash
conda activate phmr_pt2.4
cd ~/Codes/egohtr-prompthmr-eval
python -m pip install -e .
```

기본 예제의 `prompthmr_repo`는 `~/Codes/PromptHMR`이며 환경에 맞게 바꿀 수 있습니다.

## 4. 실행 방법

### 이미지 모드

`configs/example.yaml`에서 **두 입력 파일 경로와 `output_dir`를 먼저 수정**하고, 평가할 프레임 `index` 또는 `timestamp_seconds`를 지정합니다. 기본 예제는 high-object 시퀀스의 2150번 프레임을 사용합니다.

```bash
egohtr-phmr-eval --config configs/example.yaml
```

실행 시 인덱스·타임스탬프를 덮어쓰거나 뷰어를 끌 수도 있습니다.

```bash
egohtr-phmr-eval --config configs/example.yaml --index 2150 --no-viewer
egohtr-phmr-eval --config configs/example.yaml --timestamp 71.6666666667 --no-viewer
```

`index`가 있으면 그 인덱스를 사용합니다. `index`가 없고 타임스탬프가 있으면 가장 가까운 프레임을 선택합니다. 예제 설정에는 두 값이 모두 있으므로, 타임스탬프 기반 선택을 원한다면 **YAML의 `index`를 삭제**해야 합니다.

처리 순서:

1. 설정에 지정된 NPZ/YAML 파일을 확인하고 RGB·GT 프레임을 추출합니다.
2. 추출한 RGB를 `rgb.png`, GT 샘플을 `eval_sample.npz`로 저장합니다.
3. Aria 카메라 내부 파라미터를 구성하여 PromptHMR의 Neutral SMPL-X를 추론합니다.
4. 시퀀스 gender에 맞춘 별도 메시 피팅 결과를 생성하고, 예측값을 SLAM Origin으로 변환합니다.
5. **피팅 이전 원본 PromptHMR 관절**로 MPJPE와 PA-MPJPE를 평가하고, 요청 시 Viser를 엽니다.

### 비디오 모드

`configs/example_video.yaml`의 `input.sequence_npz`, `input.sequence_config`, `output_dir`를 수정합니다. `video.start_index`는 포함하고, `video.stop_index`는 **포함하지 않는** 끝 인덱스입니다. 기본적으로 전체 시퀀스를 대상으로 합니다.

```bash
egohtr-phmr-eval --config configs/example_video.yaml
```

일부 연속 프레임만 평가하려면:

```bash
egohtr-phmr-eval --config configs/example_video.yaml --start-index 2140 --stop-index 2160 --no-viewer
```

현재 GitHub 구현의 실제 입력 처리 방식은 다음과 같습니다.

```text
지정된 EgoHTR 형식 NPZ
    ├── RGB 연속 프레임 ─→ 중간 MP4 저장 ─→ PromptHMR 공식 load_frames() 디코딩
    ├── GT 관절 / SMPL-X 파라미터 ─────────────────────────────┐
    └── 카메라 pose / calibration / timestamps ───────────────┤
                                                             ▼
       Detector / SAM2 Tracker → ViTPose → Image Head → Temporal Head
                                           ↓
                     World HPS Estimation → Post-optimization
                                           ↓
                            SLAM Origin 복원 및 GT 비교
                                           ↓
                                    output_dir 저장
```

**주의:** 현재 비디오 모드는 NPZ의 RGB를 임시 MP4로 인코딩하여 `output_dir/prompthmr_video_work/`에 저장한 뒤 공식 로더로 다시 읽습니다. 메모리에서 NPZ 프레임을 바로 PromptHMR에 넘기는 구현이 아닙니다. 별도 JPG 시퀀스를 모델 입력으로 만들지는 않습니다. 소스 프레임 수와 디코딩 프레임 수를 비교하고, 원본 인덱스·타임스탬프를 GT 대응에 사용합니다.

카메라 angular velocity는 EgoHTR pose에서 계산합니다. 공식 `world_hps_estimation`과 `post_optimization`은 유지하되, EgoHTR 카메라 궤적을 기준으로 고정하기 위해 카메라 보정·평활화 및 SPEC 회전은 비활성화합니다. 사람 translation 최적화는 유지하며, 공식 world 단계의 고정 축 회전과 floor offset을 역변환하여 결과를 SLAM Origin으로 되돌립니다. 관련 설정과 변환 기록은 `video_summary.json`에 남깁니다.

평가 대상 사람의 트랙은 GT BODY_25 투영 bbox와 검출·추적 bbox의 IoU로 선택합니다(`video.track_id`로 수동 지정 가능). 트래킹 공백 구간은 temporal 추론에서 분리하고, 제외 프레임과 사유를 기록합니다.

## 5. 좌표계 및 평가 지표

### SLAM Origin

공통 평가 좌표계는 SLAM Origin입니다. 변환 표기는 `T_target_source`를 사용합니다.

```text
RGB 카메라 → Aria 장치:       aria_obs_T_device_camera
Aria 장치 → SLAM Origin:      aria_obs_T_origin_device[index]

T_origin_camera =
    aria_obs_T_origin_device[index] @ aria_obs_T_device_camera
```

이미지 모드에서는 이 행렬을 PromptHMR 카메라 좌표계의 원본 BODY_25 관절과 피팅된 예측 메시에 적용합니다. GT는 저장된 `smplx_exomc_joints`를 `smplx_to_openpose` 인덱스로 BODY_25에 매핑하여, 이미 SLAM Origin에 있다고 가정하고 사용합니다. GT 메시는 저장된 GT SMPL-X 파라미터로 재생성합니다.

원본 GT 관절은 별도의 `T_origin_exoW`, 축 순열, root subtraction 등을 거쳐 보정하지 않습니다. **절대 MPJPE**에도 정렬이나 스케일 보정이 들어가지 않습니다. 단, **PA-MPJPE는 지표 정의상 평가용 복사본을 Procrustes 정렬**합니다.

`aria_obs_T_device_camera`는 2nd Aria RGB 외부 파라미터로 해석합니다. 내부 calibration 비교와 RGB 투영 결과는 이를 지지하지만, 해당 high-object 시퀀스의 VRS calibration으로 직접 검증하지 못했다는 한계는 남아 있습니다. 예제 k=2150의 저장 RGB 정보는 1328×1328, 초점거리 608.4px, 주점 (664, 664)입니다.

### 지표 정의

- **MPJPE:** SLAM Origin에서 정렬하지 않은 예측 관절과 GT 관절의 프레임별 평균 유클리드 거리(mm).
- **PA-MPJPE:** 매 프레임의 예측 BODY_25 관절을 GT에 회전·이동·균일 스케일로 정렬한 뒤 측정(mm). 반사는 허용하지 않습니다.
- **WA-MPJPE(비디오):** 연속된 윈도우 **전체**를 기준으로 하나의 회전·이동 정렬을 구한 뒤 오차 측정(mm).
- **W-MPJPE(비디오):** 각 윈도우의 **처음 2프레임**을 기준으로 회전·이동 정렬을 구하고 동일한 정렬을 해당 윈도우 전체에 적용한 뒤 측정(mm).

현재 구현에서 **WA/W 모두 스케일 정렬을 사용하지 않습니다.** 기본 윈도우는 100프레임이며 겹치지 않는 연속 구간을 평가합니다. 추적이 끊긴 구간은 분리됩니다. 이는 TRAM/WHAM의 전체 윈도우 정렬과 초기 프레임 정렬 방식에 대응하는 구현입니다.

이미지와 비디오 모두 **원본 Neutral PromptHMR 예측 관절**로 지표를 계산합니다. gender fitting은 비교용 메시를 생성할 뿐 지표에 사용하지 않습니다.

## 6. 출력 파일 설명

### 이미지 출력

| 파일 | 내용 |
|---|---|
| `rgb.png` | 평가에 사용한 2nd Aria RGB 프레임 |
| `eval_sample.npz` | 선택 프레임의 원본 GT 관절 및 SMPL-X 파라미터 |
| `gt_body25_mesh.npz` | BODY_25 GT 관절, 재생성한 GT 메시·faces |
| `prediction.npz`, `output.pt` | 원본 Neutral PromptHMR 추론 결과 |
| `gender_fit/` | 시퀀스 gender로 맞춘 별도 표시용 예측 및 피팅 보고서 |
| `evaluation.npz` | 카메라 변환, 예측·GT 관절, 관절별 오차 및 평균 지표 |
| `mpjpe_body25.csv`, `mpjpe_body25.txt` | 관절별/전체 MPJPE 및 텍스트 요약 |
| `pa_mpjpe_body25.csv` | Procrustes 정렬 후 관절별 PA 오차 |
| `manifest.json` | 입력 프레임, 좌표계 가정, 변환, 피팅·평가 정의 |

### 비디오 출력

| 파일 | 내용 |
|---|---|
| `video_evaluation.npz` | 원본 인덱스·타임스탬프, GT·예측 BODY_25, SMPL-X, 카메라 pose, 프레임별 오차 |
| `prediction_vertices_slam_origin.npy` | 평가 대상 프레임의 SLAM Origin 예측 메시 vertex |
| `prediction_vertices_fitted_<gender>_slam_origin.npy` | gender 피팅을 적용한 표시용 메시(선택) |
| `prediction_fitted_<gender>_video.npz` | 표시용 피팅 결과·파라미터(선택) |
| `video_gender_fit_<gender>_report.json` | gender 피팅 설정 및 오차 진단(선택) |
| `video_frame_metrics.csv` | 원본 프레임 인덱스별 MPJPE / PA-MPJPE |
| `video_windows.json` | 구간별 WA/W-MPJPE 및 정렬 변환 |
| `video_excluded_frames.json` | 추적 누락 등 제외된 프레임과 사유 |
| `video_summary.json` | 집계 지표, 트랙 매칭, 카메라 설정, 후처리 내역 |
| `prompthmr_video_work/` | MP4와 공식 로더 작업 파일 |

비디오 결과에는 GT 메시를 다시 생성할 수 있도록 GT 파라미터와 관절을 저장합니다. **비디오용 시간 동기화 Viser 타임라인은 아직 연결되지 않았습니다.**

## 7. 시각화 및 후속 작업

이미지 모드의 Viser는 **예측 메시(빨강), GT 메시(파랑), SLAM Origin 축, 2nd Aria 카메라 pose, RGB 프러스텀**을 표시합니다. 카메라는 OpenCV 좌표축(+X 오른쪽, +Y 아래, +Z 전방)을 따르고 장면은 +Z를 위쪽으로 둡니다.

이미 생성한 평가 결과에서 비교 이미지 PNG를 만들 때는 추론을 다시 실행할 필요가 없습니다.

```bash
egohtr-phmr-figure \
  --evaluation-dir runs/exo-high-object-k3700-qualitative \
  --output runs/exo-high-object-k3700-qualitative/qualitative_comparison.png
```

비교 이미지는 SLAM Origin 측면 시점, 보정된 RGB 메시 투영, GT 기준 앞/오른쪽/왼쪽/뒤 포즈 비교로 구성됩니다. 하단 메시 패널은 동일한 GT pelvis 기준 카메라 대상·방향·정사영 스케일을 사용하며, 메시 좌표는 변경하지 않습니다.

기존 비교 이미지의 하단 메시 패널에서 좌우 여백만 줄이려면:

```bash
egohtr-phmr-crop-figure \
  --figure runs/exo-high-object-k3700-qualitative/qualitative_comparison.png \
  --rgb runs/exo-high-object-k3700-qualitative/rgb.png
```

이 명령은 이미지 후처리만 수행하며, 메시는 다시 렌더링하거나 변환하지 않습니다.

기존 Neutral 예측을 다시 추론하지 않고 별도로 Male SMPL-X에 피팅하려면:

```bash
egohtr-phmr-fit-gender \
  --prediction runs/exo-high-object-k2150-final/prediction.npz \
  --output-dir runs/exo-high-object-k2150-final/gender_fit \
  --prompthmr-repo ~/Codes/PromptHMR \
  --smplx-model-folder ~/Codes/PromptHMR/data/body_models/smplx \
  --gender male --device cuda --pose-prior-weight 1e-3
```

피터는 먼저 원본 Neutral 모델이 저장된 메시를 재생성하는지 확인합니다. 전역 회전·이동은 고정한 채 body pose와 shape를 기존 값에 대한 정규화 항과 함께 최적화합니다. 원본 예측·GT 파일은 수정하지 않습니다.
