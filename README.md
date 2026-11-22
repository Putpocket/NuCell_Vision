# NuCell Vision · Model Evaluation

컴퓨터비전 팀 프로젝트의 공통 평가 저장소다. BBBC038 현미경 이미지에서 핵 영역을 분할하는 고전 영상처리, U-Net, Mask R-CNN 결과를 같은 데이터 분할과 기준으로 비교한다.

## 담당 범위

작성자인 나재민은 평가 코드와 행정, 모델 성능 비교, 성공·실패 사례 분석을 담당한다. 모델 학습 코드는 각 담당자가 관리하며, 평가에는 담당자가 공유한 학습 완료 체크포인트를 사용한다.

## 저장소 구성

```text
.
├── data/                         # 로컬 데이터, Git에 올리지 않음
│   └── stage1_train/
├── splits/                       # 공통 train/val/test 이미지 ID
├── scripts/                      # 데이터 점검 및 분석 보조 스크립트
├── tests/                        # 평가·Mask R-CNN 코드 점검
├── evaluate_unet.py              # U-Net 체크포인트 평가
├── evaluate_maskrcnn.py          # Mask R-CNN 체크포인트 평가
├── evaluate_classical.py         # 고전 기법 예측 및 지표 평가
├── compare_models.py              # 모델별 결과 비교 및 실패 순위 작성
├── make_qualitative_grid.py       # 성공·실패 사례 그림 생성
├── train_unet_improved.py        # U-Net 담당자 학습 코드
├── train_maskrcnn.py             # Mask R-CNN 담당자 학습 코드
├── maskrcnn_model.py
├── maskrcnn_utils.py
├── requirements.txt
└── .gitignore
```

체크포인트, 원본 데이터, 평가 결과는 저장소에 커밋하지 않는다. 큰 파일은 팀에서 합의한 서버나 파일 공유 경로로 전달한다.

## 데이터 준비

[BBBC038 공식 페이지](https://bbbc.broadinstitute.org/BBBC038)에서 `stage1_train.zip`을 내려받아 저장소 루트에 압축을 푼다. 폴더 구조는 다음과 같아야 한다.

```text
data/stage1_train/<이미지 ID>/images/<이미지 ID>.png
data/stage1_train/<이미지 ID>/masks/<핵 ID>.png
splits/train.txt
splits/val.txt
splits/test.txt
```

공통 분할은 학습 469장, 검증 100장, 테스트 101장이다. `splits/` 파일은 모델 간 동일해야 하며, 변경하지 않는다. 테스트 101장은 최종 성능 측정 전에 학습·임계값 선택·개선 실험에 사용하지 않는다.

### 데이터 점검

저장소 루트에서 실행한다.

```powershell
python scripts/check_maskrcnn_data.py
```

점검기는 분할 중복과 학습·검증 이미지 및 마스크를 확인한다. 테스트 이미지와 주석은 열지 않는다. 2026-11-19 로컬 점검에서는 학습 469장·핵 20,059개, 검증 100장·핵 4,949개를 확인했고, 512 입력 변환에서 사라진 마스크는 없었다. 이 결과는 데이터 무결성 확인이며, 모델 평가 결과는 아니다.

## 실행 환경

비교 기준 서버 환경은 Debian 13, Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4, RTX 4060이다. CUDA 휠을 먼저 설치한 뒤 나머지 패키지를 설치한다.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import numpy, torch, torchvision; print(numpy.__version__, torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Windows에서 준비하는 경우에도 Python 3.11 환경을 사용한다. 팀원이 실행한 Windows 환경에서는 PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA 사용 가능(`True`)까지 확인했다. 학습 서버에서의 실행 여부는 서버에서 위 버전·CUDA 확인 명령과 코드 점검을 별도로 수행해 기록한다.

## 학습 및 체크포인트 전달

비교 전에 U-Net 담당자는 `results/unet_new/best.pth`, Mask R-CNN 담당자는 학습 결과 폴더의 `best.pth`를 공유한다. 아래는 예시 경로이며, 파일은 Git에 커밋하지 않는다.

U-Net 학습 예시:

```bash
python -u train_unet_improved.py --loss bce --epochs 100 --patience 15 --min-delta 0.0005 --output results/unet_new
```

Mask R-CNN 학습 예시:

```bash
python -u train_maskrcnn.py --epochs 100 --patience 15 --min-delta 0.0005 --output results/maskrcnn
```

비교 기본 실험은 증강을 끈다. 모델별 배치 크기와 손실은 구조에 맞게 설정한다. 각 학습 기록에 실제 종료 에폭, 최고 검증 Dice 에폭, 검증 Dice, 체크포인트 경로를 남긴다.

## 평가 순서

### 1. 검증 데이터로 실행 확인

공유받은 체크포인트 경로에 맞게 명령을 실행한다. 출력 폴더는 새 경로여야 한다.

```bash
python evaluate_unet.py --checkpoint results/unet_new/best.pth --split val --output results/unet_val --save-images
python evaluate_maskrcnn.py --checkpoint results/maskrcnn/best.pth --split val --output results/maskrcnn_val --save-images
```

### 2. 최종 테스트 평가

두 모델의 설정을 검증 세트에서 확정한 다음 테스트 세트에서 각 모델을 한 번 평가한다. 테스트 결과를 확인해 임계값·전처리·모델 설정을 다시 고르지 않는다.

```bash
python evaluate_unet.py --checkpoint results/unet_new/best.pth --split test --output results/unet_test --save-images
python evaluate_maskrcnn.py --checkpoint results/maskrcnn/best.pth --split test --output results/maskrcnn_test --save-images
```

두 평가기는 이미지 비율을 유지해 긴 변을 512로 맞추고 중앙 패딩한다. 추론 뒤 패딩을 제거해 원본 해상도로 복원하고, 핵 마스크 합집합 기준으로 이미지별 Dice·IoU 및 핵 개수 오차를 계산한다. 처리 시간은 배치 1, 워밍업 뒤 GPU 동기화를 포함하며, 메모리 내 이미지 입력부터 후처리까지 측정하고 디스크 파일 입출력은 제외한다. 모델별 파라미터 수와 평가 조건도 `summary.json`에 남긴다.

각 평가 결과 폴더에는 `per_image.csv`, `summary.json`, `example_ids.json`이 생성된다. `--save-images`를 지정하면 원본 해상도의 예측·정답·overlay도 저장한다.

U-Net 개수 측정은 이진 마스크 합집합에서 8방향 연결요소를 센다. 서로 닿은 핵은 하나로 셀 수 있으므로 개수 지표 해석 시 이 제한을 기록한다. U-Net은 기본 시그모이드 임계값 0.5를 사용하고 Watershed는 적용하지 않는다.

## 고전 기법 포함 성능 비교

고전 기법 파일 `classical_model.py`가 저장소 루트에 있어야 한다. `evaluate_classical.py`는 그 예측 함수를 불러 같은 이미지 ID에서 정답 마스크와 비교하고 표준 결과를 만든다. 결과 폴더에는 `per_image.csv`, `summary.json`, `example_ids.json`이 생성된다. `--save-images`를 지정하면 성공·실패 사례 그림용 원본 크기 예측·정답 마스크도 저장한다. 필수 CSV 열은 `image_id`, `dice`, `iou`, `count_abs_error`, `time_ms`다.

먼저 검증 데이터에서 실행한다. 출력 폴더는 기존에 없는 새 경로여야 한다.

```bash
python evaluate_classical.py --split val --output results/classical_val --save-images
```

설정을 확정한 뒤 최종 테스트를 한 번 실행한다. 테스트 평가에는 `--final-test`를 함께 지정한다.

```bash
python evaluate_classical.py --split test --final-test --output results/classical_test --save-images
```

`compare_models.py`는 모델을 평가하는 코드가 아니라, 각 모델의 표준 평가 결과를 모아 비교하는 코드다.

검증 결과로 설정을 확인하고 비교한다.

```bash
python compare_models.py \
  --model classical=results/classical_val \
  --model unet=results/unet_val \
  --model maskrcnn=results/maskrcnn_val \
  --output results/comparison_val
```

모델 설정을 확정한 뒤 테스트 세트에서 최종 평가를 수행하고 결과를 비교한다. 테스트 결과를 확인한 뒤 설정이나 임계값을 다시 조정하지 않는다.

```bash
python compare_models.py \
  --model classical=results/classical_test \
  --model unet=results/unet_test \
  --model maskrcnn=results/maskrcnn_test \
  --output results/comparison_test
```

비교 도구는 모델 간 split과 이미지 ID가 일치하는지 확인한다. 결과 폴더에 다음 파일을 저장한다.

- `model_comparison.csv`: 평균 Dice·IoU, 핵 개수 MAE, 이미지당 시간, 파라미터 수
- `per_image_comparison.csv`: 동일 이미지에서 모델별 지표
- `failure_analysis.csv`: 모델 평균 Dice가 낮은 순서의 이미지별 결과
- `summary.json`: 사용 split, 요약 지표와 공통 저성능 이미지 ID

## 실패 사례 시각화와 분석

모든 평가기를 `--save-images`와 함께 실행한 후 원본 데이터 및 각 모델 결과 경로를 전달한다.

```bash
python make_qualitative_grid.py \
  --data-dir data/stage1_train \
  --model classical=results/classical_test \
  --model unet=results/unet_test \
  --model maskrcnn=results/maskrcnn_test \
  --output results/qualitative_test
```

모델 평균 Dice가 높은 성공 사례 3장과 낮은 실패 사례 3장을 기본으로 저장한다. 필요하면 `--success-count`와 `--failure-count`로 사례 수를 늘릴 수 있다(각각 최소 3). 출력은 `success_case_1.png`부터 `success_case_3.png`, `failure_case_1.png`부터 `failure_case_3.png`와 선택 정보를 담은 `selected_cases.json`이다. 그림을 확인해 작은 핵, 맞닿은 핵, 흐릿한 경계, 염색·조명 차이, 마스크 손실 등 실제 원인을 구분해 보고서에 기록한다. 개선 실험은 우선 검증 데이터에서 수행하고, 개선 전후 조건과 수치를 기록한 다음 테스트 평가 설정을 고정한다.

## 실험 기록

공통 실행 환경과 데이터 분할·점검 결과는 보고서에 한 번 기록한다. 모델별 체크포인트, 학습 설정과 최고 검증 에폭을 정리한다. 개선 전후 비교는 검증 데이터로 수행하고, 설정 확정 후 테스트 데이터는 한 번 평가해 평균 Dice·IoU, 핵 개수 오차, 처리 시간을 기록한다. 성공·실패 사례는 각 3개 이상 제시하고 원인을 분석한다.

이미지별 상세 결과는 평가 결과 파일에 보관한다. 코드 기여 내역은 GitHub 커밋 기록으로 확인한다.

데이터 출처는 [BBBC038 공식 페이지](https://bbbc.broadinstitute.org/BBBC038)의 인용 안내를 따른다. 데이터셋은 CC0로 제공되며, 데이터와 학습 가중치를 GitHub 저장소에 올리지 않는다.

코드 작성·검토에 OpenAI Codex를 사용했다. 보고서에는 도구와 용도를 기재하고 실행 결과 및 코드 동작은 담당자가 직접 확인·설명한다.
