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
├── classical_model.py            # classic 브랜치에서 가져오는 고전 기법 예측 함수
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

### 학습 환경

모델 학습은 Debian 13, Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4 환경의 서버에서 진행한다. 학습 환경을 설정할 때 CUDA 휠을 먼저 설치한 뒤 나머지 패키지를 설치한다.

### 최종 평가 환경

최종 평가는 Windows 11, Python 3.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA 12.4 환경에서 진행한다.

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

U-Net V1은 데이터 증강 없이 학습하고, V2는 데이터 증강을 적용한다. 두 버전은 같은 검증 데이터로 비교한다. Mask R-CNN도 개선 전후 체크포인트가 있으면 같은 검증 데이터에서 평가한다. 모델별 배치 크기와 손실은 구조에 맞게 설정한다. 각 학습 기록에 실제 종료 에폭, 최고 검증 Dice 에폭, 검증 Dice, 체크포인트 경로를 남긴다.

## 평가 순서

### 1. 검증 데이터 평가 및 개선 비교

공유받은 체크포인트 경로에 맞게 실행한다. 고전 기법은 학습하지 않으며, 기존 설정인 `auto`와 변경 설정인 `bright`를 같은 검증 세트에서 각각 평가하고 결과를 비교한다. U-Net은 V1(증강 없음)과 V2(증강 적용)를 비교한다. Mask R-CNN은 개선 전후 체크포인트가 모두 있을 때 각각 평가하고, 개선 버전이 없으면 기존 결과를 공통으로 사용한다. 기존 출력 경로를 재사용하면 결과 파일이 덮어써진다.

```bash
# 고전 기법 기존·변경 설정 비교
python evaluate_classical.py --split val --polarity bright --output results/classical_bright_val --save-images
python evaluate_classical.py --split val --polarity auto --output results/classical_auto_val --save-images

# U-Net V1·V2
python evaluate_unet.py --checkpoint unet_v1_best.pth --split val --output results/unet_v1_val --save-images
python evaluate_unet.py --checkpoint unet_v2_best.pth --split val --output results/unet_v2_val --save-images

# Mask R-CNN: 체크포인트가 두 버전 모두 있을 때 각각 실행
python evaluate_maskrcnn.py --checkpoint results/maskrcnn_v1/best.pth --split val --output results/maskrcnn_v1_val --save-images
python evaluate_maskrcnn.py --checkpoint results/maskrcnn_v2/best.pth --split val --output results/maskrcnn_v2_val --save-images

# 고전 기법 기존 auto·변경 bright 설정 비교
python compare_models.py \
  --model classical_bright=results/classical_bright_val \
  --model classical_auto=results/classical_auto_val \
  --output results/classical_polarity_comparison_val

# 세 방법 비교: 고전 기법은 검증에서 선택한 auto 설정 사용
python compare_models.py \
  --model classical=results/classical_auto_val \
  --model unet=results/unet_v1_val \
  --model maskrcnn=results/maskrcnn_v1_val \
  --output results/comparison_v1_val

python compare_models.py \
  --model classical=results/classical_auto_val \
  --model unet=results/unet_v2_val \
  --model maskrcnn=results/maskrcnn_v2_val \
  --output results/comparison_v2_val
```

Mask R-CNN V2 체크포인트가 없으면 두 번째 비교 명령의 `maskrcnn` 경로를 `results/maskrcnn_v1_val`로 바꾼다.

### 2. 최종 테스트 평가

검증 결과를 바탕으로 각 방법의 최종 설정과 체크포인트를 확정한 뒤, 테스트 세트에서 각 방법을 한 번 평가한다. 테스트 결과를 확인해 임계값·전처리·모델 설정을 다시 고르지 않는다. 고전 기법은 검증에서 선택한 `auto` 설정을 사용한다. 체크포인트 경로는 최종 선택한 파일 경로로 바꾼다.

```bash
python evaluate_classical.py --split test --final-test --polarity auto --output results/classical_test --save-images
python evaluate_unet.py --checkpoint unet_final_best.pth --split test --output results/unet_test --save-images
python evaluate_maskrcnn.py --checkpoint maskrcnn_final_best.pth --split test --output results/maskrcnn_test --save-images

python compare_models.py \
  --model classical=results/classical_test \
  --model unet=results/unet_test \
  --model maskrcnn=results/maskrcnn_test \
  --output results/comparison_test
```
두 평가기는 이미지 비율을 유지해 긴 변을 512로 맞추고 중앙 패딩한다. 추론 뒤 패딩을 제거해 원본 해상도로 복원하고, 핵 마스크 합집합 기준으로 이미지별 Dice·IoU 및 핵 개수 오차를 계산한다. 처리 시간은 배치 1, 워밍업 뒤 동기화를 포함하며, 메모리 내 이미지 입력부터 후처리까지 측정하고 디스크 파일 입출력은 제외한다. 모델별 파라미터 수와 평가 조건도 `summary.json`에 남긴다.

각 평가 결과 폴더에는 `per_image.csv`, `summary.json`, `example_ids.json`이 생성된다. `--save-images`를 지정하면 원본 해상도의 예측·정답·overlay도 저장한다.

U-Net 개수 측정은 이진 마스크 합집합에서 8방향 연결요소를 센다. 서로 닿은 핵은 하나로 셀 수 있으므로 개수 지표 해석 시 이 제한을 기록한다. U-Net은 기본 시그모이드 임계값 0.5를 사용하고 Watershed는 적용하지 않는다.

## 고전 기법 포함 성능 비교

고전 기법은 Otsu 이진화, 거리 변환, Watershed로 핵 영역을 나누고 개수를 센다. `classical_model.py`가 예측을 담당하고 `evaluate_classical.py`가 같은 이미지 ID의 정답과 비교해 공통 결과 형식으로 저장한다. 결과 폴더에는 `per_image.csv`, `summary.json`, `example_ids.json`이 생성되며, `--save-images` 지정 시 예측·정답·overlay도 저장한다. 필수 CSV 열은 `image_id`, `dice`, `iou`, `count_abs_error`, `time_ms`다.

고전 기법에는 학습이나 데이터 증강이 없다. 기존 `auto` 설정과 변경한 `bright` 설정을 같은 검증 세트에서 비교한다.

- `auto` 기존 설정: 이진화 결과의 테두리에서 흰색 비율을 확인해 필요하면 마스크를 반전한다.
- `bright` 변경 설정: 핵이 밝고 배경이 어둡다고 가정해 Otsu 이진화 결과를 사용한다.
- 두 설정 모두 입력 크기 512, `seed_ratio=0.7`, `min_area=16`을 사용한다. 설정 차이는 `polarity`뿐이다.

검증 100장 비교에서 기존 `auto`의 평균 Dice/IoU/핵 개수 오차는 0.7848/0.6728/13.28이었고, 변경한 `bright`는 0.6772/0.5837/17.95였다. 따라서 이 검증 결과에서는 기존 `auto`가 더 나았다. 이미지별 비교에서는 기존 `auto`가 변경 `bright`보다 Dice가 16장에서 높았고 84장에서는 같았다. 변경 `bright`가 더 높았던 이미지는 없었다. 예를 들어 한 이미지에서는 `auto`의 Dice가 `bright`보다 높았지만, 핵을 과다 분할해 개수 오차는 더 컸다. 정량 결과와 정성 사례를 함께 해석한다.

`evaluate_classical.py`에서 `--polarity bright` 또는 `--polarity auto`를 지정해 평가한다. `--polarity`를 생략하면 기본값은 `auto`다. 검증 비교 명령과 최종 테스트 실행은 [평가 순서](#평가-순서)에 정리했다. 최종 테스트에는 검증에서 선택한 설정을 사용한다.
## 실패 사례 시각화와 분석

### 검증 결과의 설정별 그림 비교

고전 기법 `auto`·`bright`, U-Net V1·V2, Mask R-CNN V1·V2의 검증 결과를 한 이미지에 나란히 표시한다. 기존 `make_qualitative_grid.py`에서 `--model`을 반복해 지정하면 원본·정답과 여섯 설정의 예측이 한 줄에 배치된다. 평가는 같은 검증 이미지 ID를 사용해야 하며, 각 결과 폴더에 `per_image.csv`, `summary.json`, `predictions/`가 있어야 한다.

```bash
python make_qualitative_grid.py \
  --data-dir data/stage1_train \
  --model classical_auto=results/classical_auto_val \
  --model classical_bright=results/classical_bright_val \
  --model unet_v1=results/unet_v1_val \
  --model unet_v2=results/unet_v2_val \
  --model maskrcnn_v1=results/maskrcnn_v1_val \
  --model maskrcnn_v2=results/maskrcnn_v2_val \
  --output results/version_comparison_val \
  --success-count 3 \
  --failure-count 3 \
  --show
```

이 명령은 여섯 설정의 이미지별 평균 Dice가 높은 성공 사례 3장과 낮은 실패 사례 3장을 고른다. 각 그림에는 `원본 | 정답 | classical_auto | classical_bright | unet_v1 | unet_v2 | maskrcnn_v1 | maskrcnn_v2` 순으로 표시하며, `results/version_comparison_val/`에 성공·실패 그림 3장씩과 사례 ID 정보가 저장된다. `--show`를 지정하면 저장이 끝난 뒤 그림이 한 장씩 차례로 열린다. `n` 또는 스페이스바는 다음 그림, `p`는 이전 그림, `q` 또는 `Esc`는 종료다. 화면에는 보기 편하게 축소해 표시하지만 저장된 PNG는 원본 크기다. 이 그림은 검증 데이터의 설정·버전 차이를 살펴보는 용도다.

모든 평가기를 `--save-images`와 함께 실행한 후 원본 데이터 및 각 모델 결과 경로를 전달한다.

```bash
python make_qualitative_grid.py \
  --data-dir data/stage1_train \
  --model classical=results/classical_test \
  --model unet=results/unet_test \
  --model maskrcnn=results/maskrcnn_test \
  --output results/qualitative_test \
  --success-count 3 \
  --failure-count 3 \
  --show
```

공통 요구사항에 맞춰 각 그림은 `원본 | 정답 | 고전 기법 | 모델 A(U-Net) | 모델 B(Mask R-CNN)` 다섯 패널로 구성한다. 모델 평균 Dice가 높은 성공 사례 3장과 낮은 실패 사례 3장을 저장하고, `--show`를 지정하면 시연 중 한 장씩 넘겨 볼 수 있다. 최종 테스트 결과에서 정성 평가 사례를 고르며, 개선 원인은 검증 결과와 실패 사례를 바탕으로 별도로 분석한다.

## 실험 기록

공통 실행 환경과 데이터 분할·점검 결과는 보고서에 한 번 기록한다. 모델별 체크포인트, 학습 설정과 최고 검증 에폭을 정리한다. 개선 전후 비교는 검증 데이터로 수행하고, 설정 확정 후 테스트 데이터는 한 번 평가해 평균 Dice·IoU, 핵 개수 오차, 처리 시간을 기록한다. 성공·실패 사례는 각 3개 이상 제시하고 원인을 분석한다.

이미지별 상세 결과는 평가 결과 파일에 보관한다. 코드 기여 내역은 GitHub 커밋 기록으로 확인한다.

데이터 출처는 [BBBC038 공식 페이지](https://bbbc.broadinstitute.org/BBBC038)의 인용 안내를 따른다. 데이터셋은 CC0로 제공되며, 데이터와 학습 가중치를 GitHub 저장소에 올리지 않는다.

코드 작성·검토에 OpenAI Codex를 사용했다. 보고서에는 도구와 용도를 기재하고 실행 결과 및 코드 동작은 담당자가 직접 확인·설명한다.
