# 고전 기법: 세포 핵 분할과 개수 세기

BBBC038 현미경 이미지에 Otsu 이진화, 거리 변환, 마커 기반 Watershed를 적용해 세포 핵을 분리하고 개수를 셉니다. 학습 없이 실행하는 고전 영상처리 방법입니다.

## 담당 범위와 파일 구조

| 파일 | 역할 |
| --- | --- |
| `classical_model.py` | 이미지 한 장을 처리하는 고전 기법과 평가 담당자용 예측 함수 |
| `predict_classical.py` | 공통 분할 목록의 이미지를 읽고 예측 마스크·핵 개수·처리 시간을 저장하는 실행 파일 |
| `README_CLASSICAL.md` | 실행 방법, 알고리즘, 결과 형식과 평가 코드 연결 방법 |

세 파일은 프로젝트 최상위, 즉 기존 `README.md`와 같은 위치에 둡니다. 기존 `scripts/`는 데이터 점검 등 보조 코드를 담는 폴더입니다.

고전 기법 담당자는 분할 알고리즘, 핵 개수와 예측 결과 제공, 개선 방법 및 실패 원인 설명을 담당합니다. 정답 로딩, IoU·Dice·개수 오차 계산, 모델 비교표와 공통 사례 그림은 평가 담당 코드에서 처리합니다.

## 실행 환경과 데이터

Python 3.11 환경에서 다음 패키지를 사용합니다.

```powershell
python -m pip install opencv-python numpy pillow matplotlib
```

이미지는 팀 U-Net·Mask R-CNN과 동일하게 Pillow로 RGB uint8 배열로 읽습니다. 데이터는 아래 구조로 준비합니다.

```text
data/stage1_train/<이미지 ID>/images/<이미지 ID>.png
data/stage1_train/<이미지 ID>/masks/<핵 ID>.png
splits/train.txt
splits/val.txt
splits/test.txt
```

공통 분할은 학습 469장, 검증 100장, 테스트 101장입니다. 분할 파일은 변경하지 않습니다. 예측 코드는 세 목록의 개수·중복 여부를 검사하고, 기본 실행에서는 검증 이미지 100장만 엽니다. 예측 코드와 분할 함수는 정답 마스크를 읽지 않습니다.

## 실행 방법

프로젝트 최상위 폴더의 터미널에서 실행합니다.

```powershell
python predict_classical.py --save-stages
```

기본 결과 경로는 `results/classical_val_auto_predictions`입니다. 기존 결과를 덮어쓰지 않으므로 같은 폴더가 있으면 새 경로를 지정합니다.

```powershell
python predict_classical.py --save-stages --output results/classical_val_auto_run3
```

`[100/100]`과 `Saved 100 predictions: ...`가 출력되면 정상 완료입니다. 폴더가 생기는 시점은 실행 도중이므로 폴더 존재만으로 완료를 판단하지 않습니다. 그림은 창을 띄우지 않고 파일로 저장합니다.

## 처리 과정

| 단계 | 처리 내용 |
| --- | --- |
| 입력 크기 조정 | 비율을 유지해 긴 변을 512로 맞추고 중앙에 검은 패딩을 넣어 512×512로 만듭니다. 축소는 INTER_AREA, 확대는 INTER_LINEAR를 사용합니다. |
| 회색조·잡음 완화 | 유효한 이미지 영역을 회색조로 바꾸고 3×3 Gaussian 필터를 적용합니다. 패딩은 Otsu 계산에서 제외합니다. |
| Otsu | 밝기 분포에서 임계값을 자동으로 계산해 핵 후보와 배경을 나눕니다. |
| 후보 정리 | 3×3 타원 커널의 opening과 최소 면적 기준으로 작은 잡음을 정리합니다. |
| 거리 변환 | 핵 후보 내부에서 가장 가까운 배경까지의 거리를 계산합니다. |
| 중심 마커 | 연결된 덩어리마다 최대 거리의 70%를 넘는 영역을 중심 후보로 사용하고 번호를 붙입니다. |
| Watershed | 중심·배경·미지 영역의 마커로 cv2.watershed를 실행해 핵 후보를 분리합니다. |
| 복원·개수 계산 | 패딩을 제거하고 핵별 번호를 최근접 보간으로 원본 해상도로 복원한 뒤, 남아 있는 핵 번호를 셉니다. |

핵 중심 후보가 적으면 여러 핵이 합쳐지고, 너무 많으면 핵 하나가 여러 개로 나뉠 수 있습니다. 흐릿하거나 밀집한 사진, 세포 주변 영역과 밝기가 비슷한 사진에서 오차가 남을 수 있습니다.

## 기본 방법과 개선 방법

| 설정 | 동작 |
| --- | --- |
| `--polarity bright` | 핵은 밝고 배경은 어둡다고 가정해 Otsu 결과를 사용합니다. |
| `--polarity auto` | 기본값입니다. 이진화 영상 테두리의 흰색 비율이 50%를 넘으면 반전합니다. |

`auto`는 밝은 배경을 핵으로 잘못 선택하는 문제를 줄이기 위한 개선 규칙입니다. 입력 영상만 사용하며 정답을 보고 방향을 고르지 않습니다. 두 방법의 필터, 중심 임계 비율과 최소 면적은 동일합니다. 테두리에 핵이 많이 걸친 사진에서는 반전 판단이 틀릴 수 있습니다.

개선 전 예측은 다음처럼 별도 폴더에 저장해 평가 담당자에게 함께 전달할 수 있습니다.

```powershell
python predict_classical.py --polarity bright --output results/classical_val_bright_predictions
```

`--seed-ratio` 기본값은 0.7, `--min-area` 기본값은 512×512 처리 공간의 16픽셀입니다. 설정 조정과 개선 전후 비교는 검증 세트에서 수행합니다.

## 저장 결과

| 경로 | 내용 |
| --- | --- |
| `predictions/<ID>/prediction.png` | 원본 크기의 핵 전체 마스크. 핵=255, 배경=0 |
| `predictions/<ID>/instances.png` | 원본 크기의 uint16 핵별 번호 이미지. 배경=0, 핵=1부터 서로 다른 번호 |
| `predictions/<ID>/stages.png` | `--save-stages` 사용 시 원본·Otsu·거리 변환·중심 마커·Watershed의 5칸 그림 |
| `inference.csv` | 이미지 ID, 예측 개수, 처리 시간, 결과 파일 경로, 임계값·반전 여부 |
| `manifest.json` | 분할, 알고리즘 설정, 입력 크기, 패키지 버전, 분할 파일 해시, 측정 조건 |

`inference.csv`는 예측 기록입니다. 정답 개수, IoU·Dice·개수 오차와 모델 간 비교는 공통 평가 단계에서 계산합니다. 데이터·결과 이미지는 기존 `.gitignore`에 따라 Git에 올리지 않고 팀이 정한 공유 경로로 전달합니다.

## 평가 담당자용 인터페이스

```python
from classical_model import predict_classical

mask, count, time_ms = predict_classical(
    image, polarity="auto", size=512, seed_ratio=0.7, min_area=16
)
```

| 입력·반환값 | 형식 |
| --- | --- |
| `image` | 원본 H×W×3, RGB 순서, NumPy uint8 배열 |
| `mask` | 원본 H×W 크기의 bool 배열. 핵=True |
| `count` | 원본 해상도에 남은 Watershed 인스턴스 수, int |
| `time_ms` | 전처리·분할·원본 크기 복원·개수 계산 시간, float |

내부 `segment_classical()`은 핵별 번호 배열과 단계별 결과를 반환합니다. `predict_classical()`은 이를 공통 평가에서 사용할 마스크·개수·시간으로 정리해 반환합니다.

평가 담당자는 기존 공통 평가 루프에서 다음처럼 연결할 수 있습니다. `score`는 평가 담당자의 기존 함수를 사용합니다.

```python
mask, count, elapsed_ms = predict_classical(image)
metrics = score(gt, mask, gt_count, count)
row = {"image_id": image_id, **metrics, "time_ms": elapsed_ms}
```

저장된 결과를 받아 평가할 때는 `inference.csv`의 개수와 시간, `predictions/<ID>/prediction.png`를 사용합니다. 이진 마스크의 연결요소를 다시 세면 Watershed 인스턴스 수와 달라질 수 있으므로 제공한 `count` 또는 인스턴스 라벨을 사용합니다.

현재 `model-evaluation` 브랜치의 공통 비교 코드는 `per_image.csv`의 `image_id`, `dice`, `iou`, `count_abs_error`, `time_ms` 열을 읽습니다. 평가 담당자가 고전 기법 함수를 연결해 이 CSV와 `summary.json`, 사례 그림용 정답 마스크를 생성해야 합니다. 예측 출력만으로 공통 비교 명령이 바로 실행되는 것은 아닙니다.

## 시간 측정과 최종 테스트

예측 실행 파일은 OpenCV 스레드 수를 1로 설정하고 첫 이미지에서 10회 워밍업한 뒤 이미지당 1회 시간을 측정합니다. 시간 범위는 메모리 내 RGB 입력부터 전처리·분할·복원·개수 계산까지이며 파일 입출력, 정답, 지표, 시각화는 제외합니다. 최종 비교 시간은 팀 평가 환경에서 같은 기준으로 측정합니다.

`PARAMETER_COUNT = 0`은 학습하는 가중치가 없다는 뜻입니다. 임계 비율·최소 면적 등 설정값이 없다는 의미는 아닙니다.

검증 세트에서 방법과 설정을 확정한 뒤 평가 담당자와 최종 테스트를 진행합니다. 실행 파일은 테스트 예측에 `--final-test`를 요구합니다.

```powershell
python predict_classical.py --split test --final-test --polarity auto --output results/classical_test_auto_predictions
```

## 참고

- [BBBC038 데이터셋](https://bbbc.broadinstitute.org/BBBC038)
- [OpenCV 거리 변환·Watershed 공식 예제](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
- [팀 공통 평가 코드](https://github.com/Putpocket/NuCell_Vision/tree/model-evaluation)

코드 초안 작성, 환경 오류 해결과 인터페이스 점검에 ChatGPT/Codex를 사용했습니다. 담당자는 처리 과정과 실패 원인을 직접 확인·설명하고 보고서에 도구와 사용 용도를 기록합니다.
