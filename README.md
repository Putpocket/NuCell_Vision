# NuCell Vision · Mask R-CNN

컴퓨터비전 미니 프로젝트의 세포 핵 분할·개수 세기를 위한 Mask R-CNN 코드다. `torchvision.models.detection.maskrcnn_resnet50_fpn`을 사전학습 가중치에서 미세조정한다. 핵마다 마스크와 박스를 예측하며, 클래스는 배경 0과 핵 1이다.

이 브랜치는 학습 실행 전의 코드다. 학습 가중치나 성능 수치는 포함하지 않는다. 실제 학습과 GPU 검증은 가상서버에서 진행한다.

작성 시 기존 로컬 환경의 PyTorch 2.5.1 / torchvision 0.20.1에서 CPU 테스트 20개를 통과했다. 별도의 합성 가중치 점검에서는 체크포인트 저장·복원 후 예측이 정확히 일치함을 확인했다. 학습과 사전학습 가중치 다운로드는 실행하지 않았다. 목표 서버의 PyTorch 2.6.0 / torchvision 0.21.0 및 GPU에서의 실행은 아직 검증하지 않았으므로 아래 점검 명령을 서버에서도 먼저 실행한다.

## U-Net과 공유하는 조건

기준은 [U-net 브랜치의 README](https://github.com/Putpocket/NuCell_Vision/blob/fb311d4ac7533f58e73991570cca1668c41cc7da/README.md)와 해당 커밋의 학습 코드다. `splits/`의 이미지 ID와 `train_unet_improved.py`는 그대로 유지했다.

| 항목 | 조건 |
|---|---|
| 데이터 | BBBC038 / 2018 Data Science Bowl `stage1_train` |
| 분할 | 학습 469장, 검증 100장, 테스트 101장. 기존 `splits/*.txt` 사용 |
| 입력 | U-Net과 같은 Pillow RGB 변환, 실제 모델 입력 512×512 |
| 크기 변환 | 비율 유지, 긴 변 512, 중앙 배치, 검은색 패딩 |
| 보간 | 이미지 축소 `INTER_AREA`, 확대 `INTER_LINEAR`, 마스크 `INTER_NEAREST_EXACT` |
| 비교 실험 증강 | 끔. `--augment`는 별도 개선 실험 |
| 학습 | 최대 100에폭, 검증 Dice 최고 가중치 선택 |
| 조기 종료 | patience 15, min-delta 0.0005. 실제 종료·최고 에폭 기록 |
| 학습 중 검증 | 512×512에서 패딩 제외, 이미지별 Dice·IoU 평균 |
| 최종 테스트 | 원본 해상도로 복원한 핵 영역의 이미지별 Dice·IoU 평균 |
| 개수 | 원본 정답 핵 수와 유효한 예측 인스턴스 수의 MAE |
| 속도 | 같은 RTX 4060, 배치 1, 워밍업 10회, GPU 동기화, 전·후처리 포함, 파일 입출력 제외 |

U-Net의 배치 4와 달리 Mask R-CNN의 기본 배치는 메모리를 고려해 1이다. 손실은 모델이 반환하는 분류·박스·RPN·마스크 손실의 합이다. AdamW, 학습률 0.0001, weight decay 0.01, seed 42를 기본으로 둔다. Mask R-CNN의 사전학습 가중치는 COCO 기반이므로 U-Net의 ImageNet 인코더와 사전학습 출처가 다르다는 점도 보고서에 적는다.

## 서버 준비

기준 서버는 Debian, RTX 4060, PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4다. Python 3.10 또는 3.11의 별도 가상환경을 권장한다. 아래는 서버에서 실행할 명령이다.

```bash
git clone --branch Mask-RCNN --single-branch https://github.com/Putpocket/NuCell_Vision.git
cd NuCell_Vision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-maskrcnn.txt
```

서버에 이미 호환 환경이 있으면 그 환경을 사용해도 된다. headless OpenCV를 사용하므로 이 가상환경에 다른 OpenCV wheel을 중복 설치하지 않는다. 학습 첫 실행에는 torchvision 사전학습 가중치를 다운로드할 네트워크 연결이 필요하다.

## 데이터 준비와 실행 전 점검

[BBBC038 공식 페이지](https://bbbc.broadinstitute.org/BBBC038)에서 `stage1_train.zip`을 받아 아래처럼 압축을 푼다. 데이터는 Git에 올리지 않는다.

```text
data/stage1_train/<이미지 ID>/images/<이미지 ID>.png
data/stage1_train/<이미지 ID>/masks/<핵 ID>.png
splits/train.txt
splits/val.txt
splits/test.txt
```

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -v
python scripts/check_maskrcnn_data.py
```

테스트는 합성 이미지로 전처리·복원·평가를 점검하고, PyTorch가 있으면 모델의 512 입력 유지·정규화·두 예측 헤드·체크포인트 구조와 Dataset 계약을 추가 확인한다. 사전학습 가중치 다운로드나 학습을 실행하지 않는다. PyTorch가 없는 환경에서는 Torch 계약 테스트가 skip된다.

데이터 점검은 학습·검증 이미지만 읽는다. 분할 중복, 이미지·마스크 오류와 512 리사이즈에서 사라지는 핵을 확인한다. 크기 변환 후 빈 마스크만 학습 target에서 제외하며, 원본 해상도 테스트의 정답 개수에는 원래 핵을 모두 포함한다. 기본 검출 상한은 이미지당 500개다. 학습·검증의 핵 개수 분포를 보고 필요하면 학습 전에 `--max-detections`를 조정한다. 후보 영역 제한 등 모델 내부 설정에 따라 이 상한보다 적게 나올 수도 있다.

## 기본 비교 학습

```bash
python -u train_maskrcnn.py --epochs 100 --patience 15 --min-delta 0.0005 --batch-size 1 --output results/maskrcnn_baseline
```

서버에서 백그라운드로 실행할 때는 `tmux` 등 세션 유지 도구를 사용한다. GPU 메모리가 부족하면 다른 GPU 프로세스를 먼저 확인한다. 입력 크기는 U-Net과의 비교 조건이므로 임의로 줄이지 않는다.

출력 폴더는 새 이름이어야 한다. 기존 실험을 덮어쓰지 않는다.

- `best.pth`: 검증 Dice가 가장 높은 에폭의 모델·설정.
- `last.pth`: 마지막으로 완료한 에폭의 모델·설정. optimizer/RNG를 포함하지 않아 학습 재개용은 아니다.
- `config.json`: 하이퍼파라미터, 환경, 파라미터 수와 분할 파일 SHA-256.
- `history.csv`: 에폭별 전체·개별 손실, 검증 Dice·IoU, 사라진 학습 마스크 수, 소요 시간.
- `summary.json`: 실제 종료 에폭, 최고 에폭과 검증 성능.

체크포인트 선택은 사소한 Dice 상승도 반영하며, 조기 종료의 개선 판정만 min-delta를 사용한다. 이는 U-Net 코드와 같은 방식이다. 테스트 이미지는 학습·체크포인트 선택에 사용하지 않는다.

## 검증 및 최종 테스트

검증 데이터에서 원본 해상도 결과와 개수 오차를 먼저 확인한다.

```bash
python evaluate_maskrcnn.py --checkpoint results/maskrcnn_baseline/best.pth --split val --output results/maskrcnn_val --save-images
```

설정을 확정한 다음에만 테스트 101장을 평가한다.

```bash
python evaluate_maskrcnn.py --checkpoint results/maskrcnn_baseline/best.pth --split test --output results/maskrcnn_test --save-images
```

평가 시에는 사전학습 가중치를 다운로드하지 않는다. COCO 모델과 같은 FrozenBatchNorm2d와 epsilon 0으로 backbone을 구성한 뒤 체크포인트를 불러온다. 체크포인트의 모델 설정과 threshold를 사용하고, 분할 파일 해시가 학습 때와 다르면 중단한다.

마스크 확률에서 패딩을 제거한 뒤 `INTER_LINEAR`로 원본 해상도로 복원하고, `probability > 0.5`로 이진화한다. 기본 객체 신뢰도 기준은 0.5다. 신뢰도와 마스크 기준은 서로 다르며, 변경하려면 검증 데이터에서 결정한 값을 학습 명령의 `--score-threshold`, `--mask-threshold`로 명시해 별도 실험으로 기록한다. 테스트 결과를 보고 조정하지 않는다.

남은 마스크의 합집합으로 Dice·IoU를 계산하고, 패딩 제거·크기 복원 후 비어 있지 않은 마스크 수로 핵 개수를 센다. 원본 정답은 핵별 마스크를 합친 영역과 핵별 마스크 파일 수다. 예측이 없으면 빈 영역·개수 0으로 평가한다. U-Net도 최종 평가에서 동일한 복원 순서·보간·이진화·지표 함수를 적용해야 한다. 현재 U-Net 학습 코드의 512 검증 수치를 원본 해상도 테스트 수치와 섞어 비교하지 않는다.

- `per_image.csv`: 이미지별 Dice·IoU·정답/예측 개수·개수 절대 오차·처리 시간.
- `summary.json`: 평균 Dice·IoU·개수 MAE·평균/중앙 처리 시간·파라미터 수·장치·평가 조건.
- `example_ids.json`: Dice 상위 3장과 하위 3장. 세 방법을 같은 이미지로 비교할 때 활용한다.
- `predictions/<ID>/`: `--save-images`일 때 원본 크기의 예측/정답 이진 마스크, 16비트 인스턴스 라벨, 예측 overlay. 라벨 영상에서 겹치는 부분은 나중 객체가 덮지만 CSV 개수는 덮기 전 유효 인스턴스 수다.

시간 측정은 메모리에 있는 RGB 이미지부터 시작해 letterbox, CPU→GPU, 모델 추론, GPU→CPU, 원본 크기 복원, 이진화, 합집합·개수 계산까지 포함한다. 파일 디코딩·저장·지표 계산·시각화는 제외한다. U-Net 측정에도 같은 범위를 적용한다. 두 모델의 총 파라미터 수를 비교하고 학습 가능 파라미터 수는 별도 참고값으로 기록한다.

## 개선 실험

기본 비교를 마친 뒤 학습 데이터에 좌우·상하 뒤집기와 90도 회전을 추가할 수 있다. 이미지와 모든 핵 마스크에 같은 변환을 적용하고 박스를 다시 계산한다.

```bash
python -u train_maskrcnn.py --epochs 100 --patience 15 --min-delta 0.0005 --augment --output results/maskrcnn_augment
```

나머지 조건을 유지하고 검증 Dice·IoU·개수 MAE의 전후 차이를 비교한다. 개선 여부가 확인되기 전에는 성능 향상을 주장하지 않는다.

## 파일 구성과 출처

- `maskrcnn_utils.py`: U-Net과 같은 letterbox, 원본 복원, 마스크 합집합, Dice·IoU, 분할 검증.
- `maskrcnn_model.py`: 핵별 Dataset, 배치 묶기, 사전학습 모델과 두 예측 헤드 교체.
- `train_maskrcnn.py`: 학습, 검증, 조기 종료, 체크포인트와 기록.
- `evaluate_maskrcnn.py`: 원본 해상도 분할·개수·속도 평가와 결과 저장.
- `scripts/check_maskrcnn_data.py`, `tests/`: 학습 전 데이터·코드 점검.

모델 구성은 [PyTorch 공식 Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)을 참고했다. letterbox 규칙과 공통 조건은 [팀 U-Net 코드](https://github.com/Putpocket/NuCell_Vision/blob/fb311d4ac7533f58e73991570cca1668c41cc7da/train_unet_improved.py)를 바탕으로 맞췄다. 방법론은 [Mask R-CNN 논문](https://arxiv.org/abs/1703.06870)을 참고한다. 데이터 사용 시 BBBC038v1과 공식 페이지의 권장 논문을 보고서에 인용한다.

코드 작성·검토에 OpenAI Codex를 사용했다. 보고서 마지막에 도구와 용도를 기재하고, 실행 결과와 코드 동작은 담당자가 직접 확인·설명한다.
