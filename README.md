# NuCell Vision · U-Net

컴퓨터비전 팀 프로젝트에서 내가 맡은 U-Net 코드다. 현미경 사진에서 핵이 있는 부분을 찾는다. 인코더는 ResNet-50을 썼고, RGB 사진을 넣으면 핵 영역을 나타내는 1채널 로짓이 나온다.

## 데이터 준비

[BBBC038](https://bbbc.broadinstitute.org/BBBC038)에서 `stage1_train.zip`을 받으면 된다. 데이터는 CC0이고 인용 방법도 같은 페이지에 나와 있다. 압축을 푼 뒤 사진과 마스크를 아래처럼 뒀다.

```text
data/stage1_train/<이미지 ID>/images/<이미지 ID>.png
data/stage1_train/<이미지 ID>/masks/<핵 ID>.png
```

670장 중 469장은 학습, 100장은 검증, 101장은 테스트로 나눴다. 이미지 ID는 `splits/`에 정리해 뒀다. Mask R-CNN도 이 파일 그대로 쓰면 된다. 테스트용 101장은 학습이나 설정을 바꿀 때 쓰지 않는다.

## 학습 방법

학습 서버에서 썼던 환경은 PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4, segmentation-models-pytorch 0.6.0이다. CUDA를 쓸 수 있는 PyTorch 환경을 준비한 뒤 저장소 폴더에서 실행하면 된다.

```bash
python -m pip install -r requirements.txt
python -u train_unet_improved.py --loss bce --epochs 100 --patience 15 --min-delta 0.0005 --output results/unet_new
```

위 명령은 증강 없이 학습한다. 증강을 켜려면 `--augment`를 붙이면 된다. 이 경우 학습 사진과 마스크를 같이 뒤집거나 90도 단위로 돌리고, 밝기와 대비는 사진에만 적용한다. 출력 폴더 이름은 실험마다 다르게 뒀다. 가장 높은 검증 Dice를 기록한 가중치는 `best.pth`에 저장된다.

배치 크기는 4, 학습률은 0.0001이다. AdamW를 쓰고 가중치 감쇠는 0.01, 시드는 42로 뒀다.

## Mask R-CNN과 비교할 때

두 모델은 같은 Debian 서버에서 같은 RTX 4060과 PyTorch·CUDA 환경을 쓴다. 데이터와 평가 조건을 공유하고 모델 코드는 각자 작성한다. U-Net은 개별 핵 마스크를 합쳐 한 장으로 만들고, Mask R-CNN은 같은 정답에서 핵마다 마스크와 박스를 만든다. 배치 크기와 손실은 각 모델에 맞게 잡는다.

- 학습 469장, 검증 100장, 테스트 101장의 이미지 ID는 `splits/`에 있는 파일 그대로 쓴다.
- 두 모델에 실제로 들어가는 크기는 512×512로 맞춘다. 사진 비율을 유지해 긴 변을 512로 맞춘 뒤 중앙에 놓고 남는 부분을 패딩한다. Mask R-CNN은 모델 내부에서 크기를 다시 바꾸는지도 확인한다.
- 최대 100에폭으로 맞춘다. 검증 Dice가 가장 높았던 가중치를 쓰고, 실제 종료 에폭과 최고 에폭을 같이 기록한다.
- 모델 비교는 둘 다 증강을 끄고 진행한다. U-Net에서 `--augment`를 켠 결과는 증강 효과를 보는 별도 실험으로 둔다.

최종 평가는 같은 테스트 101장에서 한다. 패딩을 걷어내고 원본 해상도로 돌린 다음 예측 핵 영역을 한 장으로 합쳐 이미지별 Dice와 IoU를 구하고 평균을 낸다. 핵 개수 오차는 따로 비교한다. U-Net은 시그모이드 값이 0.5보다 큰 픽셀을 핵으로 보고 Watershed는 쓰지 않는다.

처리 시간은 두 모델을 같은 GPU에서 배치 1로 돌려 잰다. 워밍업 뒤 GPU 동기화를 하고 전처리부터 후처리까지 재며, 파일을 읽고 쓰는 시간은 뺀다. 두 모델의 파라미터 수도 같이 기록한다.

공통 평가에서 U-Net 가중치를 쓸 때는 `build_model(pretrained=False)`로 모델을 만든 뒤 `best.pth`를 불러오면 된다.
