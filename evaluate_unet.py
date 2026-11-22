#!/usr/bin/env python3
"""NuCell 데이터셋에서 U-Net 및 외부 모델의 분할 결과를 평가합니다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SIZE = 512
PROJECT_ROOT = Path(__file__).resolve().parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# 학습된 U-Net 체크포인트가 공유되면 아래 예시 경로를 사용합니다.
# UNET_CHECKPOINT = PROJECT_ROOT / "results/unet_new/best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="원본 해상도에서 핵 분할 모델의 Dice, IoU, 핵 개수 오차를 비교합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/stage1_train",
                        help="BBBC038 압축 해제 데이터 폴더")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "splits",
                        help="train/val/test.txt가 있는 폴더")
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="평가 split. 설정 실험은 val, 최종 평가는 test")
    parser.add_argument("--checkpoint", "--unet-checkpoint", dest="checkpoint", type=Path, required=True,
                        help="U-Net best.pth 경로")
    parser.add_argument("--output", type=Path, required=True,
                        help="새 결과 폴더 경로 (기존 폴더는 덮어쓰지 않음)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="U-Net 시그모이드 확률 threshold (기본값: 0.5)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmup", type=int, default=10,
                        help="측정 전 워밍업 횟수 (기본값: 10, 측정에 포함하지 않음)")
    parser.add_argument("--save-images", action="store_true",
                        help="원본 크기 예측, 정답, overlay 이미지를 결과 폴더에 저장")
    return parser.parse_args()


def read_splits(folder: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Mask R-CNN 평가 코드와 같은 split 개수·중복·해시 점검을 합니다."""
    # 고정된 이미지 수와 겹치지 않는 분할인지 먼저 확인합니다.
    splits: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for name, expected in (("train", 469), ("val", 100), ("test", 101)):
        path = folder / f"{name}.txt"
        if not path.is_file():
            raise ValueError(f"Split 파일을 찾을 수 없습니다: {path}")
        raw = path.read_bytes()
        ids = [line.strip() for line in raw.decode("utf-8-sig").splitlines()]
        if len(ids) != expected or len(set(ids)) != expected or any(not value for value in ids):
            raise ValueError(f"{name}.txt에는 중복 없는 ID {expected}개가 있어야 합니다.")
        if any("/" in value or "\\" in value or value in (".", "..") for value in ids):
            raise ValueError(f"{name}.txt의 이미지 ID는 폴더 이름이어야 합니다.")
        splits[name] = ids
        hashes[name] = hashlib.sha256(raw).hexdigest()
    groups = [set(splits[name]) for name in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Train/val/test split에 중복 이미지 ID가 있습니다.")
    return splits, hashes


def letterbox(array: np.ndarray, is_mask: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """학습 코드와 같은 비율 유지 리사이즈 및 중앙 패딩을 적용합니다."""
    # 비율은 유지하고 남는 부분은 검은색으로 채웁니다.
    height, width = array.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    if is_mask:
        interpolation = cv2.INTER_NEAREST_EXACT
    elif scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(array, (new_width, new_height), interpolation=interpolation)
    top = (SIZE - new_height) // 2
    left = (SIZE - new_width) // 2
    shape = (SIZE, SIZE) + array.shape[2:]
    canvas = np.zeros(shape, dtype=array.dtype)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, (top, left, new_height, new_width)


def load_sample(data_dir: Path, image_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    """이미지와 개별 핵 마스크를 읽고 합친 정답 및 인스턴스 수를 반환합니다."""
    # 개별 마스크 파일은 합쳐 Dice·IoU용 정답을 만들고, 핵 수는 따로 셉니다.
    folder = data_dir / image_id
    image_paths = sorted(path for path in (folder / "images").iterdir()
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    mask_paths = sorted(path for path in (folder / "masks").iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(image_paths) != 1:
        raise ValueError(f"이미지 파일은 한 장이어야 합니다: {folder / 'images'}")
    if not mask_paths:
        raise ValueError(f"정답 핵 마스크가 없습니다: {folder / 'masks'}")

    with Image.open(image_paths[0]) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    merged = np.zeros((height, width), dtype=bool)
    instance_count = 0
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"마스크를 읽지 못했거나 이미지와 크기가 다릅니다: {mask_path}")
        binary = mask > 0
        if binary.any():
            instance_count += 1
            merged |= binary
    return image, merged, instance_count


def make_unet(checkpoint: Path, device_name: str):
    """저장소 학습 코드와 같은 U-Net 구조에 best.pth를 불러옵니다."""
    import torch
    import segmentation_models_pytorch as smp

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 요청했지만 CUDA를 사용할 수 없습니다.")
    device = torch.device(device_name)
    # 사전학습 가중치 없이 학습 구조만 만든 뒤 체크포인트를 적용합니다.
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    # PyTorch 버전에 따라 weights_only 인자가 지원되지 않을 수 있습니다.
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return torch, model, device


def predict_unet(torch, model, device, image: np.ndarray, threshold: float):
    """전처리부터 원본 해상도 복원까지 시간을 재고 이진 마스크를 만듭니다."""
    # 학습 때와 같은 정규화로 추론하고 패딩을 제거해 원본 크기로 되돌립니다.
    started = time.perf_counter()
    boxed, (top, left, new_height, new_width) = letterbox(image)
    normalized = (boxed.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
    # 패딩 영역을 제거하고 원본 해상도로 복원한 다음 회의록의 0.5 기준을 적용합니다.
    cropped = probabilities[top:top + new_height, left:left + new_width]
    original_height, original_width = image.shape[:2]
    restored = cv2.resize(cropped, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    binary = restored > threshold
    predicted_count = int(cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)[0] - 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return binary, predicted_count, (time.perf_counter() - started) * 1000


def read_external_prediction(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """고전 기법이나 Mask R-CNN이 저장한 원본 크기 이진 마스크를 읽습니다."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"예측 마스크를 읽을 수 없습니다: {path}")
    if mask.shape != expected_shape:
        raise ValueError(f"예측 마스크 크기는 원본과 같아야 합니다: {path}, {mask.shape} != {expected_shape}")
    return mask > 0


def score(gt: np.ndarray, pred: np.ndarray, gt_count: int,
          predicted_count: int) -> dict[str, float | int]:
    """픽셀 단위 점수와 연결요소 기반 핵 개수 오차를 계산합니다."""
    if gt.shape != pred.shape:
        raise ValueError(f"정답과 예측 크기가 다릅니다: {gt.shape} != {pred.shape}")
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    # Mask R-CNN 공통 평가 코드와 같은 epsilon으로 빈 마스크도 안정적으로 계산합니다.
    epsilon = 1e-7
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    # 연결된 핵은 하나로 세므로 개수는 근사치이며, Dice/IoU와 별도로 해석합니다.
    return {
        "dice": dice, "iou": iou, "true_count": gt_count, "predicted_count": predicted_count,
        "count_abs_error": abs(predicted_count - gt_count),
        "precision": precision, "recall": recall,
        "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    # 상대 경로도 실행 위치와 관계없이 저장소 루트 기준으로 해석합니다.
    for key in ("data_dir", "split_dir", "output", "checkpoint"):
        path = getattr(args, key)
        if path is not None and not path.is_absolute():
            setattr(args, key, PROJECT_ROOT / path)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold 값은 0과 1 사이여야 합니다.")
    if args.warmup < 1:
        raise ValueError("--warmup은 1 이상이어야 합니다.")
    # 분할 목록을 검증하고 선택한 val/test 이미지 ID만 평가합니다.
    splits, split_hashes = read_splits(args.split_dir)
    ids = splits[args.split]
    torch, model, device = make_unet(args.checkpoint, args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    # Mask R-CNN 평가 코드처럼 첫 이미지를 사용해 워밍업하고 동기화합니다.
    warmup_image, _, _ = load_sample(args.data_dir, ids[0])
    for _ in range(args.warmup):
        predict_unet(torch, model, device, warmup_image, args.threshold)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    args.output.mkdir(parents=True, exist_ok=False)
    # 이미지별 지표와 선택적 시각화 파일을 모읍니다.
    rows: list[dict[str, object]] = []
    for image_id in ids:
        image, gt, gt_count = load_sample(args.data_dir, image_id)
        # 파일 읽기와 정답 로딩을 측정 구간에서 제외합니다.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pred, predicted_count, elapsed_ms = predict_unet(torch, model, device, image, args.threshold)
        metrics = score(gt, pred, gt_count, predicted_count)
        rows.append({"image_id": image_id, **metrics, "time_ms": elapsed_ms})
        if args.save_images:
            image_dir = args.output / "predictions" / image_id
            image_dir.mkdir(parents=True)
            cv2.imwrite(str(image_dir / "prediction.png"), pred.astype(np.uint8) * 255)
            cv2.imwrite(str(image_dir / "ground_truth.png"), gt.astype(np.uint8) * 255)
            overlay = image.copy()
            overlay[pred] = (0.5 * overlay[pred] + 0.5 * np.array([255, 64, 64])).astype(np.uint8)
            cv2.imwrite(str(image_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    write_csv(args.output / "per_image.csv", rows)
    ranked = sorted(rows, key=lambda row: float(row["dice"]))
    report = {
        "split": args.split, "images": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "count_mae": float(np.mean([row["count_abs_error"] for row in rows])),
        "mean_time_ms": float(np.mean([row["time_ms"] for row in rows])),
        "median_time_ms": float(np.median([row["time_ms"] for row in rows])),
        "params": parameter_count,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "checkpoint": str(args.checkpoint), "score_threshold": args.threshold,
        "warmup": args.warmup, "split_sha256": split_hashes,
        "count_method": "8-connected components on the binary union; touching nuclei may merge",
        "timing_scope": "RGB uint8 in RAM -> letterbox -> H2D -> model -> D2H -> unpad -> bilinear resize -> threshold -> connected components; excludes file I/O and metric calculation",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    examples = {"lowest_dice": [row["image_id"] for row in ranked[:3]],
                "highest_dice": [row["image_id"] for row in ranked[-3:][::-1]]}
    (args.output / "example_ids.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, cv2.error) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
