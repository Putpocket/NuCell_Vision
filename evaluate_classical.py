#!/usr/bin/env python3
"""Evaluate the classical nuclei segmenter with the shared image-level metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from classical_model import PARAMETER_COUNT, predict_classical


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="고전 기법의 원본 해상도 Dice·IoU·핵 개수·처리 시간을 평가합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/stage1_train")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True,
                        help="결과 저장 폴더 (기존 결과는 덮어씀)")
    parser.add_argument("--polarity", choices=("bright", "auto"), default="auto")
    parser.add_argument("--seed-ratio", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--save-images", action="store_true",
                        help="정성 비교 그림에 필요한 원본 크기 예측·정답 마스크 저장")
    parser.add_argument("--final-test", action="store_true",
                        help="설정 확정 후 최종 테스트를 실행할 때 명시")
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("테스트 평가는 설정 확정 후 --final-test를 함께 지정해야 합니다.")
    if not 0 < args.seed_ratio < 1 or args.min_area < 1 or args.warmup < 1:
        parser.error("seed-ratio는 0과 1 사이, min-area와 warmup은 1 이상이어야 합니다.")
    return args


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_splits(folder: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    # 이미지 수, 중복 ID, 분할 간 겹침을 확인하고 파일 해시를 기록합니다.
    splits: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for name, expected in (("train", 469), ("val", 100), ("test", 101)):
        path = folder / f"{name}.txt"
        if not path.is_file():
            raise ValueError(f"분할 파일을 찾을 수 없습니다: {path}")
        raw = path.read_bytes()
        ids = [line.strip() for line in raw.decode("utf-8-sig").splitlines()]
        if len(ids) != expected or len(set(ids)) != expected or any(not value for value in ids):
            raise ValueError(f"{name}.txt에는 중복 없는 이미지 ID {expected}개가 있어야 합니다.")
        if any("/" in value or "\\" in value or value in (".", "..") for value in ids):
            raise ValueError(f"{name}.txt의 이미지 ID는 폴더 이름이어야 합니다.")
        splits[name] = ids
        hashes[name] = hashlib.sha256(raw).hexdigest()
    groups = [set(splits[name]) for name in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("train/val/test 분할에 중복 이미지 ID가 있습니다.")
    return splits, hashes


def image_paths(data_dir: Path, image_id: str) -> tuple[list[Path], Path]:
    folder = data_dir / image_id
    image_dir, mask_dir = folder / "images", folder / "masks"
    paths = sorted(path for path in image_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return paths, mask_dir


def load_rgb(data_dir: Path, image_id: str) -> np.ndarray:
    paths, _ = image_paths(data_dir, image_id)
    if len(paths) != 1:
        raise ValueError(f"이미지 파일은 한 장이어야 합니다: {data_dir / image_id / 'images'}")
    with Image.open(paths[0]) as source:
        return np.asarray(source.convert("RGB"))


def load_sample(data_dir: Path, image_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    # 이미지와 핵별 정답을 읽어 합집합 마스크와 정답 핵 수를 만듭니다.
    paths, mask_dir = image_paths(data_dir, image_id)
    mask_paths = sorted(path for path in mask_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) != 1:
        raise ValueError(f"이미지 파일은 한 장이어야 합니다: {data_dir / image_id / 'images'}")
    if not mask_paths:
        raise ValueError(f"정답 핵 마스크가 없습니다: {mask_dir}")

    with Image.open(paths[0]) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    target = np.zeros((height, width), dtype=bool)
    count = 0
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"마스크를 읽지 못했거나 이미지와 크기가 다릅니다: {mask_path}")
        binary = mask > 0
        if binary.any():
            count += 1
            target |= binary
    if count == 0:
        raise ValueError(f"유효한 정답 핵 마스크가 없습니다: {mask_dir}")
    return image, target, count


def score(target: np.ndarray, prediction: np.ndarray, true_count: int,
          predicted_count: int) -> dict[str, float | int]:
    # 픽셀 겹침 점수와 별도의 핵 개수 오차를 계산합니다.
    if prediction.shape != target.shape:
        raise ValueError("예측 마스크와 정답 마스크의 크기가 다릅니다.")
    tp = int(np.logical_and(target, prediction).sum())
    fp = int(np.logical_and(~target, prediction).sum())
    fn = int(np.logical_and(target, ~prediction).sum())
    epsilon = 1e-7
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "true_count": true_count,
        "predicted_count": predicted_count,
        "count_abs_error": abs(predicted_count - true_count),
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.data_dir = resolve_from_root(args.data_dir)
    args.split_dir = resolve_from_root(args.split_dir)
    args.output = resolve_from_root(args.output)
    splits, split_hashes = read_splits(args.split_dir)
    image_ids = splits[args.split]

    config = dict(model="classical_otsu_distance_watershed", split=args.split,
                  input_size=512, polarity=args.polarity, seed_ratio=args.seed_ratio,
                  min_area=args.min_area, warmup=args.warmup, params=PARAMETER_COUNT,
                  split_sha256=split_hashes,
                  timing_scope="RGB uint8 in RAM -> letterbox -> classical segmentation -> original-size mask/count; excludes file I/O, ground truth and metrics")

    # OpenCV 스레드 수를 고정하고 평가 전에 같은 첫 이미지로 워밍업합니다.
    cv2.setNumThreads(1)
    # 워밍업에는 이미지 픽셀만 사용하고 정답 마스크는 읽지 않습니다.
    warmup_image = load_rgb(args.data_dir, image_ids[0])
    options = dict(polarity=args.polarity, size=512,
                   seed_ratio=args.seed_ratio, min_area=args.min_area)
    for _ in range(args.warmup):
        predict_classical(warmup_image, **options)

    args.output.mkdir(parents=True, exist_ok=True)
    # 예측 결과를 정답과 비교해 모델 공통 형식으로 저장합니다.
    rows: list[dict[str, object]] = []
    for number, image_id in enumerate(image_ids, 1):
        image, target, true_count = load_sample(args.data_dir, image_id)
        prediction, predicted_count, elapsed_ms = predict_classical(image, **options)
        metrics = score(target, prediction, true_count, predicted_count)
        rows.append({"image_id": image_id, **metrics, "time_ms": float(elapsed_ms)})

        if args.save_images:
            image_dir = args.output / "predictions" / image_id
            image_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_dir / "prediction.png"), prediction.astype(np.uint8) * 255)
            cv2.imwrite(str(image_dir / "ground_truth.png"), target.astype(np.uint8) * 255)
            overlay = image.copy()
            overlay[prediction] = (0.5 * overlay[prediction] +
                                   0.5 * np.array([255, 64, 64])).astype(np.uint8)
            cv2.imwrite(str(image_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"[{number}/{len(image_ids)}] {image_id[:12]} | Dice={metrics['dice']:.4f} "
              f"IoU={metrics['iou']:.4f} | nuclei={predicted_count}/{true_count} | "
              f"{elapsed_ms:.2f} ms", flush=True)

    write_csv(args.output / "per_image.csv", rows)
    dice_values = [float(row["dice"]) for row in rows]
    iou_values = [float(row["iou"]) for row in rows]
    count_errors = [float(row["count_abs_error"]) for row in rows]
    times = [float(row["time_ms"]) for row in rows]
    summary = {
        **config,
        "images": len(rows),
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "count_mae": float(np.mean(count_errors)),
        "mean_time_ms": float(np.mean(times)),
        "median_time_ms": float(np.median(times)),
        "prediction_count_method": "number of nonempty Watershed instance labels before binary merge",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ranked = sorted(rows, key=lambda row: float(row["dice"]))
    examples = {
        "lowest_dice": [row["image_id"] for row in ranked[:3]],
        "highest_dice": [row["image_id"] for row in ranked[-3:][::-1]],
    }
    (args.output / "example_ids.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"평가 완료: {len(rows)}장 ({args.split}) / {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, cv2.error, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
