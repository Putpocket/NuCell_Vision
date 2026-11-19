#!/usr/bin/env python3
"""원본, 정답, 모델 결과를 나란히 배치해 성공·실패 사례 그림을 만듭니다."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PANEL_WIDTH = 360
PANEL_HEIGHT = 390
TILE_HEIGHT = 330


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="평가 결과 폴더들에서 성공 1건과 실패 3건을 골라 비교 그림을 저장합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage1_train"))
    parser.add_argument("--model", action="append", required=True, metavar="NAME=DIR",
                        help="save-images로 생성한 predictions/ 폴더가 포함된 평가 폴더")
    parser.add_argument("--output", type=Path, required=True,
                        help="새 그림 출력 폴더")
    parser.add_argument("--failure-count", type=int, default=3,
                        help="실패 사례 수 (기본값: 3)")
    return parser.parse_args()


def read_results(name: str, folder: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    csv_path = folder / "per_image.csv"
    summary_path = folder / "summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise ValueError(f"{name}: per_image.csv 또는 summary.json이 없습니다: {folder}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "dice"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{name}: per_image.csv에 image_id, dice 열이 필요합니다.")
    by_id = {row["image_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{name}: 중복 image_id가 있습니다.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return by_id, summary


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def fit_tile(rgb: np.ndarray) -> np.ndarray:
    """가로세로 비율을 유지해 흰색 배경 중앙에 그림을 맞춥니다."""
    height, width = rgb.shape[:2]
    scale = min(PANEL_WIDTH / width, TILE_HEIGHT / height)
    resized = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    tile = np.full((TILE_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    top = (TILE_HEIGHT - resized.shape[0]) // 2
    left = (PANEL_WIDTH - resized.shape[1]) // 2
    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return tile


def draw_case(image_id: str, model_dirs: dict[str, Path],
              rows_by_model: dict[str, dict[str, dict[str, str]]],
              data_dir: Path, destination: Path) -> None:
    """한 이미지에 대해 원본·정답·각 모델 출력을 가로로 연결합니다."""
    source_dir = data_dir / image_id / "images"
    source_paths = sorted(source_dir.glob("*.png"))
    if len(source_paths) != 1:
        raise ValueError(f"원본 이미지가 한 장이 아닙니다: {source_dir}")

    panels: list[tuple[str, np.ndarray]] = [("Original", load_rgb(source_paths[0]))]
    first_name = next(iter(model_dirs))
    base = model_dirs[first_name] / "predictions" / image_id
    gt_path = base / "ground_truth.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError(f"정답 이미지가 없습니다. 각 평가기를 --save-images로 실행하세요: {gt_path}")
    panels.append(("Ground truth", cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)))

    for name, folder in model_dirs.items():
        prediction_path = folder / "predictions" / image_id / "prediction.png"
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        if prediction is None:
            raise ValueError(f"{name}: 예측 이미지가 없습니다: {prediction_path}")
        row = rows_by_model[name][image_id]
        label = f"{name} Dice {float(row['dice']):.3f}"
        panels.append((label, cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)))

    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH * len(panels), 3), 255, dtype=np.uint8)
    for column, (label, rgb) in enumerate(panels):
        x = column * PANEL_WIDTH
        cv2.putText(canvas, label[:36], (x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        tile = fit_tile(rgb)
        canvas[42:42 + TILE_HEIGHT, x:x + PANEL_WIDTH] = tile
    cv2.putText(canvas, f"Image ID: {image_id}", (8, PANEL_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise OSError(f"그림을 저장하지 못했습니다: {destination}")


def main() -> int:
    args = parse_args()
    if args.failure_count < 3:
        raise ValueError("과제 기준에 맞게 --failure-count는 최소 3이어야 합니다.")
    data_dir = args.data_dir if args.data_dir.is_absolute() else Path.cwd() / args.data_dir
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output

    model_dirs: dict[str, Path] = {}
    rows_by_model: dict[str, dict[str, dict[str, str]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"잘못된 --model 형식: {item} (NAME=DIR 형식 필요)")
        name, raw_path = item.split("=", 1)
        folder = Path(raw_path)
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        if not name or name in model_dirs:
            raise ValueError(f"모델 이름이 비어 있거나 중복되었습니다: {name}")
        rows_by_model[name], summaries[name] = read_results(name, folder)
        model_dirs[name] = folder

    names = list(model_dirs)
    first = names[0]
    reference_ids = set(rows_by_model[first])
    split = summaries[first].get("split")
    if any(set(rows_by_model[name]) != reference_ids for name in names[1:]):
        raise ValueError("모델들의 평가 이미지 ID가 일치하지 않습니다.")
    for name in names[1:]:
        other_split = summaries[name].get("split")
        if split is not None and other_split != split:
            raise ValueError(f"평가 split이 일치하지 않습니다: {split} != {other_split}")
    if len(reference_ids) < args.failure_count + 1:
        raise ValueError("성공 사례 1개와 요청한 실패 사례를 고를 만큼 평가 이미지가 적습니다.")

    # 세 방법 평균 Dice가 가장 높은 사례 1개와 낮은 실패 사례들을 고릅니다.
    mean_scores = {
        image_id: sum(float(rows_by_model[name][image_id]["dice"]) for name in names) / len(names)
        for image_id in reference_ids
    }
    ranked = sorted(mean_scores, key=lambda image_id: (mean_scores[image_id], image_id))
    success_id = ranked[-1]
    failure_ids = ranked[:args.failure_count]

    output.mkdir(parents=True, exist_ok=False)
    draw_case(success_id, model_dirs, rows_by_model, data_dir, output / "success_case.png")
    for index, image_id in enumerate(failure_ids, start=1):
        draw_case(image_id, model_dirs, rows_by_model, data_dir,
                  output / f"failure_case_{index}.png")
    selection = {
        "split": split,
        "success_case": {"image_id": success_id, "mean_dice": mean_scores[success_id]},
        "failure_cases": [{"image_id": image_id, "mean_dice": mean_scores[image_id]}
                          for image_id in failure_ids],
        "models": names,
    }
    (output / "selected_cases.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"성공 사례: {success_id}; 실패 사례 {len(failure_ids)}개 저장: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, cv2.error) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
