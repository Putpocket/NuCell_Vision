#!/usr/bin/env python3
"""여러 모델의 표준 평가 결과를 모아 성능을 비교하고 실패 사례를 정리합니다."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REQUIRED_COLUMNS = ("image_id", "dice", "iou", "count_abs_error", "time_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U-Net, Mask R-CNN, 고전 기법의 per_image.csv 결과를 비교합니다."
    )
    parser.add_argument("--model", action="append", required=True, metavar="NAME=DIR",
                        help="per_image.csv와 summary.json이 있는 평가 결과 폴더. 반복 지정 가능")
    parser.add_argument("--output", type=Path, required=True,
                        help="새 비교 결과 폴더 (기존 폴더는 덮어쓰지 않음)")
    return parser.parse_args()


def read_model(name: str, folder: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    """각 모델의 평가 CSV와 요약 JSON을 읽고 필요한 열을 확인합니다."""
    # 비교에 필요한 지표 열이 빠졌거나 이미지 ID가 중복되면 중단합니다.
    csv_path = folder / "per_image.csv"
    summary_path = folder / "summary.json"
    if not csv_path.is_file():
        raise ValueError(f"{name}: 평가 CSV를 찾을 수 없습니다: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{name}: per_image.csv에 필수 열이 없습니다: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{name}: 평가 CSV에 결과 행이 없습니다.")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"{name}: per_image.csv에 중복 image_id가 있습니다.")

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        # 고전 기법 평가가 summary.json을 만들지 않는 경우에도 CSV만으로 비교합니다.
        summary = {}
    return rows, summary


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"잘못된 숫자 값이 있습니다: {key}")
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    # 각 모델의 표준 평가 결과 폴더를 읽습니다.
    models: dict[str, tuple[list[dict[str, str]], dict[str, object]]] = {}
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"잘못된 --model 형식: {item} (NAME=DIR 형식 필요)")
        name, raw_folder = item.split("=", 1)
        if not name or name in models:
            raise ValueError(f"모델 이름이 비어 있거나 중복되었습니다: {name}")
        folder = Path(raw_folder)
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        models[name] = read_model(name, folder)

    # 결과의 이미지 ID와 split이 모두 같아야 공정한 직접 비교로 인정합니다.
    first_name = next(iter(models))
    reference_rows, reference_summary = models[first_name]
    reference_ids = {row["image_id"] for row in reference_rows}
    split = reference_summary.get("split")
    for name, (rows, summary) in models.items():
        if {row["image_id"] for row in rows} != reference_ids:
            raise ValueError(f"{name}: 비교 기준 모델 {first_name}와 이미지 ID 구성이 다릅니다.")
        current_split = summary.get("split")
        if split is not None and current_split is not None and current_split != split:
            raise ValueError(f"평가 split이 다릅니다: {first_name}={split}, {name}={current_split}")
        if len(rows) != len(reference_rows):
            raise ValueError(f"{name}: 평가 이미지 수가 기준 모델과 다릅니다.")

    # 모델별 행을 ID로 찾아 공통 비교표와 난이도순 실패 사례를 만듭니다.
    rows_by_id = {
        name: {row["image_id"]: row for row in rows}
        for name, (rows, _) in models.items()
    }
    # 같은 이미지 ID를 기준으로 모델별 지표를 한 행에 모읍니다.
    detailed_rows: list[dict[str, object]] = []
    for image_id in sorted(reference_ids):
        combined: dict[str, object] = {"image_id": image_id}
        dice_values = []
        for name in models:
            row = rows_by_id[name][image_id]
            dice = float(row["dice"])
            dice_values.append(dice)
            for metric in ("dice", "iou", "true_count", "predicted_count", "count_abs_error", "time_ms"):
                if metric in row and row[metric] != "":
                    combined[f"{name}_{metric}"] = row[metric]
        combined["mean_dice_across_models"] = sum(dice_values) / len(dice_values)
        combined["lowest_model_dice"] = min(dice_values)
        detailed_rows.append(combined)

    # 모든 모델에서 평균 Dice가 낮은 이미지부터 보면 공통 실패 유형을 찾기 쉽습니다.
    failures = sorted(detailed_rows, key=lambda row: (float(row["mean_dice_across_models"]),
                                                       str(row["image_id"])))
    # 모델별 평균 성능과 계산 파라미터 수를 요약합니다.
    comparison = []
    for name, (rows, summary) in models.items():
        comparison.append({
            "model": name,
            "split": summary.get("split", split or "unknown"),
            "images": len(rows),
            "mean_dice": mean(rows, "dice"),
            "mean_iou": mean(rows, "iou"),
            "count_mae": mean(rows, "count_abs_error"),
            "mean_time_ms": mean(rows, "time_ms"),
            "params": summary.get("params", ""),
        })

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "model_comparison.csv", comparison)
    write_csv(output / "per_image_comparison.csv", detailed_rows)
    write_csv(output / "failure_analysis.csv", failures)
    report = {
        "split": split or "unknown",
        "images": len(reference_ids),
        "models": {row["model"]: {key: value for key, value in row.items() if key != "model"}
                   for row in comparison},
        "common_low_dice_image_ids": [row["image_id"] for row in failures[:10]],
        "inputs": {name: str(Path(args.model[index].split("=", 1)[1]))
                   for index, name in enumerate(models)},
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"{len(models)}개 모델, {len(reference_ids)}개 이미지 비교 완료")
    print(f"비교 결과: {output}")
    for row in sorted(comparison, key=lambda item: float(item["mean_dice"]), reverse=True):
        print(f"{row['model']}: Dice={row['mean_dice']:.4f}, IoU={row['mean_iou']:.4f}, "
              f"개수 MAE={row['count_mae']:.2f}, 시간={row['mean_time_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
