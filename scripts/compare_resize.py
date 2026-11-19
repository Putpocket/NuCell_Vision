from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE = Path("data/stage1_train")
SPLITS = Path("splits")
SIZES = [256, 512]

rows = []

# 테스트 데이터는 사용하지 않음
for split in ["train", "val"]:

    ids = (SPLITS / f"{split}.txt").read_text().splitlines()

    for image_id in tqdm(ids, desc=split):

        folder = BASE / image_id
        image_path = next((folder / "images").glob("*.png"))

        with Image.open(image_path) as img:
            w, h = img.size

        # 개별 정답 마스크 불러오기
        for path in sorted((folder / "masks").glob("*.png")):

            mask = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE
            )

            mask = (mask > 0).astype(np.uint8)
            original_area = np.count_nonzero(mask)

            # 해상도별 비교
            for size in SIZES:

                scale = min(size / w, size / h)

                new_w = max(1, round(w * scale))
                new_h = max(1, round(h * scale))

                resized = cv2.resize(
                    mask,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST_EXACT
                )

                resized_area = np.count_nonzero(resized)

                # 원본 해상도로 복원
                restored = cv2.resize(
                    resized,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST_EXACT
                )

                # 원본과 복원 이미지 비교
                intersection = np.count_nonzero(
                    mask & restored
                )

                union = np.count_nonzero(
                    mask | restored
                )

                iou = intersection / union

                restored_area = np.count_nonzero(restored)

                area_error = abs(
                    restored_area - original_area
                ) / original_area

                rows.append({
                    "split": split,
                    "image_id": image_id,
                    "width": w,
                    "height": h,
                    "size": size,
                    "original_area": original_area,
                    "resized_area": resized_area,
                    "lost": resized_area == 0,
                    "tiny": 0 < resized_area < 16,
                    "iou": iou,
                    "area_error": area_error
                })

# 결과 저장
df = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/resize_comparison.csv",
    index=False
)

# 해상도별 통계 출력
for split in ["train", "val"]:

    for size in SIZES:

        data = df[
            (df["split"] == split) &
            (df["size"] == size)
        ]

        print(f"\n===== {split.upper()} / {size}x{size} =====")
        print("전체 핵:", len(data))
        print("Lost:", data["lost"].sum())
        print("Tiny:", data["tiny"].sum())
        print("Mean IoU:", round(data["iou"].mean(), 4))
        print("P10 IoU:", round(data["iou"].quantile(0.1), 4))
        print("IoU below 0.8:", (data["iou"] < 0.8).sum())
        print("Area error (%):",
              round(data["area_error"].mean() * 100, 2))
