from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE = Path("data/stage1_train")
SIZE = 512

stats = {
    "images": 0,
    "masks": 0,
    "invalid_images": 0,
    "invalid_masks": 0,
    "empty_masks": 0,
    "size_mismatch": 0,
    "overlapping_images": 0,
    "transparent_images": 0,
    "lost_after_resize": 0
}

for folder in tqdm(sorted(BASE.iterdir())):
    images = list((folder / "images").glob("*.png"))
    masks = sorted((folder / "masks").glob("*.png"))

    if not images:
        continue

    stats["images"] += 1

    # 원본 이미지 검사
    try:
        with Image.open(images[0]) as img:
            image = np.array(img.convert("RGBA"))
    except Exception:
        stats["invalid_images"] += 1
        continue

    h, w = image.shape[:2]

    if np.any(image[:, :, 3] != 255):
        stats["transparent_images"] += 1

    # 비율을 유지한 리사이즈 크기 계산
    scale = min(SIZE / w, SIZE / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    combined = np.zeros((h, w), dtype=bool)
    overlap = False

    # 개별 정답 마스크 검사
    for path in masks:
        stats["masks"] += 1

        mask = cv2.imread(
            str(path), cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            stats["invalid_masks"] += 1
            continue

        if mask.shape != (h, w):
            stats["size_mismatch"] += 1
            continue

        mask = mask > 0

        if not mask.any():
            stats["empty_masks"] += 1
            continue

        # 마스크 간 겹침 확인
        if np.any(combined & mask):
            overlap = True

        combined |= mask

        # 리사이즈 시 작은 핵 손실 검사
        resized = cv2.resize(
            mask.astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST_EXACT
        )

        if not resized.any():
            stats["lost_after_resize"] += 1

    if overlap:
        stats["overlapping_images"] += 1

print("\n===== 데이터셋 무결성 검사 =====")

for key, value in stats.items():
    print(f"{key:25s}: {value}")
