from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd

BASE = Path("data/stage1_train")

sizes = Counter()
channels = Counter()
counts = []

# 전체 이미지 및 정답 마스크 조사
for folder in sorted(BASE.iterdir()):
    images = list((folder / "images").glob("*.png"))
    masks = list((folder / "masks").glob("*.png"))

    if not images:
        continue

    with Image.open(images[0]) as img:
        sizes[img.size] += 1
        channels[img.mode] += 1

    counts.append(len(masks))

# 데이터셋 기본 통계
print("===== 데이터셋 통계 =====")
print("전체 이미지:", len(counts))
print("전체 세포 핵:", sum(counts))
print("최소 핵 개수:", min(counts))
print("최대 핵 개수:", max(counts))
print("평균 핵 개수:", round(sum(counts) / len(counts), 2))

# 해상도별 이미지 개수
print("\n===== 이미지 해상도 =====")

for size, count in sizes.most_common():
    print(f"{size[0]}x{size[1]}: {count}장")

# 이미지 색상 채널 확인
print("\n===== 이미지 채널 =====")

for mode, count in channels.items():
    print(f"{mode}: {count}장")

# 엑셀 메타데이터 분석
print("\n===== 메타데이터 =====")

excel = pd.ExcelFile("data/metadata.xlsx")

for sheet in excel.sheet_names:
    df = pd.read_excel(excel, sheet_name=sheet)

    print("시트:", sheet)
    print("크기:", df.shape)
    print("컬럼:", df.columns.tolist())
    print(df.head().to_string(index=False))
