"""Check annotated training/validation data on CPU without starting a model."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # 학습·검증 데이터만 읽어 split과 512 변환 시 핵 소실을 확인합니다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/stage1_train')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'splits')
    args = parser.parse_args()
    import numpy as np
    from maskrcnn_utils import letterbox, load_sample, prepare_instances, read_splits
    splits = read_splits(args.split_dir)
    report = {'split_counts': {k: len(v) for k, v in splits.items()}}
    for name in ('train', 'val'):
        # 각 이미지의 핵 수와 크기 변환 뒤 사라지는 마스크 수를 집계합니다.
        counts, dropped_ids, lost = [], [], 0
        for image_id in splits[name]:
            _, masks = load_sample(args.data_dir / image_id)
            resized = np.stack([letterbox(mask, is_mask=True)[0] for mask in masks])
            _, _, dropped = prepare_instances(resized)
            counts.append(len(masks))
            if dropped:
                dropped_ids.append(image_id)
                lost += dropped
        report[name] = dict(images=len(counts), min_nuclei=min(counts), max_nuclei=max(counts),
                            total_nuclei=sum(counts), masks_lost_at_512=lost,
                            images_with_lost_masks=dropped_ids)
    # 테스트 정답은 열지 않았다는 점을 결과에 함께 남깁니다.
    report['note'] = ('Test images/annotations were not opened. Choose detection limits using train/val only. '
                      'Original-resolution evaluation retains every original GT nucleus.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
