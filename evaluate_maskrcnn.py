"""Original-resolution evaluation; test use must follow validation-only tuning."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    # 체크포인트, 데이터 분할, 결과 저장 위치를 받습니다.
    parser = argparse.ArgumentParser(description='Evaluate nuclei masks/counts at original resolution.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/stage1_train')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'splits')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error('warmup must be positive')
    return args


def main():
    args = parse_args()
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from maskrcnn_model import build_maskrcnn
    from maskrcnn_utils import (binary_metrics, iter_prediction_masks, letterbox,
                                load_sample, prediction_union, read_splits)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    # 학습 때 사용한 분할 해시와 같은지 확인한 뒤 모델을 복원합니다.
    splits = read_splits(args.split_dir)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['size'] != 512 or config['num_classes'] != 2:
        raise ValueError('Unsupported checkpoint configuration')
    actual_hashes = {name: hashlib.sha256((args.split_dir / f'{name}.txt').read_bytes()).hexdigest()
                     for name in ('train', 'val', 'test')}
    if config['split_sha256'] != actual_hashes:
        raise ValueError('Split files differ from those used during training')
    args.output.mkdir(parents=True, exist_ok=False)
    model = build_maskrcnn(False, config['max_detections'], config['score_threshold']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def predict(image):
        padded, geometry = letterbox(image)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().div_(255).to(device)
        output = model([tensor])[0]
        masks = output['masks'][:, 0].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        union, count = prediction_union(masks, scores, config['score_threshold'],
                                        config['mask_threshold'], geometry)
        return union, count, masks, scores, geometry

    # 첫 이미지로 워밍업한 다음 선택한 분할의 예측·지표를 기록합니다.
    rows = []
    with torch.inference_mode():
        warmup_image, _ = load_sample(args.data_dir / splits[args.split][0])
        for _ in range(args.warmup):
            predict(warmup_image)
        synchronize()
        for image_id in tqdm(splits[args.split], desc=f'Evaluate {args.split}'):
            # File decoding/GT loading is outside the timed inference pipeline.
            image, gt_instances = load_sample(args.data_dir / image_id)
            gt = gt_instances.any(axis=0)
            synchronize()
            start = time.perf_counter()
            pred, count, masks, scores, geometry = predict(image)
            synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            dice, iou = binary_metrics(pred, gt)
            rows.append(dict(image_id=image_id, dice=float(dice), iou=float(iou),
                             true_count=len(gt_instances), predicted_count=count,
                             count_abs_error=abs(count - len(gt_instances)), time_ms=elapsed_ms))
            if args.save_images:
                folder = args.output / 'predictions' / image_id
                folder.mkdir(parents=True)
                labels = np.zeros(pred.shape, dtype=np.uint16)
                for i, mask in enumerate(iter_prediction_masks(
                        masks, scores, config['score_threshold'], config['mask_threshold'], geometry), 1):
                    labels[mask] = i
                cv2.imwrite(str(folder / 'prediction.png'), pred.astype(np.uint8) * 255)
                cv2.imwrite(str(folder / 'ground_truth.png'), gt.astype(np.uint8) * 255)
                # Labels are a visualization: overlapping detections overwrite earlier labels.
                # Predicted_count in CSV counts retained instances before this visualization.
                cv2.imwrite(str(folder / 'instances.png'), labels)
                overlay = image.copy()
                overlay[pred] = (0.5 * overlay[pred] + 0.5 * np.array([255, 64, 64])).astype(np.uint8)
                cv2.imwrite(str(folder / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 비교 스크립트가 읽는 이미지별 CSV와 요약 JSON을 저장합니다.
    with (args.output / 'per_image.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(split=args.split, images=len(rows),
                   mean_dice=float(np.mean([r['dice'] for r in rows])),
                   mean_iou=float(np.mean([r['iou'] for r in rows])),
                   count_mae=float(np.mean([r['count_abs_error'] for r in rows])),
                   mean_time_ms=float(np.mean([r['time_ms'] for r in rows])),
                   median_time_ms=float(np.median([r['time_ms'] for r in rows])),
                   params=sum(p.numel() for p in model.parameters()),
                   device=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint['epoch'],
                   score_threshold=config['score_threshold'], mask_threshold=config['mask_threshold'],
                   max_detections=config['max_detections'], warmup=args.warmup,
                   timing_scope='RGB uint8 in RAM -> letterbox -> H2D -> model -> D2H -> '
                                'unpad -> bilinear probability resize -> threshold -> union/count; excludes I/O/metrics',
                   split_sha256=actual_hashes)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    ranked = sorted(rows, key=lambda r: r['dice'])
    examples = dict(lowest_dice=[r['image_id'] for r in ranked[:3]],
                    highest_dice=[r['image_id'] for r in ranked[-3:][::-1]])
    (args.output / 'example_ids.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
