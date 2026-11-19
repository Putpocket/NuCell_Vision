"""Server training entry point. Nothing runs on import."""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Train nuclei Mask R-CNN with fixed U-Net splits.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/stage1_train')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'splits')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min-delta', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-detections', type=int, default=500)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--augment', action='store_true', help='Train-only flips/90-degree rotations; off for comparison.')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.max_detections) < 1:
        parser.error('epochs, patience, batch-size and max-detections must be positive')
    if args.epochs > 100:
        parser.error('epochs must not exceed the shared comparison limit of 100')
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error('Invalid workers, lr, weight-decay or min-delta')
    if not all(math.isfinite(x) for x in
               (args.lr, args.weight_decay, args.min_delta, args.score_threshold, args.mask_threshold)):
        parser.error('Numeric settings must be finite')
    if not (0 <= args.score_threshold <= 1 and 0 < args.mask_threshold < 1):
        parser.error('Invalid probability thresholds')
    return args


def validate(model, loader, device, score_threshold, mask_threshold):
    import numpy as np
    import torch
    from maskrcnn_utils import binary_metrics, prediction_union

    model.eval()
    dices, ious = [], []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            predictions = model([image.to(device) for image in images])
            for prediction, target, meta in zip(predictions, targets, metadata):
                masks = prediction['masks'][:, 0].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                pred, _ = prediction_union(masks, scores, score_threshold, mask_threshold)
                gt = target['masks'].numpy().any(axis=0)
                region = meta['geometry'].region
                dice, iou = binary_metrics(pred[region], gt[region])
                dices.append(dice)
                ious.append(iou)
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    args = parse_args()  # --help works without torch or a dataset.
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from maskrcnn_model import NucleiInstanceDataset, build_maskrcnn, collate_instances
    from maskrcnn_utils import read_splits

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check the server PyTorch/CUDA installation.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    splits = read_splits(args.split_dir)
    train_data = NucleiInstanceDataset(args.data_dir, splits['train'], args.augment)
    val_data = NucleiInstanceDataset(args.data_dir, splits['val'])
    loader_kwargs = dict(num_workers=args.workers, pin_memory=device.type == 'cuda',
                         collate_fn=collate_instances)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_kwargs)
    # Refuse to overwrite a previous experiment, before downloading pretrained weights.
    args.output.mkdir(parents=True, exist_ok=False)
    model = build_maskrcnn(True, args.max_detections, args.score_threshold).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(size=512, model='maskrcnn_resnet50_fpn', num_classes=2,
                  params=sum(p.numel() for p in model.parameters()),
                  trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
                  split_sha256={name: hashlib.sha256((args.split_dir / f'{name}.txt').read_bytes()).hexdigest()
                                for name in ('train', 'val', 'test')},
                  validation_space='512 letterbox, padding excluded',
                  tiny_mask_policy='drop only empty resized training targets; original test count unchanged')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best, stop_best, wait, best_epoch, history = -1.0, -1.0, 0, 0, []
    start = time.perf_counter()
    print(json.dumps(config, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        model.train()
        totals, samples, dropped = {}, 0, 0
        for images, targets, metadata in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}: {losses}')
            loss.backward()
            optimizer.step()
            n = len(images)
            for key, value in dict(loss_total=loss, **losses).items():
                totals[key] = totals.get(key, 0.0) + value.detach().item() * n
            samples += n
            dropped += sum(meta['dropped'] for meta in metadata)
        dice, iou = validate(model, val_loader, device, args.score_threshold, args.mask_threshold)
        row = dict(epoch=epoch, **{k: v / samples for k, v in totals.items()},
                   val_dice=dice, val_iou=iou, dropped_train_masks=dropped,
                   time_sec=time.perf_counter() - tick)
        history.append(row)
        with (args.output / 'history.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        state = dict(model=model.state_dict(), config=config, epoch=epoch, val_dice=dice, val_iou=iou)
        if dice > best:
            best, best_epoch = dice, epoch
            torch.save(state, args.output / 'best.pth')
        torch.save(state, args.output / 'last.pth')
        if dice > stop_best + args.min_delta:
            stop_best, wait = dice, 0
        else:
            wait += 1
        summary = dict(completed_epochs=epoch, best_epoch=best_epoch, best_val_dice=best,
                       best_val_iou=history[best_epoch - 1]['val_iou'],
                       training_time_sec=time.perf_counter() - start, params=config['params'])
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Epoch {epoch}: loss={row["loss_total"]:.4f} val_dice={dice:.4f} '
              f'val_iou={iou:.4f} dropped_masks={dropped} patience={wait}/{args.patience}', flush=True)
        if wait >= args.patience:
            break


if __name__ == '__main__':
    main()
