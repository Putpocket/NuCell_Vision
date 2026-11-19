import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

import segmentation_models_pytorch as smp


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/stage1_train"

SIZE = 512
BATCH_SIZE = 4
LR = 1e-4
SEED = 42
DEVICE = "cuda"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# 이미지 비율 유지 및 여백 추가
def letterbox(array, is_mask=False):
    h, w = array.shape[:2]
    scale = min(SIZE / w, SIZE / h)

    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    if is_mask:
        method = cv2.INTER_NEAREST_EXACT
    elif scale < 1:
        method = cv2.INTER_AREA
    else:
        method = cv2.INTER_LINEAR

    resized = cv2.resize(array, (nw, nh), interpolation=method)

    top = (SIZE - nh) // 2
    left = (SIZE - nw) // 2

    shape = (SIZE, SIZE) + array.shape[2:]
    result = np.zeros(shape, dtype=array.dtype)
    result[top:top + nh, left:left + nw] = resized

    valid = np.zeros((SIZE, SIZE), dtype=np.float32)
    valid[top:top + nh, left:left + nw] = 1

    return result, valid


class NucleiDataset(Dataset):

    def __init__(self, split):
        path = ROOT / "splits" / f"{split}.txt"
        self.ids = path.read_text().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        folder = DATA_DIR / self.ids[index]
        path = next((folder / "images").glob("*.png"))

        with Image.open(path) as img:
            image = np.array(img.convert("RGB"))

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 개별 핵 마스크를 이진 마스크로 통합
        for path in sorted((folder / "masks").glob("*.png")):
            nucleus = cv2.imread(str(path), 0)

            if nucleus is None or nucleus.shape != (h, w):
                raise ValueError(f"Invalid mask: {path}")

            mask |= (nucleus > 0).astype(np.uint8)

        if not mask.any():
            raise ValueError(f"Empty mask: {folder.name}")

        image, valid = letterbox(image)
        mask, _ = letterbox(mask, is_mask=True)

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        valid = torch.from_numpy(valid).unsqueeze(0)

        return image, mask, valid


def build_model(pretrained=True):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        state = weights.get_state_dict(
            progress=True, check_hash=True
        )

        state.pop("fc.weight", None)
        state.pop("fc.bias", None)

        model.encoder.load_state_dict(state, strict=True)

    return model


def save_graphs(history, output, best_epoch):
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_dice"], label="Dice")
    plt.plot(df["epoch"], df["val_iou"], label="IoU")
    plt.axvline(
        best_epoch, color="gray",
        linestyle="--", label="Best epoch"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("U-Net Validation Performance")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "validation.png", dpi=300)
    plt.close()


class AugmentedDataset(NucleiDataset):
    def __init__(self, split, augment=False):
        super().__init__(split)
        self.augment = augment and split == 'train'

    def __getitem__(self, index):
        image, mask, valid = super().__getitem__(index)
        if not self.augment:
            return image, mask, valid

        # 영상과 정답, 유효 영역에 동일한 기하 변환 적용
        items = [image, mask, valid]
        for dim in (-1, -2):
            if random.random() < 0.5:
                items = [x.flip(dim) for x in items]
        k = random.randrange(4)
        image, mask, valid = [x.rot90(k, (-2, -1)) for x in items]

        # 밝기와 대비는 RGB 영상에만 적용 (패딩 유지)
        if random.random() < 0.5:
            mean = image.new_tensor(MEAN).view(3, 1, 1)
            std = image.new_tensor(STD).view(3, 1, 1)
            rgb = image * std + mean
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(-0.10, 0.10)
            rgb = ((rgb - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
            image = torch.where(valid.bool(), (rgb - mean) / std, image)

        return image.contiguous(), mask.contiguous(), valid.contiguous()


def loss_fn(logits, mask, valid, mode):
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    bce = ((bce * valid).sum(dims) / valid.sum(dims).clamp_min(1)).mean()
    if mode == 'bce':
        return bce

    # 유효 영역에서만 이미지별 Soft Dice Loss 계산
    prob = logits.sigmoid() * valid
    target = mask * valid
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + 1e-7) / (denom + 1e-7)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def run_epoch(model, loader, mode, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_dice = total_iou = 0.0

    for image, mask, valid in tqdm(loader, desc='Train' if training else 'Val'):
        image, mask, valid = [x.to(DEVICE) for x in (image, mask, valid)]
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = loss_fn(logits, mask, valid, mode)
            if not torch.isfinite(loss):
                raise RuntimeError('Invalid loss')
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * image.size(0)
        if not training:
            pred = (logits.sigmoid() > 0.5).float() * valid
            gt = mask * valid
            inter = (pred * gt).sum((1, 2, 3))
            area = pred.sum((1, 2, 3)) + gt.sum((1, 2, 3))
            total_dice += ((2 * inter + 1e-7) / (area + 1e-7)).sum().item()
            total_iou += ((inter + 1e-7) / (area - inter + 1e-7)).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--loss', choices=['bce', 'bce_dice'], default='bce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.min_delta < 0:
        parser.error('Invalid training arguments')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_data = AugmentedDataset('train', args.augment)
    val_data = AugmentedDataset('val')
    test_ids = (ROOT / 'splits/test.txt').read_text().splitlines()
    groups = [set(train_data.ids), set(val_data.ids), set(test_ids)]
    if ([len(train_data), len(val_data), len(test_ids)] != [469, 100, 101]
            or [len(g) for g in groups] != [469, 100, 101]
            or len(set.union(*groups)) != 670):
        raise RuntimeError('Unexpected or overlapping dataset splits')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    model = build_model(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    params = sum(p.numel() for p in model.parameters())
    if params != 32521105:
        raise RuntimeError(f'Baseline model mismatch: {params}')

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), seed=SEED, size=SIZE, batch_size=BATCH_SIZE,
                  lr=LR, params=params, model='Unet/ResNet50/1 channel')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    print('Model: U-Net / ResNet50 / 1 channel')
    print('Parameters:', params)
    print('Train:', len(train_data), '| Val:', len(val_data))
    print('Augmentation:', args.augment, '| Loss:', args.loss)
    history = []
    best = stop_best = -1.0
    wait = best_epoch = 0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss, _, _ = run_epoch(model, train_loader, args.loss, optimizer)
        val_loss, dice, iou = run_epoch(model, val_loader, args.loss)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_dice=dice, val_iou=iou,
                            time_sec=time.perf_counter() - tick))
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Dice: {dice:.4f} | IoU: {iou:.4f}')
        if dice > best:
            best, best_epoch = dice, epoch
            torch.save(model.state_dict(), output / 'best.pth')
            print('Best model saved')
        save_graphs(history, output, best_epoch)

        if dice > stop_best + args.min_delta:
            stop_best, wait = dice, 0
        else:
            wait += 1
        print(f'Early Stopping: {wait}/{args.patience}')
        if wait >= args.patience:
            break

    torch.save(model.state_dict(), output / 'last.pth')
    summary = (f'Parameters: {params}\nCompleted Epochs: {len(history)}\n'
               f'Best Epoch: {best_epoch}\nBest Dice: {best:.4f}\n'
               f'Best IoU: {history[best_epoch - 1]["val_iou"]:.4f}\n'
               f'Training Time: {time.perf_counter() - start:.1f}s\n')
    (output / 'summary.txt').write_text(summary)
    print(summary)


if __name__ == '__main__':
    main()
