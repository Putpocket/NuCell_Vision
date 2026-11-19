"""Shared preprocessing/metrics; no torch import or model downloads."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    original_height: int
    original_width: int
    height: int
    width: int
    top: int
    left: int
    size: int

    @property
    def region(self):
        return (slice(self.top, self.top + self.height),
                slice(self.left, self.left + self.width))


def letterbox(array, is_mask=False, size=512):
    """Match U-net letterbox rounding, interpolation, zero padding exactly."""
    h, w = array.shape[:2]
    if min(h, w, size) < 1:
        raise ValueError('Image dimensions and size must be positive')
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    method = (cv2.INTER_NEAREST_EXACT if is_mask else
              cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized = cv2.resize(array, (nw, nh), interpolation=method)
    top, left = (size - nh) // 2, (size - nw) // 2
    output = np.zeros((size, size) + array.shape[2:], dtype=array.dtype)
    output[top:top + nh, left:left + nw] = resized
    return output, Geometry(h, w, nh, nw, top, left, size)


def restore_probability(probability, geometry):
    if probability.shape != (geometry.size, geometry.size):
        raise ValueError('Prediction does not match letterbox dimensions')
    cropped = probability[geometry.region].astype(np.float32)
    return cv2.resize(cropped, (geometry.original_width, geometry.original_height),
                      interpolation=cv2.INTER_LINEAR)


def prepare_instances(masks):
    """Remove masks lost in resizing; exclusive max coordinates keep 1px boxes valid."""
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 3:
        raise ValueError('Masks must have shape [N,H,W]')
    keep, boxes = [], []
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            keep.append(i)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return (masks[keep], np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            len(masks) - len(keep))


def load_sample(folder):
    from PIL import Image
    folder = Path(folder)
    paths = sorted((folder / 'images').glob('*.png'))
    if len(paths) != 1:
        raise ValueError(f'Expected one image: {folder}')
    # Same decoder/conversion as U-Net, including high-bit-depth PNG behavior.
    with Image.open(paths[0]) as source:
        image = np.array(source.convert('RGB'))
    masks = []
    for path in sorted((folder / 'masks').glob('*.png')):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2] or not mask.any():
            raise ValueError(f'Invalid or empty mask: {path}')
        masks.append((mask > 0).astype(np.uint8))
    if not masks:
        raise ValueError(f'No annotated nuclei: {folder}')
    return image, np.stack(masks)


def read_splits(folder, expected=(469, 100, 101)):
    result = {}
    for name, count in zip(('train', 'val', 'test'), expected):
        ids = (Path(folder) / f'{name}.txt').read_text(encoding='utf-8').splitlines()
        if len(ids) != count or len(set(ids)) != count or any(not x.strip() for x in ids):
            raise ValueError(f'Unexpected count, blank or duplicate IDs in {name}')
        if any('/' in x or '\\' in x or x in ('.', '..') for x in ids):
            raise ValueError('Image IDs must be directory names')
        result[name] = ids
    groups = list(map(set, result.values()))
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Dataset splits overlap')
    return result


def iter_prediction_masks(masks, scores, score_threshold, mask_threshold, geometry=None):
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError('Expected masks [N,H,W] and scores [N]')
    for mask, score in zip(masks, scores):
        if score < score_threshold:
            continue
        probability = restore_probability(mask, geometry) if geometry else mask
        binary = probability > mask_threshold
        if binary.any():
            yield binary


def prediction_union(masks, scores, score_threshold=0.5, mask_threshold=0.5, geometry=None):
    shape = ((geometry.original_height, geometry.original_width)
             if geometry else masks.shape[-2:])
    union = np.zeros(shape, dtype=bool)
    count = 0
    for binary in iter_prediction_masks(masks, scores, score_threshold, mask_threshold, geometry):
        union |= binary
        count += 1
    return union, count


def binary_metrics(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    area = prediction.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    eps = 1e-7
    return ((2 * intersection + eps) / (area + eps),
            (intersection + eps) / (area - intersection + eps))
