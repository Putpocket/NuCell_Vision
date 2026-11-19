"""Torch-specific Dataset and Mask R-CNN construction."""
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from maskrcnn_utils import letterbox, load_sample, prepare_instances


class NucleiInstanceDataset(Dataset):
    def __init__(self, data_dir, ids, augment=False):
        self.data_dir, self.ids, self.augment = data_dir, list(ids), augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, original_masks = load_sample(self.data_dir / self.ids[index])
        image, geometry = letterbox(image)
        masks = np.stack([letterbox(m, is_mask=True)[0] for m in original_masks])
        if self.augment:
            # Same geometry for RGB and every instance; boxes are computed afterwards.
            for axis in (0, 1):
                if random.random() < 0.5:
                    image = np.flip(image, axis=axis)
                    masks = np.flip(masks, axis=axis + 1)
            k = random.randrange(4)
            image = np.rot90(image, k, axes=(0, 1))
            masks = np.rot90(masks, k, axes=(1, 2))
        image = image.astype(np.float32) / 255.0
        masks, boxes, dropped = prepare_instances(masks)
        target = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.ones(len(masks), dtype=torch.int64),
            'masks': torch.from_numpy(np.ascontiguousarray(masks)),
            'image_id': torch.tensor(index),
            'area': torch.from_numpy(masks.sum(axis=(1, 2)).astype(np.float32)),
            'iscrowd': torch.zeros(len(masks), dtype=torch.int64),
        }
        # Do not externally ImageNet-normalize: model.transform handles it once.
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        metadata = dict(geometry=geometry, image_id=self.ids[index],
                        original_count=len(original_masks), dropped=dropped)
        return tensor, target, metadata


def collate_instances(batch):
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def build_maskrcnn(pretrained=True, max_detections=500, score_threshold=0.5):
    options = dict(min_size=512, max_size=512, box_detections_per_img=max_detections,
                   box_score_thresh=score_threshold)
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights='DEFAULT', weights_backbone=None, **options)
    else:
        # torchvision's weights=None builder switches to ordinary BatchNorm.
        # Reconstruct the COCO architecture explicitly, including eps=0, offline.
        backbone = resnet_fpn_backbone(
            backbone_name='resnet50', weights=None,
            norm_layer=partial(FrozenBatchNorm2d, eps=0.0), trainable_layers=3,
        )
        model = MaskRCNN(backbone, num_classes=2, **options)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, 2)
    return model
