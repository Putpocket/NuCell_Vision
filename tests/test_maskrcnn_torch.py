"""Server-side contract checks: no pretrained downloads and no training."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HAS_TORCH = (importlib.util.find_spec('torch') is not None
             and importlib.util.find_spec('torchvision') is not None)


@unittest.skipUnless(HAS_TORCH, 'PyTorch/torchvision not installed; run these checks on the server')
class TorchContractTests(unittest.TestCase):
    """PyTorch 모델 구성과 Dataset 반환 형식의 약속을 확인합니다."""

    def test_offline_checkpoint_model_keeps_coco_frozen_norm_and_freezing(self):
        # 사전학습 다운로드 없이 COCO 체크포인트와 호환되는 층 구성을 확인합니다.
        import torch
        from torchvision.ops.misc import FrozenBatchNorm2d
        from maskrcnn_model import build_maskrcnn
        model = build_maskrcnn(pretrained=False)
        norms = [m for m in model.modules() if isinstance(m, FrozenBatchNorm2d)]
        self.assertGreater(len(norms), 0, 'Offline reconstruction changed COCO backbone normalization')
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        self.assertTrue(all(m.eps == 0 for m in norms))
        self.assertFalse(model.backbone.body.conv1.weight.requires_grad)
        self.assertTrue(model.backbone.body.layer4[0].conv1.weight.requires_grad)

    def test_model_normalizes_once_and_keeps_actual_input_at_512(self):
        # 모델 내부 정규화가 한 번만 적용되고 입력 크기가 512로 유지되는지 봅니다.
        import torch
        from maskrcnn_model import build_maskrcnn
        model = build_maskrcnn(pretrained=False)
        model.eval()
        with torch.inference_mode():
            transformed, _ = model.transform([torch.full((3, 512, 512), 0.5)])
        self.assertEqual(tuple(transformed.tensors.shape), (1, 3, 512, 512))
        self.assertAlmostEqual(transformed.tensors[0, 0, 10, 10].item(), (0.5 - 0.485) / 0.229, places=5)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        self.assertEqual(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels, 2)

    def test_dataset_returns_individual_masks_boxes_and_unnormalized_rgb(self):
        # Dataset이 RGB 이미지와 핵별 마스크·박스를 기대한 형식으로 반환하는지 봅니다.
        from maskrcnn_model import NucleiInstanceDataset, collate_instances
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a/images').mkdir(parents=True)
            (root / 'a/masks').mkdir()
            cv2.imwrite(str(root / 'a/images/a.png'), np.full((512, 512, 3), 128, np.uint8))
            for i, (y, x) in enumerate([(5, 7), (20, 30)]):
                mask = np.zeros((512, 512), np.uint8)
                mask[y, x] = 255
                cv2.imwrite(str(root / f'a/masks/{i}.png'), mask)
            sample = NucleiInstanceDataset(root, ['a'])[0]
            image, target, metadata = sample
            self.assertAlmostEqual(image[0, 0, 0].item(), 128 / 255, places=6)
            self.assertEqual(target['masks'].shape[0], 2)
            self.assertEqual(target['labels'].tolist(), [1, 1])
            self.assertEqual(target['boxes'].tolist(), [[7, 5, 8, 6], [30, 20, 31, 21]])
            self.assertEqual(metadata['original_count'], 2)
            self.assertEqual(len(collate_instances([sample, sample])[0]), 2)


if __name__ == '__main__':
    unittest.main()
