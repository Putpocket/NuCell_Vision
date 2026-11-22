import tempfile
import unittest
import ast
from pathlib import Path

import cv2
import numpy as np

try:
    import maskrcnn_utils as u
except ModuleNotFoundError:
    u = None


class PreprocessingTests(unittest.TestCase):
    """전처리, 복원, 분할 검증과 마스크 평가 규칙을 확인합니다."""

    def setUp(self):
        self.assertIsNotNone(u, 'Mask R-CNN preprocessing has not been implemented')

    def test_odd_padding_and_resize_match_unet(self):
        # 홀수 패딩과 리사이즈 결과가 U-Net 전처리와 같은지 비교합니다.
        image = np.full((3, 8, 3), 200, np.uint8)
        out, geom = u.letterbox(image)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual((geom.height, geom.width, geom.top, geom.left), (192, 512, 160, 0))
        self.assertTrue((out[:160] == 0).all())
        self.assertTrue((out[160:352] == 200).all())

    def test_restore_removes_padding_before_resize(self):
        # 패딩을 먼저 제거한 뒤 원본 크기로 복원하는지 확인합니다.
        _, geom = u.letterbox(np.zeros((3, 8, 3), np.uint8))
        probability = np.zeros((512, 512), np.float32)
        probability[geom.top:geom.top + geom.height, :] = 0.75
        restored = u.restore_probability(probability, geom)
        self.assertEqual(restored.shape, (3, 8))
        np.testing.assert_allclose(restored, 0.75)

    def test_instances_stay_separate_and_single_pixel_box_has_area(self):
        # 핵 마스크가 분리되고 한 픽셀 박스도 유효한 크기를 갖는지 확인합니다.
        masks = np.zeros((2, 512, 512), np.uint8)
        masks[0, 5, 7] = 1
        masks[1, 20:25, 30:40] = 1
        boxed_masks, boxes, dropped = u.prepare_instances(masks)
        self.assertEqual(boxed_masks.shape, (2, 512, 512))
        self.assertEqual(dropped, 0)
        np.testing.assert_array_equal(boxes, [[7, 5, 8, 6], [30, 20, 40, 25]])

    def test_empty_resized_instances_are_removed(self):
        # 크기 변환 후 빈 마스크만 학습 대상에서 제외되는지 확인합니다.
        masks = np.zeros((2, 512, 512), np.uint8)
        masks[1, 1:3, 1:3] = 1
        result, boxes, dropped = u.prepare_instances(masks)
        self.assertEqual((len(result), len(boxes), dropped), (1, 1, 1))

    def test_overlapping_predictions_are_merged_not_added(self):
        # 겹친 예측 영역은 합집합으로 만들고 인스턴스 수는 따로 세는지 확인합니다.
        masks = np.zeros((2, 4, 4), np.float32)
        masks[0, :2, :2] = 0.8
        masks[1, :2, :2] = 0.9
        union, count = u.prediction_union(masks, np.array([0.9, 0.9]), 0.5, 0.5)
        self.assertEqual((int(union.sum()), count), (4, 2))
        self.assertEqual(u.binary_metrics(union, union), (1.0, 1.0))

    def test_zero_predictions_and_score_filter(self):
        # 예측이 없거나 신뢰도 기준 미달이면 빈 마스크로 처리하는지 확인합니다.
        union, count = u.prediction_union(np.zeros((0, 4, 4)), np.zeros(0), 0.5, 0.5)
        self.assertEqual((union.shape, int(union.sum()), count), ((4, 4), 0, 0))
        gt = np.zeros((4, 4), bool)
        gt[1, 1] = True
        dice, iou = u.binary_metrics(union, gt)
        self.assertLess(dice, 1e-6)
        self.assertLess(iou, 1e-6)
        union, count = u.prediction_union(np.ones((1, 4, 4)), np.array([0.4]), 0.5, 0.5)
        self.assertEqual((int(union.sum()), count), (0, 0))

    def test_partial_overlap_metrics_are_image_metrics(self):
        # 부분적으로 겹치는 픽셀 영역의 Dice와 IoU를 확인합니다.
        pred = np.array([[1, 1, 0]], bool)
        gt = np.array([[0, 1, 1]], bool)
        dice, iou = u.binary_metrics(pred, gt)
        self.assertAlmostEqual(dice, 0.5, places=6)
        self.assertAlmostEqual(iou, 1 / 3, places=6)

    def test_duplicate_or_overlapping_splits_rejected(self):
        # split 내부 중복과 split 간 이미지 중복을 거절하는지 확인합니다.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'train.txt').write_text('a\nb\n')
            (path / 'val.txt').write_text('b\n')
            (path / 'test.txt').write_text('c\n')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                u.read_splits(path, expected=(2, 1, 1))

    def test_read_rgb_and_masks_preserves_instance_count(self):
        # 이미지 RGB 순서와 개별 핵 마스크 수가 보존되는지 확인합니다.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'images').mkdir()
            (path / 'masks').mkdir()
            cv2.imwrite(str(path / 'images/a.png'), np.full((4, 6, 3), (10, 20, 30), np.uint8))
            mask = np.zeros((4, 6), np.uint8)
            mask[1, 2] = 255
            cv2.imwrite(str(path / 'masks/one.png'), mask)
            image, masks = u.load_sample(path)
            np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
            self.assertEqual((masks.shape, int(masks.sum())), ((1, 4, 6), 1))

    def test_letterbox_matches_existing_unet_for_landscape_portrait_and_downscale(self):
        # 가로·세로·축소 입력에서 U-Net과 letterbox 결과가 일치하는지 확인합니다.
        source = Path(__file__).resolve().parents[1] / 'train_unet_improved.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'letterbox')
        namespace = dict(np=np, cv2=cv2, SIZE=512)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<upstream letterbox>', 'exec'), namespace)
        rng = np.random.default_rng(12)
        for height, width in [(3, 8), (101, 333), (1025, 777), (777, 1025), (512, 512), (1, 2000)]:
            image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for array, is_mask in [(image, False), ((image[:, :, 0] > 127).astype(np.uint8), True)]:
                expected, valid = namespace['letterbox'](array, is_mask)
                actual, geometry = u.letterbox(array, is_mask)
                np.testing.assert_array_equal(actual, expected)
                region = np.zeros((512, 512), np.float32)
                region[geometry.region] = 1
                np.testing.assert_array_equal(region, valid)

    def test_repository_splits_are_fixed_and_disjoint(self):
        # 저장소의 공통 분할 크기와 분할 간 독립성을 확인합니다.
        groups = u.read_splits(Path(__file__).resolve().parents[1] / 'splits')
        self.assertEqual([len(groups[n]) for n in ('train', 'val', 'test')], [469, 100, 101])

    def test_original_count_excludes_empty_masks_after_unpadding(self):
        # 패딩 영역만 차지한 예측은 원본 복원 후 핵 수에서 제외하는지 확인합니다.
        _, geometry = u.letterbox(np.zeros((3, 8, 3), np.uint8))
        masks = np.zeros((2, 512, 512), np.float32)
        masks[0, :100] = 1  # Prediction exists only in padding: not a nucleus.
        masks[1, 160:352] = 1
        union, count = u.prediction_union(masks, np.array([0.9, 0.9]), geometry=geometry)
        self.assertEqual((union.shape, int(union.sum()), count), ((3, 8), 24, 1))

    def test_high_bit_depth_rgb_decode_matches_unet_pillow_conversion(self):
        # 높은 비트 PNG도 U-Net과 같은 Pillow RGB 변환 결과를 내는지 확인합니다.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'images').mkdir()
            (path / 'masks').mkdir()
            cv2.imwrite(str(path / 'images/a.png'), np.full((2, 2), 257, np.uint16))
            cv2.imwrite(str(path / 'masks/one.png'), np.full((2, 2), 255, np.uint8))
            image, _ = u.load_sample(path)
            np.testing.assert_array_equal(image[0, 0], [255, 255, 255])


if __name__ == '__main__':
    unittest.main()
