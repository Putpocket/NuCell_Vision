import subprocess
import sys
import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    """학습·평가 명령 옵션이 잘못된 입력을 안전하게 거절하는지 확인합니다."""

    def test_more_than_100_epochs_is_rejected_without_starting_training(self):
        from train_maskrcnn import parse_args
        with patch.object(sys, 'argv', ['train_maskrcnn.py', '--epochs', '101', '--output', 'unused']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_training_help_does_not_load_model(self):
        # 도움말 호출만으로 학습이나 가중치 다운로드가 시작되지 않아야 합니다.
        process = subprocess.run([sys.executable, str(ROOT / 'train_maskrcnn.py'), '--help'],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--epochs', process.stdout)
        self.assertIn('--output', process.stdout)

    def test_evaluation_help_does_not_load_model(self):
        # 평가 스크립트의 주요 옵션을 학습 없이 확인합니다.
        process = subprocess.run([sys.executable, str(ROOT / 'evaluate_maskrcnn.py'), '--help'],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--checkpoint', process.stdout)

    def test_training_rejects_invalid_epochs_before_loading_data(self):
        # 유효하지 않은 에폭 수는 데이터 로딩 전에 오류로 처리합니다.
        process = subprocess.run([sys.executable, str(ROOT / 'train_maskrcnn.py'),
                                  '--epochs', '0', '--output', 'unused'],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn('epochs', process.stderr)


if __name__ == '__main__':
    unittest.main()
