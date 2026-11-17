"""Exercise configuration and the public training/evaluation entry points."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.server import discover_jobs
from utils.config import parse_with_config


class ConfigTest(unittest.TestCase):
    def test_cli_overrides_preset_and_validates_keys(self):
        def parser():
            p = argparse.ArgumentParser()
            p.add_argument('--epochs', type=int, required=True)
            p.add_argument('--models', nargs='+', choices=['ggcnn', 'ggcnn2'])
            p.add_argument('--resume', action='store_true')
            return p

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps(dict(epochs=20, models=['ggcnn', 'ggcnn2'], resume=True)))
            args = parse_with_config(parser(), ['--config', str(config), '--epochs', '1', '--models', 'ggcnn2'])
            self.assertEqual((args.epochs, args.models, args.resume), (1, ['ggcnn2'], True))
            self.assertEqual(args.config, str(config))
            for invalid in ({'epohs': 20}, {'epochs': 'invalid'}, {'epochs': 20, 'resume': 'false'}):
                config.write_text(json.dumps(invalid))
                with self.subTest(config=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_with_config(parser(), ['--config', str(config)])

    def test_dashboard_discovers_queued_models_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'comparison_protocol.json').write_text(json.dumps(dict(
                main_models=['ggcnn2', 'dual_concat'], ablation=['dual_concat', 'early_concat'], seeds=[42, 43])))
            jobs = discover_jobs([root, root])
            self.assertEqual(len(jobs), 6)
            self.assertIn('early_concat_s43', [j[2] for j in jobs])


class WorkflowTest(unittest.TestCase):
    def test_train_resume_and_evaluate_via_public_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = root / 'scenes/scene_0000/realsense'
            for folder in ('rgb', 'depth', 'rect'):
                (cam / folder).mkdir(parents=True)
            cv2.imwrite(str(cam / 'rgb/0000.png'), np.full((48, 64, 3), 120, np.uint8))
            cv2.imwrite(str(cam / 'depth/0000.png'), np.full((48, 64), 1000, np.uint16))
            np.save(cam / 'rect/0000.npy', [[32, 24, 42, 28, 10, .8, 1]])
            row = dict(sample='scenes/scene_0000/realsense/0000', object_id='scene_0000')
            split = root / 'split.json'
            # Deliberately shared synthetic frame: plumbing test, not a benchmark.
            split.write_text(json.dumps({part: [row] for part in ('train', 'val', 'test_seen')}))
            config = root / 'preset.json'
            preset = json.loads((ROOT / 'configs/ggcnn2_depth.json').read_text())
            preset.update(data=str(root), split=str(split), out=str(root / 'resumed'),
                          size=64, epochs=1, batch=1, threads=2, device='cpu')
            config.write_text(json.dumps(preset))
            env = dict(os.environ, GRASPNET_TARGET_CACHE=str(root / 'targets'), MPLCONFIGDIR=str(root / 'matplotlib'))

            def run(entry, *args):
                result = subprocess.run([sys.executable, str(ROOT / entry), *args], cwd=ROOT,
                                        env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run('train.py', '--config', str(config))
            run('train.py', '--config', str(config), '--epochs', '2', '--resume')
            run('run.py', 'train', '--config', str(config), '--epochs', '2', '--out', str(root / 'continuous'))
            resumed = torch.load(root / 'resumed/last.pt', map_location='cpu', weights_only=False)
            continuous = torch.load(root / 'continuous/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['epoch'], 2)
            self.assertEqual(len(resumed['history']), 2)
            for key, value in resumed['model'].items():
                torch.testing.assert_close(value, continuous['model'][key], rtol=0, atol=0)
            run('evaluate.py', '--config', str(config), '--checkpoint', str(root / 'resumed/best.pt'),
                '--out', str(root / 'resumed/test_seen'), '--part', 'test_seen', '--no-visuals')
            result = root / 'resumed/test_seen'
            metrics = json.loads((result / 'metrics.json').read_text())
            self.assertEqual(len(metrics), 11)
            self.assertTrue(all(row['n_images'] == 1 for row in metrics))
            self.assertEqual(len((result / 'metrics.csv').read_text().splitlines()), 12)
            self.assertEqual(len(json.loads((result / 'predictions.json').read_text())), 11)
            self.assertTrue((result / 'robustness.png').is_file())


if __name__ == '__main__':
    unittest.main()
