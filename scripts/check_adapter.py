"""Synthetic interface checks only: these are NOT dataset experiment results."""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import GraspNet, load_rectangles
from models import build_model, model_loss
from prepare_split import create


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    camera = root / 'scenes/scene_0000/realsense'
    for folder in ('rgb', 'depth', 'rect'):
        (camera / folder).mkdir(parents=True)
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[:, :, 2] = 255
    depth = np.full((48, 64), 1000, np.uint16)
    depth[0, 0] = 0
    cv2.imwrite(str(camera / 'rgb/0000.png'), rgb)
    cv2.imwrite(str(camera / 'depth/0000.png'), depth)
    np.save(camera / 'rect/0000.npy', [[32,24,42,24,10,0.8,1]])
    rs = load_rectangles(camera / 'rect/0000.npy')
    np.testing.assert_allclose(rs[0], [[22,19],[42,19],[42,29],[22,29]])
    split = root / 'split.json'
    split.write_text(json.dumps({'train':[dict(sample='scenes/scene_0000/realsense/0000',
                                               object_id='scene_0000')]}))
    ds = GraspNet(root, split, 'train', 64)
    x, y, rs, color, d = ds.sample(0)
    assert d.max() == 1.0 and d.min() == 0
    assert color[20,20].tolist() == [255,0,0]
    np.testing.assert_allclose(rs[0].mean(0), [32,32])
    assert x.shape == (4,64,64) and y[0].sum() > 0
    model = build_model('ggcnn2', 'foreground_balanced')
    output = model(torch.from_numpy(x[None]))
    loss = model_loss(model, output, torch.from_numpy(y[None]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    try:
        create(root)
        raise AssertionError('Incomplete dataset accepted')
    except FileNotFoundError:
        pass
    with patch.object(Path, 'is_file', return_value=True):
        full = create(root)
    counts = {k:len(v) for k,v in full.items() if k != 'protocol'}
    assert counts == dict(train=23040,val=2560,test_seen=7680,test_similar=7680,test_novel=7680)
    sets = [{r['scene_id'] for r in full[k]} for k in counts]
    assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])

report = dict(status='passed', real_data_test=False, training_completed=False,
              checks=['rectangle_width_and_coordinates', 'millimetres_to_metres',
                      'rgb_channel_order', 'letterbox_coordinates',
                      'GGCNN2_forward_backward_finite', 'reject_missing_files',
                      'full_split_counts_and_scene_disjointness'],
              planned_full_counts=counts)
(ROOT / 'audit/adapter_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
