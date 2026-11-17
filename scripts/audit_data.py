"""Check all selected RGB/depth/rectangle files before permitting training."""
import hashlib,json,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from dataset import load_rectangles

parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
split=json.loads((ROOT/'split.json').read_text())
counts={}; hashes={}; empty=[]; scene_sets={}; missing_test=[]
for part,rows in split.items():
    if part=='protocol':continue
    counts[part]=len(rows);scene_sets[part]=set()
    for row in rows:
        p=ROOT/'data'/row['sample'];cam=p.parent
        rgb=cam/'rgb'/(p.name+'.png');depth=cam/'depth'/(p.name+'.png');rect=cam/'rect'/(p.name+'.npy')
        image=cv2.imread(str(rgb));d=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
        if image is None or d is None or image.shape[:2]!=d.shape:
            raise ValueError('Invalid image pair: '+row['sample'])
        if args.training_only and part.startswith('test_') and not rect.exists():
            missing_test.append(row['sample'])
        else:
            r=load_rectangles(rect)
            if not len(r):empty.append(row['sample'])
        h=hashlib.sha256(rgb.read_bytes()).hexdigest()
        if h in hashes and hashes[h]!=part:raise ValueError('Identical RGB across partitions')
        hashes[h]=part;scene_sets[part].add(row['scene_id'])
for i,(part,ss) in enumerate(scene_sets.items()):
    for other in list(scene_sets)[i+1:]:
        if ss&scene_sets[other]:raise ValueError('Scene leakage')
report=dict(training_only=args.training_only,missing_test_labels=missing_test,counts=counts,scene_disjoint=True,rgb_hash_disjoint=True,empty_label_samples=empty,
            physical_object_disjoint=False,frame_stride=split['protocol']['frame_stride'])
(ROOT/('audit/training_data_audit.json' if args.training_only else 'audit/data_audit.json')).write_text(json.dumps(report,indent=2))
if empty:raise ValueError('Empty labels found. Review before training; do not silently drop samples.')
print(json.dumps(report,indent=2))
