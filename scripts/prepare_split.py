"""Create a scene-disjoint manifest only when ALL requested files exist."""
import argparse
import json
from pathlib import Path


def create(root, camera='realsense', stride=1, training_ready=False):
    if stride < 1 or stride > 256:
        raise ValueError('stride must be 1..256')
    # Fixed project validation reservation within official train scenes 0..99.
    val = set(range(9, 100, 10))
    groups = dict(train=[s for s in range(100) if s not in val], val=sorted(val),
                  test_seen=list(range(100,130)), test_similar=list(range(130,160)),
                  test_novel=list(range(160,190)))
    result = {'protocol': {'dataset': 'GraspNet-1Billion', 'camera': camera,
                          'frame_stride': stride, 'grouping': 'scene',
                          'validation': 'project-defined, every tenth official train scene'}}
    missing = []
    for part, scenes in groups.items():
        rows = []
        for scene in scenes:
            for frame in range(0, 256, stride):
                camera_path = Path('scenes') / ('scene_%04d' % scene) / camera
                for folder, ext in [('rgb', 'png'), ('depth', 'png'), ('rect', 'npy')]:
                    p = root / camera_path / folder / ('%04d.%s' % (frame, ext))
                    if not p.is_file() and not (training_ready and part.startswith("test_") and folder == "rect"):
                        missing.append(str(p))
                rows.append(dict(sample=str(camera_path / ('%04d' % frame)),
                                 scene_id=scene, object_id='scene_%04d' % scene))
        result[part] = rows
    if missing:
        raise FileNotFoundError('%d required files missing; first: %s. No partial split written.' %
                                (len(missing), missing[0]))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--camera', choices=['realsense', 'kinect'], default='realsense')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument("--training-ready", action="store_true", help="Require all images and train/val labels; retain ALL planned test rows even if test labels are pending")
    a = p.parse_args()
    split = create(a.data, a.camera, a.stride, a.training_ready)
    with a.out.open('x') as f:
        json.dump(split, f, indent=2)
    print({k:len(v) for k,v in split.items() if k != 'protocol'})
