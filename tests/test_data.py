import sys,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import cv2
import numpy as np
from download_data import selected
from prepare_split import create
from dataset import GraspNet
from grasp import targets


class DataTest(unittest.TestCase):
    def test_mirror_label_layout_and_frame_filter(self):
        expected=Path('scenes/scene_0017/realsense/rect/0016.npy')
        self.assertEqual(selected('rect_labels/scene_0017/realsense/0016.npy',16),expected)
        self.assertEqual(selected('scenes/scene_0017/realsense/rect/0016.npy',16),expected)
        self.assertIsNone(selected('rect_labels/scene_0017/realsense/0017.npy',16))
        self.assertIsNone(selected('rect_labels/scene_0017/kinect/0016.npy',16))

    def test_selected_frame_scene_partitions(self):
        with patch.object(Path,'is_file',return_value=True):s=create(Path('/unused'),stride=16)
        self.assertEqual([len(s[k]) for k in ('train','val','test_seen','test_similar','test_novel')],[1440,160,480,480,480])
        sets=[{r['scene_id'] for r in s[k]} for k in ('train','val','test_seen','test_similar','test_novel')]
        self.assertEqual(len(set.union(*sets)),190)
        self.assertTrue(all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]))

    def test_cached_targets_preserve_all_rotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cam=root/'scenes/scene_0000/realsense'
            for f in ('rgb','depth','rect'):(cam/f).mkdir(parents=True)
            cv2.imwrite(str(cam/'rgb/0000.png'),np.ones((48,64,3),np.uint8)*120)
            cv2.imwrite(str(cam/'depth/0000.png'),np.ones((48,64),np.uint16)*1000)
            np.save(cam/'rect/0000.npy',[[32,24,42,28,10,.8,1],[30,20,25,21,5,.7,2]])
            sp=root/'split.json';sp.write_text(json.dumps({'train':[dict(sample='scenes/scene_0000/realsense/0000',object_id='scene_0000')]}))
            ds=GraspNet(root,sp,'train',64)
            _,_,rs=ds.base(0)
            for k in range(4):
                expected=targets(rs,64)
                np.testing.assert_array_equal(ds.target_for(0,rs,k),expected)
                ds._target_cache={}
                np.testing.assert_array_equal(ds.target_for(0,rs,k),expected)
                rs=np.stack([rs[...,1],63-rs[...,0]],-1)


if __name__=='__main__':unittest.main()
