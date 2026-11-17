"""Create architectures; historical identifiers stay stable for checkpoints."""
from legacy.tiny_unet import Detector
from losses import LOSS_PROFILES

NAMES = ('grconvnet', 'ggcnn2', 'ggcnn2_depth', 'ggcnn', 'ggcnn_depth', 'dual_concat', 'early_concat')



def build_model(name='tiny_unet', loss_profile='original'):
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f'Unknown loss profile: {loss_profile}')
    if name == 'tiny_unet' and loss_profile != 'original':
        raise ValueError('Foreground balancing is for GR-ConvNet only')
    if name == 'tiny_unet':
        return Detector()
    if name in ('dual_concat', 'early_concat'):
        from .fusion import DualGraspNet
        model = DualGraspNet(use_gate=False, early_fusion=name == 'early_concat')
        model.architecture = name
        model.loss_profile = loss_profile
        return model
    if name in ('ggcnn', 'ggcnn_depth'):
        from .ggcnn import GGCNNAdapter
        model = GGCNNAdapter(depth_only=name == 'ggcnn_depth')
        model.loss_profile = loss_profile
        return model
    if name in ('ggcnn2', 'ggcnn2_depth'):
        from .ggcnn2 import GGCNN2Adapter
        model = GGCNN2Adapter(depth_only=name == 'ggcnn2_depth')
        model.loss_profile = loss_profile
        return model
    if name == 'grconvnet':
        from .grconvnet import GRConvNet
        model = GRConvNet()
        model.loss_profile = loss_profile
        return model
    raise ValueError(f'Unknown model {name!r}')
