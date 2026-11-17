"""Named architectures and output protocols. Legacy checkpoints default to tiny_unet."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from grasp import Detector, box, loss_fn, decode

NAMES = ('grconvnet', 'ggcnn2', 'ggcnn2_depth', 'ggcnn', 'ggcnn_depth', 'dual_concat', 'early_concat')


LOSS_PROFILES = ('original', 'foreground_balanced')

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
        from .ggcnn import GGCNNRGBD
        model = GGCNNRGBD(depth_only=name == 'ggcnn_depth')
        model.loss_profile = loss_profile
        return model
    if name in ('ggcnn2', 'ggcnn2_depth'):
        from .ggcnn2 import GGCNN2RGBD
        model = GGCNN2RGBD(depth_only=name == 'ggcnn2_depth')
        model.loss_profile = loss_profile
        return model
    if name == 'grconvnet':
        from .grconvnet import GRConvNet
        model = GRConvNet()
        model.loss_profile = loss_profile
        return model
    raise ValueError(f'Unknown model {name!r}')


def model_name(model):
    return getattr(model, 'architecture', 'tiny_unet')


def metadata(model):
    name = model_name(model)
    info = {'model_name': name, 'output_protocol': 'legacy_logits_v1',
            'loss_profile': getattr(model, 'loss_profile', 'original')}
    if name in ('ggcnn', 'ggcnn_depth', 'dual_concat', 'early_concat'):
        info.update(output_protocol=name + '_linear_half_size_v1', pretrained=False)
    if name in ('grconvnet', 'ggcnn2', 'ggcnn2_depth'):
        source_name = 'ggcnn2' if name == 'ggcnn2_depth' else name
        path = Path(__file__).resolve().parents[1] / ('third_party/' + source_name + '/SOURCE.json')
        import json
        source = json.loads(path.read_text())
        for file, hashes in source['files'].items():
            if hashlib.sha256((path.parent / file).read_bytes()).hexdigest() != hashes['local_sha256']:
                raise ValueError(f'Vendored source changed: {file}')
        info.update(output_protocol=name + '_linear_half_size_v1',
                    upstream_commit=source['commit'], upstream_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    pretrained=False, network_dropout_probability=0.0)
    return info


def grconv_targets(y):
    """Upstream uses cos(0)=1 on background; width is normalized by half size."""
    target = y.clone()
    target[:, 1] = torch.where(y[:, 0] > 0, y[:, 1], torch.ones_like(y[:, 1]))
    target[:, 3] = (y[:, 3] * 2).clamp(0, 1)
    return target


def model_loss(model, output, target):
    if model_name(model) == 'tiny_unet':
        return loss_fn(output, target)
    target = grconv_targets(target)
    if getattr(model, 'loss_profile', 'original') == 'foreground_balanced':
        # Project adaptation: equal foreground/background quality weight;
        # angle/width matter only where a grasp is annotated.
        mask = target[:, 0:1]
        quality = torch.nn.functional.smooth_l1_loss(
            output[:, 0:1].contiguous(), mask.contiguous(), reduction='none')
        fg = mask.sum().clamp_min(1)
        bg = (1 - mask).sum().clamp_min(1)
        loss = .5 * ((quality * mask).sum() / fg + (quality * (1-mask)).sum() / bg)
        for i in range(1, 4):
            error = torch.nn.functional.smooth_l1_loss(
                output[:, i:i+1].contiguous(), target[:, i:i+1].contiguous(), reduction='none')
            loss = loss + (error * mask).sum() / fg
        return loss
    if model_name(model) in ('ggcnn', 'ggcnn_depth', 'ggcnn2', 'ggcnn2_depth'):
        return sum(torch.nn.functional.mse_loss(output[:, i:i+1], target[:, i:i+1]) for i in range(4))
    # Exactly the four unmasked Smooth-L1 terms in upstream GraspModel.compute_loss.
    return sum(torch.nn.functional.smooth_l1_loss(output[:, i:i+1].contiguous(), target[:, i:i+1].contiguous()) for i in range(4))


def decode_output(model, output):
    if model_name(model) == 'tiny_unet':
        return decode(output)
    o = output.detach().cpu().numpy()
    # Linear heads: no sigmoid. scipy mode nearest matches skimage gaussian default.
    q = gaussian_filter(o[0], 2, mode='nearest')
    angle = gaussian_filter(np.arctan2(o[2], o[1]) / 2, 2, mode='nearest')
    width = gaussian_filter(o[3] * (o.shape[-1] / 2), 1, mode='nearest')
    y, x = np.unravel_index(q.argmax(), q.shape)
    w = max(2.0, float(width[y, x]))
    return box([x, y], float(angle[y, x]), w, w / 2), q
