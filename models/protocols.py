"""Model identifiers and checkpoint/source metadata."""
import hashlib
from pathlib import Path

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
