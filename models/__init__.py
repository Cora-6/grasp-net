"""Public compatibility API. Implementations live in named modules."""
from .protocols import metadata, model_name

__all__ = ['build_model', 'NAMES', 'metadata', 'model_name', 'model_loss',
           'grconv_targets', 'LOSS_PROFILES', 'decode_output']

def __getattr__(name):
    if name in ('build_model', 'NAMES'):
        from . import registry
        return getattr(registry, name)
    if name in ('model_loss', 'grconv_targets', 'LOSS_PROFILES'):
        import losses
        return getattr(losses, name)
    if name == 'decode_output':
        from evaluation.decoder import decode_output
        return decode_output
    raise AttributeError(name)
