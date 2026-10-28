"""Load the exact model source recorded in a checkpoint, including after relocation."""
import importlib.util
from pathlib import Path
import sys

from common import sha256


def model_classes(config, training_source=None):
    expected = config.get('code_sha256', {})
    source = Path(training_source) if training_source is not None else Path(__file__).parent
    if 'ufm_model.py' not in expected:
        raise ValueError('Checkpoint has no model source fingerprint.')
    # An archived training snapshot must include every recorded source file.
    names = expected if training_source is not None else ['ufm_model.py']
    for name in names:
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Invalid training source filename.')
        path = source / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected[name]:
            raise ValueError('Model code does not match training checkpoint: ' + name)
    if training_source is None:
        from ufm_model import UFMConfig, UFMRec
        return UFMConfig, UFMRec
    module_name = '_ufm_training_' + expected['ufm_model.py']
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, source / 'ufm_model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # dataclasses resolves the defining module.
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    module = sys.modules[module_name]
    return module.UFMConfig, module.UFMRec
