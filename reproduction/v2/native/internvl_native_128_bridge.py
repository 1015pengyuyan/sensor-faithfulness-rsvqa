"""Load the preserved 128-token InternVL adapter and its corrected raw-token trace."""
import hashlib
import importlib.util
from pathlib import Path
import sys

PINNED = {
    'native_internvl_128.py': '9dc924ec2585d5f562e2723e35b88d51ad396c884bf5c676f9fc58e7efdbcfb6',
    'internvl_trace.py': '34242c2ed428473ecf59d7d50c3004aa3cb2d0a45a23a87f48f681c5361721e1',
}


def load_pinned_implementation():
    root = Path(__file__).resolve().parent / 'pinned_internvl_128'
    for name, digest in PINNED.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('fixed InternVL implementation hash differs: ' + name)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('pinned_internvl_matrix128', root / 'native_internvl_128.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InternVLNative128Adapter:
    @classmethod
    def from_pretrained(cls, model_path, max_new_tokens=128):
        if max_new_tokens != 128:
            raise ValueError('frozen unified protocol requires max_new_tokens=128')
        native = load_pinned_implementation()
        return native.NativeInternAdapter(model_path, max_new_tokens=128)
