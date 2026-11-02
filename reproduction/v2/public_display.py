"""Display-only font policy; never transforms scientific values."""
import hashlib
import json
from pathlib import Path


def apply_public_display(font_root=None):
    import matplotlib
    from matplotlib import font_manager
    manifest=json.loads((Path(__file__).parent/'PUBLIC_DISPLAY.json').read_text(encoding='utf-8'))
    root=Path(font_root) if font_root is not None else Path(matplotlib.get_data_path())/'fonts/ttf'
    for r in manifest['files']:
        p=root/r['file']
        if not p.is_file():raise FileNotFoundError('Missing fixed public font: '+r['file'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Public font identity differs: '+r['file'])
        font_manager.fontManager.addfont(str(p))
    matplotlib.rcParams.update({'font.family':[manifest['family']],
        'font.sans-serif':[manifest['family']],'mathtext.fontset':'dejavusans',
        'pdf.fonttype':42,'svg.fonttype':'none'})
    for r in manifest['files']:
        props=font_manager.FontProperties(family=manifest['family'],weight=r['weight'])
        resolved=Path(font_manager.findfont(props,fallback_to_default=False))
        if hashlib.sha256(resolved.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Font resolver substituted an unverified face')
    return {'display_version':manifest['display_version'],'family':manifest['family']}
