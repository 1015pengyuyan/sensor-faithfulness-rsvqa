"""Rebuild accepted quantitative Figures 2–4. No dataset pixels or estimation."""
import argparse
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True,help='Extracted release root')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
src=a.data/'figures/source_data/source_data'
if not (src/'original_ALL_ENDPOINTS.json').is_file():
    raise FileNotFoundError('Missing exact article figure-source bundle')
for sub in ['figures','qa','tables']:(a.out/sub).mkdir(parents=True,exist_ok=True)
os.environ['SF_FIGURE_DATA']=str(src.resolve())
os.environ['SF_FIGURE_OUT']=str(a.out.resolve())
sys.path.insert(0,str(Path(__file__).resolve().parent/'article_figures'))
import build_figures
from public_display import apply_public_display
apply_public_display()
build_figures.capability()
build_figures.conflict()
build_figures.tables()
import build_v14_displays
apply_public_display()
rates=build_v14_displays.cr(src/'position_closeout/POSITION_COMMON_SET_RATES.csv')
outcomes=build_v14_displays.cr(src/'position_closeout/POSITION_OUTCOME_TRANSITIONS.csv')
build_v14_displays.figure(rates,outcomes)
(a.out/'NUMERIC_DISPLAY_INPUTS.json').write_text(json.dumps({
    'estimates':build_figures.legacy.SOURCES,'counts':build_figures.OUTCOMES,
    'display_version':'public-display-v2-dejavu','new_estimates':False},indent=2)+'\n',encoding='utf-8')
print('Public DejaVu display; accepted values unchanged; no inference, draws or V14.1 pixel-byte identity claim.')
