"""CPU-only replay of saved evidence. No inference or random draw generation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sensor_faithfulness.evaluation.parser import parse_answer
from public_io import sha, check_images


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    if not rows:
        raise ValueError('No rows: missing data are not replaced with synthetic results')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def verify(data):
    manifests = sorted((data / 'manifests').glob('*_FILES.json'))
    if not manifests:
        raise FileNotFoundError('Extract the release attachments first; no release manifest found')
    count = 0
    for mf in manifests:
        for r in json.loads(mf.read_text(encoding='utf-8'))['files']:
            p = data / r['path']
            if not p.is_file() or p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
                raise ValueError('File integrity mismatch: ' + r['path'])
            count += 1
    return {'verified_files': count}


def route_record(row, fixture):
    refs=fixture.get('image_refs', [])
    observed=row.get('ordered_image_identity', [])
    trace=row.get('processor_trace') or {}
    visual=trace.get('model_input_visual_tensor_keys',trace.get('visual_tensor_keys',[]))
    actual=trace.get('actual_model_generate') or {}
    check_images(refs,observed,visual)
    if not refs and actual.get('visual_tensor_keys'):
        raise ValueError('Text-only underlying generate received visual tensors')
    for ref,im in zip(refs,observed):
        if ref['relative_path']!=im.get('relative_path'):
            raise ValueError('Image relative identity/order mismatch')
    pixel=[x.get('pixel_sha256') for x in observed]
    aliases=[trace[k] for k in ('image_pixel_sha256','image_sha256') if isinstance(trace.get(k),list)]
    if len(aliases)==2 and aliases[0]!=aliases[1]:
        raise ValueError('Conflicting observed pixel identity aliases')
    if pixel and all(pixel) and aliases and pixel!=aliases[0]:
        raise ValueError('Processor observed pixel order differs')


def routes(data,out):
    stages=json.loads((data/'main/STAGES.json').read_text(encoding='utf-8'))
    reports=[]
    for s in stages:
        fixtures={r['fixture_id']:r for r in map(json.loads,(data/s['public_fixture']).open(encoding='utf-8'))}
        main={};replay=[];ids=set()
        for r in map(json.loads,(data/s['public_predictions']).open(encoding='utf-8')):
            if r['call_id'] in ids:raise ValueError('Duplicate call ID')
            ids.add(r['call_id']);route_record(r,fixtures[r['fixture_id']])
            if r['call_type']=='main':
                if r['fixture_id'] in main:raise ValueError('Repeated scientific input')
                main[r['fixture_id']]=r
            else:replay.append(r)
        if set(main)!=set(fixtures):raise ValueError('Fixture/scientific ID coverage differs')
        for r in replay:
            original=main[r['fixture_id']]
            if any(r.get(k)!=original.get(k) for k in ['raw_output','parsed_answer','parser_status','strict_json_valid']):
                raise ValueError('Replay differs')
        if len(replay)!=s['replay_calls']:raise ValueError('Replay count differs')
        reports.append(dict(model=s['model'],dataset=s['dataset'],scientific=len(main),replays=len(replay),
                            image_identity_and_order='PASS_SAVED_OBSERVATIONS',replay='EXACT',
                            image_file_bytes='not redistributed; archived hashes/observations checked'))
    write_csv(out/'SAVED_ROUTE_AND_REPLAY_CHECK.csv',reports)
    return {'stages':len(reports),'saved_image_identity_and_order':'PASS','replay_calls':sum(x['replays'] for x in reports)}


def position_routes(data,out):
    """Check all extra-position records separately from the main matrix."""
    main_fixture={r['fixture_id']:r for r in map(json.loads,(data/'fixtures/POSITION_MAIN_2998.jsonl').open(encoding='utf-8'))}
    alt_fixture={r['fixture_id']:r for r in map(json.loads,(data/'fixtures/POSITION_ALT_2998.jsonl').open(encoding='utf-8'))}
    reports=[]
    for path in sorted((data/'position/predictions').glob('*.jsonl')):
        ff=main_fixture if path.stem.endswith('_main') else alt_fixture
        science={};replays=[];ids=set()
        for r in map(json.loads,path.open(encoding='utf-8')):
            if r['call_id'] in ids:raise ValueError('Duplicate extra-position call ID')
            ids.add(r['call_id']);route_record(r,ff[r['fixture_id']])
            parsed=parse_answer(r['raw_output'])
            if (parsed.answer,parsed.parser_status,parsed.strict_json_valid)!=(r.get('parsed_answer'),r['parser_status'],r['strict_json_valid']):
                raise ValueError('Position parser differs')
            if r['call_type']=='main':science[r['fixture_id']]=r
            else:replays.append(r)
        if set(science)!=set(ff) or len(ids)!=3024 or len(replays)!=26:
            raise ValueError('Position input/replay coverage differs')
        for r in replays:
            if any(r.get(k)!=science[r['fixture_id']].get(k) for k in ['raw_output','parsed_answer','parser_status','strict_json_valid']):
                raise ValueError('Position replay differs')
        reports.append(dict(stage=path.stem,scientific=len(science),replays=len(replays),saved_route='PASS',parser='EXACT',replay='EXACT'))
    if len(reports)!=11:raise ValueError('Expected ten v8/v9 stages plus earlier LLaVA alternative')
    write_csv(out/'POSITION_ROUTE_AND_REPLAY_CHECK.csv',reports)
    return dict(v8_v9_scientific=29980,v8_v9_replays=260,earlier_llava_scientific=2998,earlier_llava_replays=26,
                stage_checks=len(reports),new_inference=0)


def score(data, out):
    """Reparse every saved main-matrix call; compare descriptive full-population BA."""
    stages = json.loads((data / 'main/STAGES.json').read_text(encoding='utf-8'))
    scientific = replay = 0
    for s in stages:
        ids = set()
        for line in (data / s['public_predictions']).open(encoding='utf-8'):
            r = json.loads(line)
            if r['call_id'] in ids:
                raise ValueError('Duplicate call ID')
            ids.add(r['call_id'])
            a = parse_answer(r.get('raw_output', ''))
            if (a.answer, a.parser_status, a.strict_json_valid) != (
                    r.get('parsed_answer'), r['parser_status'], r['strict_json_valid']):
                raise ValueError('Saved parser result differs: ' + r['call_id'])
        if len(ids) != s['total_calls']:
            raise ValueError('Incomplete stage')
        scientific += s['unique_scientific_inputs']; replay += s['replay_calls']
    import pandas as pd
    frame = pd.read_csv(data / 'analysis/original/NORMALIZED_SCIENTIFIC_ROWS.csv', keep_default_na=False)
    results = []
    for dataset in ('m4', 'bigearthnet'):
        expected = {r['model']: r for r in read_csv(data / f'analysis/original/MAIN_TABLE_{dataset}_primary.csv')}
        for model, group in frame[(frame.dataset == dataset) & (frame.condition == 'A0') &
                                  (frame.primary.astype(str).str.lower().isin(['true', '1']))].groupby('model'):
            recalls = [float((group[group.gold == g].answer == g).mean()) for g in ('yes', 'no')]
            ba = sum(recalls) / 2
            if abs(ba - float(expected[model]['a0_balanced_accuracy'])) > 1e-12:
                raise ValueError('A0 scoring mismatch: ' + model)
            results.append(dict(dataset=dataset, model=model, a0_balanced_accuracy=ba,
                                yes_recall=recalls[0], no_specificity=recalls[1], records=len(group)))
    if (scientific, replay, len(frame)) != (139092, 416, 139092):
        raise ValueError('Scientific/replay accounting differs')
    write_csv(out / 'RECOMPUTED_A0.csv', results)
    # Reconstruct every frozen point estimate and eligible denominator from saved
    # scientific rows; uncertainty remains the accepted cached draw arrays.
    from frozen_metrics import build_book
    import math
    norm = frame.to_dict('records')
    excluded = {r['physical_source'] for r in read_csv(data/'analysis/original/SOURCE_MAPPING.csv')
                if r['dataset']=='bigearthnet' and r['source_exclusion'].lower()=='true'}
    expected = json.loads((data/'analysis/original/ALL_ENDPOINTS.json').read_text(encoding='utf-8'))
    index = {(r['dataset'],r['cohort'],r['model'],r['metric']):r for r in expected}
    points=[]
    for dataset, cohort in [('m4','primary'),('bigearthnet','primary'),('bigearthnet','source_exclusion')]:
        rr=[r for r in norm if r['dataset']==dataset and (cohort!='source_exclusion' or r['source'] not in excluded)]
        sources=sorted({r['source'] for r in rr})
        for model in sorted({r['model'] for r in rr}):
            book=build_book([r for r in rr if r['model']==model],sources,model=='earthmind_4b')
            for name,m in book.metrics.items():
                value=sum((n.sum()/d.sum() if d.sum() else float('nan'))*weight for n,d,weight in m['parts'])
                original=index[(dataset,cohort,model,name)]
                if not ((math.isnan(value) and original['estimate'] is None) or
                        (original['estimate'] is not None and abs(value-original['estimate'])<1e-12)):
                    raise ValueError('Point-estimate reconstruction mismatch: '+str((dataset,cohort,model,name)))
                for key in ['eligible_units','eligible_physical_sources','eligible_target_questions','eligible_source_directions']:
                    if key in m['meta'] and m['meta'][key]!=original[key]:
                        raise ValueError('Eligibility denominator mismatch: '+key)
                points.append(dict(dataset=dataset,cohort=cohort,model=model,metric=name,estimate=value))
    write_csv(out/'RECOMPUTED_ALL_POINT_ESTIMATES.csv',points)
    return dict(scientific_records=scientific, replay_records=replay, parser_mismatches=0,
                a0_model_dataset_checks=len(results),all_point_estimates_verified=len(points))


def fixtures(data):
    count = 0
    for p in sorted((data / 'fixtures').glob('*.jsonl')):
        ids = set()
        for line in p.open(encoding='utf-8'):
            r = json.loads(line)
            if 'fixture_id' not in r:  # Pairing/replay maps have their own preserved schema.
                continue
            if r['fixture_id'] in ids:
                raise ValueError('Duplicate fixture ID: ' + p.name)
            ids.add(r['fixture_id'])
            refs = r.get('image_refs', [])
            if len(refs) not in (0, 1, 2):
                raise ValueError('Unexpected image count')
            if any(not x.get('sha256') or not x.get('relative_path') for x in refs):
                raise ValueError('Missing public image identity')
            # Metadata-only: actual third-party images are acquired separately.
            count += 1
    if not count:
        raise FileNotFoundError('No fixture metadata')
    return dict(fixture_rows=count, image_bytes_check='NOT_RUN: third-party images not redistributed')


def tables(data, out):
    for ds in ('m4', 'bigearthnet'):
        rows = read_csv(data / f'analysis/original/MAIN_TABLE_{ds}_primary.csv')
        write_csv(out / f'MAIN_TABLE_{ds}.csv', rows)
        lines = ['| Model | A0 balanced accuracy [95% CI] |', '|---|---|']
        for r in rows:
            vals = [float(r['a0_balanced_accuracy' + k]) * 100 for k in ('', '_ci95_lower', '_ci95_upper')]
            lines.append(f"| {r['model']} | {vals[0]:.2f} [{vals[1]:.2f}, {vals[2]:.2f}] |")
        (out / f'MAIN_TABLE_{ds}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'main_tables': 2, 'new_estimation': False}


def cached_intervals(data, out):
    """Verify accepted intervals against saved endpoint draws, never resample."""
    import numpy as np
    results = []
    for p in sorted((data / 'analysis/original/bootstrap').glob('*.json')):
        meta = json.loads(p.read_text(encoding='utf-8'))
        z = np.load(p.with_suffix('.npz'), allow_pickle=False)
        for i, name in enumerate(meta['metric_names']):
            a = z[f'v{i}']; valid = a[np.isfinite(a)]
            lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            r = meta['metrics'][i]
            if len(valid)!=r['valid_bootstrap'] or len(a)-len(valid)!=r['invalid_zero_denominator_bootstrap']:
                raise ValueError('Cached valid/zero-denominator accounting differs: '+name)
            for got, expected in [(lo, r['ci95_lower']), (hi, r['ci95_upper'])]:
                if got is not None and abs(float(got) - float(expected)) > 1e-12:
                    raise ValueError('Cached interval mismatch: ' + name)
            results.append(dict(cache=p.stem, metric=name, lower=lo, upper=hi,
                                valid=len(valid), invalid=len(a)-len(valid)))
    if not results:
        raise FileNotFoundError('Missing saved endpoint draws; will not generate new draws')
    write_csv(out / 'CACHED_INTERVAL_CHECK.csv', results)
    return {'cached_endpoints_verified': len(results), 'new_random_draws': 0}


def figures(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from public_display import apply_public_display
    apply_public_display()
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 8})
    for ds in ('m4', 'bigearthnet'):
        rows = read_csv(data / f'analysis/original/MAIN_TABLE_{ds}_primary.csv')
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for i, r in enumerate(rows):
            x, l, h = [100 * float(r['a0_balanced_accuracy'+k]) for k in ('','_ci95_lower','_ci95_upper')]
            ax.errorbar(x, i, xerr=[[x-l], [h-x]], fmt='o', color='#326B96', capsize=2)
        ax.set_yticks(range(8), [r['model'] for r in rows]); ax.invert_yaxis()
        ax.set(xlabel='A0 balanced accuracy (%) — descriptive order', xlim=(0,100))
        fig.tight_layout()
        for ext in ('pdf','svg'):
            fig.savefig(out / f'{ds}_capability.{ext}')
        plt.close(fig)
    return {'vector_figures_rebuilt': 4, 'inference': 0}


def demo(out):
    cases = [('{"answer":"yes"}', 'yes'), ('No.', 'no'),
             ('{"answer":"yes|no|abstain"}', None), ('abstain', 'abstain')]
    for raw, answer in cases:
        if parse_answer(raw).answer != answer:
            raise ValueError('Frozen parser synthetic check failed')
    check_images([], [], [])
    write_csv(out / 'SYNTHETIC_SCORING.csv', [{'raw': x, 'answer': a} for x,a in cases])
    return {'synthetic_cases': len(cases), 'dataset_images_used': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['demo','verify','score','fixtures','routes','position-routes','tables','cached-intervals','figures','all'])
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.action != 'demo' and a.data is None:
        p.error('--data must point to extracted release attachments')
    actions = ['demo','verify','fixtures','routes','position-routes','score','cached-intervals','tables','figures'] if a.action == 'all' else [a.action]
    report = {}
    for name in actions:
        f = globals()[name.replace('-', '_')]
        report[name] = f(a.out) if name == 'demo' else f(a.data) if name in ('verify','fixtures') else f(a.data,a.out)
    (a.out / 'CPU_REPRODUCTION_REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
