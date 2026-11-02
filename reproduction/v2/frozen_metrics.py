"""Accepted statistical definitions, mechanically extracted; no draw generator.
Original analyze.py SHA256: d978d1d10ad2a21d0ea4ddaf7a0c221070119aaa8aac76432473781bb186ff0f
"""
from collections import Counter, defaultdict
import numpy as np
SENSORS=("optical","sar")

def ratio_draws(w, num, den):
    n, d = w @ np.asarray(num, dtype=float), w @ np.asarray(den, dtype=float)
    return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)

def interval(vals):
    valid = vals[np.isfinite(vals)]
    if not len(valid): return None, None, 0, len(vals)
    lo, hi = np.quantile(valid, [.025, .975], method="linear")
    return float(lo), float(hi), len(valid), len(vals) - len(valid)

class Book:
    def __init__(self, sources):
        self.sources = sources; self.idx = {s: i for i, s in enumerate(sources)}; self.metrics = {}

    def add(self, name, events, values, unit="target_question_direction", kind="frozen", **tags):
        assert name not in self.metrics
        n = np.zeros(len(self.sources)); d = np.zeros_like(n)
        assert len(events) == len(values)
        for r, v in zip(events, values):
            n[self.idx[r["source"]]] += v; d[self.idx[r["source"]]] += 1
        meta = dict(metric=name, denominator_unit=unit, analysis_role=kind,
                    eligible_units=len(events), eligible_physical_sources=sum(d > 0),
                    eligible_target_questions=len({r['target'] for r in events}),
                    eligible_source_directions=len({(r['source'],r['sensor']) for r in events}), **tags)
        self.metrics[name] = dict(parts=[(n, d, 1.0)], meta=meta)

    def composite(self, name, children, **tags):
        parts=[]
        for child, weight in children:
            parts.extend((n, d, c * weight) for n, d, c in self.metrics[child]["parts"])
        self.metrics[name] = dict(parts=parts, meta=dict(metric=name, denominator_unit="composite_of_stated_arm_ratios", analysis_role="frozen", **tags))

    def bootstrap(self, w):
        parts = [p for m in self.metrics.values() for p in m['parts']]
        n = np.column_stack([p[0] for p in parts]); d = np.column_stack([p[1] for p in parts])
        bn = w.astype(float) @ n; bd = w.astype(float) @ d
        ratios = np.divide(bn, bd, out=np.full_like(bn, np.nan), where=bd > 0)
        out, draws = [], {}; j = 0
        for name, m in self.metrics.items():
            count = len(m['parts']); value = np.zeros(w.shape[0]); point = 0.0
            arms=[]
            for k, (pn, pd, weight) in enumerate(m['parts']):
                value += ratios[:, j+k] * weight
                point += (pn.sum() / pd.sum() if pd.sum() else float('nan')) * weight
                arms.append(dict(numerator=float(pn.sum()),denominator=int(pd.sum()),weight=weight,
                                 eligible_physical_sources=int(sum(pd>0))))
            j += count
            lo,hi,valid,invalid=interval(value)
            out.append(dict(m['meta'], estimate=point, ci95_lower=lo, ci95_upper=hi,
                            valid_bootstrap=valid, invalid_zero_denominator_bootstrap=invalid,
                            requested_bootstrap=len(w), resampled_physical_sources=len(self.sources),
                            arm_denominators=arms))
            draws[name]=value
        return out,draws,n,d

def population(book, stem, rows, kind="frozen"):
    for metric, vals in [
        ('accuracy',[r['correct'] for r in rows]),('abstain',[r['abstain'] for r in rows]),
        ('parse_failure',[r['parse_failure'] for r in rows]),('strict_json',[int(r['strict_json']) for r in rows]),
        ('yes_output',[int(r['answer']=='yes') for r in rows]),('no_output',[int(r['answer']=='no') for r in rows])]:
        book.add(stem+'/'+metric, rows, vals,kind=kind)
    for label,metric in [('yes','yes_recall'),('no','no_specificity')]:
        sub=[r for r in rows if r['gold']==label]
        book.add(stem+'/'+metric, sub,[r['correct'] for r in sub],kind=kind)
    book.composite(stem+'/balanced_accuracy',[(stem+'/yes_recall',.5),(stem+'/no_specificity',.5)])

def build_book(rows, sources, named):
    book=Book(sources)
    population(book,'main/A0',[r for r in rows if r['condition']=='A0' and r['primary']])
    population(book,'main/conflict',[r for r in rows if r['condition'] in ('O1','O2','O3') and r['primary']])
    for c in ('O1','O2','O3'):
        population(book,'main/'+c,[r for r in rows if r['condition']==c and r['primary']])
    if not named:
        for s in SENSORS:
            for c in ('REQUESTED_ONLY','NONREQUESTED_ONLY','TEXT_ONLY'):
                population(book,'aux/'+s+'/'+c,[r for r in rows if r['sensor']==s and r['condition']==c])
    cells=defaultdict(dict)
    for r in rows:
        if r['condition'] in ('A0','C1','O1','O2','O3'):
            cells[(r['target'],r['sensor'],r['position'])][r['condition']]=r
    events={}
    for key, c in cells.items():
        assert all(x in c for x in ('A0','O1','O2','O3'))
        a0=c['A0']
        if not a0['correct']: continue
        gold=a0['gold']; opposite='no' if gold=='yes' else 'yes'
        op=[c[x] for x in ('O1','O2','O3')]
        v=dict(meta=a0,O=sum(r['answer']==opposite for r in op)/3,
               R=sum(r['answer']==gold for r in op)/3,
               pf=sum(r['parse_failure'] for r in op)/3,ab=sum(r['abstain'] for r in op)/3)
        if 'C1' in c:
            v.update(C=int(c['C1']['answer']==opposite),D=v['O']-int(c['C1']['answer']==opposite),
                     C_changed=int(c['C1']['answer']!=gold),C_pf=c['C1']['parse_failure'],C_ab=c['C1']['abstain'])
            for i,r in enumerate(op,1):v['D_rep'+str(i)]=int(r['answer']==opposite)-v['C']
        events[key]=v
    positions=('named',) if named else ('1','2')
    for sensor in SENSORS:
        for pos in positions:
            pop=[r for r in rows if r['condition']=='A0' and r['sensor']==sensor and r['position']==pos]
            population(book,f'cell/{sensor}/{pos}/A0',pop)
            ev=[e for (t,s,p),e in events.items() if s==sensor and p==pos]
            meta=[e['meta'] for e in ev]
            stem=f'cell/{sensor}/{pos}'
            for name,key in [('CIFR','O'),('RSR','R'),('conflict_parse_failure','pf'),('conflict_abstain','ab')]:
                book.add(stem+'/'+name,meta,[e[key] for e in ev],sensor=sensor,position=pos)
            for gold in ('yes','no'):
                eg=[e for e in ev if e['meta']['gold']==gold]
                for name,key in [('CIFR','O'),('RSR','R')]:
                    book.add(stem+'/'+name+'/gold_'+gold,[e['meta'] for e in eg],[e[key] for e in eg],kind='exploratory_stratified',sensor=sensor,position=pos,gold=gold)
            control=[e for e in ev if 'D' in e]
            for name,key in [('opposite_reverse','O'),('second_congruent_reverse','C'),('D','D'),
                             ('second_congruent_other_change','C_changed'),('second_congruent_parse_failure','C_pf'),('second_congruent_abstain','C_ab')]:
                book.add(stem+'/control/'+name,[e['meta'] for e in control],[e[key] for e in control],sensor=sensor,position=pos)
            for i in (1,2,3):
                book.add(stem+'/control/D_rep'+str(i),[e['meta'] for e in control],[e['D_rep'+str(i)] for e in control],sensor=sensor,position=pos)
        main_ev=[e for (t,s,p),e in events.items() if s==sensor and e['meta']['primary']]
        for metric,key in [('CIFR','O'),('RSR','R')]:
            book.add('main/'+sensor+'/'+metric,[e['meta'] for e in main_ev],[e[key] for e in main_ev],sensor=sensor)
    pooled=[e for e in events.values() if e['meta']['primary']]
    for metric,key in [('CIFR','O'),('RSR','R'),('eligible_conflict_parse_failure','pf'),('eligible_conflict_abstain','ab')]:
        book.add('main/'+metric,[e['meta'] for e in pooled],[e[key] for e in pooled])
    book.composite('main/CIFR_ASYM_SAR_minus_optical', [('main/sar/CIFR',1),('main/optical/CIFR',-1)],
                   eligibility_note='direction-specific A0-correct sets; difference of ratios, not joint-direction eligibility')
    if not named:
        targets=sorted({r['target'] for r in rows})
        for sensor in SENSORS:
            common=[(events[(t,sensor,'1')],events[(t,sensor,'2')]) for t in targets if (t,sensor,'1') in events and (t,sensor,'2') in events]
            book.add('position/'+sensor+'/CIFR_slot2_minus_slot1',[a['meta'] for a,b in common],[b['O']-a['O'] for a,b in common],unit='target_question',sensor=sensor)
            common=[(a,b) for a,b in common if 'D' in a and 'D' in b]
            book.add('position/'+sensor+'/D_slot2_minus_slot1',[a['meta'] for a,b in common],[b['D']-a['D'] for a,b in common],unit='target_question',sensor=sensor)
        common4=[{(s,p):events[(t,s,p)] for s in SENSORS for p in ('1','2')} for t in targets if all((t,s,p) in events for s in SENSORS for p in ('1','2'))]
        book.add('interaction/CIFR_SAR_minus_optical',[e[('optical','1')]['meta'] for e in common4],
                 [(e[('sar','2')]['O']-e[('sar','1')]['O'])-(e[('optical','2')]['O']-e[('optical','1')]['O']) for e in common4],unit='target_question')
        common4=[e for e in common4 if all('D' in v for v in e.values())]
        book.add('interaction/D_SAR_minus_optical',[e[('optical','1')]['meta'] for e in common4],
                 [(e[('sar','2')]['D']-e[('sar','1')]['D'])-(e[('optical','2')]['D']-e[('optical','1')]['D']) for e in common4],unit='target_question')
        book.add('control/overall_four_cell_common_D',[e[('optical','1')]['meta'] for e in common4],
                 [sum(v['D'] for v in e.values())/4 for e in common4],unit='target_question')
    return book

def descriptive(rows, by_class=False):
    groups=defaultdict(list)
    for r in rows:
        for gold in (r['gold'],'ALL'):
            k=(r['sensor'],r['position'],r['condition'],gold,r['class_id'] if by_class else 'ALL')
            groups[k].append(r)
            if r['condition'] in ('O1','O2','O3'):
                groups[(r['sensor'],r['position'],'ALL_OPPOSITE',gold,r['class_id'] if by_class else 'ALL')].append(r)
    out=[]
    for (s,p,c,g,cl), rr in sorted(groups.items()):
        n=len(rr); count=Counter(r['answer'] or 'parse_failure' for r in rr); fmt=Counter(r['parser_status'] for r in rr)
        d=dict(sensor=s,position=p,condition=c,gold=g,class_id=cl,
               class_name=rr[0]['class_name'] if by_class else 'ALL',assigned_records=n,
               target_questions=len({r['target'] for r in rr}),physical_sources=len({r['source'] for r in rr}),
               correct=sum(r['correct'] for r in rr),yes=count['yes'],no=count['no'],
               abstain=count['abstain'],parse_failure=count['parse_failure'],
               strict_json=sum(r['strict_json'] for r in rr),recovered=fmt['recovered'],
               ambiguous=fmt['ambiguous'],missing=fmt['missing'],
               majority_label_accuracy=max(sum(r['gold']=='yes' for r in rr),sum(r['gold']=='no' for r in rr))/n,
               analysis_role='exploratory_stratified' if by_class or g!='ALL' else 'descriptive_frozen_population')
        for k in ('correct','yes','no','abstain','parse_failure','strict_json','recovered'): d[k+'_rate']=d[k]/n
        out.append(d)
    return out
