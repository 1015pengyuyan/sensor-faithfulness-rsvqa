"""Post-hoc CPU robustness functions; no random sampling or model runtime."""
from collections import defaultdict
import numpy as np

def family_tails(size, alpha):
    if not isinstance(size,int) or size < 1 or not 0 < alpha < 1:
        raise ValueError('invalid fixed family')
    return alpha/(2*size), 1-alpha/(2*size)

def ratio_on_saved_draws(weights, numerators, denominators):
    w=np.asarray(weights,dtype=float)
    num=w@np.asarray(numerators,dtype=float)
    den=w@np.asarray(denominators,dtype=float)
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)

def summarize(values, tails=(.025,.975)):
    v=np.asarray(values,dtype=float);good=v[np.isfinite(v)]
    if not len(good):return dict(lower=None,upper=None,valid=0,invalid=len(v))
    lo,hi=np.quantile(good,tails,method='linear')
    return dict(lower=float(lo),upper=float(hi),valid=len(good),invalid=len(v)-len(good))

def select_events(rows, sensor, position):
    cells=defaultdict(dict);ro={}
    for r in rows:
        if r['sensor']!=sensor:continue
        if r['condition']=='REQUESTED_ONLY':
            if r['target'] in ro:raise ValueError('duplicate requested-only')
            ro[r['target']]=r
        elif r['position']==position and r['condition'] in ('A0','C1','O1','O2','O3'):
            if r['condition'] in cells[r['target']]:raise ValueError('duplicate condition')
            cells[r['target']][r['condition']]=r
    out=[]
    for target,c in sorted(cells.items()):
        if target not in ro:raise ValueError('missing corresponding requested-only')
        if 'A0' not in c or not all(x in c for x in ('O1','O2','O3')):raise ValueError('incomplete opposite cell')
        if 'C1' not in c:continue
        base=c['A0'];one=ro[target]
        if (base['source'],base['gold'])!=(one['source'],one['gold']):raise ValueError('requested-only identity mismatch')
        if not int(base['correct']) or not int(one['correct']):continue
        reverse='no' if base['gold']=='yes' else 'yes'
        o=sum(c[x]['answer']==reverse for x in ('O1','O2','O3'))/3
        same=int(c['C1']['answer']==reverse)
        out.append(dict(target=target,source=base['source'],sensor=sensor,position=position,gold=base['gold'],O=o,C=same,D=o-same,
                        opposite_parse_failure=sum(c[x]['answer'] in ('',None) for x in ('O1','O2','O3'))/3,
                        opposite_abstain=sum(c[x]['answer']=='abstain' for x in ('O1','O2','O3'))/3,
                        congruent_parse_failure=int(c['C1']['answer'] in ('',None)),congruent_abstain=int(c['C1']['answer']=='abstain')))
    return out
