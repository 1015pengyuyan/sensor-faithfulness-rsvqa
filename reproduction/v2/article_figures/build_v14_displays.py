import os
"""Render accepted estimates only: no model calls, estimation or resampling."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(os.environ['SF_FIGURE_OUT'])
PROJECT=HERE.parents[2]
AD=PROJECT/'model_admission_20261017'
BASE=HERE.parents[1]/'paper_revision_20261026/v13_evidence_complete'
V1=AD/'v13_phase1_position_analysis_20261028_v1/outputs'
V2=AD/'v13_phase1_position_analysis_20261028_v2_final_cpu_closeout/outputs'
DATA=Path(os.environ['SF_FIGURE_DATA'])
FIG=HERE/'figures'
QA=HERE/'qa'
sys.path.insert(0,str(HERE/'tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cr(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def num(v): return float(v) if v not in ('',None,'None','nan') else None
def pct(v):
    v=num(v)
    if v is None:return '--'
    x=v*100
    return '0.00' if abs(x)<.0000001 else (f'{x:.4f}' if abs(x)<.005 else f'{x:.2f}')
def ci(r,p='estimate',lo='ci_low',hi='ci_high'):
    return f'{pct(r[p])} [{pct(r[lo])}, {pct(r[hi])}]'
def label(m):
    return {'idefics2_8b':'Idefics2','qwen25_vl_7b':'Qwen2.5-VL-7B','qwen3_vl_8b':'Qwen3-VL-8B',
        'internvl35_8b':'InternVL3.5','qwen25_vl_3b':'Qwen2.5-VL-3B','mantis_siglip_llama3_8b':'Mantis'}[m]

def prepare():
    if not V2.is_dir():
        records=json.loads((DATA/'V14_POSITION_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        assert all(sha(HERE/r['copy'])==r['sha256'] for r in records)
        return  # Portable source bundle: use the already locked local copies.
    records=[]
    for root,names in [(V2,['POSITION_COMMON_SET_RATES.csv','POSITION_OUTCOME_TRANSITIONS.csv',
        'POSITION_FORMAT_STATUS.csv','QWEN3B_REUSE_LAYER_AUDIT.json','LLAVA_BRIDGE_ARCHIVE_AUDIT.json',
        'PAPER_CLAIM_UPDATE_TABLE.csv','RUN_RECEIPT.json','ANALYSIS_INPUT_MANIFEST.json','OUTPUT_HASHES.json']),
        (V1,['POSITION_ENDPOINTS.csv','A0_FULL_POPULATION.csv','CONDITION_OUTCOMES.csv','CROSS_RUN_REUSE_AUDIT.csv',
        'RUN_RECEIPT.json','ANALYSIS_INPUT_MANIFEST.json'])]:
        sub=DATA/('position_closeout' if root==V2 else 'position_original');sub.mkdir(exist_ok=True)
        for name in names:
            src=root/name;dst=sub/name
            if dst.exists():assert sha(src)==sha(dst)
            else:shutil.copy2(src,dst)
            records.append(dict(source=str(src),copy=str(dst.relative_to(HERE)),bytes=src.stat().st_size,sha256=sha(src)))
    dump(DATA/'V14_POSITION_SOURCE_MANIFEST.json',records)
    if not (QA/'PRESERVED_V13.json').exists():
        dump(QA/'PRESERVED_V13.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(BASE.rglob('*')) if p.is_file()])

CORE=['idefics2_8b','qwen25_vl_7b','qwen3_vl_8b','internvl35_8b']
BLUE='#326B96'; ORANGE='#B36D30'; GRAY='#555B65'; PURPLE='#826B9D'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'lines.linewidth':.9,'legend.frameon':False,'axes.unicode_minus':False})

def figure(rates,outcomes):
    chosen=[next(r for r in rates if r['model_id']==m and r['cohort']=='primary' and r['requested_sensor']==s
        and r['endpoint']=='paired_slot2_minus_slot1_D_opposite_minus_C1') for m in CORE for s in ['optical','sar']]
    assert len(chosen)==8 and len({(r['model_id'],r['requested_sensor']) for r in chosen})==8
    fig,axs=plt.subplots(1,2,figsize=(7.16,3.25),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.21,right=.98,bottom=.22,top=.82,wspace=.66)
    a,b=axs
    for i,r in enumerate(chosen):
        y=7-i; x,l,h=[100*float(r[k]) for k in ['estimate','ci_low','ci_high']]
        color=BLUE if r['requested_sensor']=='optical' else ORANGE
        a.errorbar(x,y,xerr=[[x-l],[h-x]],fmt='o' if r['requested_sensor']=='optical' else 's',
                   color=color,markersize=3.5,capsize=2,lw=.9)
        a.text(1.025,y,f"{r['eligible_questions']}/{r['eligible_sources']}",transform=a.get_yaxis_transform(),va='center',fontsize=7)
    a.set_yticks(range(7,-1,-1),[label(r['model_id'])+'  '+('Opt' if r['requested_sensor']=='optical' else 'SAR') for r in chosen])
    a.set_ylim(-.6,7.6);a.set_xlim(-10,12);a.set_xticks([-10,-5,0,5,10]);a.tick_params(axis='y',length=0,pad=4)
    a.spines['left'].set_visible(False);a.axvline(0,color='.55',ls='--',lw=.7);a.grid(axis='x',color='.92',lw=.45)
    a.set_xlabel('D: slot 2 - slot 1 (pp)')
    a.text(1.025,1.035,'n q/s',transform=a.transAxes,fontsize=7,ha='left')
    a.set_title('Common eligibility at both positions',fontsize=7,pad=14)
    for ax,ch in [(a,'a'),(b,'b')]:
        ax.annotate(ch,(0,1),xycoords='axes fraction',xytext=(-13,15),textcoords='offset points',
                    weight='bold',fontsize=8,annotation_clip=False)
    cats=['target_label_answer','reverse_label_answer','abstain','parse_failure']
    names=['Target label','Reverse label','Abstain','Parser failure'];cols=[BLUE,ORANGE,GRAY,PURPLE]
    selected=[]
    for y,slot in [(5.3,'1'),(2.7,'2')]:
        rows=[next(r for r in outcomes if r['model_id']=='qwen25_vl_7b' and r['cohort']=='primary'
            and r['requested_sensor']=='sar' and r['gold_stratum']=='all' and r['actual_requested_slot']==slot
            and r['category']==c) for c in cats]
        assert sum(int(r['count']) for r in rows)==219
        assert all(r['eligible_qsensor_units']=='73' and r['eligible_sources']=='69' for r in rows)
        left=0
        for j,r in enumerate(rows):
            width=float(r['rate'])*100
            b.barh(y,width,left=left,height=1.0,color=cols[j],label=names[j] if slot=='1' else None)
            left+=width
        b.text(0,y-.83,' / '.join(r['count'] for r in rows),fontsize=7,ha='left',va='top')
        selected.extend(rows)
    b.set_ylim(-.6,7.6);b.set_xlim(0,100);b.set_xticks([0,50,100]);b.set_yticks([5.3,2.7],['Slot 1','Slot 2'])
    b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    b.set_title('Qwen7 SAR: 71 gold-no, 2 gold-yes',fontsize=7,pad=14)
    b.set_xlabel('Opposite-condition answers (%)')
    b.legend(loc='lower left',bbox_to_anchor=(-.04,-.38),ncol=2,fontsize=7,columnspacing=1.0,handlelength=1.0)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/'fig4_position_alignment.json',overlay_svg=QA/'fig4_position_alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(FIG/'fig4_position.pdf')
    fig.savefig(FIG/'fig4_position.svg')
    fig.savefig(FIG/'fig4_position.png',dpi=600)
    plt.close(fig)
    dump(FIG/'POSITION_FIGURE_SOURCE.json',dict(figure='fig4_position',pointwise=True,new_statistics=False,
        panel_a_source='source_data/position_closeout/POSITION_COMMON_SET_RATES.csv',panel_a=chosen,
        panel_b_source='source_data/position_closeout/POSITION_OUTCOME_TRANSITIONS.csv',panel_b=selected,
        source_hashes={n:sha(DATA/'position_closeout'/n) for n in ['POSITION_COMMON_SET_RATES.csv','POSITION_OUTCOME_TRANSITIONS.csv']}))

def table(name,caption,lab,headers,rows,width='textwidth'):
    lines=[r'{\scriptsize\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{'+('l'*len(headers))+'}',
        r'\caption{'+caption+r'}\label{'+lab+r'}\\',r'\toprule',
        ' & '.join(headers)+r'\\\midrule\endfirsthead',r'\toprule',
        ' & '.join(headers)+r'\\\midrule\endhead']
    lines += [' & '.join(str(c) for c in row)+r'\\' for row in rows]
    lines += [r'\bottomrule\end{longtable}}']
    (HERE/'tables'/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def tables(rates,outcomes):
    absolute=[]
    for cohort,title in [('primary','BEN main population'),('source_exclusion','BEN whole-source exclusion')]:
        rows=[]
        for m in CORE:
            for s in ['optical','sar']:
                for slot in ['1','2']:
                    rr=[next(r for r in rates if (r['model_id'],r['cohort'],r['requested_sensor'],r['actual_requested_slot'],r['endpoint'])
                        ==(m,cohort,s,slot,ep)) for ep in ['opposite_reverse_C1subset','c1_reverse','D_opposite_minus_C1']]
                    r=rr[0]
                    rows.append([label(m),s,slot,*[ci(x) for x in rr],f"{r['eligible_questions']}/{r['eligible_sources']}",f"{r['gold_yes_qsensor']}/{r['gold_no_qsensor']}"])
        filename=f'position_absolute_{cohort}.tex';absolute.append('\\input{tables/'+filename[:-4]+'}')
        table(filename,title+': absolute opposite (O), second-congruent (C) reverse-answer rates and D on the same common-A0-and-C1 set at both slots. Rates and pointwise 95\\% intervals are in \\% (D in pp); n q/s is eligible questions/sources and y/n is their gold composition. All intervals are exploratory and unadjusted.',
              'tab:position_absolute' if cohort=='primary' else 'tab:position_absolute_exclusion',
              ['Model','Request','Slot','O [95\\% CI]','C [95\\% CI]','D [95\\% CI]','n q/s','Gold y/n'],rows)
    (HERE/'tables/appendix_position_absolute.tex').write_text('\n'.join(absolute)+'\n',encoding='utf-8')
    endpoints=cr(DATA/'position_original/POSITION_ENDPOINTS.csv');includes=[]
    for cohort,title in [('primary','Main population'),('source_exclusion','Whole-source exclusion')]:
        rows=[]
        for m in CORE:
            for s in ['optical','sar']:
                for ep in ['paired_slot2_minus_slot1_CIFR','paired_slot2_minus_slot1_RSR','paired_slot2_minus_slot1_D_opposite_minus_C1_reverse']:
                    r=next(r for r in endpoints if (r['model_id'],r['cohort'],r['requested_sensor'],r['endpoint'])==(m,cohort,s,ep))
                    short='CIFR' if ep.endswith('CIFR') else 'RSR' if ep.endswith('RSR') else 'D'
                    rows.append([label(m),s,short,ci(r),f"{r['eligible_questions']}/{r['eligible_sources']}",f"{r['gold_yes_units']}/{r['gold_no_units']}",r['invalid_zero_denominator_draws']])
            for r in endpoints:
                if r['model_id']==m and r['cohort']==cohort and 'interaction' in r['endpoint']:
                    metric='CIFR' if r['endpoint'].endswith('CIFR') else 'RSR' if r['endpoint'].endswith('RSR') else 'D'
                    rows.append([label(m),'four-cell','Interaction '+metric,ci(r),f"{r['eligible_questions']}/{r['eligible_sources']}",f"{r['gold_yes_questions']}/{r['gold_no_questions']}",r['invalid_zero_denominator_draws']])
        name=f'position_contrasts_{cohort}.tex';includes.append('\\input{tables/'+name[:-4]+'}')
        table(name,title+': direct paired slot-2-minus-slot-1 contrasts and four-cell sensor-by-position interactions (SAR slot contrast minus optical slot contrast). Values and pointwise 95\\% intervals are pp. Eligibility is question-specific, common across both positions, and additionally common across sensors for interaction. Zero-denominator draws are reported, not replaced.',
              'tab:position_contrasts' if cohort=='primary' else 'tab:position_contrasts_exclusion',
              ['Model','Request','Endpoint','Difference [95\\% CI]','n q/s','Gold y/n','Invalid draws'],rows)
    (HERE/'tables/appendix_position_contrasts.tex').write_text('\n'.join(includes)+'\n',encoding='utf-8')
    includes=[]
    for cohort,title in [('primary','Main population'),('source_exclusion','Whole-source exclusion')]:
        rows=[]
        for m in CORE:
            for s in ['optical','sar']:
                for slot in ['1','2']:
                    rr=[next(r for r in outcomes if (r['model_id'],r['cohort'],r['requested_sensor'],r['gold_stratum'],r['actual_requested_slot'],r['category'])
                        ==(m,cohort,s,'all',slot,c)) for c in ['target_label_answer','reverse_label_answer','abstain','parse_failure']]
                    r=rr[0]
                    assert sum(int(x['count']) for x in rr)==int(r['denominator_calls'])
                    rows.append([label(m),s,slot,*[pct(x['rate'])+' ('+x['count']+')' for x in rr],
                        f"{r['eligible_qsensor_units']}/{r['eligible_sources']}",f"{r['gold_yes_qsensor']}/{r['gold_no_qsensor']}"])
        name=f'position_outcomes_{cohort}.tex';includes.append('\\input{tables/'+name[:-4]+'}')
        table(name,title+': opposite-condition outcomes on the same common A0-correct set at both slots. Cells give \\% (answer counts); there are three fixed opposite replacements per eligible question. Target/reverse/abstain/parser failure are mutually exclusive and exhaustive. n q/s and gold y/n describe questions, not the tripled call count. Format status is tabulated separately in source data.',
            'tab:position_outcomes' if cohort=='primary' else 'tab:position_outcomes_exclusion',
            ['Model','Request','Slot','Target','Reverse','Abstain','Parser failure','n q/s','Gold y/n'],rows)
    (HERE/'tables/appendix_position_outcomes.tex').write_text('\n'.join(includes)+'\n',encoding='utf-8')
    # Only previously accepted estimates; no new Qwen3B paired contrast is calculated here.
    rows=[]
    for r in endpoints:
        if r['model_id']=='mantis_siglip_llama3_8b' and r['cohort']=='primary' and r['endpoint'].startswith('paired_slot2_minus_slot1_'):
            ep=r['endpoint'].replace('paired_slot2_minus_slot1_','').replace('D_opposite_minus_C1_reverse','D')
            rows.append(['Mantis',r['requested_sensor'],ep,ci(r),f"{r['eligible_questions']}/{r['eligible_sources']}"])
    table('crossrun_mantis.tex','Cross-run Mantis position supplement: historical Main and corrected Alternative, conditional on the frozen bridge and effective-configuration checks. Pointwise exploratory intervals in pp. A limited bridge does not establish full-run stability. LLaVA has no authenticated bridge for paired inference.',
        'tab:crossrun',['Model','Request','Endpoint','Difference [95\\% CI]','n q/s'],rows)
    rows=[]
    for r in endpoints:
        if r['model_id']=='qwen25_vl_3b' and r['cohort']=='primary' and r['endpoint'] in [f'slot_{i}_{e}' for i in [1,2] for e in ['CIFR','RSR','D_opposite_minus_C1_reverse']]:
            parts=r['endpoint'].split('_'); ep='D' if parts[2]=='D' else parts[2]
            rows.append([r['requested_sensor'],parts[1],ep,ci(r),f"{r['eligible_questions']}/{r['eligible_sources']}"])
    if rows:
        table('crossrun_qwen3b.tex','Qwen2.5-VL-3B cross-run supplement: accepted position-specific estimates with each slot\'s own A0 eligibility. The final audit qualifies effective-component reuse, but these absolute rates are not a direct paired position contrast. No new paired endpoint is calculated in this integration. Rates are \\% (D in pp); intervals are pointwise exploratory.',
            'tab:crossrun_qwen3b',['Request','Slot','Endpoint','Estimate [95\\% CI]','n q/s'],rows)
    else:
        (HERE/'tables/crossrun_qwen3b.tex').write_text(r'\noindent\textbf{Qwen2.5-VL-3B reuse coverage.} The final audit qualifies the historical-Main/corrected-Alternative data as a cross-run supplement: the frozen native components, effective configuration and full 2,998-input pairing match, with an exact 20-input bridge. The accepted position endpoint table did not include Qwen3B, and this integration does not calculate a new paired endpoint. Reuse qualification is not full-run stability evidence; no paired position claim is added.'+'\n',encoding='utf-8')
    (HERE/'tables/appendix_crossrun.tex').write_text('\\input{tables/crossrun_mantis}\n\\input{tables/crossrun_qwen3b}\n',encoding='utf-8')
    E=json.loads((DATA/'original_ALL_ENDPOINTS.json').read_text(encoding='utf-8'))
    models=['qwen25_vl_3b','qwen25_vl_7b','qwen3_vl_8b','mantis_siglip_llama3_8b','idefics2_8b','llava_onevision_7b','internvl35_8b','earthmind_4b']
    names=['Qwen2.5-VL-3B','Qwen2.5-VL-7B','Qwen3-VL-8B','Mantis','Idefics2','LLaVA','InternVL3.5','EarthMind']
    includes=[]
    for ds in ['m4','bigearthnet']:
        rows=[]
        for m,name in zip(models,names):
            for s in ['optical','sar']:
                pos='named' if m=='earthmind_4b' else '1' if ds=='m4' or s=='optical' else '2'
                rr=[next(r for r in E if (r['dataset'],r['cohort'],r['model'],r['metric'])==(ds,'primary',m,f'cell/{s}/{pos}/control/{ep}')) for ep in ['opposite_reverse','second_congruent_reverse','D']]
                r=rr[0]
                rows.append([name,s,*[ci(x,lo='ci95_lower',hi='ci95_upper') for x in rr],f"{r['eligible_target_questions']}/{r['eligible_physical_sources']}"])
        name=f'primary_rates_{ds}.tex';includes.append('\\input{tables/'+name[:-4]+'}')
        table(name,('M4-SAR' if ds=='m4' else 'BigEarthNet.txt')+': fixed-Main opposite (O), second-congruent (C) reverse-answer rates and D on identical eligible questions. Rates are \\% (D in pp); intervals are pointwise 95\\% source-cluster intervals. Different models retain different eligible populations.',
            'tab:primary_'+ds,['Checkpoint','Request','O [95\\% CI]','C [95\\% CI]','D [95\\% CI]','n q/s'],rows)
    (HERE/'tables/appendix_primary_rates.tex').write_text('\n'.join(includes)+'\n',encoding='utf-8')

if __name__=='__main__':
    prepare()
    rates=cr(DATA/'position_closeout/POSITION_COMMON_SET_RATES.csv')
    outcomes=cr(DATA/'position_closeout/POSITION_OUTCOME_TRANSITIONS.csv')
    figure(rates,outcomes);tables(rates,outcomes)
    print('Accepted estimates rendered: eight paired D marks, eight outcome segments; no inference or random draws.')
