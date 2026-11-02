import os
"""Three question-led article figures from accepted estimates and finite counts."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import build_displays as legacy
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE=Path(os.environ['SF_FIGURE_OUT'])
DATA=Path(os.environ['SF_FIGURE_DATA'])
FIG=HERE/'figures'
QA=HERE/'qa'
MODELS=legacy.MODELS;LABEL=legacy.LABEL;BLUE=legacy.BLUE;ORANGE=legacy.ORANGE;GRAY=legacy.GRAY
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
OUTCOMES=[]

def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}_alignment.json',
        overlay_svg=QA/f'{name}_alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=len(fig.axes)>1,strict=True)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=600)
    fig.savefig(FIG/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def task():
    p=legacy.jr('DESIGN_IMAGE_PROVENANCE.json');refs=[p['target_ref']]+[r['ref'] for r in p['conditions']]
    images=[]
    for i,ref in enumerate(refs):
        image_path=DATA/f'design_{i}.jpg';assert legacy.sha(image_path)==ref['sha256']
        with Image.open(image_path) as im:images.append(np.array(im.convert('RGB')))
    fig,ax=plt.subplots(figsize=(7.16,3.55));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.012,.967,'a',fontsize=8,weight='bold')
    headers=['Requested SAR','Same-label optical','Opposite-label optical']
    for i,(im,title) in enumerate(zip(images,headers)):
        x=.035+i*.215
        ax.text(x+.0875,.93,title,ha='center',fontsize=7)
        # Same raw image bytes; square image plate without cropping or contrast changes.
        ax.imshow(im,extent=(x,x+.175,.56,.913),interpolation='nearest',aspect='auto')
        label=p['target_label'] if i==0 else p['conditions'][i-1]['distractor_label']
        ax.text(x+.0875,.515,'Bridge label: '+label,ha='center',fontsize=6.6,color='.35')
    ax.text(.72,.875,'Question',fontsize=7,weight='bold')
    ax.text(.72,.815,'Is there a Bridge\nin the requested\nSAR evidence?',fontsize=7,va='top',linespacing=1.5)
    ax.text(.72,.575,'Only the requested\nimage determines\nthe target label.',fontsize=6.7,va='top',linespacing=1.4,color='.35')
    ax.text(.012,.405,'b',fontsize=8,weight='bold')
    ax.text(.07,.405,'Matched replacements: keep requested image and question constant',fontsize=7)
    def box(x,y,w,h,t,c,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.008',lw=.65,edgecolor=c,facecolor=fill))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=6.7,linespacing=1.3)
    box(.035,.14,.205,.15,'Congruent baseline A0\nCorrect-answer eligibility',GRAY,'#F4F5F6')
    box(.32,.255,.235,.105,'Second congruent C1\nSame-label replacement',BLUE,'#EFF5F9')
    box(.32,.065,.235,.105,'Opposite replacements\nThree distinct images',ORANGE,'#FBF4EC')
    box(.65,.14,.18,.15,'D = O - C\nExtra reverse answers',GRAY,'#F4F5F6')
    for start,end in [((.25,.24),(.305,.305)),((.25,.19),(.305,.115)),((.568,.305),(.635,.24)),((.568,.115),(.635,.19))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':.65,'color':'.45'})
    ax.text(.925,.28,'Controls',fontsize=6.7,weight='bold',ha='center')
    ax.text(.925,.21,'Requested-only\nText-only',fontsize=6.7,ha='center',va='center',linespacing=1.5)
    save(fig,'fig1_task')

def model_axis(ax,show=True,xlim=(0,100)):
    legacy.axes(ax,False,xlim)
    ax.set_yticklabels(LABEL if show else ['']*8)
    ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='y',labelsize=6.7)

def capability():
    name='fig2_capability';fig,axs=plt.subplots(2,2,figsize=(7.16,4.75))
    fig.subplots_adjust(left=.225,right=.985,bottom=.14,top=.89,wspace=.22,hspace=.48)
    colors=['#4C8B85','#CBCED3','#D6A05D','#846CA7']
    labels=['Correct','Incorrect','Abstain','Unparseable']
    outcome_rows=legacy.cr('original_CONDITION_GOLD_OUTCOMES.csv')
    for j,ds in enumerate(['m4','bigearthnet']):
        title='M4-SAR' if ds=='m4' else 'BigEarthNet.txt'
        ax=axs[0,j];model_axis(ax,j==0,xlim=(-2,102));ax.set_xlabel('Recognition rate (%)')
        legacy.panel(ax,chr(97+j),title+' / recognition')
        for i,m in enumerate(MODELS):
            for metric,off,col,mark,lab in [('yes_recall',-.13,BLUE,'o','Yes recall'),('no_specificity',.13,GRAY,'s','No specificity')]:
                r=legacy.endpoint(ds,m,'main/A0/'+metric)
                legacy.dot(ax,i+off,r['estimate'],r['ci95_lower'],r['ci95_upper'],col,mark,lab if i==0 and j==0 else None)
                legacy.record(name,'original_ALL_ENDPOINTS.json',dict(dataset=ds,model=m,metric=r['metric']),r['estimate'],r['ci95_lower'],r['ci95_upper'])
        ax=axs[1,j];model_axis(ax,j==0,xlim=(-2,102));ax.set_xlabel('Share of baseline answers (%)')
        legacy.panel(ax,chr(99+j),title+' / answer composition')
        for i,m in enumerate(MODELS):
            rs=[r for r in outcome_rows if r['dataset']==ds and r['cohort']=='primary' and r['model']==m
                and r['condition']=='A0' and r['gold']=='ALL' and r['class_id']=='ALL'
                and r['position']==legacy.pos(ds,m,r['sensor'])]
            assert len(rs)==2,(ds,m,len(rs))
            n=sum(int(r['assigned_records']) for r in rs)
            correct=sum(int(r['correct']) for r in rs);abstain=sum(int(r['abstain']) for r in rs);bad=sum(int(r['parse_failure']) for r in rs)
            counts=[correct,n-correct-abstain-bad,abstain,bad];assert min(counts)>=0
            OUTCOMES.append(dict(figure=name,dataset=ds,model=m,source='original_CONDITION_GOLD_OUTCOMES.csv',
                sha256=legacy.sha(DATA/'original_CONDITION_GOLD_OUTCOMES.csv'),conditions=[dict(sensor=r['sensor'],position=r['position']) for r in rs],
                n=n,correct=counts[0],incorrect=counts[1],abstain=counts[2],unparseable=counts[3],
                operation='incorrect = assigned - correct - abstain - parse_failure; descriptive counts only'))
            left=0
            for c,col,lab in zip(counts,colors,labels):
                width=c/n*100;ax.barh(i,width,left=left,height=.61,color=col,edgecolor='white',linewidth=.3,label=lab if i==0 and j==0 else None,zorder=3);left+=width
            for metric,expected in [('accuracy',correct/n),('abstain',abstain/n),('parse_failure',bad/n)]:
                r=legacy.endpoint(ds,m,'main/A0/'+metric);assert abs(r['estimate']-expected)<1e-12
                legacy.record(name,'original_ALL_ENDPOINTS.json',dict(dataset=ds,model=m,metric=r['metric']),r['estimate'],r['ci95_lower'],r['ci95_upper'])
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.59,1.005),fontsize=6.8)
    fig.legend(*axs[1,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.59,.005),fontsize=6.8)
    save(fig,name)

def conflict():
    name='fig3_conflict';fig,axs=plt.subplots(1,2,figsize=(7.16,3.65))
    fig.subplots_adjust(left=.225,right=.905,bottom=.19,top=.87,wspace=.68)
    for j,(ds,ax) in enumerate(zip(['m4','bigearthnet'],axs)):
        legacy.axes(ax,False);ax.set_yticklabels(LABEL if j==0 else ['']*8);ax.tick_params(axis='y',labelsize=6.7)
        legacy.panel(ax,chr(97+j),'M4-SAR' if ds=='m4' else 'BigEarthNet.txt')
        ax.set_xticks([-20,0,20,40]);ax.set_xlabel('Extra reverse answers D (pp)')
        for i,m in enumerate(MODELS):
            ns=[]
            for sensor,off,col,mark in [('optical',-.13,BLUE,'o'),('sar',.13,ORANGE,'s')]:
                r=legacy.endpoint(ds,m,f'cell/{sensor}/{legacy.pos(ds,m,sensor)}/control/D')
                legacy.dot(ax,i+off,r['estimate'],r['ci95_lower'],r['ci95_upper'],col,mark,('Optical' if sensor=='optical' else 'SAR') if i==0 and j==0 else None)
                legacy.record(name,'original_ALL_ENDPOINTS.json',dict(dataset=ds,model=m,metric=r['metric']),r['estimate'],r['ci95_lower'],r['ci95_upper'],r['eligible_target_questions'])
                ns.append(r['eligible_target_questions'])
            ax.text(1.035,i,f'{ns[0]}/{ns[1]}',transform=ax.get_yaxis_transform(),va='center',fontsize=6.3)
        ax.text(1.035,-.60,'n Opt/SAR',transform=ax.get_yaxis_transform(),fontsize=6.3)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.58,.01),fontsize=7)
    save(fig,name)

def tables():
    legacy.tables() # Preserve complete family table from the accepted inputs.
    rows=[r'\begin{table}[t]\centering\scriptsize',
      r'\caption{Baseline recognition on the full assigned population. BA is balanced accuracy (\%) with pointwise 95\% target-source-cluster intervals. Abstention and unparseable output count as errors. Model order is held constant across datasets; the table does not imply a statistically separated ranking.}',
      r'\label{tab:ability}\setlength{\tabcolsep}{1.5pt}\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr}\toprule',
      r'Checkpoint & M4 BA [95\% CI] & BEN BA [95\% CI]\\\midrule']
    main={ds:legacy.cr(f'original_MAIN_TABLE_{ds}_primary.csv') for ds in ['m4','bigearthnet']}
    for m,label,params in zip(MODELS,LABEL,['3B','7B','8B','8B','8B','7B','8B','4B']):
        cells=[label]
        for ds in ['m4','bigearthnet']:
            r=next(r for r in main[ds] if r['model']==m)
            cells.append(legacy.ci(r,'a0_balanced_accuracy','a0_balanced_accuracy_ci95_lower','a0_balanced_accuracy_ci95_upper'))
        rows.append(' & '.join(cells)+r'\\')
    rows+=[r'\bottomrule\end{tabular*}\end{table}']
    (HERE/'tables/main_ability.tex').write_text('\n'.join(rows),encoding='utf-8')
    f=legacy.jr('posthoc_MULTIPLICITY_F32.json');s=legacy.jr('posthoc_REQUESTED_ONLY_SENSITIVITY.json')
    rows=[r'\begin{table*}[t]\centering\scriptsize',
      r'\caption{Post-hoc qualifications to primary SAR differences (pp). Expanded intervals use the complete 32-comparison family; capability-subset estimates require both A0 and requested-only correctness and have unadjusted 95\% intervals. n is the eligible question count in the capability subset. These columns describe different populations; EarthMind single-image eligibility is not applicable.}',
      r'\label{tab:sensitivity}\setlength{\tabcolsep}{3pt}\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr}\toprule',
      r' & \multicolumn{3}{c}{M4-SAR} & \multicolumn{3}{c}{BigEarthNet.txt}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
      r'Checkpoint & Expanded interval & Capability D [95\% CI] & n & Expanded interval & Capability D [95\% CI] & n\\\midrule']
    for m,label in zip(MODELS,LABEL):
        cells=[label]
        for ds in ['m4','bigearthnet']:
            r=next(r for r in f if (r['dataset'],r['model'],r['sensor'])==(ds,m,'sar'))
            cells.append('['+legacy.fmt(r['bonferroni_lower'])+', '+legacy.fmt(r['bonferroni_upper'])+']')
            if m=='earthmind_4b':cells+=['--','--']
            else:
                q=next(r for r in s if (r['dataset'],r['cohort'],r['model'],r['sensor'],r['position'])==(ds,'primary',m,'sar',legacy.pos(ds,m,'sar')))
                cells+=[legacy.ci(q,'D','D_ci95_lower','D_ci95_upper'),str(q['eligible_questions'])]
        rows.append(' & '.join(cells)+r'\\')
    rows+=[r'\bottomrule\end{tabular*}\end{table*}']
    (HERE/'tables/main_sensitivity.tex').write_text('\n'.join(rows),encoding='utf-8')

if __name__=='__main__':
    task();capability();conflict();tables()
    (FIG/'FIGURE_SOURCE_DATA.json').write_text(json.dumps(legacy.SOURCES,indent=2),encoding='utf-8')
    (FIG/'OUTCOME_COMPOSITION_SOURCE.json').write_text(json.dumps(OUTCOMES,indent=2),encoding='utf-8')
    print('Three figures;',len(legacy.SOURCES),'source-linked estimates;',len(OUTCOMES),'full-population count decompositions. No new inference or resampling.')
