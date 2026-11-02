import os
"""Journal displays from immutable accepted estimates. No inference or resampling."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import numpy as np

HERE=Path(os.environ['SF_FIGURE_OUT'])
DATA=Path(os.environ['SF_FIGURE_DATA'])
FIG=HERE/'figures'
QA=HERE/'qa'
PROJECT=HERE.parents[2]
sys.path.insert(0,str(HERE/'tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS=['qwen25_vl_3b','qwen25_vl_7b','qwen3_vl_8b','mantis_siglip_llama3_8b','idefics2_8b','llava_onevision_7b','internvl35_8b','earthmind_4b']
LABEL=['Qwen2.5-VL-3B','Qwen2.5-VL-7B','Qwen3-VL-8B','Mantis-8B','Idefics2-8B','LLaVA-OneVision-7B','InternVL3.5-8B','EarthMind-4B']
SHORT=['Q2.5-3B','Q2.5-7B','Q3-8B','Mantis','Idefics2','LLaVA','InternVL3.5','EarthMind']
BLUE='#326B96';ORANGE='#B36D30';GRAY='#555B65';PURPLE='#826B9D'; SOURCES=[]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'lines.linewidth':.9,'legend.frameon':False,'axes.unicode_minus':False})

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def jr(name):return json.loads((DATA/name).read_text(encoding='utf-8-sig'))
def cr(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
E=jr('original_ALL_ENDPOINTS.json')
def pos(ds,m,s):return 'named' if m=='earthmind_4b' else ('1' if ds=='m4' or s=='optical' else '2')
def endpoint(ds,m,metric):
    rows=[r for r in E if (r['dataset'],r['cohort'],r['model'],r['metric'])==(ds,'primary',m,metric)]
    assert len(rows)==1,(ds,m,metric)
    return rows[0]
def record(fig,source,selector,p,lo,hi,n=None):
    SOURCES.append(dict(figure=fig,source=source,sha256=sha(DATA/source),selector=selector,
                        estimate=p,lower=lo,upper=hi,eligible_questions=n))
def dot(ax,y,p,l,h,color,marker='o',label=None):
    if None in (p,l,h):return
    p,l,h=[float(x)*100 for x in (p,l,h)]
    ax.plot([l,h],[y,y],color=color,lw=.9,zorder=2)
    ax.plot([l,l],[y-.045,y+.045],color=color,lw=.6)
    ax.plot([h,h],[y-.045,y+.045],color=color,lw=.6)
    ax.plot(p,y,marker=marker,ms=3.2,color=color,label=label,zorder=3)
def axes(ax,show=True,xlim=(-20,45)):
    ax.set_yticks(range(8),SHORT if show else ['']*8);ax.set_ylim(7.65,-.7)
    ax.tick_params(axis='y',length=0,pad=4);ax.spines['left'].set_visible(False)
    ax.set_xlim(*xlim);ax.axvline(0,color='.60',ls='--',lw=.6)
    ax.grid(axis='x',color='.92',lw=.45)
def panel(ax,letter,title):
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-13,10),textcoords='offset points',
        weight='bold',fontsize=8,ha='left',va='bottom',annotation_clip=False)
    ax.text(.5,1.055,title,transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'_alignment.json'),
        overlay_svg=QA/(name+'_alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=len(fig.axes)>1,strict=True)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=600)
    fig.savefig(FIG/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def design():
    # Rebuild a delivered project from its already verified image bytes and provenance.
    provenance=DATA/'ILLUSTRATIVE_IMAGE_PROVENANCE.json'
    if provenance.is_file():
        refs=json.loads(provenance.read_text(encoding='utf-8'))['image_refs']
        images=[]
        for i,ref in enumerate(refs):
            cached=next(DATA.glob(f'illustrative_{i}.*'))
            assert sha(cached)==ref['sha256']
            with Image.open(cached) as im:images.append(np.array(im.convert('RGB')))
        return draw_design(images)
    # Illustration is selected by deterministic row order and local image availability, never by outputs.
    fixture=PROJECT/'model_admission_20261017/unified_matrix_128_eight_model_20261018_v1/fixtures_m4/RETROSPECTIVE_CONTROL_FIXTURE.jsonl'
    image_root=PROJECT/'visual_polish/source_images/m4sar'
    rows=[json.loads(x) for x in fixture.read_text(encoding='utf-8').splitlines() if x]
    row=None
    for candidate in sorted(rows,key=lambda r:r['fixture_id']):
        refs=candidate.get('image_refs',[])
        if len(refs)==2 and candidate.get('condition_id')=='A0_CONGRUENT' and candidate.get('requested_sensor')=='sar' and candidate.get('requested_image_position')==1:
            if all((image_root/r['relative_path']).is_file() for r in refs):row=candidate;break
    assert row is not None,'No verified illustrative files; do not synthesize remote-sensing imagery'
    refs=sorted(row['image_refs'],key=lambda r:r['slot'])
    images=[]
    for i,ref in enumerate(refs):
        src=image_root/ref['relative_path'];assert sha(src)==ref['sha256']
        dst=DATA/('illustrative_'+str(i)+src.suffix);shutil.copyfile(src,dst)
        with Image.open(dst) as im:images.append(np.array(im.convert('RGB')))
    (DATA/'ILLUSTRATIVE_IMAGE_PROVENANCE.json').write_text(json.dumps({'fixture_id':row['fixture_id'],
        'fixture_sha256':sha(fixture),'selection':'first sorted SAR A0 row with both already-local image files; no output used',
        'purpose':'input illustration only; images are not claimed to share a physical scene; no predictions depicted',
        'image_refs':refs},indent=2),encoding='utf-8')
    draw_design(images)

def draw_design(images):
    fig,ax=plt.subplots(figsize=(7.16,2.75));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.015,.94,'a',weight='bold',fontsize=8);ax.text(.06,.94,'Requested evidence stays fixed',fontsize=7)
    # Pure display resizing; no contrast manipulation, crop, segmentation or generated pixels.
    ax.imshow(images[0],extent=(.025,.21,.42,.90),aspect='auto',interpolation='nearest')
    ax.imshow(images[1],extent=(.25,.435,.42,.90),aspect='auto',interpolation='nearest')
    ax.text(.118,.355,'Requested SAR',ha='center',color=ORANGE)
    ax.text(.342,.355,'Non-requested optical',ha='center',color=BLUE)
    ax.text(.23,.24,'Example inputs; no prediction shown',ha='center',fontsize=6.5,color='.35')
    ax.text(.50,.94,'b',weight='bold',fontsize=8);ax.text(.545,.94,'Measure excess reversal, not any change',fontsize=7)
    def box(x,y,w,h,text,color,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.01',
                    edgecolor=color,facecolor=fill,lw=.65))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=6.7,linespacing=1.35)
    box(.52,.64,.20,.19,'Baseline A0\nSame-label evidence',GRAY,'#F4F5F6')
    box(.77,.64,.20,.19,'Second congruent\nAnother same-label image',BLUE,'#EFF5F9')
    box(.52,.32,.20,.19,'Opposite evidence\n3 fixed replacements',ORANGE,'#FBF4EC')
    box(.77,.32,.20,.19,'D = O - C\nSame reverse-answer event',GRAY,'#F4F5F6')
    for p,q in [((.73,.735),(.755,.735)),((.62,.62),(.62,.53)),((.73,.415),(.755,.415)),((.87,.62),(.87,.53))]:
        ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',lw=.7,color='.4'))
    ax.text(.745,.15,'Question and requested image unchanged across replacements',ha='center',fontsize=6.5)
    ax.text(.745,.065,'Gold labels define scoring; they are not supplied as answers',ha='center',fontsize=6.5,color='.35')
    save(fig,'fig1_design')

def conflicts():
    name='fig2_conflict';fig,axs=plt.subplots(1,2,figsize=(7.16,3.7))
    fig.subplots_adjust(left=.17,right=.90,bottom=.19,top=.86,wspace=.64)
    for j,(ds,ax) in enumerate(zip(('m4','bigearthnet'),axs)):
        axes(ax,j==0);panel(ax,chr(97+j),'M4-SAR' if ds=='m4' else 'BigEarthNet.txt')
        ax.set_xticks([-20,0,20,40]);ax.set_xlabel('Opposite - second congruent (pp)')
        for i,m in enumerate(MODELS):
            ns=[]
            for sensor,off,col,mark in [('optical',-.13,BLUE,'o'),('sar',.13,ORANGE,'s')]:
                metric=f'cell/{sensor}/{pos(ds,m,sensor)}/control/D';r=endpoint(ds,m,metric)
                dot(ax,i+off,r['estimate'],r['ci95_lower'],r['ci95_upper'],col,mark,('Optical' if sensor=='optical' else 'SAR') if i==0 and j==0 else None)
                record(name,'original_ALL_ENDPOINTS.json',dict(dataset=ds,model=m,metric=metric),r['estimate'],r['ci95_lower'],r['ci95_upper'],r['eligible_target_questions'])
                ns.append(r['eligible_target_questions'])
            ax.text(1.035,i,f'{ns[0]}/{ns[1]}',transform=ax.get_yaxis_transform(),va='center',fontsize=6.3)
        ax.text(1.035,-.60,'n O/S',transform=ax.get_yaxis_transform(),fontsize=6.3)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.54,.01),fontsize=7)
    save(fig,name)

def ability():
    name='fig3_failure_modes';fig,axs=plt.subplots(2,2,figsize=(7.16,4.65))
    fig.subplots_adjust(left=.17,right=.97,bottom=.12,top=.92,wspace=.25,hspace=.40)
    for row in range(2):
        for j,ds in enumerate(('m4','bigearthnet')):
            ax=axs[row,j];axes(ax,j==0,xlim=(-2,102));ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Rate (%)')
            panel(ax,chr(97+row*2+j),('M4-SAR' if ds=='m4' else 'BigEarthNet.txt')+(' / task capability' if row==0 else ' / non-answer outcomes'))
            metrics=[('yes_recall',BLUE,'o','Yes recall'),('no_specificity',GRAY,'s','No specificity')] if row==0 else [('abstain',ORANGE,'o','Abstain'),('parse_failure',PURPLE,'s','Parse failure')]
            for i,m in enumerate(MODELS):
                for off,(metric,col,mark,display) in zip((-.13,.13),metrics):
                    r=endpoint(ds,m,'main/A0/'+metric)
                    dot(ax,i+off,r['estimate'],r['ci95_lower'],r['ci95_upper'],col,mark,display if i==0 and j==0 else None)
                    record(name,'original_ALL_ENDPOINTS.json',dict(dataset=ds,model=m,metric=r['metric']),r['estimate'],r['ci95_lower'],r['ci95_upper'])
    hh=[];ll=[]
    for ax in axs[:,0]:
        h,l=ax.get_legend_handles_labels();hh+=h;ll+=l
    fig.legend(hh,ll,loc='lower center',ncol=4,bbox_to_anchor=(.55,.005),fontsize=6.7)
    save(fig,name)

def robustness():
    name='fig4_qualifications';fig,axs=plt.subplots(2,2,figsize=(7.16,4.9))
    fig.subplots_adjust(left=.17,right=.90,bottom=.12,top=.92,wspace=.63,hspace=.39)
    f=jr('posthoc_MULTIPLICITY_F32.json');s=jr('posthoc_REQUESTED_ONLY_SENSITIVITY.json')
    for row in range(2):
        for j,ds in enumerate(('m4','bigearthnet')):
            ax=axs[row,j];axes(ax,j==0,xlim=(-25,45));ax.set_xticks([-20,0,20,40]);ax.set_xlabel('SAR extra reverse answers (pp)')
            panel(ax,chr(97+row*2+j),('M4-SAR' if ds=='m4' else 'BigEarthNet.txt')+(' / F32 expanded intervals' if row==0 else ' / additional capability eligibility'))
            for i,m in enumerate(MODELS):
                if row==1 and m=='earthmind_4b':
                    ax.text(5,i,'N/A',fontsize=6.5,va='center',color='.4');continue
                if row==0:
                    r=next(r for r in f if (r['dataset'],r['model'],r['sensor'])==(ds,m,'sar'))
                    p,l,h=[r[k] for k in ('estimate','bonferroni_lower','bonferroni_upper')];src='posthoc_MULTIPLICITY_F32.json'
                else:
                    r=next(r for r in s if (r['dataset'],r['cohort'],r['model'],r['sensor'],r['position'])==(ds,'primary',m,'sar',pos(ds,m,'sar')))
                    p,l,h=[r[k] for k in ('D','D_ci95_lower','D_ci95_upper')];src='posthoc_REQUESTED_ONLY_SENSITIVITY.json'
                dot(ax,i,p,l,h,ORANGE,'s');n=r['eligible_questions']
                ax.text(1.035,i,str(n),transform=ax.get_yaxis_transform(),va='center',fontsize=6.4)
                record(name,src,dict(dataset=ds,model=m,sensor='sar',position=pos(ds,m,'sar')),p,l,h,n)
            ax.text(1.035,-.60,'n',transform=ax.get_yaxis_transform(),fontsize=6.4)
    fig.text(.55,.015,'Top: post-hoc approximate multiplicity expansion. Bottom: unadjusted 95% intervals in a different population.',ha='center',fontsize=6.3)
    save(fig,name)

def fmt(x):
    if x is None:return '--'
    x=float(x)*100
    return '0.00' if abs(x)<.005 else f'{x:.2f}'
def ci(r,p='estimate',lo='ci95_lower',hi='ci95_upper'):
    return fmt(r[p])+' ['+fmt(r[lo])+', '+fmt(r[hi])+']'
def tables():
    rows=['\\begin{table}[t]\\centering\\scriptsize',
        '\\caption{Baseline balanced accuracy (BA, \\%) with pointwise 95\\% target-source-cluster intervals. Each dataset is sorted by its own point estimate for display, not significance. Abstention and parse failure count as errors; full cohorts are retained (M4: 840 question--direction records; BEN: 642). Q2.5/Q3 denote the corresponding Qwen families; full checkpoint names are in Table~\\ref{tab:models}.}',
        '\\label{tab:ability}\\setlength{\\tabcolsep}{2pt}\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrlr}\\toprule',
        '\\multicolumn{2}{c}{M4-SAR} & \\multicolumn{2}{c}{BigEarthNet.txt}\\\\',
        '\\cmidrule(lr){1-2}\\cmidrule(lr){3-4}',
        'Model & BA [95\\% CI] & Model & BA [95\\% CI]\\\\\\midrule']
    main={ds:cr('original_MAIN_TABLE_'+ds+'_primary.csv') for ds in ('m4','bigearthnet')}
    ordered={ds:sorted(main[ds],key=lambda r:-float(r['a0_balanced_accuracy'])) for ds in main}
    for i in range(8):
        cells=[]
        for ds in ('m4','bigearthnet'):
            r=ordered[ds][i]
            cells += [SHORT[MODELS.index(r['model'])],ci(r,'a0_balanced_accuracy','a0_balanced_accuracy_ci95_lower','a0_balanced_accuracy_ci95_upper')]
        rows.append(' & '.join(cells)+r'\\')
    rows += ['\\bottomrule\\end{tabular*}\\end{table}']
    (HERE/'tables/main_ability.tex').write_text('\n'.join(rows),encoding='utf-8')
    rows=['\\begin{table*}[t]\\centering\\scriptsize',
        '\\caption{Complete post-hoc F32 family at the primary interfaces. D and interval limits are in percentage points; n is eligible questions/physical target sources. B denotes approximate Bonferroni-expanded percentile intervals, not exact simultaneous coverage. Original pointwise 95\\% intervals appear in Fig.~\\ref{fig:conflict}. Numerical residuals below display precision are printed as zero without using their sign for inference.}',
        '\\label{tab:f32}\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrll}\\toprule',
        'Dataset & Checkpoint & Optical D [B] & n q/s & SAR D [B] & n q/s\\\\\\midrule']
    f=jr('posthoc_MULTIPLICITY_F32.json')
    for ds in ('m4','bigearthnet'):
        for m,name in zip(MODELS,LABEL):
            a,b=[next(r for r in f if (r['dataset'],r['model'],r['sensor'])==(ds,m,sensor)) for sensor in ('optical','sar')]
            cells=['M4' if ds=='m4' else 'BEN',name,ci(a,'estimate','bonferroni_lower','bonferroni_upper'),f"{a['eligible_questions']}/{a['eligible_physical_sources']}",
                   ci(b,'estimate','bonferroni_lower','bonferroni_upper'),f"{b['eligible_questions']}/{b['eligible_physical_sources']}"]
            rows.append(' & '.join(cells)+r'\\')
        if ds=='m4':rows.append('\\midrule')
    rows+=['\\bottomrule\\end{tabular*}\\end{table*}']
    (HERE/'tables/appendix_f32.tex').write_text('\n'.join(rows),encoding='utf-8')
    identities=cr('original_INPUT_PROTOCOL_IDENTITIES.csv')
    rows=['\\begin{table*}[t]\\centering\\scriptsize',
        '\\caption{Fixed checkpoint revisions. The same revision is used for both datasets. Necessary native templates and visual processors remain model-specific.}',
        '\\label{tab:models}\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lll}\\toprule',
        'Checkpoint & Repository revision & Interface\\\\\\midrule']
    for m,name in zip(MODELS,LABEL):
        revisions={r['model_revision'] for r in identities if r['model']==m};assert len(revisions)==1
        rows.append(name+' & \\texttt{'+revisions.pop()+'} & '+('Named SAR/RGB' if m=='earthmind_4b' else 'Ordered native multi-image')+r'\\')
    rows+=['\\bottomrule\\end{tabular*}\\end{table*}']
    (HERE/'tables/appendix_models.tex').write_text('\n'.join(rows),encoding='utf-8')

if __name__=='__main__':
    design();conflicts();ability();robustness();tables()
    (FIG/'FIGURE_SOURCE_DATA.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Four figures,',len(SOURCES),'source-linked quantitative marks, three tables; no new estimation')
