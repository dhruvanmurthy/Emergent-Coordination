#!/usr/bin/env python3
"""Rebuild thesis figures from values reported in THESIS_REPORT(1).md.
No episode-level dataset is available. Diagrams are explanatory reconstructions.
Requires Python 3, matplotlib, numpy, and Pillow. Run from any directory.
"""
from pathlib import Path
import csv, hashlib, json, math, shutil, subprocess, textwrap, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
for folder in ('png', 'svg', 'pdf'): (ROOT / folder).mkdir(exist_ok=True)

def load_report_font():
    variants = [
        ('LMRoman10-Regular.otf', 'normal', 'normal'),
        ('LMRoman10-Bold.otf', 'bold', 'normal'),
        ('LMRoman10-Italic.otf', 'normal', 'italic'),
        ('LMRoman10-BoldItalic.otf', 'bold', 'italic'),
    ]
    kpsewhich = shutil.which('kpsewhich')
    regular_path = None
    for filename, weight, style in variants:
        if kpsewhich:
            result = subprocess.run(
                [kpsewhich, filename], capture_output=True, text=True, check=False
            )
            path = result.stdout.strip()
            if result.returncode != 0 or not path:
                raise RuntimeError(f'Could not locate report font file: {filename}')
        else:
            path = font_manager.findfont(
                font_manager.FontProperties(
                    family='Latin Modern Roman', weight=weight, style=style
                ),
                fallback_to_default=False,
            )
        font_manager.fontManager.addfont(path)
        if weight == 'normal' and style == 'normal':
            regular_path = path
    return regular_path

REPORT_FONT_PATH = load_report_font()
plt.rcParams.update({'font.family':'Latin Modern Roman','font.size':11,'mathtext.fontset':'cm',
    'pdf.fonttype':3,'ps.fonttype':3,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#283542','text.color':'#283542',
    'xtick.color':'#52616D','ytick.color':'#52616D','svg.fonttype':'none',
    'axes.titleweight':'bold','savefig.facecolor':'white','savefig.dpi':600})
BLUE, TEAL, AMBER, RED, GRAY = '#245D8F','#217A71','#CC8A2C','#B84949','#718392'
INK, LIGHT, BORDER = '#283542','#F2F5F7','#C7D1D9'
COND=['Single-agent','Independent multi-agent','Prompted coordination']
COL=[BLUE,TEAL,AMBER]
OUTCOL=[TEAL,AMBER,RED]
DATA={
 'source':'THESIS_REPORT(1).md; revised report supplied 17 October 2026',
 'scope':'Report-level values only; repository and episode artifacts not supplied.',
 'condition_names':COND,
 'final_outcomes':{'success':[64,60,63],'partial':[0,4,1],'failure':[0,0,0],'n':[64,64,64]},
 'scripted_outcomes':{'conditions':['Single-agent','Random valid-action','Independent multi-agent','Prompted coordination'],'success':[16,6,16,16],'partial':[4,10,4,4],'failure':[0,4,0,0],'n':[20]*4},
 'tool_counts':{'tools':['Retrieve file','Search symbol','Apply patch','Run tests','Finalize ticket'],'counts':[[58,25,136,133,64],[66,19,132,113,64],[51,21,135,100,64]]},
 'retry':{'first_attempt_completed':[57,53,59],'affected_ids':[7,11,5],'failed_attempts':[18,27,11],'further_failed_retries':[11,16,6],'recovered_invalid_responses':[34,35,27],'final_episodes_with_invalid_response':[26,30,25],'failure_reasons':[42,13,1]},
 'mi':{'conditions':COND[1:],'n':[64,58],'bits':[0.046617,0.040041],'low':[0.001735,0.0],'high':[0.197643,0.125826],'permutation_p':[0.207171,0.386454],'entropy_bits':[0.3373,0.1257],'all64_coordinator_point':0.038524},
 'profile_tables':{'independent':[['finalized','updated',22,2],['finalized','verified',1,1],['updated','finalized',37,1]],'coordinator_paired':[['finalized','retrieved',3,0],['finalized','updated',11,1],['finalized','verified',6,0],['updated','finalized',37,0]],'coordinator_all64':[['finalized','no_actions',5,0],['finalized','retrieved',3,0],['finalized','updated',11,1],['finalized','verified',6,0],['no_actions','finalized',1,0],['updated','finalized',37,0]]},
 'contrasts':{'labels':['Single − independent','Single − prompted','Prompted − independent'],'difference_pp':[6.25,1.5625,4.6875],'reported_low_pp':[0,0,0],'reported_high_pp':[15.62,4.69,12.50]},
 'participation':{'agent0_calls':[210,228],'agent1_calls':[184,143],'mean_imbalance_percent':[6.81,33.08],'mean_tvd':[0.480,0.884],'both_active_n':[64,58],'single_active_n':[0,6]},
 'partial_cases':[['Discount order · independent · case A',5,'Primary defect'],['Discount order · independent · case B',6,'Primary defect'],['Invoice total · independent · case A',5,'Tertiary defect'],['Invoice total · independent · case B',5,'Tertiary defect'],['Invoice total · prompted',4,'Tertiary defect']],
 'families':['Billing','Flag parser','Inventory','Math operations','Ranking','Retry policy','Scheduler','String utilities'],
 'family_success_counts':[[8,8,8,8,8,8,8,8],[4,8,8,8,8,8,8,8],[7,8,8,8,8,8,8,8]],
 'agent_patch_test_counts':{'independent':None,'coordinator':{'agent0_patch':109,'agent1_patch':26,'agent0_tests':36,'agent1_tests':64}}
}
FIGS=[]

def canvas(title, subtitle='', size=(12,7.2), diagram=False):
    fig=plt.figure(figsize=size)
    fig.text(.055,.955,title,fontsize=19,fontweight='bold',va='top')
    if subtitle: fig.text(.055,.905,subtitle,fontsize=10.7,color='#52616D',va='top')
    if diagram:
        ax=fig.add_axes([.035,.14,.93,.70]); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        return fig,ax
    return fig

def box(ax,x,y,w,h,title,body='',color=BLUE,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',facecolor=LIGHT,edgecolor=color,lw=1.3))
    if body:
        ax.text(x+w/2,y+h*.73,title,ha='center',va='center',weight='bold',fontsize=fs,color=color)
        ax.text(x+w/2,y+h*.35,body,ha='center',va='center',fontsize=fs-1,linespacing=1.5)
    else: ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,weight='bold',color=color,linespacing=1.4)

def arrow(ax,p,q,label=None,color=GRAY,rad=0,style='-|>'):
    start=ax.transData.transform(p);end=ax.transData.transform(q)
    boxes=[patch for patch in ax.patches if isinstance(patch,FancyBboxPatch)]
    patchA=next((patch for patch in boxes if patch.contains_point(start)),None)
    patchB=next((patch for patch in boxes if patch.contains_point(end)),None)
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=13,color=color,lw=1.35,
                                 connectionstyle=f'arc3,rad={rad}',patchA=patchA,patchB=patchB,
                                 shrinkA=0 if patchA else 2,shrinkB=0 if patchB else 2))
    if label:
        ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.02,label,ha='center',fontsize=9,color=INK,
                bbox=dict(facecolor='white',edgecolor='none',pad=1))

def save(fig,slug,title,section,caption,note='',kind='reported data',recommended=True):
    num=len(FIGS)+1
    stem=f'{num:02d}_{slug}'
    fig.savefig(ROOT/'png'/f'{stem}.png',dpi=600,bbox_inches='tight',pad_inches=0.12)
    fig.savefig(ROOT/'svg'/f'{stem}.svg',bbox_inches='tight',pad_inches=0.12)
    fig.savefig(ROOT/'pdf'/f'{stem}.pdf',bbox_inches='tight',pad_inches=0.12)
    FIGS.append(dict(number=num,filename=stem,title=title,section=section,kind=kind,
                     recommended='Core' if recommended else 'Optional',caption=caption))
    plt.close(fig)

def polish(ax,gridaxis='x'):
    ax.set_axisbelow(True);ax.grid(axis=gridaxis,color='#E2E7EB',lw=.7)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(BORDER)
    ax.tick_params(length=0,pad=8)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(c-h),100*(c+h)

def entropy_and_mi(rows):
    a=np.array([r[2:] for r in rows],dtype=float);n=a.sum();py=a.sum(axis=0)/n;px=a.sum(axis=1)/n
    h=-sum(v*math.log2(v) for v in py if v)
    mi=sum(a[i,j]/n*math.log2((a[i,j]/n)/(px[i]*py[j])) for i,j in np.argwhere(a>0))
    return h,mi

# 01: Study boundary. Two panels avoid suggesting an empirical performance bridge.
fig,ax=canvas('Two phases, distinct experimental questions','Study overview and the boundary of the evidence',diagram=True)
box(ax,.04,.50,.41,.34,'PHASE 1 · GROUP GUESSING','Private integer guesses\nShared directional feedback\nConvergence and trajectory inspection',BLUE,13)
box(ax,.55,.50,.41,.34,'PHASE 2 · TOOL SELECTION','Supplied, state-filtered actions\nSynthetic defect-state checks\nCompletion, action use, stopping',TEAL,13)
box(ax,.04,.13,.41,.23,'Evidence boundary','Adapted task; incomplete run archive\nNo full paper-protocol replication',BLUE)
box(ax,.55,.13,.41,.23,'Evidence boundary','192 final episodes; one model/provider\nNo unrestricted repair evaluation',TEAL)
arrow(ax,(.245,.49),(.245,.37));arrow(ax,(.755,.49),(.755,.37))
ax.text(.50,.94,'Shared motivation: make multi-agent behavior observable',ha='center',fontsize=13,weight='bold')
ax.text(.50,.02,'Task and measurement change between phases; outcome rates are not directly comparable.',ha='center',fontsize=10)
save(fig,'two_phase_overview','Two-phase study overview','§§1.4, 4.1, 10.1','The two phases share a research motivation but differ in task, observation unit, and analysis. Phase 2 is a methodological extension rather than a direct performance comparison or replication.',kind='conceptual diagram')

# 02: Phase 1 information topology.
fig,ax=canvas('Phase 1: private guesses and shared feedback','Sum-mode information flow; agents do not observe peers’ guesses',diagram=True)
for i,x in enumerate([.05,.37,.69]):
    box(ax,x,.64,.25,.19,f'Agent {i}','Own guess history only',BLUE)
box(ax,.26,.30,.48,.17,'Group evaluation','Sum guesses; compare with hidden target',TEAL,12)
for i,x in enumerate([.05,.37,.69]):
    target_x=[.36,.50,.64][i]
    arrow(ax,(x+.125,.63),(target_x,.46),f'guess {i}')
box(ax,.26,.04,.48,.16,'Shared categorical feedback','Too high · Too low · Correct',AMBER,12)
arrow(ax,(.5,.30),(.5,.21))
ax.plot([.245,.015,.015,.815],[.12,.12,.94,.94],color=GRAY,lw=1.2)
for x in [.175,.495,.815]: arrow(ax,(x,.94),(x,.84))
ax.text(.50,.985,'If incorrect, next round: own prior guesses + the same group feedback',ha='center',fontsize=10)
save(fig,'phase1_feedback','Private-guess/shared-feedback loop','§§2.2–2.4','Illustrative three-agent sum-mode loop. The evaluator observes guesses and the hidden target; each agent receives its own history and shared categorical feedback. Correct feedback terminates the game.',kind='conceptual diagram',recommended=False)

# 03: Architecture with a single action-selection adapter.
fig,ax=canvas('Phase 2 architecture and artifact flow','Sequential action selection over one shared synthetic environment',diagram=True,size=(12,8))
box(ax,.05,.80,.27,.15,'Fixture catalog','8 families · 16 variants',TEAL)
box(ax,.40,.80,.26,.15,'Frozen run design','Settings + manifest',BLUE)
box(ax,.40,.57,.26,.15,'Sweep runner','Assignment, progress, retries',BLUE)
box(ax,.04,.34,.29,.16,'Policy adapter','Shared state + recent actions',BLUE)
box(ax,.40,.34,.26,.16,'LLM client','One structured action per decision',AMBER)
box(ax,.74,.34,.23,.16,'Run audit','Progress / errors / aggregate',GRAY)
box(ax,.04,.09,.29,.16,'Synthetic environment','Validate action; update fixture state',TEAL)
box(ax,.40,.09,.26,.16,'Canonical episode files','Events · summary · fixture',TEAL)
box(ax,.74,.09,.23,.16,'Derived outputs','CSV/Parquet · analysis · plots',GRAY)
arrow(ax,(.32,.875),(.40,.875));arrow(ax,(.53,.80),(.53,.73));arrow(ax,(.40,.61),(.185,.51))
arrow(ax,(.33,.42),(.40,.42),style='<->');arrow(ax,(.66,.64),(.85,.51))
arrow(ax,(.13,.34),(.13,.25));arrow(ax,(.25,.25),(.25,.34));arrow(ax,(.33,.17),(.40,.17));arrow(ax,(.66,.17),(.74,.17))
ax.text(.10,.295,'action',ha='right',fontsize=8,color=GRAY)
ax.text(.27,.295,'state',ha='left',fontsize=8,color=GRAY)
ax.plot([.041,.015,.015,.031],[.87,.87,.17,.17],color=TEAL,lw=1.2)
ax.text(.53,.27,'Tool event after validation',ha='center',fontsize=9,color=GRAY)
save(fig,'phase2_architecture','Phase 2 execution and artifact architecture','§§5.1–5.6','A sweep invokes a policy adapter that requests one action at a time from the model, using shared task state and a supplied valid-action list. The environment validates and applies actions; canonical episode records support exports and analysis. Nominal actors are not separate persistent agent contexts.',kind='architecture diagram')

# 04: Validation funnel.
fig,ax=canvas('Action selection, validation, and retry boundaries','Model-response validity and environment-call validity are different stages',diagram=True,size=(12,8))
box(ax,.05,.77,.26,.17,'State-filtered action list','Includes exact catalog patches\nand replacement text',TEAL)
box(ax,.39,.77,.25,.17,'Model response','One JSON action selection',BLUE)
box(ax,.39,.47,.25,.17,'Policy validation','JSON shape + exact allow-list\nmatch + repetition checks',BLUE)
box(ax,.39,.14,.25,.17,'Environment step','Schema, references, state, budget\nthen fixture-state transition',TEAL)
box(ax,.73,.47,.23,.17,'Response recovery','Retry invalid selection\nStrict exhaustion → error',AMBER)
arrow(ax,(.31,.85),(.39,.85));arrow(ax,(.515,.77),(.515,.65));arrow(ax,(.515,.47),(.515,.32),'accepted')
arrow(ax,(.64,.55),(.73,.55),'rejected');arrow(ax,(.85,.65),(.64,.85),rad=.2)
box(ax,.04,.12,.26,.23,'Final-trace upstream audit','96 invalid responses recovered\n81 of 192 final episodes',AMBER)
box(ax,.73,.12,.23,.23,'Submitted tool events','1,181 recorded events\n0 environment-invalid calls\n403/403 patches accepted',TEAL)
arrow(ax,(.64,.225),(.73,.225))
save(fig,'action_validation','Action-selection and validation boundaries','§§5.2, 5.4, 9.4','Policy validation rejects or retries model responses before actions reach the environment. The 96 recoverable invalid responses in final traces are distinct from zero invalid environment calls. Allow-list selection exposes exact patch replacements, so patch acceptance measures compliance within a constrained interface.',kind='block diagram')

# 05: Matched design.
fig,ax=canvas('Final-run experimental design','Matched fixture/repetition identities across the three policies',diagram=True)
box(ax,.04,.73,.25,.17,'8 fixture families','2 variants per family',TEAL)
box(ax,.38,.73,.25,.17,'16 fixture variants','4 repetitions per variant',BLUE)
box(ax,.72,.73,.25,.17,'64 matched assignments','Fixture + repetition + seed ID',BLUE)
arrow(ax,(.29,.815),(.38,.815));arrow(ax,(.63,.815),(.72,.815))
for i,x in enumerate([.04,.375,.71]):
    box(ax,x,.30,.25,.23,COND[i],'64 final episodes\n12 tool-call budget\nStrict LLM handling',COL[i])
ax.plot([.845,.845,.165],[.72,.63,.63],color=GRAY,lw=1.2)
for x in [.165,.50,.835]: arrow(ax,(x,.63),(x,.54))
box(ax,.28,.01,.44,.16,'192 completed final episodes','187 success · 5 partial · 0 failure',TEAL,12)
for x,target_x in zip([.165,.50,.835],[.36,.50,.64]):
    arrow(ax,(x,.29),(target_x,.16))
save(fig,'matched_design','Balanced matched-assignment study design','§§5.5, 9.1','The final run covers 16 engineered fixture variants and four repetitions per policy–variant cell. The three policies share fixture/repetition/seed identities, yielding 64 episodes per policy and 192 completed episodes. Shared seed identities do not make model responses deterministic.',kind='study-design diagram')

# 06: Actual actor-assignment semantics.
fig,ax=canvas('What differs between the model-backed policies','One shared state and policy-level history; no persistent per-agent memory',diagram=True)
for i,x in enumerate([.035,.365,.695]):
    box(ax,x,.66,.27,.20,COND[i],['One nominal actor','Two alternating actor IDs','Coordinator selects actor ID'][i],COL[i],12)
    box(ax,x,.30,.27,.24,['Actor 0 every turn','Actor 0 → 1 → 0 → 1','Actor 0 or 1 each turn'][i],
        ['Model selects tool + arguments','Adapter fixes actor; model selects\ntool + arguments','Model selects tool + arguments\nand attributes action to an actor'][i],COL[i])
    arrow(ax,(x+.135,.65),(x+.135,.55))
box(ax,.12,.015,.76,.16,'Shared task state + last six selected actions + valid-action list',
    'Sequential model requests; no direct agent messaging or separate private contexts',GRAY,12)
for x in [.17,.50,.83]:arrow(ax,(x,.29),(x,.19))
save(fig,'policy_assignment','Policy and actor-assignment comparison','§§5.3–5.4','Single-agent actions use actor 0. Independent mode alternates actor IDs by adapter rule; prompted coordination lets one action-selection process choose the actor ID. Role wording and actor-assignment authority change together, preventing a prompt-only causal interpretation.',kind='policy diagram')

# 07: abstraction.
fig,ax=canvas('From tool events to a coarse action profile','The priority rule discards order and can hide earlier work',diagram=True)
tools=['retrieve_file / search_symbol','run_tests','apply_patch','finalize_ticket','invalid call']
cats=['retrieve','verify','update','finalize','error_or_noop']
for i,(t,c) in enumerate(zip(tools,cats)):
    y=.81-i*.145
    box(ax,.02,y,.32,.10,t,fs=10);box(ax,.405,y,.18,.10,c,color=TEAL,fs=10);arrow(ax,(.345,y+.05),(.40,y+.05))
box(ax,.68,.56,.29,.35,'Highest priority present wins','error/no-op > finalized\n> updated > verified\n> retrieved > no actions',AMBER,12)
arrow(ax,(.59,.55),(.67,.70))
box(ax,.66,.12,.32,.30,'Compression example','Retrieve → patch → test → finalize\nOnly finalize\nBoth become: finalized',RED,11)
save(fig,'action_abstraction','Action abstraction and information loss','§§5.6–5.7','Tool events map to five action categories; a priority rule then assigns one per-agent profile. The two illustrative traces both become finalized, demonstrating how terminal-action priority masks prior retrieval, update, and verification. The example is conceptual, not an observed episode sequence.',kind='analysis diagram')

# 08: scripted pilot.
d=DATA['scripted_outcomes'];fig=canvas('Scripted pilot: episode outcomes','Pipeline validation; these actions were not generated by an LLM')
ax=fig.add_axes([.26,.24,.68,.57]);left=np.zeros(4);y=np.arange(4)
for key,color in zip(['success','partial','failure'],OUTCOL):
    vals=np.array(d[key]);ax.barh(y,vals,left=left,color=color,height=.58,label=key.title())
    for i,v in enumerate(vals):
        if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if key!='partial' else INK,weight='bold')
    left+=vals
ax.set(yticks=y,yticklabels=d['conditions'],xlim=(0,20),xlabel='Episodes per scripted condition (n = 20)');ax.invert_yaxis();polish(ax)
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.34),ncol=3,frameon=False)
save(fig,'scripted_pilot_outcomes','Scripted pilot outcomes','§§7.1–7.4','The 80-episode scripted pilot recorded 54 successes, 22 partials, and four failures. The three non-random policies shared 16/20 success outcomes; the random valid-action policy had 6/20. These are pipeline checks, not LLM treatment estimates.',recommended=False)

# 09: final outcomes and Wilson.
fig=canvas('Final-run completion by policy','Success, partial, and failure remain separate outcomes; 64 episodes per policy')
ax=fig.add_axes([.27,.46,.67,.35]);y=np.arange(3);succ=np.array([64,60,63]);partial=np.array([0,4,1]);
ax.barh(y,succ,color=TEAL,height=.53,label='Success');ax.barh(y,partial,left=succ,color=AMBER,height=.53,label='Partial')
for i,(s,p) in enumerate(zip(succ,partial)):
    ax.text(s/2,i,f'{s} success',ha='center',va='center',color='white',weight='bold')
    if p:ax.text(s+p/2,i,str(p),ha='center',va='center',fontsize=10)
ax.set(yticks=y,yticklabels=COND,xlim=(0,64),xlabel='Completed final episodes');ax.invert_yaxis();polish(ax)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.25),ncol=2,frameon=False)
ax2=fig.add_axes([.27,.19,.40,.18]);rates=succ/64*100
for i,(k,color) in enumerate(zip(succ,COL)):
    lo,hi=wilson(k,64);ax2.errorbar(rates[i],i,xerr=[[rates[i]-lo],[hi-rates[i]]],fmt='o',color=color,capsize=4)
    ax2.text(1.10,i,f'{rates[i]:.2f}%  [{lo:.2f}, {hi:.2f}]',transform=ax2.get_yaxis_transform(),ha='left',va='center',fontsize=9,clip_on=False)
ax2.set(yticks=[],xlim=(82,101),xlabel='Success rate (%) · marginal 95% Wilson interval');ax2.invert_yaxis();polish(ax2)
save(fig,'final_outcomes','Final outcomes and marginal success-rate intervals','§9.2','The final dataset contains 187 successes and five partial outcomes, with no failures. Dots and whiskers show success rates and two-sided 95% Wilson intervals calculated from reported counts. Marginal Wilson intervals do not adjust for repeated fixtures; they are not a paired policy comparison.')

# 10: paired contrasts.
fig=canvas('Paired differences in success rate','Reported 95% percentile intervals from fixture-variant cluster resampling')
ax=fig.add_axes([.32,.24,.60,.55]);d=DATA['contrasts'];yy=np.arange(3)
for i,(v,lo,hi) in enumerate(zip(d['difference_pp'],d['reported_low_pp'],d['reported_high_pp'])):
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COL[i],capsize=5,ms=7)
    ax.text(17,i,f'+{v:.2f} pp\n[{lo:.2f}, +{hi:.2f}]',va='center',fontsize=10)
ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set(yticks=yy,yticklabels=d['labels'],xlim=(-.7,22),xlabel='Difference in success rate (percentage points)');ax.invert_yaxis();polish(ax)
save(fig,'paired_contrasts','Matched-policy success-rate contrasts','§9.2','Point differences use matched reported outcomes. Whiskers reproduce the report’s 95% fixture-variant cluster bootstrap intervals (10,000 resamples); all include zero. These intervals were not independently rerun from episode-level data, and no causal or superiority claim follows.',note='Bootstrap intervals reproduced from report; raw paired records unavailable.',recommended=False)

# 11: tool composition in calls/episode, not a dubious efficiency index.
fig=canvas('Tool-call volume and composition','Recorded calls in final persisted episodes; failed-attempt work is excluded')
ax=fig.add_axes([.26,.27,.68,.52]);a=np.array(DATA['tool_counts']['counts'])/64;left=np.zeros(3)
colors=[BLUE,'#7FABC6',TEAL,AMBER,GRAY]
for j,(name,color) in enumerate(zip(DATA['tool_counts']['tools'],colors)):
    vals=a[:,j];ax.barh(np.arange(3),vals,left=left,color=color,height=.55,label=name)
    for i,v in enumerate(vals):
        if v>.65:ax.text(left[i]+v/2,i,f'{v:.2f}',ha='center',va='center',color='white' if j in [0,2,4] else INK,fontsize=10)
    left+=vals
for i,total in enumerate(left):ax.text(total+.09,i,f'{total:.2f}',va='center',weight='bold')
ax.set(yticks=np.arange(3),yticklabels=COND,xlim=(0,7.2),xlabel='Mean tool calls per final episode');ax.invert_yaxis();polish(ax)
ax.legend(loc='lower center',bbox_to_anchor=(.48,-.38),ncol=3,frameon=False,fontsize=10)
save(fig,'tool_call_mix','Tool-call volume and composition','§9.3','Stacked bar lengths are mean calls per final episode, calculated by dividing each reported tool count by 64. Totals are 6.50, 6.16, and 5.80 calls. Fewer calls alone do not demonstrate greater efficiency because five episodes stopped with unresolved defects.')

# 12: observed stopping positions only; no synthetic intermediate trajectories.
fig=canvas('Five partial episodes stopped before the call limit','Every partial outcome occurred in billing and ended with a required defect unresolved')
ax=fig.add_axes([.36,.23,.58,.57]);cases=DATA['partial_cases'];y=np.arange(5)
for i,(_,calls,defect) in enumerate(cases):
    ax.barh(i,12,color=LIGHT,height=.48);ax.barh(i,calls,color=AMBER,height=.48)
    ax.scatter(calls,i,s=55,color=RED,zorder=3);ax.text(calls+.18,i,f'Finalize at {calls} · {12-calls} calls unused',va='center',fontsize=10)
ax.axvline(12,color=GRAY,ls='--');ax.set(yticks=y,yticklabels=[c[0] for c in cases],xlim=(0,12.6),xlabel='Tool-call position within the 12-call budget');ax.invert_yaxis();polish(ax)
save(fig,'premature_finalization','Premature finalization in the five partial episodes','§9.2.1','The reported partial episodes finalized after 4–6 calls despite a 12-call limit. Discount-order cases left the primary defect unresolved; invoice-total cases left the tertiary defect unresolved. Case A/B are display labels, not episode IDs. Bars show stopping positions only, not intermediate action histories.')

# 13: retries, separate final sample from executions.
fig=canvas('Collection reliability and whole-episode retries','Task outcomes and technical collection completion use different denominators')
ax=fig.add_axes([.26,.50,.67,.29]);r=DATA['retry'];y=np.arange(3)
ax.barh(y,r['first_attempt_completed'],color=TEAL,height=.55,label='Completed on first attempt');ax.barh(y,r['affected_ids'],left=r['first_attempt_completed'],color=AMBER,height=.55,label='Required a whole-episode retry')
for i,(s,p) in enumerate(zip(r['first_attempt_completed'],r['affected_ids'])):
    ax.text(s/2,i,str(s),color='white',ha='center',va='center',weight='bold');ax.text(s+p/2,i,str(p),ha='center',va='center')
ax.set(yticks=y,yticklabels=COND,xlim=(0,64),xlabel='Manifest IDs (64 per policy)');ax.invert_yaxis();polish(ax);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.26),ncol=2,frameon=False,fontsize=9)
ax2=fig.add_axes([.26,.18,.67,.16]);completed=np.array([64]*3);failed=np.array(r['failed_attempts'])
ax2.barh(y,completed,color=BLUE,height=.55,label='Final completed executions');ax2.barh(y,failed,left=completed,color=RED,height=.55,label='Failed collection attempts')
ax2.legend(loc='lower center',bbox_to_anchor=(.5,1.05),ncol=2,frameon=False,fontsize=9)
for i,f in enumerate(failed):ax2.text(64+f/2,i,str(f),color='white',ha='center',va='center');ax2.text(64+f+.7,i,f'{64+f} total',va='center',fontsize=10)
ax2.set(yticks=[],xlim=(0,102),xlabel='Episode executions (192 completed + 56 failed = 248 total)');ax2.invert_yaxis();polish(ax2)
save(fig,'retry_burden','First-attempt completion and collection burden','§9.4','The first sweep completed 169/192 manifest IDs; 23 initially failed IDs eventually completed. Whole-episode restarts produced 56 failed-attempt records and 248 total executions. The final task-outcome sample remains 192 episodes. Lower-panel red segments count failed attempts, not task failures.')

# 14: MI diagnostic uncertainty.
fig=canvas('Exploratory action-profile mutual information','Dependence under a coarse encoding; these are not validated coordination scores')
ax=fig.add_axes([.29,.34,.63,.42]);d=DATA['mi']
for i in range(2):
    v=d['bits'][i];ax.errorbar(v,i,xerr=[[v-d['low'][i]],[d['high'][i]-v]],fmt='o',color=COL[i+1],capsize=5,ms=7)
    ax.text(.217,i,f"n = {d['n'][i]}\np = {d['permutation_p'][i]:.3f}",va='center',fontsize=10)
ax.scatter([d['all64_coordinator_point']],[1.30],marker='D',s=40,color=GRAY)
ax.text(.075,1.30,'All-64 sensitivity: 0.038524 bits (point only)',va='center',fontsize=9)
ax.set(yticks=[0,1],yticklabels=d['conditions'],xlim=(-.005,.255),ylim=(1.55,-.45),xlabel='Mutual information (bits); 95% episode-bootstrap intervals');polish(ax)
fig.text(.29,.20,'Outcome entropy: independent 0.3373 bits; paired coordinator 0.1257 bits.\nSingle-agent: no two-agent estimate. Bootstrap and permutation diagnostics do not preserve fixture clusters.',fontsize=10,linespacing=1.6)
save(fig,'mutual_information','Exploratory MI estimates and eligibility sensitivity','§9.5','Whiskers reproduce the reported 500-resample percentile intervals; p-values are from 250 unrestricted outcome permutations. The diamond retains all 64 coordinator episodes by adding no_actions profiles and has no reported interval. These diagnostics neither compare policies inferentially nor establish coordination.',note='Intervals/p-values reproduced; point estimates checked from reported contingency tables.')

# 15: actual joint-profile outcome frequencies.
fig=canvas('What drives the joint-profile MI estimates','Priority-compressed profile pairs; counts are final episodes')
for j,key in enumerate(['independent','coordinator_paired']):
    rows=DATA['profile_tables'][key];ax=fig.add_axes([.30,.54-j*.35,.62,.25]);yy=np.arange(len(rows));s=np.array([r[2] for r in rows]);p=np.array([r[3] for r in rows])
    ax.barh(yy,s,color=TEAL,height=.58);ax.barh(yy,p,left=s,color=AMBER,height=.58)
    for i,(ss,pp) in enumerate(zip(s,p)):
        ax.text(ss+pp+.4,i,f'{ss} S / {pp} P',va='center',fontsize=9)
    ax.set(yticks=yy,yticklabels=[f'{r[0]} / {r[1]}' for r in rows],xlim=(0,44),xlabel='Episode count');ax.invert_yaxis();polish(ax)
    ax.set_title(('Independent multi-agent · n = 64' if j==0 else 'Prompted coordination · paired n = 58'),loc='left',fontsize=11,pad=9)
save(fig,'joint_profile_counts','Joint-profile/outcome frequency distributions','§9.5','Each profile pair is ordered agent 0 / agent 1. Green denotes success (S), amber partial (P). Independent-policy partials occupy all three observed cells (2, 1, 1); the coordinator paired sample has one partial in finalized/updated. Every pair includes finalized, exposing the terminal-actor component of the encoding.',recommended=False)

# 16: participation, distinct denominators.
fig=canvas('Actor participation and tool-mix differences','Descriptive attribution patterns; assignment rules differ between policies',size=(12,8))
p=DATA['participation'];labels=['Independent','Prompted']
axes=[fig.add_axes([.16,.56,.29,.22]),fig.add_axes([.65,.56,.28,.22]),fig.add_axes([.16,.23,.29,.20]),fig.add_axes([.65,.23,.28,.20])]
ax=axes[0];tot=np.array(p['agent0_calls'])+np.array(p['agent1_calls']);share=np.array(p['agent0_calls'])/tot*100
ax.barh([0,1],share,color=BLUE,label='Actor 0');ax.barh([0,1],100-share,left=share,color='#9EBBCF',label='Actor 1')
for i,v in enumerate(share):ax.text(v/2,i,f'{v:.1f}%',ha='center',va='center',color='white');ax.text(v+(100-v)/2,i,f'{100-v:.1f}%',ha='center',va='center')
ax.set(yticks=[0,1],yticklabels=labels,xlim=(0,100),xlabel='Share of pooled tool calls (%)');ax.invert_yaxis();ax.set_title('A · Pooled actor workload',loc='left',fontsize=11);ax.legend(ncol=2,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.25));polish(ax)
for ax,vals,title,xlabel,lim,fmt in [(axes[1],p['mean_imbalance_percent'],'B · Mean episode workload imbalance',r'Mean $|n_0 - n_1| / (n_0 + n_1)$ (%)',40,'{:.2f}%'),(axes[2],p['mean_tvd'],'C · Mean tool-mix distance','Mean total variation distance',1.05,'{:.3f}'),(axes[3],p['single_active_n'],'D · Single-active-actor episodes','Episodes out of 64',8,'{:.0f}/64')]:
    ax.barh([0,1],vals,color=[TEAL,AMBER],height=.53)
    for i,v in enumerate(vals):ax.text(v+lim*.03,i,fmt.format(v),va='center',fontsize=10)
    ax.set(yticks=[0,1],yticklabels=labels,xlim=(0,lim),xlabel=xlabel);ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=10);polish(ax)
for ax in axes:
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_fontsize(9)
fig.text(.09,.105,'TVD uses both-active episodes only: independent n = 64; prompted n = 58. Imbalance uses all 64 per policy.',fontsize=9)
save(fig,'actor_participation','Actor workload and tool-mix attribution','§9.5.1','Pooled actor shares and mean per-episode imbalance are different summaries. Prompted coordination has greater mean imbalance and tool-mix TVD, but it also permits model-selected actor attribution. TVD excludes six single-active coordinator episodes. These results do not demonstrate emergent specialization.')

# 17: per-family n is constant and explicit.
fig=canvas('Completion by fixture family and policy','Every non-billing family had full success; all five partial outcomes were in billing')
ax=fig.add_axes([.24,.20,.67,.60]);arr=np.array(DATA['family_success_counts']).T
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('completion',['#F5E1B8','#B9D8CE',TEAL])
ax.imshow(arr/8*100,cmap=cmap,vmin=0,vmax=100,aspect='auto')
for i in range(8):
    for j in range(3):ax.text(j,i,f'{arr[i,j]}/8',ha='center',va='center',color='white' if arr[i,j]>=7 else INK,weight='bold')
ax.set(xticks=[0,1,2],xticklabels=['Single-agent','Independent','Prompted'],yticks=np.arange(8),yticklabels=DATA['families']);ax.tick_params(length=0,pad=10)
for spine in ax.spines.values():spine.set_visible(False)
ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which='minor',color='white',lw=3);ax.tick_params(which='minor',length=0)
save(fig,'fixture_family_outcomes','Fixture-family success counts by condition','§9.5.1','Each cell contains eight episodes: two fixture variants × four repetitions. Billing success counts were 8/8, 4/8, and 7/8; all seven other families had 8/8 successes in each condition. These engineered families are not a random sample of software-repair tasks.')

# 18: state transition branches; outcome checks are conceptual, not executed tests.
fig,ax=canvas('Synthetic episode lifecycle and outcome rules','Fixture-state evaluation; run_tests does not execute patched Python code',diagram=True)
box(ax,.04,.73,.24,.17,'Not started','Fresh copied fixture state',BLUE)
box(ax,.40,.73,.25,.17,'In progress','Valid actions update observations\nand/or simulated defect state',TEAL)
arrow(ax,(.28,.815),(.40,.815),'task progress')
box(ax,.40,.40,.25,.17,'Evaluate terminal state','Finalize or budget exhaustion',GRAY)
arrow(ax,(.525,.73),(.525,.58))
for start_x,(x,t,b,c) in zip([.44,.525,.61],[(.025,'Success','All required defects fixed;\nall fixture checks pass',TEAL),(.37,'Partial','Some progress; incomplete fixture;\nno catastrophic regression',AMBER),(.715,'Failure','Otherwise, under the\nenvironment’s outcome rules',RED)]):
    box(ax,x,.035,.26,.23,t,b,c,12);arrow(ax,(start_x,.39),(x+.13,.25))
save(fig,'episode_lifecycle','Synthetic state transitions and terminal outcomes','§§5.1–5.2','A fresh fixture enters an in-progress state as actions make task progress. Finalization or budget exhaustion triggers classification using predefined defect and regression rules. Synthetic success is an environment outcome, not proof that executed software was repaired.',kind='state diagram',recommended=False)

# Export exact chart inputs, mapping, captions, and a machine-checkable summary.
(ROOT/'source'/'figure_data.json').write_text(json.dumps(DATA,indent=2)+'\n')
with (ROOT/'figure_manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(FIGS[0]));writer.writeheader();writer.writerows(FIGS)
captions=['THESIS FIGURES — PLACEMENT AND CAPTIONS','',
 'All 18 figures use Latin Modern Roman to match the thesis and are supplied as 600-dpi PNG, editable vector SVG, and vector PDF.',
 'Numbers below identify asset files; renumber figures after insertion into the thesis.',
 'Source: revised THESIS_REPORT(1).md. Underlying repository/episode files were not supplied.',
 'Diagrams reconstruct the described design. No missing raw data or trajectories were invented.','']
for f in FIGS:
    captions += [f"Figure asset {f['number']:02d}. {f['title']}",f"Suggested placement: {f['section']} | {f['recommended']} | {f['kind']}",f"Files: png/{f['filename']}.png; svg/{f['filename']}.svg; pdf/{f['filename']}.pdf",f['caption'],'']
(ROOT/'CAPTIONS_AND_PLACEMENT.txt').write_text('\n'.join(captions))

# QA: reported values and contingency-table estimates; no fabricated episode records.
checks={}
checks['final_outcomes_total']=sum(DATA['final_outcomes']['success'])+sum(DATA['final_outcomes']['partial'])==192
checks['tool_counts_total']=sum(map(sum,DATA['tool_counts']['counts']))==1181
checks['patch_total']=sum(row[2] for row in DATA['tool_counts']['counts'])==403
checks['retry_accounting']=sum(DATA['retry']['failed_attempts'])==56 and sum(DATA['retry']['first_attempt_completed'])==169
for key,expected in [('independent',.046617),('coordinator_paired',.040041),('coordinator_all64',.038524)]:
    h,mi=entropy_and_mi(DATA['profile_tables'][key]);checks['MI_'+key]=bool(abs(mi-expected)<.0000005)
    checks['computed_'+key]={'entropy_bits':h,'mi_bits':mi}
checks['wilson_intervals_percent']={COND[i]:wilson(k,64) for i,k in enumerate([64,60,63])}
for f in FIGS:
    with Image.open(ROOT/'png'/f"{f['filename']}.png") as im:
        assert im.width>=6000 and im.height>=3000
        im.verify()
    assert (ROOT/'pdf'/f"{f['filename']}.pdf").is_file()
assert all(v for k,v in checks.items() if isinstance(v,bool))
(ROOT/'source'/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')

# Overview is an index, not a substitute for full-size images.
cellw,cellh,cols=600,420,3;rows=math.ceil(len(FIGS)/cols)
sheet=Image.new('RGB',(cols*cellw,rows*cellh),'#E8EDF0');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype(REPORT_FONT_PATH,17)
for i,f in enumerate(FIGS):
    im=Image.open(ROOT/'png'/f"{f['filename']}.png").convert('RGB');im.thumbnail((580,365))
    x=(i%cols)*cellw;y=(i//cols)*cellh
    sheet.paste(im,(x+(cellw-im.width)//2,y+10))
    title=f"{f['number']:02d} · {f['title']}"
    for j,line in enumerate(textwrap.wrap(title,53)):
        draw.text((x+12,y+373+j*21),line,font=font,fill=INK)
sheet.save(ROOT/'FIGURE_OVERVIEW.png')

checksum_files = [
    ROOT/'CAPTIONS_AND_PLACEMENT.txt', ROOT/'FIGURE_OVERVIEW.png',
    ROOT/'THESIS_REVIEW_NOTES.txt', ROOT/'figure_manifest.csv',
    ROOT/'source'/'figure_data.json', Path(__file__), ROOT/'source'/'verification.json',
]
checksum_files.extend((ROOT/folder/f"{figure['filename']}.{extension}")
                      for figure in FIGS for folder, extension in
                      [('png', 'png'), ('svg', 'svg'), ('pdf', 'pdf')])
checksums = {
    path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in checksum_files
}
(ROOT/'source'/'file_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
print(json.dumps({'figures':len(FIGS),'formats':['PNG 600 dpi','SVG','vector PDF'],'font':'Latin Modern Roman','checks_passed':True,'output':str(ROOT)}))
