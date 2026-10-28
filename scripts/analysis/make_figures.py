from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
R=Path(__file__).resolve().parents[2]
OUT=R/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
# 1 comment separate
c=pd.read_csv(R/'results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv').query('seed==42').sort_values('macro_f1')
fig,ax=plt.subplots(figsize=(8,3.7));bars=ax.barh(c.model.astype(str),c.macro_f1,color='#5a788f')
for b,v in zip(bars,c.macro_f1):ax.text(v+.001,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=9)
ax.set_xlim(0,1.07);ax.set_xlabel('Macro F1');ax.set_title('Standalone comment benchmark (held-out n = 1,226)');fig.tight_layout();fig.savefig(OUT/'comment_benchmark_macro_f1.png');plt.close(fig)
# 2 video 3 seeds, selected models
s=pd.read_csv(R/'results/video/gold_adjudicated_v1.1/METRICS_BY_SEED.csv')
sel=['text','equal_vat','visual','audio','vat_lr','healnet_adapted_2024','siglip2_vta_probe']
m=s[s.model.isin(sel)]; a=m.groupby('model').macro_f1.agg(['mean','std']).reindex(sel)
labels=['Extracted text','Equal VAT','Visual','Acoustic','VAT logistic stacking','HEALNet (local)','SigLIP2 VTA probe']
fig,ax=plt.subplots(figsize=(9,4.6));i=np.arange(len(a));ax.bar(i,a['mean'],yerr=a['std'],capsize=4,color=['#496b86','#427b7b','#9fa5ad','#b6b9bf','#aa8d6a','#9b7897','#6c8e72'])
ax.set_xticks(i,labels,rotation=20,ha='right');ax.set_ylim(0,.69);ax.set_ylabel('Macro F1');ax.set_title('Video-window evaluation: means and SD across three training seeds');ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(OUT/'video_three_seed_stability.png');plt.close(fig)
# 3 confusion matrices correct latest
p=pd.read_csv(R/'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv').query('seed==42')
fig,axes=plt.subplots(1,2,figsize=(8.7,3.65))
for ax,key,title in zip(axes,['text','equal_vat'],['Extracted text','Equal VAT']):
 cm=confusion_matrix(p.label_final,p[key+'_prediction'],labels=[0,1]); im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max());ax.set_xticks([0,1],['0','1']);ax.set_yticks([0,1],['0','1']);ax.set_xlabel('Predicted');ax.set_ylabel('Reference');ax.set_title(title)
 for r in range(2):
  for c in range(2):ax.text(c,r,str(cm[r,c]),ha='center',va='center',color='white' if cm[r,c]>.6*cm.max() else '#222')
fig.suptitle('Seed 42, same 3,951 source-disjoint video windows');fig.tight_layout();fig.savefig(OUT/'video_confusion_matrices_seed42.png');plt.close(fig)
# 4 effect interval
f=pd.read_csv(R/'results/statistics/FOCAL_FOUR_100000.csv')
print('focal headers',list(f.columns))
# avoid implicit values if unfamiliar
fig,ax=plt.subplots(figsize=(7.9,2.7));diff=-.0033623179238975;lo=-.0178765858028778;hi=.0115901332544028
ax.errorbar([diff],[0],xerr=[[diff-lo],[hi-diff]],fmt='o',capsize=6,color='#3b6d84',linewidth=2);ax.axvline(0,ls='--',color='#666');ax.set_yticks([0],['Equal VAT − text']);ax.set_xlim(-.029,.023);ax.set_ylim(-.7,.7);ax.set_xlabel('Macro F1 difference, clustered 95% bootstrap CI');ax.set_title('Paired video comparison (214 source groups, seed 42)');ax.text(0.02,.07,'Δ = −0.00336; permutation p = 0.6608',transform=ax.transAxes,fontsize=9);fig.tight_layout();fig.savefig(OUT/'focal_paired_difference_seed42.png');plt.close(fig)
(OUT/'README.md').write_text('''# Figures\n\nThese four images were generated from the saved **v1.1 final metric tables** by `scripts/analysis/make_figures.py`, without refitting any model. All values use the aligned 3,951-window population for video figures; the text benchmark is a separate held-out comment cohort. The paired interval is conditional on the 214 source groups and already-fitted predictions. No old manuscript figure panels are retained.\n''')
