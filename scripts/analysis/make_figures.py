"""Generate publication-readable figures from frozen v1.1 and V4 revision tables.

Enforces >10pt typography at print size, removes outer box borders and top/right
spines, and avoids (a)/(b)/(c) panel letters per journal formatting rules.
Does not modify the archived V4 PNG snapshots in results/statistics/revision_20261108/probability/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

R = Path(__file__).resolve().parents[2]
OUT = R / 'results/figures'
REV_OUT = OUT / 'revision_20261108'
OUT.mkdir(parents=True, exist_ok=True)
REV_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.edgecolor': 'none',
    'savefig.edgecolor': 'none',
})

# 1. Standalone comment benchmark Macro F1 (seed 42, n = 1,226)
c = pd.read_csv(R / 'results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv').query('seed == 42').sort_values('macro_f1')
fig, ax = plt.subplots(figsize=(8.2, 3.9))
bars = ax.barh(c.model.astype(str), c.macro_f1, color='#5a788f')
for b, v in zip(bars, c.macro_f1):
    ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f'{v:.4f}', va='center', fontsize=10.5)
ax.set_xlim(0, 1.09)
ax.set_xlabel('Macro F1')
ax.set_title('Standalone comment benchmark (held-out n = 1,226)')
fig.tight_layout()
fig.savefig(OUT / 'comment_benchmark_macro_f1.png')
plt.close(fig)

# 2. Video 3-seed stability across selected models (n = 3,951)
s = pd.read_csv(R / 'results/video/gold_adjudicated_v1.1/METRICS_BY_SEED.csv')
sel = ['text', 'equal_vat', 'visual', 'audio', 'vat_lr', 'healnet_adapted_2024', 'siglip2_vta_probe']
m = s[s.model.isin(sel)]
a = m.groupby('model').macro_f1.agg(['mean', 'std']).reindex(sel)
labels = ['Extracted text', 'Equal VAT', 'Visual', 'Acoustic', 'VAT logistic stacking', 'HEALNet (local)', 'SigLIP2 VTA probe']
fig, ax = plt.subplots(figsize=(9.2, 4.8))
i = np.arange(len(a))
ax.bar(i, a['mean'], yerr=a['std'], capsize=4, color=['#496b86', '#427b7b', '#9fa5ad', '#b6b9bf', '#aa8d6a', '#9b7897', '#6c8e72'])
ax.set_xticks(i, labels, rotation=20, ha='right')
ax.set_ylim(0, 0.69)
ax.set_ylabel('Macro F1')
ax.set_title('Video-window evaluation: means and SD across three training seeds')
ax.grid(axis='y', alpha=0.15)
fig.tight_layout()
fig.savefig(OUT / 'video_three_seed_stability.png')
plt.close(fig)

# 3. Confusion matrices (seed 42, n = 3,951)
p = pd.read_csv(R / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv').query('seed == 42')
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
for ax, key, title in zip(axes, ['text', 'equal_vat'], ['Extracted text', 'Equal VAT']):
    cm = confusion_matrix(p.label_final, p[key + '_prediction'], labels=[0, 1])
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1], ['0', '1'])
    ax.set_yticks([0, 1], ['0', '1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Reference')
    ax.set_title(title)
    for r_i in range(2):
        for c_i in range(2):
            ax.text(c_i, r_i, str(cm[r_i, c_i]), ha='center', va='center', fontsize=11,
                    color='white' if cm[r_i, c_i] > 0.6 * cm.max() else '#222')
fig.suptitle('Seed 42, same 3,951 source-disjoint video windows', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'video_confusion_matrices_seed42.png')
plt.close(fig)

# 4. Focal paired difference interval (seed 42, 214 source groups)
f = pd.read_csv(R / 'results/statistics/FOCAL_FOUR_100000.csv')
row = f[(f.candidate == 'equal_vat') & (f.baseline == 'text')].iloc[0]
diff, lo, hi = float(row.delta_macro_f1), float(row.ci_low), float(row.ci_high)
fig, ax = plt.subplots(figsize=(8.0, 2.9))
ax.errorbar([diff], [0], xerr=[[diff - lo], [hi - diff]], fmt='o', capsize=6, color='#3b6d84', linewidth=2)
ax.axvline(0, ls='--', color='#666')
ax.set_yticks([0], ['Equal VAT − text'])
ax.set_xlim(-0.029, 0.023)
ax.set_ylim(-0.7, 0.7)
ax.set_xlabel('Macro F1 difference, clustered 95% bootstrap CI')
ax.set_title('Paired video comparison (214 source groups, seed 42)')
ax.text(0.02, 0.08, f'Δ = {diff:.5f}; permutation p = {float(row.permutation_p):.4f}', transform=ax.transAxes, fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / 'focal_paired_difference_seed42.png')
plt.close(fig)

# 5. Publication-sized reliability curves from V4 revision bins (without overwriting V4 archive PNGs)
bins = pd.read_csv(R / 'results/statistics/revision_20261108/probability/RELIABILITY_BINS.csv')
label_map = {
    'text': 'Extracted text',
    'equal_vat': 'Equal VAT',
    'vat_lr': 'VAT-LR',
    'sigmoid_equal_vat': 'Sigmoid-calibrated Equal VAT',
    'convex_vat': 'Convex VAT pool',
}
for weighting in ['window', 'source']:
    # 3-model reliability curve
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    for model in ['text', 'equal_vat', 'vat_lr']:
        q = bins[(bins.seed == 42) & (bins.model == model) & (bins.weighting == weighting) & (bins.n > 0)]
        line, = ax.plot(q.mean_probability, q.observed_positive, 'o-', ms=5, label=label_map[model])
        small = q[q['sparse']]
        ax.scatter(small.mean_probability, small.observed_positive, facecolors='none', edgecolors=line.get_color(), s=110, linewidths=1.5)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f'Seed-42 reliability ({weighting}-weighted)')
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, 'Open rings: sparse bins (<20 windows or <5 source groups)', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REV_OUT / f'reliability_seed42_{weighting}_pub.png')
    plt.close(fig)

    # 5-model Table 6 reliability curve
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    for model in ['text', 'equal_vat', 'vat_lr', 'sigmoid_equal_vat', 'convex_vat']:
        q = bins[(bins.seed == 42) & (bins.model == model) & (bins.weighting == weighting) & (bins.n > 0)]
        line, = ax.plot(q.mean_probability, q.observed_positive, 'o-', ms=5, label=label_map[model])
        small = q[q['sparse']]
        ax.scatter(small.mean_probability, small.observed_positive, facecolors='none', edgecolors=line.get_color(), s=110, linewidths=1.5)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f'Table 6 probability controls ({weighting}-weighted, seed 42)')
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, 'Open rings: sparse bins (<20 windows or <5 source groups)', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REV_OUT / f'reliability_table6_{weighting}_pub.png')
    plt.close(fig)

(OUT / 'README.md').write_text(
    '# Figures\n\n'
    'These illustrations are generated deterministically from the frozen **v1.1-GoldAdjudicated** and **V4 revision** '
    'result tables by `python scripts/analysis/make_figures.py`, without refitting any model.\n\n'
    '- `comment_benchmark_macro_f1.png`: Standalone comment benchmark held-out Macro F1 (`n = 1,226`, seed 42).\n'
    '- `video_three_seed_stability.png`: Aligned video-window Macro F1 means and SD across three training seeds (`42, 2026, 2027`).\n'
    '- `video_confusion_matrices_seed42.png`: Paired seed-42 confusion matrices for Extracted Text and Equal VAT (`n = 3,951`).\n'
    '- `focal_paired_difference_seed42.png`: Focal seed-42 paired Macro F1 contrast and 100,000-replicate clustered 95% bootstrap CI (`214` source groups).\n'
    '- `revision_20261108/reliability_*.png`: Publication-formatted (`>10pt`, 300 DPI, borderless) window-weighted and source-weighted reliability curves for seed-42 primary and Table 6 probability-control models. Archived V4 PNGs in `results/statistics/revision_20261108/probability/` remain untouched.\n'
)
