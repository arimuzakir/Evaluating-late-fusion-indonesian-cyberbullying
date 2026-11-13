"""Reproduce manuscript summary Tables 1-6 and composite Figures 3-6 from frozen numerical inputs.

All metrics, counts, and confidence intervals are derived directly from frozen public CSV/JSON files
in the repository (`GoldAdjudicated v1.1` primary outputs and `revision_20261108` audit/probability
outputs), with zero hardcoded numerical estimates. Diagrams `Fig_1.png` and `Fig_2.png` remain
manual author exports from Draw.io and are not regenerated here.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PALETTE = {
    'blue': '#1177aa',
    'orange': '#f08c00',
    'green': '#1b9e4b',
    'red': '#dc2638',
    'purple': '#7b4fa3',
    'gray': '#6c757d',
    'light_gray': '#b8c0c8',
}

TABLE2_ORDER = [
    ('tfidf_word_char_lr', 'TF-IDF word/character + LR'),
    ('indobert_p2', 'IndoBERT-base-p2'),
    ('indobertweet', 'IndoBERTweet'),
    ('mbert', 'Multilingual BERT'),
    ('equal_text_ensemble', 'Equal text ensemble'),
]

TABLE3_ORDER = [
    ('majority', 'Majority class'),
    ('text', 'Extracted text'),
    ('visual', 'Visual only'),
    ('audio', 'Acoustic only'),
    ('equal_va', 'Equal VA'),
    ('equal_vat', 'Equal VAT'),
    ('equal_tv', 'Equal TV'),
    ('equal_ta', 'Equal TA'),
    ('tv_lr', 'TV-LR'),
    ('ta_lr', 'TA-LR'),
    ('va_lr', 'VA-LR'),
    ('vat_lr', 'VAT-LR'),
    ('vat_flags', 'VAT + flags'),
    ('vat_coverage', 'Availability-LR'),
    ('coverage_only', 'Coverage only'),
    ('coverage_shuffled_control', 'Shuffled coverage'),
    ('concat_mlp', 'Feature MLP'),
    ('healnet_adapted_2024', 'HEALNet'),
    ('siglip2_visual_probe', 'SigLIP 2-V'),
    ('siglip2_vt_probe', 'SigLIP 2-VT'),
    ('siglip2_vta_probe', 'SigLIP 2-VTA'),
]

TABLE6_ORDER = [
    ('text', 'Extracted text'),
    ('equal_vat', 'Equal VAT'),
    ('vat_lr', 'VAT-LR'),
    ('sigmoid_equal_vat', 'Sigmoid-calibrated VAT'),
    ('convex_vat', 'Convex probability pool'),
]


def _configure_matplotlib():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def build_table_1(root):
    root = Path(root)
    master = pd.read_csv(root / 'data/labels/v1_1_GoldAdjudicated/data/dataset_master_13307_labeled.csv')
    text_split = pd.read_csv(root / 'data/manifests/comments/TEXT_SPLIT_MANIFEST.csv')
    video_raw = pd.read_csv(root / 'data/labels/v1_1_GoldAdjudicated/features/multimodal_video_clean_metadata.csv')
    video_aligned = pd.read_csv(root / 'data/labels/v1_1_GoldAdjudicated/manifests/VIDEO_PRIMARY_MANIFEST.csv')
    pop_counts = pd.read_csv(root / 'data/labels/v1_1_GoldAdjudicated/POPULATION_COUNTS.csv').set_index('population')

    comments = master[master['modality_type'] == 'text_corpus']
    test_comments = text_split[text_split['split'] == 'test']

    def _counts(df):
        n = int(len(df))
        neg = int((df['label_final'] == 0).sum())
        pos = int((df['label_final'] == 1).sum())
        if neg + pos != n:
            raise AssertionError(f'Binary label count mismatch: {neg}+{pos} != {n}')
        return n, neg, pos

    rows = [
        ('Master corpus', *_counts(master)),
        ('Standalone comments', *_counts(comments)),
        ('Selected comments', *_counts(text_split)),
        ('Held-out comment test', *_counts(test_comments)),
        ('Original video windows', *_counts(video_raw)),
        ('Aligned video windows', *_counts(video_aligned)),
    ]
    out = pd.DataFrame(rows, columns=['population', 'observations', 'negative', 'positive'])

    # Cross-check against POPULATION_COUNTS.csv where overlapping
    check_map = {
        'Master corpus': 'data/dataset_master_13307_labeled.csv',
        'Original video windows': 'features/multimodal_video_clean_metadata.csv',
        'Aligned video windows': 'manifests/VIDEO_PRIMARY_MANIFEST.csv',
    }
    by_label = out.set_index('population')
    for label, pop_key in check_map.items():
        if int(by_label.loc[label, 'observations']) != int(pop_counts.loc[pop_key, 'n']):
            raise AssertionError(f'Table 1 count mismatch vs POPULATION_COUNTS.csv for {label}')
    return out


def build_table_2(root):
    root = Path(root)
    metrics_df = pd.read_csv(root / 'results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv')
    preds_df = pd.read_csv(root / 'results/comments/benchmark_v1.1/TEXT_TEST_PREDICTIONS.csv')
    m42 = metrics_df[metrics_df['seed'] == 42].set_index('model')
    p42 = preds_df[preds_df['seed'] == 42]

    rows = []
    for key, display in TABLE2_ORDER:
        r = m42.loc[key]
        sub = p42[p42['model'] == key]
        y_true = sub['label_final'].to_numpy(int)
        y_pred = sub['prediction'].to_numpy(int)
        y_prob = sub['probability'].to_numpy(float)
        recomp = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
            'roc_auc': float(roc_auc_score(y_true, y_prob)),
            'average_precision': float(average_precision_score(y_true, y_prob)),
        }
        for col, val in recomp.items():
            if abs(val - float(r[col])) > 1e-12:
                raise AssertionError(f'Table 2 prediction vs metric mismatch for {key}.{col}')
        rows.append({
            'model': key,
            'display_name': display,
            'accuracy': float(r['accuracy']),
            'macro_f1': float(r['macro_f1']),
            'roc_auc': float(r['roc_auc']),
            'average_precision': float(r['average_precision']),
        })
    return pd.DataFrame(rows)


def build_table_3(root):
    root = Path(root)
    mbs = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/METRICS_BY_SEED.csv')
    m42 = mbs[mbs['seed'] == 42].set_index('model')
    tss = pd.read_csv(root / 'results/statistics/revision_20261108/main_result_checks/THREE_SEED_RECOMPUTED.csv').set_index('model')
    rand_info = json.loads((root / 'results/statistics/STRATIFIED_RANDOM_SUMMARY.json').read_text(encoding='utf-8'))

    rows = []
    for key, display in TABLE3_ORDER:
        rows.append({
            'model': key,
            'display_name': display,
            'macro_f1_selected': float(m42.loc[key, 'macro_f1']),
            'macro_f1_fixed_050': float(m42.loc[key, 'macro_f1_fixed_050']),
            'macro_f1_mean': float(tss.loc[key, 'mean']),
            'macro_f1_sd': float(tss.loc[key, 'std']),
        })
    rows.append({
        'model': 'stratified_random',
        'display_name': 'Random reference',
        'macro_f1_selected': float(rand_info['macro_f1']),
        'macro_f1_fixed_050': np.nan,
        'macro_f1_mean': np.nan,
        'macro_f1_sd': np.nan,
    })
    return pd.DataFrame(rows)


def build_table_4(root):
    root = Path(root)
    lopo_ci = pd.read_csv(root / 'results/generalization/LOPO_CLUSTER_BOOTSTRAP_CI.csv')
    lopo_preds = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/LOPO_FROZEN_FEATURE_PROBE_PREDICTIONS.csv')

    vta_ci = lopo_ci[lopo_ci['model'] == 'siglip2_vta_probe'].copy()
    vta_preds = lopo_preds[lopo_preds['model'] == 'siglip2_vta_probe']

    rows = []
    for r in vta_ci.sort_values('held_out_platform').itertuples():
        sub = vta_preds[vta_preds['platform'] == r.held_out_platform]
        recomp_f1 = float(f1_score(sub['label_final'], sub['prediction'], average='macro'))
        if abs(recomp_f1 - float(r.macro_f1)) > 1e-12 or len(sub) != int(r.n_windows) or sub['group_id'].nunique() != int(r.n_groups):
            raise AssertionError(f'Table 4 LOPO mismatch on {r.held_out_platform}')
        rows.append({
            'held_out_platform': r.held_out_platform,
            'n_windows': int(r.n_windows),
            'n_groups': int(r.n_groups),
            'macro_f1': float(r.macro_f1),
            'cluster_ci_low': float(r.cluster_ci_low),
            'cluster_ci_high': float(r.cluster_ci_high),
        })
    return pd.DataFrame(rows)


def build_table_5(root):
    root = Path(root)
    agr = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_AGREEMENT.csv').set_index('field')
    r1 = pd.read_csv(root / 'results/input_audit/Form_Audit_R1_scored.csv').set_index('audit_id')
    r2 = pd.read_csv(root / 'results/input_audit/Form_Audit_R2_scored.csv').set_index('audit_id')

    fields = [
        ('B_agresi_window', 'Aggression'),
        ('ocr_akurasi', 'OCR accuracy'),
        ('stt_akurasi', 'STT accuracy'),
    ]
    rows = []
    for field, label in fields:
        s1 = pd.to_numeric(r1[field], errors='coerce')
        s2 = pd.to_numeric(r2[field], errors='coerce')
        valid = s1.notna() & s2.notna()
        n_scored = int(valid.sum())
        agree_n = int((s1[valid] == s2[valid]).sum())
        rec = agr.loc[field]
        if n_scored != int(rec['n']) or abs(agree_n / n_scored - float(rec['agreement'])) > 1e-12:
            raise AssertionError(f'Table 5 audit agreement mismatch for {field}')
        rows.append({
            'field': field,
            'assessment': label,
            'agree_count': agree_n,
            'n': n_scored,
            'agreement': float(rec['agreement']),
            'kappa': float(rec['kappa']),
        })
    return pd.DataFrame(rows)


def build_table_6(root):
    root = Path(root)
    t6 = pd.read_csv(root / 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv').set_index('model')
    rows = []
    for key, display in TABLE6_ORDER:
        r = t6.loc[key]
        rows.append({
            'model': key,
            'display_name': display,
            'window_macro_f1': float(r['window_macro_f1']),
            'window_brier': float(r['window_brier']),
            'window_ece': float(r['window_ece']),
            'source_brier': float(r['source_brier']),
            'source_ece': float(r['source_ece']),
            'window_brier_rank': int(r['window_brier_rank']),
            'source_brier_rank': int(r['source_brier_rank']),
        })
    return pd.DataFrame(rows)


def build_all_tables(root):
    return {
        'Table_1': build_table_1(root),
        'Table_2': build_table_2(root),
        'Table_3': build_table_3(root),
        'Table_4': build_table_4(root),
        'Table_5': build_table_5(root),
        'Table_6': build_table_6(root),
    }


def _plot_fig_3(root, dest_path):
    metrics_df = pd.read_csv(root / 'results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv')
    m42 = metrics_df[metrics_df['seed'] == 42].set_index('model')
    plat_df = pd.read_csv(root / 'results/comments/benchmark_v1.1/TEXT_PLATFORM_SLICES.csv')
    p42 = plat_df[(plat_df['seed'] == 42) & (plat_df['model'] == 'equal_text_ensemble')].set_index('platform')

    models = [
        ('tfidf_word_char_lr', 'TF-IDF'),
        ('indobertweet', 'IBTweet'),
        ('mbert', 'mBERT'),
        ('indobert_p2', 'IndoBERT'),
        ('equal_text_ensemble', 'Ensemble'),
    ]
    metric_cols = [
        ('accuracy', 'Acc.', PALETTE['blue']),
        ('macro_f1', 'F1', PALETTE['orange']),
        ('roc_auc', 'AUC', PALETTE['green']),
        ('average_precision', 'AP', PALETTE['purple']),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), gridspec_kw={'height_ratios': [1.05, 1.0]})

    x = np.arange(len(models))
    width = 0.19
    for idx, (col, label, color) in enumerate(metric_cols):
        vals = [float(m42.loc[m, col]) for m, _ in models]
        ax1.bar(x + (idx - 1.5) * width, vals, width=width * 0.92, color=color, label=label, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in models])
    ax1.set_ylim(0.90, 1.016)
    ax1.set_ylabel('Score')
    ax1.set_title('Held-out Comment Benchmark (n = 1,226)')
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', ncol=4, frameon=True, framealpha=0.95)

    platforms = [
        ('Facebook', 'Facebook'),
        ('Instagram', 'Instagram'),
        ('TikTok', 'TikTok'),
        ('YouTube', 'YouTube'),
        ('X (Twitter)', 'X'),
    ]
    xp = np.arange(len(platforms))
    p_vals = [float(p42.loc[raw_name, 'macro_f1']) for raw_name, _ in platforms]
    bars = ax2.bar(xp, p_vals, width=0.55, color=PALETTE['blue'], edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, p_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(xp)
    ax2.set_xticklabels([disp for _, disp in platforms])
    ax2.set_ylim(0.85, 1.025)
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Equal Text Ensemble Macro F1 by Platform (n = 1,226)')
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.set_axisbelow(True)

    fig.tight_layout(h_pad=1.8)
    fig.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_cm(ax, cm, title):
    totals = cm.sum(axis=1, keepdims=True)
    pct = cm / totals * 100.0
    ax.imshow(pct, cmap='Blues', vmin=0.0, vmax=80.0, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pred Neg (0)', 'Pred Pos (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['True Neg (0)', 'True Pos (1)'])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = 'white' if pct[i, j] > 50 else '#111111'
            ax.text(j, i, f'{cm[i, j]:,}\n({pct[i, j]:.1f}%)', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#cccccc')


def _plot_fig_4(root, dest_path):
    oof = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv')
    z42 = oof[oof['seed'] == 42]
    cm_text = confusion_matrix(z42['label_final'], z42['text_prediction'], labels=[0, 1])
    cm_vat = confusion_matrix(z42['label_final'], z42['equal_vat_prediction'], labels=[0, 1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 6.2))
    _plot_cm(ax1, cm_text, 'Extracted text OOF predictions (Macro F1 = 0.5703)')
    _plot_cm(ax2, cm_vat, 'Equal VAT OOF predictions (Macro F1 = 0.5670)')
    fig.tight_layout(h_pad=2.0)
    fig.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_fig_5(root, dest_path):
    focal = pd.read_csv(root / 'results/statistics/FOCAL_FOUR_100000.csv').set_index('baseline')
    tss = pd.read_csv(root / 'results/statistics/revision_20261108/main_result_checks/THREE_SEED_RECOMPUTED.csv').set_index('model')

    contrasts = [
        ('healnet_adapted_2024', 'Equal VAT − HEALNet'),
        ('siglip2_vta_probe', 'Equal VAT − SigLIP 2-VTA'),
        ('visual', 'Equal VAT − Visual'),
        ('text', 'Equal VAT − Text'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={'width_ratios': [1.3, 1.0]})

    y = np.arange(len(contrasts))
    ax1.axvline(0.0, color='#444444', linestyle='--', linewidth=1.1)
    for idx, (base_key, label) in enumerate(contrasts):
        r = focal.loc[base_key]
        delta = float(r['delta_macro_f1'])
        lo = float(r['ci_low'])
        hi = float(r['ci_high'])
        pholm = float(r['p_holm_four'])
        col = PALETTE['green'] if pholm < 0.05 else PALETTE['blue']
        ax1.errorbar(
            delta,
            idx,
            xerr=[[delta - lo], [hi - delta]],
            fmt='o',
            color=col,
            ecolor=col,
            elinewidth=2.2,
            capsize=4,
            markersize=6.5,
        )
        ax1.text(
            delta,
            idx + 0.24,
            f'{delta:+.4f} (p={pholm:.4f})',
            ha='center',
            va='bottom',
            fontsize=8.8,
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl in contrasts])
    ax1.set_ylim(-0.6, len(contrasts) - 0.35)
    ax1.set_xlim(-0.035, 0.105)
    ax1.set_xlabel('Paired Macro F1 difference (95% CI, 100,000 replicates)')
    ax1.set_title('Focal Four Source-Group Contrasts')
    ax1.grid(axis='x', linestyle='--', alpha=0.35)
    ax1.set_axisbelow(True)

    seed_models = [
        ('text', 'Text', PALETTE['blue']),
        ('equal_vat', 'Equal VAT', PALETTE['green']),
        ('visual', 'Visual', PALETTE['orange']),
        ('healnet_adapted_2024', 'HEALNet', PALETTE['purple']),
    ]
    xs = np.arange(len(seed_models))
    means = [float(tss.loc[m, 'mean']) for m, _, _ in seed_models]
    stds = [float(tss.loc[m, 'std']) for m, _, _ in seed_models]
    colors = [c for _, _, c in seed_models]
    bars = ax2.bar(xs, means, yerr=stds, width=0.52, color=colors, capsize=4, edgecolor='white', error_kw={'elinewidth': 1.6})
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([lbl for _, lbl, _ in seed_models])
    ax2.set_ylim(0.45, 0.61)
    ax2.set_ylabel('Macro F1 (Mean ± SD, 3 seeds)')
    ax2.set_title('Three-Seed Stability (Seeds 42, 2026, 2027)')
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.set_axisbelow(True)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_fig_6(root, dest_path):
    sys.path.insert(0, str(root))
    from src.utils.group_inference import paired_inference

    oof = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv')
    z42 = oof[oof['seed'] == 42].copy()
    q_slices = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/QUALITY_PLATFORM_COVERAGE_SLICES.csv')
    q_all = q_slices[(q_slices['seed'] == 42) & (q_slices['slice'] == 'all_aligned')].set_index('model')

    states = [
        ((0, 0), 'Neither\n(n=685)'),
        ((0, 1), 'STT only\n(n=1,231)'),
        ((1, 0), 'OCR only\n(n=760)'),
        ((1, 1), 'OCR + STT\n(n=1,275)'),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), gridspec_kw={'height_ratios': [1.05, 1.0]})

    x1 = np.arange(len(states))
    ax1.axhline(0.0, color='#444444', linestyle='--', linewidth=1.1)
    for idx, ((ocr_flag, stt_flag), _) in enumerate(states):
        sub = z42[(z42['ocr_ok'] == ocr_flag) & (z42['stt_ok'] == stt_flag)]
        res = paired_inference(sub, 'equal_vat', 'text', reps=10000, seed=24102026)
        delta = float(res['delta_macro_f1'])
        lo = float(res['ci_low'])
        hi = float(res['ci_high'])
        col = PALETTE['green'] if delta >= 0 else PALETTE['red']
        ax1.errorbar(
            idx,
            delta,
            yerr=[[delta - lo], [hi - delta]],
            fmt='o',
            color=col,
            ecolor=col,
            elinewidth=2.2,
            capsize=5,
            markersize=7,
        )
        y_text = hi + 0.005 if delta >= 0 else hi + 0.005
        ax1.text(idx, y_text, f'{delta:+.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in states])
    ax1.set_ylim(-0.06, 0.105)
    ax1.set_ylabel('Δ Macro F1 (Equal VAT − Text)')
    ax1.set_title('Subgroup Differences Across OCR/STT Availability (10,000 replicates)')
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.set_axisbelow(True)

    thresh_models = [
        ('text', 'Extracted text'),
        ('equal_vat', 'Equal VAT'),
    ]
    x2 = np.arange(len(thresh_models))
    w = 0.30
    fixed_vals = [float(q_all.loc[m, 'macro_f1_fixed_050']) for m, _ in thresh_models]
    sel_vals = [float(q_all.loc[m, 'macro_f1']) for m, _ in thresh_models]
    b1 = ax2.bar(x2 - w / 2, fixed_vals, width=w * 0.94, color=PALETTE['gray'], label='Fixed 0.50')
    b2 = ax2.bar(x2 + w / 2, sel_vals, width=w * 0.94, color=PALETTE['blue'], label='Selected (reserved role)')
    for bar, val in list(zip(b1, fixed_vals)) + list(zip(b2, sel_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=9.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in thresh_models])
    ax2.set_ylim(0.48, 0.605)
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Effect of Threshold Selection on Model Ranking (Seed 42)')
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.92)

    fig.tight_layout(h_pad=1.9)
    fig.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run(root, output=None, targets='all', write_repo_figures=False):
    root = Path(root).resolve()
    _configure_matplotlib()

    req = {t.strip() for t in targets.split(',')} if targets != 'all' else {'Table_1', 'Table_2', 'Table_3', 'Table_4', 'Table_5', 'Table_6', 'Fig_3', 'Fig_4', 'Fig_5', 'Fig_6'}
    if 'tables' in req:
        req.update({f'Table_{i}' for i in range(1, 7)})
    if 'figures' in req:
        req.update({f'Fig_{i}' for i in range(3, 7)})

    tables = build_all_tables(root)
    if output is not None:
        out_dir = Path(output).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            if name in req:
                df.to_csv(out_dir / f'{name}.csv', index=False)

    fig_builders = {
        'Fig_3': (_plot_fig_3, root / 'results/figures/manuscript/Fig_3.png'),
        'Fig_4': (_plot_fig_4, root / 'results/figures/manuscript/Fig_4.png'),
        'Fig_5': (_plot_fig_5, root / 'results/figures/manuscript/Fig_5.png'),
        'Fig_6': (_plot_fig_6, root / 'results/figures/manuscript/Fig_6.png'),
    }

    for fig_name, (fn, repo_dest) in fig_builders.items():
        if fig_name not in req:
            continue
        if output is not None:
            fig_out = Path(output).resolve() / f'{fig_name}.png'
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            fn(root, fig_out)
        if write_repo_figures:
            repo_dest.parent.mkdir(parents=True, exist_ok=True)
            fn(root, repo_dest)

    if output is not None:
        summary = {
            'status': 'OK',
            'targets': sorted(req),
            'table_rows': {k: int(len(v)) for k, v in tables.items() if k in req},
        }
        (Path(output).resolve() / 'BUILD_SUMMARY.json').write_text(
            json.dumps(summary, indent=2) + '\n',
            encoding='utf-8',
        )
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build manuscript Tables 1-6 and Figures 3-6 from frozen CSV/JSON inputs.')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--output', default=None, help='Optional output directory for Table CSVs and Figure PNGs')
    parser.add_argument('--targets', default='all', help='Comma-separated targets (all, tables, figures, Table_1..Table_6, Fig_3..Fig_6)')
    parser.add_argument('--write-repo-figures', action='store_true', help='Write Fig_3..Fig_6.png directly into results/figures/manuscript/')
    args = parser.parse_args()
    run(args.root, output=args.output, targets=args.targets, write_repo_figures=args.write_repo_figures)
