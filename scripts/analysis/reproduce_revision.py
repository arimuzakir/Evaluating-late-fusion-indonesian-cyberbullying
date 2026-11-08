"""CPU reproduction of V4 reviewer-revision analyses from public repository inputs.

Reconstructs role counts, decision thresholds, no-text fallback confusion matrices,
retrospective MDE diagnostics, window-weighted and source-weighted Brier/ECE and
paired bootstrap intervals, seed-42 probability controls (Table 6), neural-head
pooled metrics and baseline reconciliation, two-partition source-allocation effects
and seed ledger, and 100-window SRSWOR audit sampling replay, agreement/kappa,
conservative finite-population hypergeometric bounds, model-assisted sensitivity
intervals, and descriptive error-transition tables.

No GPU training, media extraction, Google Drive mounts, or interactive prompts are used.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import norm, hypergeom

ROLES = ['fit', 'early_stop', 'meta_fit', 'calibration', 'test']
GRID = np.round(np.arange(0.1, 0.90001, 0.025), 3)
SEED = 6112026
BOOTSTRAP_REPS = 10000
HEAD_EPOCHS = [15, 30, 45]
HEAD_WEIGHT_DECAYS = [0.001, 0.01]
PARTITION_SEEDS = [1042, 2042]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def decisions(y, p):
    return float(f1_score(y, p, average='macro', labels=[0, 1], zero_division=0))


def threshold(y, p):
    s = np.array([decisions(y, np.asarray(p) >= t) for t in GRID])
    return float(min(GRID[np.isclose(s, s.max(), rtol=0, atol=1e-12)], key=lambda x: (abs(x - 0.5), x)))


def counts(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def f1counts(c):
    tn, fp, fn, tp = np.moveaxis(np.asarray(c, float), -1, 0)
    return 0.5 * (
        np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
        + np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    )


def paired_ci(d, a, b, B=BOOTSTRAP_REPS, seed=SEED):
    keys, inv = np.unique(d.group_id.astype(str), return_inverse=True)
    ca = np.zeros((len(keys), 4))
    cb = ca.copy()
    np.add.at(ca, (inv, 2 * d.label_final.to_numpy(int) + d[a].to_numpy(int)), 1)
    np.add.at(cb, (inv, 2 * d.label_final.to_numpy(int) + d[b].to_numpy(int)), 1)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, B, 256):
        ix = rng.integers(0, len(keys), (min(256, B - start), len(keys)))
        values.extend(f1counts(ca[ix].sum(1)) - f1counts(cb[ix].sum(1)))
    return dict(
        delta=float(f1counts(ca.sum(0)) - f1counts(cb.sum(0))),
        ci_low=float(np.quantile(values, 0.025)),
        ci_high=float(np.quantile(values, 0.975)),
        groups=len(keys),
        replicates=B,
        scope='exploratory conditional on fitted models',
    )


def pseudonymise_df(df):
    out = df.copy()
    for c in ['group_id', 'video_id']:
        if c in out.columns:
            out[c] = out[c].map(lambda x: x if str(x).startswith('src_') else 'src_' + hashlib.sha256(str(x).encode()).hexdigest()[:16])
    return out


def save_csv(out_dir, name, df, pseudonymise=True):
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = pseudonymise_df(df) if pseudonymise else df
    data.to_csv(p, index=False)
    return p


def save_json(out_dir, name, obj):
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return p


def finite_success_bounds(N, n, k, alpha):
    assert all(isinstance(v, (int, np.integer)) for v in [N, n, k])
    assert 0 < n <= N and 0 <= k <= n and 0 < alpha < 1
    left, right = k, N - n + k
    lo, hi = left, right
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(k - 1, N, mid, n) >= alpha / 2:
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = left, right
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(k, N, mid, n) >= alpha / 2:
            lo = mid
        else:
            hi = mid - 1
    return int(lower), int(lo)


def run(root, output_dir, B=BOOTSTRAP_REPS):
    root = Path(root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    rm = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/ROLE_MANIFEST.csv', dtype={'master_id': str, 'group_id': str})

    # 1. Roles, thresholds, Table 3 recomputed, no-text fallback, and MDE
    z42 = oof[oof.seed == 42]
    m = rm.merge(z42[['master_id', 'label_final', 'ocr_ok', 'stt_ok']], on='master_id', validate='many_to_one')
    role_rows = []
    for (fold, role), a in m.groupby(['outer_fold', 'role']):
        t = a[(a.ocr_ok == 1) | (a.stt_ok == 1)]
        role_rows.append(dict(
            fold=int(fold), role=role, windows=len(a), groups=a.group_id.nunique(),
            negative=int((a.label_final == 0).sum()), positive=int(a.label_final.sum()),
            text_windows=len(t), text_groups=t.group_id.nunique(),
            text_positive=int(t.label_final.sum()), text_negative=int((t.label_final == 0).sum()),
        ))
    save_csv(out, 'ROLE_COUNTS.csv', pd.DataFrame(role_rows))

    tcols = [c for c in oof if c.endswith('_threshold')]
    all_thresh = oof.groupby(['seed', 'outer_fold'])[tcols].first().reset_index()
    save_csv(out, 'ALL_THRESHOLDS.csv', all_thresh)

    models_21 = [c[:-12] for c in oof if c.endswith('_probability')]
    perf = []
    for model in models_21:
        for seed, a in oof.groupby('seed'):
            tn, fp, fn, tp = counts(a.label_final, a[model + '_prediction'])
            perf.append(dict(
                model=model, seed=int(seed),
                pooled_macro_f1=decisions(a.label_final, a[model + '_prediction']),
                fixed_05_macro_f1=decisions(a.label_final, a[model + '_probability'] >= 0.5),
                tn=tn, fp=fp, fn=fn, tp=tp,
            ))
    perf_df = pd.DataFrame(perf)
    save_csv(out, 'TABLE3_RECOMPUTED.csv', perf_df)
    save_csv(out, 'THREE_SEED_RECOMPUTED.csv', perf_df.groupby('model').pooled_macro_f1.agg(['mean', 'std']).reset_index())

    fb_rows, fb_dec = [], []
    for seed, q in oof.groupby('seed'):
        missing = q[(q.ocr_ok == 0) & (q.stt_ok == 0)]
        for fold, a in [('pooled', missing)] + list(missing.groupby('outer_fold')):
            for model in models_21:
                tn, fp, fn, tp = counts(a.label_final, a[model + '_prediction'])
                fb_rows.append(dict(
                    seed=int(seed), fold=fold, model=model, n=len(a), groups=a.group_id.nunique(),
                    positive=int(a.label_final.sum()), negative=int((a.label_final == 0).sum()),
                    threshold='fold-specific' if fold == 'pooled' else float(a[model + '_threshold'].iloc[0]),
                    tn=tn, fp=fp, fn=fn, tp=tp, macro_f1=decisions(a.label_final, a[model + '_prediction']),
                ))
            if fold != 'pooled':
                tval = float(a.text_threshold.iloc[0])
                pred_val = int(0.5 >= tval)
                fb_dec.append(dict(
                    seed=int(seed), fold=int(fold), n=len(a), text_probability=0.5, threshold=tval,
                    decision='all_positive' if pred_val else 'all_negative',
                    predicted_positive=int(a.text_prediction.sum()),
                    predicted_negative=int((a.text_prediction == 0).sum()),
                    rule='probability >= threshold; no OCR/STT means zero text feature vector',
                ))
    save_csv(out, 'NO_TEXT_FALLBACK.csv', pd.DataFrame(fb_rows))
    save_csv(out, 'TEXT_FALLBACK_DECISIONS.csv', pd.DataFrame(fb_dec))

    focal = pd.read_csv(root / 'results/statistics/FOCAL_FOUR_100000.csv')
    frow = focal[(focal.candidate == 'equal_vat') & (focal.baseline == 'text')].iloc[0]
    se = float(frow.bootstrap_se)
    mde = float((norm.ppf(0.975) + norm.ppf(0.8)) * se)
    delta_f1 = decisions(z42.label_final, z42.equal_vat_prediction) - decisions(z42.label_final, z42.text_prediction)
    save_json(out, 'ROUNDING_AND_MDE.json', dict(
        f1_text=decisions(z42.label_final, z42.text_prediction),
        f1_equal_vat=decisions(z42.label_final, z42.equal_vat_prediction),
        delta_macro_f1=delta_f1,
        rounding_note='compute paired differences from unrounded scores; pooled OOF is not an unweighted fold mean',
        bootstrap_se=se, target_power=0.8, alpha=0.05, sidedness='two-sided',
        formula='(norm.ppf(1-alpha/2)+norm.ppf(target_power))*bootstrap_se',
        mde_unadjusted=mde, bootstrap_replicates=int(frow.replicates),
        source='ARCHIVED_FOCAL_FOUR_100000.csv: equal_vat versus text',
        scope='retrospective conditional normal-approximation sensitivity diagnostic; not Holm-adjusted, prospective power or equivalence',
    ))

    # 2. Probability analysis (21 models x 3 seeds) + Seed-42 calibration controls (Table 6)
    brier_rows, rel_bins = [], []
    for seed, z in oof.groupby('seed'):
        group_n = z.groupby('group_id').master_id.transform('size').to_numpy()
        for model in models_21:
            p = z[model + '_probability'].to_numpy(float)
            y = z.label_final.to_numpy(int)
            for weighting, w in [('window', np.ones(len(z))), ('source', 1 / group_n)]:
                for j in range(10):
                    mask = (p >= j / 10) & ((p < (j + 1) / 10) if j < 9 else (p <= 1))
                    idx = np.flatnonzero(mask)
                    mass = w[idx].sum()
                    ng = z.iloc[idx].group_id.nunique()
                    rel_bins.append(dict(
                        seed=int(seed), model=model, weighting=weighting, bin=j,
                        left=j / 10, right=(j + 1) / 10, right_inclusive=(j == 9),
                        n=len(idx), groups=ng,
                        mean_probability=float(np.average(p[idx], weights=w[idx])) if mass else np.nan,
                        observed_positive=float(np.average(y[idx], weights=w[idx])) if mass else np.nan,
                        sparse=(len(idx) < 20 or ng < 5),
                    ))
        g = z[['group_id', 'label_final']].copy()
        g['n'] = 1
        for model in models_21:
            g[model] = (z[model + '_probability'] - z.label_final) ** 2
        a = g.groupby('group_id').sum(numeric_only=True)
        rng = np.random.default_rng(SEED)
        for model in models_21:
            diff = (a[model] - a.text).to_numpy()
            n_arr = a.n.to_numpy()
            vwin, vsource = [], []
            for start in range(0, B, 256):
                ix = rng.integers(0, len(a), (min(256, B - start), len(a)))
                vwin.extend(diff[ix].sum(1) / n_arr[ix].sum(1))
                vsource.extend((diff / n_arr)[ix].mean(1))
            for name, point, values in [('window', diff.sum() / n_arr.sum(), vwin), ('source', (diff / n_arr).mean(), vsource)]:
                brier_rows.append(dict(
                    seed=int(seed), model=model, reference='text', weighting=name,
                    brier=float(a[model].sum() / n_arr.sum() if name == 'window' else (a[model] / a.n).mean()),
                    delta=float(point), ci_low=float(np.quantile(values, 0.025)), ci_high=float(np.quantile(values, 0.975)),
                    replicates=B, scope='conditional paired source bootstrap; secondary descriptive family',
                ))
    brier_prior = pd.DataFrame(brier_rows)
    bins_prior = pd.DataFrame(rel_bins)

    # Calibration controls (seed 42)
    raw_ctrl = pd.read_csv(root / 'results/learned_fusion/ROLE_PROBABILITIES_SEED42.csv', dtype={'master_id': str, 'group_id': str})
    ctrl_thresholds = pd.read_csv(root / 'results/learned_fusion/CONTROL_THRESHOLDS.csv')
    orig42 = oof[oof.seed == 42].set_index('master_id')
    z_ctrl = raw_ctrl[raw_ctrl.role == 'test'].set_index('master_id').loc[orig42.index].reset_index()
    controls = ['sigmoid_equal_vat', 'convex_vat']
    models_t6 = ['text', 'equal_vat', 'vat_lr'] + controls
    for model in controls:
        selected = ctrl_thresholds[ctrl_thresholds.model == model].set_index('fold')
        z_ctrl[model + '_threshold'] = z_ctrl.fold_model.map(selected.threshold)
    for model in ['text', 'equal_vat', 'vat_lr']:
        z_ctrl[model + '_threshold'] = orig42[model + '_threshold'].to_numpy()
    cols_ctrl = ['master_id', 'group_id', 'label_final', 'ocr_ok', 'stt_ok', 'seed', 'fold_model'] + [
        m_name + sfx for m_name in models_t6 for sfx in ['_probability', '_prediction', '_threshold']
    ]
    save_csv(out, 'CALIBRATION_CONTROL_TEST_PREDICTIONS.csv', z_ctrl[cols_ctrl])
    save_csv(out, 'CALIBRATION_CONTROL_THRESHOLDS.csv', ctrl_thresholds[ctrl_thresholds.model.isin(controls)].assign(seed=42))

    ctrl_fb = []
    missing_ctrl = z_ctrl[(z_ctrl.ocr_ok == 0) & (z_ctrl.stt_ok == 0)]
    for fold, q in [('pooled', missing_ctrl)] + list(missing_ctrl.groupby('fold_model')):
        for model in controls:
            tn, fp, fn, tp = counts(q.label_final, q[model + '_prediction'])
            ctrl_fb.append(dict(
                seed=42, fold=fold, model=model, n=len(q), groups=q.group_id.nunique(),
                positive=int(q.label_final.sum()), negative=int((q.label_final == 0).sum()),
                threshold='fold-specific' if fold == 'pooled' else float(q[model + '_threshold'].iloc[0]),
                tn=tn, fp=fp, fn=fn, tp=tp, macro_f1=decisions(q.label_final, q[model + '_prediction']),
                scope='Archived control predictions; absent-extraction text branch remains 0.5 before probability combination',
            ))
    save_csv(out, 'CALIBRATION_CONTROL_FALLBACK.csv', pd.DataFrame(ctrl_fb))

    y_ctrl = z_ctrl.label_final.to_numpy(int)
    group_n_ctrl = z_ctrl.groupby('group_id', sort=False).master_id.transform('size').to_numpy()
    losses = pd.DataFrame({'group_id': z_ctrl.group_id, 'n': 1})
    for model in models_t6:
        losses[model] = (z_ctrl[model + '_probability'].to_numpy() - y_ctrl) ** 2
    a_ctrl = losses.groupby('group_id', sort=False).sum(numeric_only=True)
    n_c = a_ctrl.n.to_numpy()
    pairs = [(model, ref) for model in controls for ref in ['text', 'equal_vat']]
    diff_c = np.stack([(a_ctrl[model] - a_ctrl[ref]).to_numpy() for model, ref in pairs], axis=1)
    rng = np.random.default_rng(SEED)
    win_d, src_d = [], []
    for start in range(0, B, 256):
        ix = rng.integers(0, len(a_ctrl), (min(256, B - start), len(a_ctrl)))
        win_d.append(diff_c[ix].sum(axis=1) / n_c[ix].sum(axis=1)[:, None])
        src_d.append((diff_c / n_c[:, None])[ix].mean(axis=1))
    dist_c = {'window': np.concatenate(win_d), 'source': np.concatenate(src_d)}
    ctrl_int_rows = []
    for j, (model, ref) in enumerate(pairs):
        for weighting in ['window', 'source']:
            val_fn = lambda name: float(a_ctrl[name].sum() / n_c.sum() if weighting == 'window' else (a_ctrl[name] / a_ctrl.n).mean())
            pt = val_fn(model) - val_fn(ref)
            lo, hi = np.quantile(dist_c[weighting][:, j], [0.025, 0.975])
            ctrl_int_rows.append(dict(
                seed=42, model=model, reference=ref, weighting=weighting,
                brier=val_fn(model), reference_brier=val_fn(ref), delta=pt,
                ci_low=float(lo), ci_high=float(hi), replicates=B,
                groups=len(a_ctrl), windows=len(z_ctrl), rng_seed=SEED,
                scope='conditional paired source bootstrap; descriptive unadjusted 95% interval; no model refitting',
            ))
    ctrl_intervals = pd.DataFrame(ctrl_int_rows)
    save_csv(out, 'CALIBRATION_CONTROL_BRIER_INTERVALS.csv', ctrl_intervals)
    save_csv(out, 'BRIER_PAIRED_INTERVALS.csv', pd.concat([brier_prior, ctrl_intervals[ctrl_intervals.reference == 'text'][brier_prior.columns]], ignore_index=True))

    t6_bins, t6_metrics = [], []
    for model in models_t6:
        p = z_ctrl[model + '_probability'].to_numpy(float)
        res = dict(seed=42, model=model, window_macro_f1=decisions(y_ctrl, z_ctrl[model + '_prediction']))
        for weighting, w in [('window', np.ones(len(z_ctrl))), ('source', 1 / group_n_ctrl)]:
            ece = 0.0
            for j in range(10):
                mask = (p >= j / 10) & ((p < (j + 1) / 10) if j < 9 else (p <= 1))
                cnt = int(mask.sum())
                grps = z_ctrl.loc[mask, 'group_id'].nunique()
                mass = w[mask].sum()
                pred_m = float(np.average(p[mask], weights=w[mask])) if mass else np.nan
                obs_m = float(np.average(y_ctrl[mask], weights=w[mask])) if mass else np.nan
                if mass:
                    ece += mass / w.sum() * abs(pred_m - obs_m)
                t6_bins.append(dict(
                    seed=42, model=model, weighting=weighting, bin=j,
                    left=j / 10, right=(j + 1) / 10, right_inclusive=(j == 9),
                    n=cnt, groups=grps, mean_probability=pred_m, observed_positive=obs_m,
                    sparse=(cnt < 20 or grps < 5),
                ))
            res[weighting + '_brier'] = float(np.average((p - y_ctrl) ** 2, weights=w))
            res[weighting + '_ece'] = float(ece)
        t6_metrics.append(res)
    t6_metrics_df = pd.DataFrame(t6_metrics)
    for weighting in ['window', 'source']:
        t6_metrics_df[weighting + '_brier_rank'] = t6_metrics_df[weighting + '_brier'].rank(method='min').astype(int)
    save_csv(out, 'TABLE6_PROBABILITY_METRICS.csv', t6_metrics_df)
    t6_bins_df = pd.DataFrame(t6_bins)
    ctrl_bins_df = t6_bins_df[t6_bins_df.model.isin(controls)]
    save_csv(out, 'CALIBRATION_CONTROL_RELIABILITY_BINS.csv', ctrl_bins_df)
    save_csv(out, 'RELIABILITY_BINS.csv', pd.concat([bins_prior, ctrl_bins_df], ignore_index=True))

    # 3. Neural head sensitivity pooled & baseline reconciliation + Repartition effects & seed ledger
    hp = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    hp_rows = []
    for name, z in hp.groupby('model'):
        hp_rows.append(dict(
            model=name,
            pooled_macro_f1=decisions(z.label_final, z.prediction),
            scope='all grid entries reported; no outer-test best model selected',
        ))
    hp_pooled = pd.DataFrame(hp_rows)
    save_csv(out, 'HEAD_SENSITIVITY_POOLED.csv', hp_pooled)

    hp_idx = hp_pooled.set_index('model')
    recon_rows = []
    for old_model, new_model in [('concat_mlp', 'concat_mlp_e30_wd0.01'), ('healnet_adapted_2024', 'healnet_e30_wd0.01')]:
        old_score = decisions(z42.label_final, z42[old_model + '_prediction'])
        new_score = float(hp_idx.loc[new_model, 'pooled_macro_f1'])
        recon_rows.append(dict(
            model=old_model, original_macro_f1=old_score, current_macro_f1=new_score,
            delta=new_score - old_score,
            scope='Separate execution environments; not an exact original replay. Causal attribution of discrepancy unresolved.',
        ))
    save_csv(out, 'HEAD_BASELINE_RECONCILIATION.csv', pd.DataFrame(recon_rows))

    rp = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/REPARTITION_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    rep_effects = []
    for split_seed in PARTITION_SEEDS:
        d_sub = rp[rp.partition_seed == split_seed]
        rep_effects.append(dict(
            partition_seed=split_seed, training_seed=42,
            text_f1=decisions(d_sub.label_final, d_sub.text_prediction),
            equal_vat_f1=decisions(d_sub.label_final, d_sub.equal_vat_prediction),
            **paired_ci(d_sub, 'equal_vat_prediction', 'text_prediction', B=B, seed=SEED),
        ))
    save_csv(out, 'REPARTITION_EFFECTS.csv', pd.DataFrame(rep_effects))

    ledger = []
    for partition in PARTITION_SEEDS:
        rules_path = root / f'results/video/revision_20261108/source_partitions/PARTITION_{partition}_ALLOCATION_RULES.json'
        rules = json.loads(rules_path.read_text())
        for fold, rule in rules.items():
            ledger.append(dict(
                partition_id=partition, search_start_seed=partition,
                effective_outer_seed=rule['outer_seed'], fold=int(fold),
                calibration_seed=rule['calibration']['seed'],
                meta_fit_seed=rule['meta_fit']['seed'],
                early_stop_seed=rule['early_stop']['seed'],
                training_seed=42,
                acceptance='first candidate satisfying window/class/platform constraints; no prediction score used',
            ))
    save_csv(out, 'PARTITION_SEED_LEDGER.csv', pd.DataFrame(ledger))

    # 4. Blinded 100-window audit: SRSWOR sampling replay, kappa agreement, design bounds, model-assisted CIs, error transitions
    r1 = pd.read_csv(root / 'results/input_audit/Form_Audit_R1_scored.csv', dtype={'audit_id': str}).set_index('audit_id')
    r2 = pd.read_csv(root / 'results/input_audit/Form_Audit_R2_scored.csv', dtype={'audit_id': str}).set_index('audit_id').loc[r1.index]
    adj = pd.read_csv(root / 'results/input_audit/Form_Adjudikasi.csv', dtype={'audit_id': str})
    allocation = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_SAMPLING_ALLOCATION.csv')
    allocation['sampling_fraction'] = allocation.n_h / allocation.N_h
    save_csv(out, 'AUDIT_SAMPLING_ALLOCATION.csv', allocation)

    # Reconstruct exact 100-window SRSWOR sample and blind order from seed 26102026 and 26102027
    frame = z42.sort_values('master_id')
    rng_samp = np.random.default_rng(26102026)
    selected = []
    for oc, st, n_cell in [(0, 0, 17), (0, 1, 31), (1, 0, 19), (1, 1, 33)]:
        cell = frame[(frame.ocr_ok == oc) & (frame.stt_ok == st)]
        sizes = cell.groupby('platform', sort=True).size()
        raw_q = sizes / sizes.sum() * n_cell
        quota = np.floor(raw_q).astype(int)
        for p_name in sorted(sizes.index, key=lambda p_name: (-(raw_q[p_name] - quota[p_name]), p_name))[:n_cell - int(quota.sum())]:
            quota[p_name] += 1
        for p_name in sizes.index:
            block = cell[cell.platform == p_name]
            selected.append(block.iloc[rng_samp.permutation(len(block))[:quota[p_name]]])
    chosen = pd.concat(selected)
    chosen = chosen.iloc[np.random.default_rng(26102027).permutation(len(chosen))].copy()
    chosen['audit_id'] = [f'AU{i:03}' for i in range(1, 101)]
    chosen = chosen.merge(allocation[['ocr_ok', 'stt_ok', 'platform', 'N_h', 'n_h']], on=['ocr_ok', 'stt_ok', 'platform'], validate='many_to_one')
    chosen['design_weight'] = chosen['N_h'] / chosen['n_h']
    key = chosen.set_index('audit_id').loc[r1.index]

    agrees = []
    for field, weight in [
        ('B_agresi_window', None), ('ocr_akurasi', 'linear'), ('stt_akurasi', 'linear'),
        ('ocr_relevansi', 'linear'), ('stt_relevansi', 'linear'), ('B_keyframe_event', 'linear')
    ]:
        a_s = pd.to_numeric(r1[field], errors='coerce')
        b_s = pd.to_numeric(r2[field], errors='coerce')
        mask = a_s.notna() & b_s.notna()
        save_csv(out, 'CONTINGENCY_' + field + '.csv', pd.crosstab(a_s[mask], b_s[mask]).rename_axis('rater1').reset_index())
        labels = [0, 1] if weight is None else [0, 1, 2]
        kappa = float(cohen_kappa_score(a_s[mask], b_s[mask], weights=weight, labels=labels)) if len(set(a_s[mask]) | set(b_s[mask])) > 1 else np.nan
        agrees.append(dict(
            field=field, n=int(mask.sum()), agreement=float((a_s[mask] == b_s[mask]).mean()),
            kappa=kappa, rater1_unscored=int(a_s.isna().sum()), rater2_unscored=int(b_s.isna().sum()),
            scope='pre-consensus paired scored ratings; NA/U excluded from kappa, not recoded as zero',
        ))
    save_csv(out, 'AUDIT_AGREEMENT.csv', pd.DataFrame(agrees))

    cf = next(c for c in ['field', 'item_field', 'kolom'] if c in adj.columns)
    cv = next(c for c in ['consensus_value', 'adjudicated_value', 'nilai_konsensus', 'final_value', 'nilai_final', 'consensus'] if c in adj.columns)
    idx_adj = adj.set_index(['audit_id', cf])[cv]
    fields = ['B_agresi_window', 'B_keyframe_event', 'ocr_akurasi', 'stt_akurasi', 'ocr_relevansi', 'stt_relevansi']
    consensus = pd.DataFrame(index=r1.index)
    for f_name in fields:
        vals = []
        for aid in r1.index:
            a_v = pd.to_numeric(r1.loc[aid, f_name], errors='coerce')
            b_v = pd.to_numeric(r2.loc[aid, f_name], errors='coerce')
            if (aid, f_name) in idx_adj:
                v_c = pd.to_numeric(idx_adj.loc[(aid, f_name)], errors='raise')
            elif pd.isna(a_v) and pd.isna(b_v):
                v_c = np.nan
            elif a_v == b_v:
                v_c = a_v
            else:
                raise ValueError(f'Missing adjudication for {aid} {f_name}')
            vals.append(v_c)
        consensus[f_name] = vals

    joined = key[['master_id', 'platform', 'ocr_ok', 'stt_ok', 'N_h', 'n_h', 'design_weight', 'label_final']].join(consensus)
    joined = joined.join(orig42[['group_id', 'text_prediction', 'equal_vat_prediction']], on='master_id', validate='many_to_one')
    joined['stratum'] = joined.platform.astype(str) + '_' + joined.ocr_ok.astype(str) + '_' + joined.stt_ok.astype(str)

    groups_u, inv_u = np.unique(joined.group_id.astype(str), return_inverse=True)
    rng_ma = np.random.default_rng(SEED)
    weights_a = joined.design_weight.to_numpy(float)
    strata_a = joined.stratum.to_numpy()
    ma_rows = []

    B_audit = 2000
    def boot_ratio(value, available, mode):
        values = []
        states = (joined.ocr_ok.astype(str) + '_' + joined.stt_ok.astype(str)).to_numpy()
        cells = strata_a if mode == 'original_20_cells' else states
        for _ in range(B_audit):
            mult = rng_ma.exponential(size=len(groups_u))[inv_u]
            w = weights_a * mult
            for h in np.unique(cells):
                m_h = cells == h
                w[m_h] *= weights_a[m_h].sum() / w[m_h].sum()
            den = w[available].sum()
            if den > 0:
                values.append(np.sum(w[available] * value[available]) / den)
        return np.quantile(values, [0.025, 0.975])

    for branch in ['ocr', 'stt']:
        available = joined[branch + '_ok'].eq(1).to_numpy()
        value = joined[branch + '_akurasi'].eq(2).to_numpy(float)
        for mode in ['original_20_cells', 'availability_4_cells_sensitivity']:
            ci = boot_ratio(value, available, mode)
            ma_rows.append(dict(
                branch=branch, estimand='faithful among available outputs',
                n=int(available.sum()), faithful=int(value[available].sum()),
                weighted_proportion=float(np.average(value[available], weights=weights_a[available])),
                ci_low=float(ci[0]), ci_high=float(ci[1]), replicates=B_audit,
                method='shared source exponential multiplier; calibrate within cells',
                strata=mode,
                limitation='exploratory model-assisted interval; singleton strata not design-identifiable; four-state analysis changes calibration constraints',
            ))
    model_assisted = pd.DataFrame(ma_rows)
    model_assisted['analysis_role'] = 'secondary_exploratory_source_dependence'
    save_csv(out, 'AUDIT_FIDELITY_MODEL_ASSISTED.csv', model_assisted)

    # Conservative hypergeometric design bounds
    alpha = 0.05
    family = sum(int(allocation[b + '_ok'].eq(1).sum()) for b in ['ocr', 'stt'])
    cell_alpha = alpha / family
    d_cells, d_summary = [], []
    for branch in ['ocr', 'stt']:
        strata_sub = allocation[allocation[branch + '_ok'] == 1]
        for h in strata_sub.itertuples():
            mask = joined.ocr_ok.eq(h.ocr_ok) & joined.stt_ok.eq(h.stt_ok) & joined.platform.eq(h.platform)
            q_h = joined[mask]
            N_h, n_h, k_h = int(h.N_h), int(h.n_h), int(q_h[branch + '_akurasi'].eq(2).sum())
            lo, hi = finite_success_bounds(N_h, n_h, k_h, cell_alpha)
            d_cells.append(dict(
                branch=branch, platform=h.platform, ocr_ok=h.ocr_ok, stt_ok=h.stt_ok,
                N_h=N_h, n_h=n_h, faithful=k_h, success_count_low=lo, success_count_high=hi,
                cell_alpha=cell_alpha, singleton=(n_h == 1),
            ))
        a_b = pd.DataFrame(d_cells)
        a_b = a_b[a_b.branch == branch]
        avail = joined[branch + '_ok'].eq(1)
        N_tot = int(a_b.N_h.sum())
        w_b = joined.loc[avail, 'design_weight'].to_numpy()
        pt_b = float(np.average(joined.loc[avail, branch + '_akurasi'].eq(2), weights=w_b))
        d_summary.append(dict(
            branch=branch,
            estimand='faithful among available outputs in the fixed aligned finite population',
            n=int(avail.sum()), faithful=int(joined.loc[avail, branch + '_akurasi'].eq(2).sum()),
            population_available=N_tot, weighted_proportion=pt_b,
            ci_low=float(a_b.success_count_low.sum() / N_tot),
            ci_high=float(a_b.success_count_high.sum() / N_tot),
            replicates=0, confidence_at_least=1 - alpha,
            method='invert hypergeometric tails per stratum; Bonferroni across both branch-stratum outcomes; sum integer success-count bounds',
            strata='original_20_cells', analysis_role='primary_conservative_finite_population',
            limitation='Conditional on reproduced SRSWOR and fixed adjudicated measurements. Arbitrary source-dependent outcome patterns are held fixed; not a confidence interval for new sources or rater error.',
        ))
    design_df = pd.DataFrame(d_summary)
    save_csv(out, 'AUDIT_FIDELITY_DESIGN_CELLS.csv', pd.DataFrame(d_cells))
    save_csv(out, 'AUDIT_FIDELITY_DESIGN_BOUNDS.csv', design_df)
    save_csv(out, 'AUDIT_FIDELITY_INTERVALS.csv', pd.concat([design_df, model_assisted], ignore_index=True))

    joined['corrected'] = (joined.text_prediction != joined.label_final) & (joined.equal_vat_prediction == joined.label_final)
    joined['new_error'] = (joined.text_prediction == joined.label_final) & (joined.equal_vat_prediction != joined.label_final)
    num_cols = [
        'master_id', 'group_id', 'platform', 'ocr_ok', 'stt_ok', 'N_h', 'n_h',
        'design_weight', 'label_final', 'text_prediction', 'equal_vat_prediction',
        'corrected', 'new_error'
    ] + fields
    save_csv(out, 'AUDIT_NUMERIC_ROWS.csv', joined[num_cols].reset_index())

    err_rows = []
    for f_name in ['ocr_akurasi', 'stt_akurasi', 'ocr_relevansi', 'stt_relevansi', 'B_keyframe_event']:
        for val, q in joined.groupby(f_name):
            w_q = q.design_weight.to_numpy()
            delta_q = (q.new_error.astype(float) - q.corrected.astype(float)).to_numpy()
            events = q.corrected | q.new_error
            err_rows.append(dict(
                field=f_name, rating=val, n=len(q), groups=q.group_id.nunique(),
                corrected=int(q.corrected.sum()), new_errors=int(q.new_error.sum()),
                discordant_groups=q.loc[events, 'group_id'].nunique(),
                weighted_error_delta=float(np.average(delta_q, weights=w_q)),
                ci_low=np.nan, ci_high=np.nan,
                ci_status='not_estimated_sparse_transitions_and_unvalidated_design_variance',
                effective_n_kish=float(w_q.sum() ** 2 / np.square(w_q).sum()),
                scope='descriptive audited subgroup only; mechanism unresolved; no significance or causal claim',
            ))
    save_csv(out, 'AUDIT_ERROR_TRANSITIONS.csv', pd.DataFrame(err_rows))

    # 5. Model role rules (23 models)
    rule_rows = []
    for model in models_21:
        fitting, epoch, scaler, detail = 'none', 'none', 'none', 'fixed arithmetic mean of branch probabilities'
        if model == 'majority':
            fitting, detail = 'fit', 'constant window prevalence in fit; decision threshold fixed .5'
        elif model in ['text', 'visual']:
            fitting, epoch = 'fit', 'early_stop'
            detail = 'generic pretrained encoder; best macro F1 at .5, earliest epoch on tie'
            if model == 'text':
                detail += '; rows without OCR and STT excluded from fit/early_stop'
        elif model == 'audio':
            fitting, scaler, detail = 'fit', 'fit', 'waveform16 logistic regression'
        elif model in ['concat_mlp', 'healnet_adapted_2024']:
            fitting, scaler, detail = 'meta_fit', 'meta_fit', 'fixed 30 epochs in original implementation; all sensitivity grid entries retained'
        elif model.startswith('siglip2_'):
            fitting, scaler, detail = 'fit', 'fit', 'frozen generic pretrained features; supervised linear probe only'
        elif model.endswith('_lr') or model in ['vat_flags', 'vat_coverage', 'coverage_only', 'coverage_shuffled_control']:
            fitting, scaler, detail = 'meta_fit', 'meta_fit', 'learned probability/coverage combiner; no outer-test model selection'
        rule_rows.append(dict(
            model=model, supervised_fit_role=fitting, epoch_selection_role=epoch,
            scaler_fit_role=scaler, threshold_selection_role='fixed .5' if model == 'majority' else 'calibration',
            details=detail, role_counts_file='ROLE_COUNTS.csv',
            historical_scope='Fixed dictionaries at execution; their original development chronology remains separately qualified',
        ))
    for model, detail in [
        ('sigmoid_equal_vat', 'Monotone branch sigmoid calibrators fitted on meta_fit; text calibrator uses available-extraction rows only; equal mean after calibration.'),
        ('convex_vat', 'Nonnegative probability weights summing to one fitted on meta_fit by weighted Brier objective; no intercept.')
    ]:
        rule_rows.append(dict(
            model=model, supervised_fit_role='meta_fit', epoch_selection_role='none',
            scaler_fit_role='none', threshold_selection_role='calibration',
            details=detail + ' Archived seed 42 only; fitting not repeated.',
            role_counts_file='ROLE_COUNTS.csv',
            historical_scope='Post hoc control lock 26 October; not full-history preregistration.',
        ))
    save_csv(out, 'MODEL_ROLE_RULES.csv', pd.DataFrame(rule_rows), pseudonymise=False)

    # 6. Comment multiplicity family
    comment_stats = pd.read_csv(root / 'results/comments/benchmark_v1.1/TEXT_PAIRED_GROUP_STATISTICS.csv')
    save_csv(out, 'COMMENT_MULTIPLICITY_FAMILY.csv', comment_stats, pseudonymise=False)

    return {
        'status': 'PASS',
        'output_dir': str(out),
        'bootstrap_replicates': B,
        'rng_seed': SEED,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce V4 reviewer revision tables on CPU.')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--output', required=True, help='Output directory for reproduced tables')
    parser.add_argument('--replicates', type=int, default=BOOTSTRAP_REPS, help='Bootstrap replicates (default 10000)')
    args = parser.parse_args()
    res = run(args.root, args.output, B=args.replicates)
    print(json.dumps(res, indent=2))
