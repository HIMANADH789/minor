"""MiniROCKET(10K) + TURS Local — matched comparison experiment.

Answers: "Does TURS Local add predictive value beyond full-capacity MiniROCKET?"

Two models, identical protocol:
  A) MiniROCKET(10K) — standalone
  B) MiniROCKET(10K) + TURS Local — combined

Both use:
  - aeon MiniRocket(n_kernels=10000, random_state=42)
  - same data splits (experiments.turs_rrmt.data.load_split)
  - same RidgeClassifierCV protocol (canonical benchmark)
  - seed 42
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_rrmt.data import load_split, DATASETS, SEED
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                              confusion_matrix, classification_report)

RESULTS = os.path.join(ROOT, "results", "minirocket_10k_turs_local")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_minirocket_10k(X_np, transformer=None, fit=False):
    """aeon MiniRocket(10K) feature extraction."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    X = X_np[:, 0, :].astype(np.float32)  # [N, T]
    if transformer is None:
        transformer = MiniRocket(random_state=SEED, n_jobs=-1)
    if fit:
        Z = transformer.fit_transform(X)
    else:
        Z = transformer.transform(X)
    return Z, transformer


def extract_turs_local(X_np, extractor=None, fit=False):
    """Existing TURS Local feature extraction (FixedPatternBank + LocalActivity)."""
    from models.turs_gld.model import GLDFeatureExtractor
    if extractor is None:
        extractor = GLDFeatureExtractor(blocks=('local',), seed=SEED)
    if fit:
        extractor.fit_train(X_np, seq_len=X_np.shape[-1])
    feats, _ = extractor.extract(X_np)
    return feats['local'], extractor


def run_one_model(ds_tag, Xtr, ytr, Xva, yva, Xte, yte, n_cls,
                  model_name, use_local=False, device=None, save_dir=None):
    """Run one model (standalone or combined) and return results."""
    t0 = time.time()

    # ── Step 1: MiniRocket(10K) features ───────────────────────────
    Zmr_tr, transformer = extract_minirocket_10k(Xtr, fit=True)
    Zmr_va, _ = extract_minirocket_10k(Xva, transformer=transformer)
    Zmr_te, _ = extract_minirocket_10k(Xte, transformer=transformer)
    t_mr = time.time() - t0    # ── Step 2: TURS Local features (if combined) ──────────────────
    t_local = 0
    if use_local:
        t1 = time.time()
        Zlocal_tr, local_ex = extract_turs_local(Xtr, fit=True)
        Zlocal_va, _ = extract_turs_local(Xva, extractor=local_ex)
        Zlocal_te, _ = extract_turs_local(Xte, extractor=local_ex)
        t_local = time.time() - t1

        # per-block standardization (same as GLD-A3)
        mu_mr = Zmr_tr.mean(0); sd_mr = np.clip(Zmr_tr.std(0), 1e-6, None)
        mu_loc = Zlocal_tr.mean(0); sd_loc = np.clip(Zlocal_tr.std(0), 1e-6, None)
        Zmr_tr_s = ((Zmr_tr - mu_mr) / sd_mr).astype(np.float32)
        Zmr_va_s = ((Zmr_va - mu_mr) / sd_mr).astype(np.float32)
        Zmr_te_s = ((Zmr_te - mu_mr) / sd_mr).astype(np.float32)
        Zloc_tr_s = ((Zlocal_tr - mu_loc) / sd_loc).astype(np.float32)
        Zloc_va_s = ((Zlocal_va - mu_loc) / sd_loc).astype(np.float32)
        Zloc_te_s = ((Zlocal_te - mu_loc) / sd_loc).astype(np.float32)

        # concatenate standardized blocks
        Ztr = np.concatenate([Zmr_tr_s, Zloc_tr_s], axis=1)
        Zva = np.concatenate([Zmr_va_s, Zloc_va_s], axis=1)
        Zte = np.concatenate([Zmr_te_s, Zloc_te_s], axis=1)
        # save scaler
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, 'block_scaler.npz'),
                            mu_mr=mu_mr, sd_mr=sd_mr, mu_loc=mu_loc, sd_loc=sd_loc)
    else:
        # standalone: standardize for fair comparison with combined model
        mu_mr = Zmr_tr.mean(0); sd_mr = np.clip(Zmr_tr.std(0), 1e-6, None)
        Ztr = ((Zmr_tr - mu_mr) / sd_mr).astype(np.float32)
        Zva = ((Zmr_va - mu_mr) / sd_mr).astype(np.float32)
        Zte = ((Zmr_te - mu_mr) / sd_mr).astype(np.float32)
        Zlocal_tr = None

    feat_dim = Ztr.shape[1]

    # ── Step 3: Combine train+val (canonical benchmark protocol) ───
    Ztrva = np.concatenate([Ztr, Zva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)

    # ── Step 4: RidgeClassifierCV (canonical protocol) ─────────────
    t1 = time.time()
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Ztrva, ytrva)
    t_ridge = time.time() - t1

    preds = clf.predict(Zte)
    probs = clf.decision_function(Zte)  # logits for reference

    # ── Step 5: Metrics ────────────────────────────────────────────
    mf1 = f1_score(yte, preds, average='macro', zero_division=0)
    wf1 = f1_score(yte, preds, average='weighted', zero_division=0)
    acc = accuracy_score(yte, preds)
    bal_acc = balanced_accuracy_score(yte, preds)
    per_class_f1 = f1_score(yte, preds, average=None, zero_division=0,
                            labels=list(range(n_cls)))
    per_class_prec = []
    per_class_rec = []
    for c in range(n_cls):
        tp = ((preds == c) & (yte == c)).sum()
        fp = ((preds == c) & (yte != c)).sum()
        fn = ((yte == c) & (preds != c)).sum()
        per_class_prec.append(round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4))
        per_class_rec.append(round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4))
    cm = confusion_matrix(yte, preds, labels=list(range(n_cls))).tolist()

    elapsed = time.time() - t0

    result = dict(
        dataset=ds_tag,
        model=model_name,
        use_local=use_local,
        n_kernels=10000,
        mr_feature_dim=int(Zmr_tr.shape[1]),
        local_feature_dim=int(Zlocal_tr.shape[1]) if use_local else 0,
        total_feature_dim=feat_dim,
        selected_alpha=float(clf.alpha_),
        cv_score=float(clf.best_score_) if hasattr(clf, 'best_score_') else None,
        test_mf1=round(mf1, 4),
        test_acc=round(acc, 4),
        test_wf1=round(wf1, 4),
        test_bal_acc=round(bal_acc, 4),
        per_class_f1=[round(float(x), 4) for x in per_class_f1],
        per_class_prec=per_class_prec,
        per_class_rec=per_class_rec,
        confusion_matrix=cm,
        time_total=round(elapsed, 2),
        time_minrocket=round(t_mr, 2),
        time_local=round(t_local, 2),
        time_ridge=round(t_ridge, 2),
    )

    # save predictions
    ds_dir = os.path.join(RESULTS, ds_tag)
    os.makedirs(ds_dir, exist_ok=True)
    np.savez_compressed(os.path.join(ds_dir, f'{model_name}_preds.npz'),
                        preds=preds, probs=probs, y_true=yte)

    return result


def run_dataset(ds_tag, force=False, log_fn=log):
    ds = load_split(ds_tag)
    n_cls = ds['n_cls']
    y_tr, y_va, y_te = ds['y_train'], ds['y_val'], ds['y_test']
    ds_dir = os.path.join(RESULTS, ds_tag)
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, 'full_results.json')

    if os.path.exists(out_path) and not force:
        try:
            cached = json.load(open(out_path))
            if 'mr_10k' in cached and 'mr_10k_local' in cached:
                log_fn(f"  [{ds_tag}] cached")
                return cached
        except Exception:
            pass

    log_fn(f"  [{ds_tag}] Running MR_10K...")
    r_mr = run_one_model(ds_tag, ds['Xtr'], y_tr, ds['Xva'], y_va,
                         ds['Xte'], y_te, n_cls, 'MR_10K', use_local=False,
                         save_dir=ds_dir)

    log_fn(f"  [{ds_tag}] Running MR_10K + TURS Local...")
    r_local = run_one_model(ds_tag, ds['Xtr'], y_tr, ds['Xva'], y_va,
                            ds['Xte'], y_te, n_cls, 'MR_10K_Local', use_local=True,
                            save_dir=ds_dir)

    result = dict(
        dataset=ds_tag,
        mr_10k=r_mr,
        mr_10k_local=r_local,
        delta_mf1=round(r_local['test_mf1'] - r_mr['test_mf1'], 4),
        delta_pct=round((r_local['test_mf1'] - r_mr['test_mf1']) / max(r_mr['test_mf1'], 1e-8) * 100, 2),
    )

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    log_fn(f"  [{ds_tag}] MR_10K: MF1={r_mr['test_mf1']:.4f} | "
           f"MR_10K+Local: MF1={r_local['test_mf1']:.4f} | "
           f"Delta={result['delta_mf1']:+.4f} ({result['delta_pct']:+.2f}%)")
    return result


def write_comparison(all_results):
    os.makedirs(RESULTS, exist_ok=True)

    rows = []
    for ds in DATASETS:
        r = all_results.get(ds, {})
        mr = r.get('mr_10k', {})
        lo = r.get('mr_10k_local', {})
        rows.append(dict(
            dataset=ds,
            minirocket_10k=mr.get('test_mf1'),
            minirocket_10k_plus_local=lo.get('test_mf1'),
            delta=r.get('delta_mf1'),
            delta_pct=r.get('delta_pct'),
            mr_acc=mr.get('test_acc'),
            local_acc=lo.get('test_acc'),
        ))

    # mean
    deltas = [r['delta'] for r in rows if r['delta'] is not None]
    mr_mf1s = [r['minirocket_10k'] for r in rows if r['minirocket_10k'] is not None]
    lo_mf1s = [r['minirocket_10k_plus_local'] for r in rows if r['minirocket_10k_plus_local'] is not None]
    rows.append(dict(
        dataset='MEAN',
        minirocket_10k=round(np.mean(mr_mf1s), 4) if mr_mf1s else None,
        minirocket_10k_plus_local=round(np.mean(lo_mf1s), 4) if lo_mf1s else None,
        delta=round(np.mean(deltas), 4) if deltas else None,
        delta_pct=round(np.mean([r['delta_pct'] for r in rows[:-1] if r['delta_pct'] is not None]), 2) if deltas else None,
        mr_acc=None, local_acc=None,
    ))

    with open(os.path.join(RESULTS, 'comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # master comparison (including 2016 results)
    hist_10k = {'ECG5000_UNBAL': 0.5938, 'ECG5000_BAL': 0.6553,
                'CWRU_UNBAL': 0.9917, 'CWRU_BAL': 0.9947}
    # load 2016 results
    r2016 = {}
    for ds in DATASETS:
        fp = os.path.join(ROOT, 'results', 'minirocket_2016', ds, 'full_results.json')
        if os.path.exists(fp):
            r2016[ds] = json.load(open(fp))
    # load GLD-A3
    gld_a3 = {}
    for ds in DATASETS:
        fp = os.path.join(ROOT, 'results', 'turs_gld', ds, 'full_results.json')
        if os.path.exists(fp):
            r = json.load(open(fp))
            gld_a3[ds] = r.get('variants', {}).get('A3', {}).get('test', {}).get('macro_f1')

    master = []
    for ds in DATASETS:
        r = all_results.get(ds, {})
        mr10 = r.get('mr_10k', {}).get('test_mf1')
        lo10 = r.get('mr_10k_local', {}).get('test_mf1')
        d10 = r.get('delta_mf1')
        mr2 = r2016.get(ds, {}).get('test_mf1')
        lo2 = gld_a3.get(ds)
        d2 = round(lo2 - mr2, 4) if lo2 is not None and mr2 is not None else None
        master.append(dict(
            dataset=ds,
            minirocket_10k=mr10,
            minirocket_10k_plus_local=lo10,
            delta_10k=d10,
            minirocket_2016=mr2,
            minirocket_2016_plus_local=lo2,
            delta_2016=d2,
            kernel_count=10000,
            combined_feature_dim=r.get('mr_10k_local', {}).get('total_feature_dim'),
        ))

    with open(os.path.join(RESULTS, 'master_comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=master[0].keys())
        w.writeheader()
        w.writerows(master)

    return rows


def write_report(all_results, rows, elapsed_s):
    os.makedirs(os.path.join(RESULTS, 'reports'), exist_ok=True)

    a = []
    a.append("# MiniROCKET(10K) + TURS Local — Experiment Report\n")
    a.append(f"*Runtime {elapsed_s/60:.1f} min · seed {SEED} · n_kernels=10000*\n")

    a.append("## 1. Objective\n")
    a.append("Determine whether TURS Local adds predictive value beyond full-capacity MiniROCKET(10K).\n")

    a.append("## 2. Implementation\n")
    a.append("- aeon `MiniRocket(random_state=42)` — default 10,000 kernels, 9,996 features")
    a.append("- TURS Local: FixedPatternBank M=128 + LocalActivity, 1,024 features")
    a.append("- Combined: 9,996 + 1,024 = 11,020 features")
    a.append("- RidgeClassifierCV with `alphas=np.logspace(-4, 4, 20)`")
    a.append("- Train+val combined for final classifier fit (canonical benchmark protocol)")
    a.append("- Per-sample z-normalization\n")

    a.append("## 3. Four-Dataset Results\n")
    a.append("| Dataset | MR_10K | MR_10K+Local | Δ | Δ% |")
    a.append("|---|---|---|---|---|")
    for row in rows:
        if row['dataset'] == 'MEAN':
            a.append(f"| **{row['dataset']}** | **{row['minirocket_10k']}** | "
                     f"**{row['minirocket_10k_plus_local']}** | "
                     f"**{row['delta']}** | **{row['delta_pct']}%** |")
        else:
            d = row['delta'] if row['delta'] is not None else 'N/A'
            p = f"{row['delta_pct']}%" if row['delta_pct'] is not None else 'N/A'
            a.append(f"| {row['dataset']} | {row['minirocket_10k']} | "
                     f"{row['minirocket_10k_plus_local']} | {d} | {p} |")
    a.append("")

    # 10K vs 2K comparison
    a.append("## 4. Capacity Comparison (Δ at 2K vs 10K)\n")
    a.append("| Dataset | Δ at 2016 kernels | Δ at 10K kernels |")
    a.append("|---|---|---|")
    d2k = {'ECG5000_UNBAL': -0.0345, 'ECG5000_BAL': 0.0049,
           'CWRU_UNBAL': 0.0000, 'CWRU_BAL': 0.0000}
    for ds in DATASETS:
        r = all_results.get(ds, {})
        d10 = r.get('delta_mf1', 'N/A')
        a.append(f"| {ds} | {d2k.get(ds, 'N/A')} | {d10} |")
    a.append("")

    # interpretation
    deltas = [r['delta'] for r in rows[:-1] if r['delta'] is not None]
    if deltas:
        mean_d = np.mean(deltas)
        pos = sum(1 for d in deltas if d > 0.005)
        neg = sum(1 for d in deltas if d < -0.005)
        neut = len(deltas) - pos - neg

        a.append("## 5. Scientific Interpretation\n")
        if mean_d > 0.005:
            verdict = "TURS Local provides measurable predictive information beyond MiniROCKET(10K)"
        elif mean_d > -0.005:
            verdict = "TURS Local provides no measurable gain beyond MiniROCKET(10K)"
        else:
            verdict = "TURS Local slightly degrades MiniROCKET(10K) performance"

        a.append(f"**Mean Δ = {mean_d:+.4f}: {verdict}**\n")
        a.append(f"- Positive Δ on {pos}/{len(deltas)} datasets")
        a.append(f"- Negative Δ on {neg}/{len(deltas)} datasets")
        a.append(f"- Neutral on {neut}/{len(deltas)} datasets\n")

        a.append("### Per-dataset\n")
        for row in rows[:-1]:
            d = row['delta']
            if d is None:
                continue
            if d > 0.005:
                interp = "TURS Local helps"
            elif d < -0.005:
                interp = "TURS Local hurts"
            else:
                interp = "No meaningful difference"
            a.append(f"- **{row['dataset']}**: Δ={d:+.4f} → {interp}")

    a.append("\n## 6. Final Conclusion\n")
    if deltas:
        mean_d = np.mean(deltas)
        if mean_d > 0.01:
            final = "STRONGLY BETTER"
        elif mean_d > 0.003:
            final = "MODERATELY BETTER"
        elif mean_d > 0.001:
            final = "MARGINALLY BETTER"
        elif mean_d > -0.001:
            final = "EQUIVALENT"
        else:
            final = "WORSE"
        a.append(f"MiniROCKET(10K) + TURS Local is **{final}** than MiniROCKET(10K).\n")
        a.append(f"Mean Δ = {mean_d:+.4f}\n")

        if mean_d < 0.001:
            a.append("**Does TURS Local retain measurable predictive value when the global representation is full-capacity MiniROCKET?**")
            a.append("\n**NO.** TURS Local does not provide measurable predictive value beyond full-capacity MiniROCKET(10K). "
                     "The local pattern bank adds 1,024 features that do not improve held-out Macro-F1 under an otherwise identical protocol.\n")
            a.append("**Should TURS Local be considered experimentally redundant?**")
            a.append("\nBased on these four datasets with matched capacity and protocol, **yes** — TURS Local is experimentally "
                     "redundant when the global representation has full MiniROCKET capacity.")
        else:
            a.append("**Does TURS Local retain measurable predictive value when the global representation is full-capacity MiniROCKET?**")
            a.append(f"\n**YES.** TURS Local provides a mean Δ of {mean_d:+.4f} beyond full-capacity MiniROCKET(10K).")

    path = os.path.join(RESULTS, 'reports', 'REPORT.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(a) + '\n')
    log(f"  report -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', default=True)
    ap.add_argument('--dataset', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    log("MiniROCKET(10K) + TURS Local experiment")

    tags = [args.dataset] if args.dataset else list(DATASETS)

    all_results = {}
    for ds in tags:
        log(f"\n=== {ds} ===")
        r = run_dataset(ds, force=args.force)
        all_results[ds] = r

    log("\nPHASE: comparison + report")
    rows = write_comparison(all_results)
    write_report(all_results, rows, time.time() - t0)

    log(f"\nCOMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
