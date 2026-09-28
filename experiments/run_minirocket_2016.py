"""MiniROCKET(2016) baseline — matched control for GLD-A3.

This is a BASELINE REPRODUCTION, not a new architecture.

Protocol matches GLD-A3 exactly:
  - aeon MiniRocket(n_kernels=2016, random_state=42)
  - same data splits (experiments.turs_rrmt.data.load_split)
  - per-block TRAIN-fitted standardization
  - DualRidge with same lambda grid and validation selection
  - seed 42

Usage:
  python experiments/run_minirocket_2016.py --all
  python experiments/run_minirocket_2016.py --dataset CWRU_UNBAL
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_rrmt.data import load_split, DATASETS, SEED
from experiments.turs_rrmt.train_eval import full_metrics, _mf1
from src.diagnostics.calibration import nll as nll_fn, ece as ece_fn
from models.turs_mgb.ridge_readout import DualRidge

RESULTS = os.path.join(ROOT, "results", "minirocket_2016")
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_dataset(ds_tag, device, force=False, log_fn=log):
    ds = load_split(ds_tag)
    n_cls = ds['n_cls']
    y_tr, y_va, y_te = ds['y_train'], ds['y_val'], ds['y_test']
    seq_len = ds['L']
    ds_dir = os.path.join(RESULTS, ds_tag)
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, 'full_results.json')

    # check cache
    if os.path.exists(out_path) and not force:
        try:
            cached = json.load(open(out_path))
            log_fn(f"  [{ds_tag}] cached: MF1={cached['test_mf1']:.4f}")
            return cached
        except Exception:
            pass

    # ── Step 1: MiniRocket feature extraction ──────────────────────
    from aeon.transformations.collection.convolution_based import MiniRocket

    t0 = time.time()
    Xtr = ds['Xtr'][:, 0, :].astype(np.float32)  # [N, T]
    Xva = ds['Xva'][:, 0, :].astype(np.float32)
    Xte = ds['Xte'][:, 0, :].astype(np.float32)

    transformer = MiniRocket(n_kernels=2016, random_state=SEED, n_jobs=-1)
    Ztr = transformer.fit_transform(Xtr)  # [N, 2016]
    Zva = transformer.transform(Xva)
    Zte = transformer.transform(Xte)
    t_extract = time.time() - t0

    log_fn(f"  [{ds_tag}] MiniRocket(2016): Ztr={Ztr.shape}, time={t_extract:.1f}s")

    # sanity check
    assert not np.any(np.isnan(Ztr)), "NaN in training features"
    assert not np.any(np.isnan(Zva)), "NaN in validation features"
    assert not np.any(np.isnan(Zte)), "NaN in test features"
    assert not np.any(np.isinf(Ztr)), "Inf in training features"

    # ── Step 2: Standardization (TRAIN stats only) ─────────────────
    mu = Ztr.mean(0)
    sd = np.clip(Ztr.std(0), 1e-6, None)
    Ztr_s = ((Ztr - mu) / sd).astype(np.float32)
    Zva_s = ((Zva - mu) / sd).astype(np.float32)
    Zte_s = ((Zte - mu) / sd).astype(np.float32)

    # save scaler
    np.savez_compressed(os.path.join(ds_dir, 'scaler.npz'), mu=mu, sd=sd)

    # ── Step 3: Ridge with lambda selection on validation ──────────
    t0 = time.time()
    best = dict(lam=None, val_mf1=-1, val_nll=float('inf'))
    for lam in LAM_GRID:
        rd = DualRidge(n_cls, lam=lam, device=device)
        rd.fit(Ztr_s, y_tr)
        pv = rd.predict_proba(Zva_s)
        m = _mf1(y_va, pv.argmax(1), n_cls)
        nl = float(nll_fn(pv, y_va))
        if (m > best['val_mf1']) or (m == best['val_mf1'] and nl < best['val_nll']):
            best = dict(lam=lam, val_mf1=round(m, 4), val_nll=round(nl, 4))

    # final fit with frozen lambda
    rd = DualRidge(n_cls, lam=best['lam'], device=device)
    rd.fit(Ztr_s, y_tr)
    pt = rd.predict_proba(Zte_s)
    pred = pt.argmax(1)
    t_ridge = time.time() - t0

    test_metrics = full_metrics(y_te, pred, n_cls)
    test_nll = round(float(nll_fn(pt, y_te)), 4)
    test_ece = round(float(ece_fn(pt, y_te)), 4)

    # save predictions
    np.savez_compressed(os.path.join(ds_dir, 'predictions.npz'),
                        probs=pt, pred=pred)
    rd.save(os.path.join(ds_dir, 'ridge.npz'))

    result = dict(
        dataset=ds_tag,
        model='MiniROCKET_2016',
        n_kernels=2016,
        feature_dim=int(Ztr.shape[1]),
        seed=SEED,
        seq_len=seq_len,
        n_cls=n_cls,
        n_train=len(y_tr),
        n_val=len(y_va),
        n_test=len(y_te),
        lam=best['lam'],
        val_mf1=best['val_mf1'],
        val_nll=best['val_nll'],
        test=test_metrics,
        test_nll=test_nll,
        test_ece=test_ece,
        test_mf1=round(test_metrics['macro_f1'], 4),
        test_acc=round(test_metrics['accuracy'], 4),
        test_wf1=round(test_metrics['weighted_f1'], 4),
        elapsed_s=round(t_extract + t_ridge, 2),
        extraction_s=round(t_extract, 2),
        ridge_s=round(t_ridge, 2),
        lambda_grid=LAM_GRID,
    )

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    log_fn(f"  [{ds_tag}] MF1={result['test_mf1']:.4f} Acc={result['test_acc']:.4f} "
           f"lam={best['lam']} dim={result['feature_dim']} ({result['elapsed_s']}s)")
    return result


def run_all(device, force=False, log_fn=log):
    all_results = {}
    for ds in DATASETS:
        log_fn(f"\n=== {ds} ===")
        r = run_dataset(ds, device, force=force, log_fn=log_fn)
        all_results[ds] = r
    return all_results


def write_comparison(all_results):
    """Compare MiniROCKET_2016 vs GLD-A3 (MiniROCKET_2016 + TURS Local)."""
    os.makedirs(RESULTS, exist_ok=True)

    # load GLD-A3 results
    gld_results = {}
    for ds in DATASETS:
        fp = os.path.join(ROOT, 'results', 'turs_gld', ds, 'full_results.json')
        if os.path.exists(fp):
            r = json.load(open(fp))
            a3 = r.get('variants', {}).get('A3', {})
            gld_results[ds] = a3

    # load historical 10K baseline
    hist_10k = {
        'ECG5000_UNBAL': 0.5938, 'ECG5000_BAL': 0.6553,
        'CWRU_UNBAL': 0.9917, 'CWRU_BAL': 0.9947
    }

    rows = []
    for ds in DATASETS:
        mr = all_results.get(ds, {})
        a3 = gld_results.get(ds, {})
        mr_mf1 = mr.get('test_mf1', float('nan'))
        a3_mf1 = a3.get('test', {}).get('macro_f1', float('nan'))
        delta = a3_mf1 - mr_mf1 if np.isfinite(mr_mf1) and np.isfinite(a3_mf1) else float('nan')
        pct = (delta / mr_mf1 * 100) if np.isfinite(mr_mf1) and mr_mf1 > 0 else float('nan')
        rows.append(dict(
            dataset=ds,
            minirocket_2016=round(mr_mf1, 4),
            gld_a3_turs_local=round(a3_mf1, 4),
            delta=round(delta, 4) if np.isfinite(delta) else None,
            relative_pct=round(pct, 2) if np.isfinite(pct) else None,
        ))

    import csv
    with open(os.path.join(RESULTS, 'comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # 10K matched comparison (GLD-A3 uses M=2016; no M=10K + TURS Local exists)
    rows_10k = []
    for ds in DATASETS:
        mr_10k = hist_10k.get(ds, float('nan'))
        a3_mf1 = gld_results.get(ds, {}).get('test', {}).get('macro_f1', float('nan'))
        # GLD-A3 uses M=2016, not M=10K, so direct 10K comparison is not available
        rows_10k.append(dict(
            dataset=ds,
            minirocket_10k=round(mr_10k, 4) if np.isfinite(mr_10k) else None,
            minirocket_10k_plus_turs_local='N/A (GLD-A3 uses M=2016)',
            note='GLD-A3 uses M=2016; no M=10K+TURS Local result exists'
        ))

    with open(os.path.join(RESULTS, 'comparison_10k.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows_10k[0].keys())
        w.writeheader()
        w.writerows(rows_10k)

    return rows, rows_10k


def write_report(all_results, rows, elapsed_s):
    os.makedirs(os.path.join(RESULTS, 'reports'), exist_ok=True)

    a = []
    a.append("# MiniROCKET(2016) Baseline Report\n")
    a.append(f"*Runtime {elapsed_s/60:.1f} min · seed {SEED} · n_kernels=2016*\n")

    a.append("## 1. Implementation\n")
    a.append("- aeon `MiniRocket(n_kernels=2016, random_state=42)`")
    a.append("- Feature dim: 2,016 (PPV only)")
    a.append("- Same data splits as GLD (experiments.turs_rrmt.data.load_split)")
    a.append("- Per-block TRAIN-fitted standardization (z-score)")
    a.append(f"- DualRidge with lambda grid {LAM_GRID}")
    a.append("- Lambda selected on validation (Macro-F1 primary, NLL tie-break)")
    a.append("- Final fit on TRAIN with frozen lambda\n")

    a.append("## 2. Four-Dataset Results\n")
    a.append("| Dataset | MF1 | Acc | WF1 | Lambda | Features | Runtime |")
    a.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        r = all_results.get(ds, {})
        a.append(f"| {ds} | {r.get('test_mf1', 'N/A')} | {r.get('test_acc', 'N/A')} | "
                 f"{r.get('test_wf1', 'N/A')} | {r.get('lam', 'N/A')} | "
                 f"{r.get('feature_dim', 'N/A')} | {r.get('elapsed_s', 'N/A')}s |")
    a.append("")

    a.append("## 3. Comparison with GLD-A3 (MiniROCKET_2016 + TURS Local)\n")
    a.append("| Dataset | MR_2016 | GLD-A3 (+Local) | Δ | Relative |")
    a.append("|---|---|---|---|---|")
    for row in rows:
        d = row['delta'] if row['delta'] is not None else 'N/A'
        p = f"{row['relative_pct']:+.2f}%" if row['relative_pct'] is not None else 'N/A'
        a.append(f"| {row['dataset']} | {row['minirocket_2016']} | "
                 f"{row['gld_a3_turs_local']} | {d} | {p} |")

    deltas = [r['delta'] for r in rows if r['delta'] is not None]
    if deltas:
        mean_d = np.mean(deltas)
        a.append(f"\n**Mean Δ: {mean_d:+.4f}**\n")

    a.append("## 4. 10K Matched Comparison\n")
    a.append("GLD-A3 uses M=2016, not M=10K. No M=10K+TURS Local result exists.\n")
    a.append("| Dataset | MR_10K | MR_10K+Local | Note |")
    a.append("|---|---|---|---|")
    hist_10k = {'ECG5000_UNBAL': 0.5938, 'ECG5000_BAL': 0.6553,
                'CWRU_UNBAL': 0.9917, 'CWRU_BAL': 0.9947}
    for ds in DATASETS:
        a.append(f"| {ds} | {hist_10k.get(ds, 'N/A')} | N/A | GLD-A3 uses M=2016 |")
    a.append("")

    a.append("## 5. Scientific Interpretation\n")
    if deltas:
        mean_d = np.mean(deltas)
        if mean_d > 0.005:
            verdict = "TURS Local provides measurable predictive information beyond MiniROCKET(2016)"
        elif mean_d > -0.005:
            verdict = "TURS Local provides no measurable gain or loss beyond MiniROCKET(2016)"
        else:
            verdict = "TURS Local slightly degrades MiniROCKET(2016) performance"
        a.append(f"**Mean Δ = {mean_d:+.4f}: {verdict}**\n")

    a.append("### Per-dataset interpretation\n")
    for row in rows:
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
    a.append("This baseline answers: does TURS Local add information beyond MiniROCKET(2016)?\n")
    if deltas:
        mean_d = np.mean(deltas)
        pos = sum(1 for d in deltas if d > 0.005)
        neg = sum(1 for d in deltas if d < -0.005)
        neut = len(deltas) - pos - neg
        a.append(f"- Positive Δ on {pos}/{len(deltas)} datasets")
        a.append(f"- Negative Δ on {neg}/{len(deltas)} datasets")
        a.append(f"- Neutral on {neut}/{len(deltas)} datasets")
        a.append(f"- Mean Δ: {mean_d:+.4f}")

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"MiniROCKET(2016) baseline. Device: {device}")

    tags = [args.dataset] if args.dataset else list(DATASETS)

    all_results = {}
    for ds in tags:
        log(f"\n=== {ds} ===")
        r = run_dataset(ds, device, force=args.force)
        all_results[ds] = r

    log("\nPHASE: comparison + report")
    rows, rows_10k = write_comparison(all_results)
    write_report(all_results, rows, time.time() - t0)

    log(f"\nCOMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
