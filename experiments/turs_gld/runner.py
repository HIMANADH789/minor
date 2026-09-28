"""TURS-GLD master runner.

Usage:
  python -m experiments.turs_gld.runner --all
  python -m experiments.turs_gld.runner --dataset ECG5000_UNBAL
  python -m experiments.turs_gld.runner --variant A7
  python -m experiments.turs_gld.runner --ablation-only    # B1/B2 + A0-A7
  python -m experiments.turs_gld.runner --report-only      # regenerate report
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_rrmt.data import load_split, DATASETS, SEED
from experiments.turs_rrmt.train_eval import full_metrics, _mf1
from src.diagnostics.calibration import nll as nll_fn, ece as ece_fn
from models.turs_mgb.ridge_readout import DualRidge

RESULTS = os.path.join(ROOT, "results", "turs_gld")
TABLE_DIR = os.path.join(RESULTS, "tables")
FIG_DIR = os.path.join(RESULTS, "figures")
REPORT_DIR = os.path.join(RESULTS, "reports")
CACHE_DIR = os.path.join(RESULTS, "cache")

LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

# ── variant definitions ─────────────────────────────────────────────
# blocks: which streams to include
# local_g4_mode: 'whole' = whole-signal G4, 'local' = local-window G4
VARIANTS = {
    # B1/B2 locality test — ONLY difference is G4 computation mode
    'B1': dict(blocks=('global', 'local', 'local_g4'), local_g4_mode='whole'),
    'B2': dict(blocks=('global', 'local', 'local_g4'), local_g4_mode='local'),
    # A0-A7 ablations
    'A0': dict(blocks=('global',)),
    'A1': dict(blocks=('local',)),
    'A2': dict(blocks=('local_g4',), local_g4_mode='local'),
    'A3': dict(blocks=('global', 'local')),
    'A4': dict(blocks=('global', 'local_g4'), local_g4_mode='local'),
    'A5': dict(blocks=('local', 'local_g4'), local_g4_mode='local'),
    'A6': dict(blocks=('global', 'local', 'local_g4'), local_g4_mode='whole'),
    'A7': dict(blocks=('global', 'local', 'local_g4'), local_g4_mode='local'),
}

ABLATION_VARIANTS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
LOCALITY_VARIANTS = ['B1', 'B2']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── feature caching ─────────────────────────────────────────────────
def _cache_path(ds_tag, variant, M_g, M_l):
    d = os.path.join(CACHE_DIR, ds_tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"feats_{variant}_Mg{M_g}_Ml{M_l}.npz")


def _save_feats(path, blocks_dict, y_dict, spec, extra=None):
    save = {}
    for k, v in blocks_dict.items():
        save[f'block_{k}'] = v
    for k, v in y_dict.items():
        save[f'y_{k}'] = v
    save['spec'] = np.array(json.dumps(spec), dtype=object)
    if extra:
        save['extra'] = np.array(json.dumps(extra), dtype=object)
    np.savez_compressed(path, **save)


def _load_feats(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith('block_'):
            out[k[6:]] = z[k]
        elif k.startswith('y_'):
            out[k[2:]] = z[k]
        elif k == 'spec':
            out['spec'] = json.loads(str(z[k]))
        elif k == 'extra':
            out['extra'] = json.loads(str(z[k]))
    return out


# ── per-block standardization ──────────────────────────────────────
class BlockStandardizer:
    """Per-block TRAIN-fitted standardization."""

    def __init__(self, block_names, block_dims):
        self.block_names = list(block_names)
        self.block_dims = block_dims
        self.mu = {}
        self.sd = {}

    def fit(self, blocks_dict):
        """blocks_dict: name -> [N, D] TRAIN features."""
        for name in self.block_names:
            Z = blocks_dict[name]
            self.mu[name] = Z.mean(0).astype(np.float64)
            self.sd[name] = np.clip(Z.std(0), 1e-6, None).astype(np.float64)

    def transform(self, blocks_dict):
        """Standardize each block. Returns dict of standardized blocks."""
        out = {}
        for name in self.block_names:
            Z = blocks_dict[name]
            out[name] = ((Z - self.mu[name]) / self.sd[name]).astype(np.float32)
        return out

    def assemble(self, blocks_dict):
        """Standardize and concatenate all blocks -> [N, total_dim]."""
        std = self.transform(blocks_dict)
        return np.concatenate([std[n] for n in self.block_names], axis=1)

    def state(self):
        return dict(
            block_names=self.block_names,
            block_dims=self.block_dims,
            mu={k: v.tolist() for k, v in self.mu.items()},
            sd={k: v.tolist() for k, v in self.sd.items()})

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.state(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            st = json.load(f)
        obj = cls(st['block_names'], st['block_dims'])
        obj.mu = {k: np.array(v) for k, v in st['mu'].items()}
        obj.sd = {k: np.array(v) for k, v in st['sd'].items()}
        return obj


# ── whole-signal G4 for B1/A6 ──────────────────────────────────────
def extract_whole_g4(extractor, X, batch=256):
    """Compute G4 multi-lag drift features over the WHOLE signal.

    For each lag, compute the full-temporal-resolution lag displacement
    signal d_lag(t) = |drift(t) - drift(t-lag)|, normalized by the
    train reference.  Then compute summary statistics (mean/std/max/min)
    over the full temporal axis.  This is the B1/A6 variant.
    """
    dev = torch.device('cpu')
    outs = []
    builder = extractor.builder
    ref_lag = extractor.transport_flavors.ref_lag.to(dev)
    ref_norm = ref_lag.abs().mean() + 1e-8
    lag_set = extractor.lag_set
    T = X.shape[-1]

    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).float().to(dev)
        views = builder(xb)  # [B, 3, T]
        drift = views[:, 2, :]  # [B, T]
        lag_signals = []
        for lag in lag_set:
            if lag < T:
                shifted = torch.roll(drift, shifts=lag, dims=1)
                lag_disp = (drift - shifted).abs() / ref_norm  # [B, T]
                lag_signals.append(lag_disp)
        if lag_signals:
            # stack lag signals: [B, n_lags, T]
            lag_stack = torch.stack(lag_signals, dim=1)
            # summary stats over time: [B, n_lags * 4]
            Z = _stats_np_torch(lag_stack)
        else:
            Z = np.zeros((xb.shape[0], 4 * len(lag_set)), np.float32)
        outs.append(Z)
    return np.concatenate(outs, 0).astype(np.float32)


def _stats_np_torch(x):
    """[B, D] or [B, D, T] torch -> [B, D*4] numpy."""
    if x.ndim == 3:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else torch.zeros_like(mu)
        mx = x.max(-1).values
        mn = x.min(-1).values
        return torch.stack([mu, sd, mx, mn], -1).reshape(x.shape[0], -1).numpy().astype(np.float32)
    elif x.ndim == 2:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else torch.zeros_like(mu)
        mx = x.max(-1).values
        mn = x.min(-1).values
        return torch.stack([mu, sd, mx, mn], -1).numpy().astype(np.float32)
    raise ValueError


# ── fit one variant ────────────────────────────────────────────────
def fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va, y_te,
                n_cls, block_names, device, lam_grid=LAM_GRID,
                variant='?'):
    """Group-standardize (TRAIN stats), select lambda (VAL), fit dual Ridge (TRAIN).

    feats_tr/va/te: dict of block_name -> [N, D] raw features
    block_names: list of block names to use
    """
    # assemble standardized feature matrices
    scaler = BlockStandardizer(block_names, {n: feats_tr[n].shape[1] for n in block_names})
    scaler.fit({n: feats_tr[n] for n in block_names})
    Ztr = scaler.assemble({n: feats_tr[n] for n in block_names})
    Zva = scaler.assemble({n: feats_va[n] for n in block_names})
    Zte = scaler.assemble({n: feats_te[n] for n in block_names})

    # lambda selection on validation
    best = dict(lam=None, val_mf1=-1, val_nll=float('inf'))
    for lam in lam_grid:
        rd = DualRidge(n_cls, lam=lam, device=device)
        rd.fit(Ztr, y_tr)
        pv = rd.predict_proba(Zva)
        m = _mf1(y_va, pv.argmax(1), n_cls)
        nl = float(nll_fn(pv, y_va))
        if (m > best['val_mf1']) or (m == best['val_mf1'] and nl < best['val_nll']):
            best = dict(lam=lam, val_mf1=round(m, 4), val_nll=round(nl, 4))

    # final fit
    rd = DualRidge(n_cls, lam=best['lam'], device=device)
    rd.fit(Ztr, y_tr)
    pt = rd.predict_proba(Zte)
    pred = pt.argmax(1)

    res = dict(
        variant=variant, blocks=list(block_names), lam=best['lam'],
        val_mf1=best['val_mf1'], val_nll=best['val_nll'],
        feature_dim=int(Ztr.shape[1]),
        block_dims={n: int(feats_tr[n].shape[1]) for n in block_names},
        test=full_metrics(y_te, pred, n_cls),
        test_nll=round(float(nll_fn(pt, y_te)), 4),
        test_ece=round(float(ece_fn(pt, y_te)), 4),
        probs=pt, pred=pred)
    return res, dict(scaler=scaler, Ztr=Ztr, Zva=Zva, Zte=Zte, readout=rd)


# ── per-dataset run ────────────────────────────────────────────────
def run_dataset(ds_tag, device, force=False, variants=None, M_g=2048, M_l=128,
                hyp=None, log_fn=log):
    ds = load_split(ds_tag)
    n_cls = ds['n_cls']
    y_tr, y_va, y_te = ds['y_train'], ds['y_val'], ds['y_test']
    seq_len = ds['L']
    ds_dir = os.path.join(RESULTS, ds_tag)
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, 'full_results.json')

    requested = variants or (ABLATION_VARIANTS + LOCALITY_VARIANTS)

    # check cache
    cached = None
    if os.path.exists(out_path) and not force:
        try:
            cached = json.load(open(out_path))
        except Exception:
            cached = None
        if cached is not None:
            missing = [v for v in requested if v not in cached.get('variants', {})]
            if not missing:
                log_fn(f"  [{ds_tag}] full results cached ({len(cached.get('variants', {}))} variants)")
                return cached
            log_fn(f"  [{ds_tag}] resume: {len(missing)} variants missing")

    # ── extract features for each variant ──────────────────────────
    from models.turs_gld.model import GLDFeatureExtractor, DEFAULT
    all_feats = {}  # variant -> {block: [N, D]}

    for v in requested:
        cfg = VARIANTS[v]
        cp = _cache_path(ds_tag, v, M_g, M_l)
        if os.path.exists(cp) and not force:
            cached_feats = _load_feats(cp)
            all_feats[v] = {k: cached_feats[k] for k in cached_feats if not k.startswith('y_') and not k.startswith('spec')}
            log_fn(f"  [{ds_tag}/{v}] features cached")
            continue

        t0 = time.time()
        blocks = cfg['blocks']
        g4_mode = cfg.get('local_g4_mode', 'local')

        # build extractor for the required blocks
        extract_blocks = list(blocks)
        if 'local_g4' in blocks and g4_mode == 'whole':
            # for whole-signal G4, we still need local bank if 'local' in blocks
            # but G4 is computed separately
            extract_blocks = [b for b in blocks if b != 'local_g4']
            if not extract_blocks:
                extract_blocks = []

        if extract_blocks:
            ex = GLDFeatureExtractor(blocks=tuple(extract_blocks),
                                     M_global=M_g, M_local=M_l, seed=SEED)
            ex.fit_train(ds['Xtr'], seq_len=seq_len)
            feats = {}
            for split, X in [('tr', ds['Xtr']), ('va', ds['Xva']), ('te', ds['Xte'])]:
                result, spec = ex.extract(X)
                feats[split] = result
        else:
            feats = {'tr': {}, 'va': {}, 'te': {}}
            spec = {}

        # whole-signal G4 for B1/A6
        if 'local_g4' in blocks and g4_mode == 'whole':
            if not extract_blocks:
                # need a minimal extractor just for the builder/ref_lag
                ex = GLDFeatureExtractor(blocks=('global',), M_global=M_g, seed=SEED)
                ex.fit_train(ds['Xtr'], seq_len=seq_len)
            for split, X in [('tr', ds['Xtr']), ('va', ds['Xva']), ('te', ds['Xte'])]:
                feats[split]['local_g4'] = extract_whole_g4(ex, X)
            spec['local_g4'] = feats['tr']['local_g4'].shape[1]

        # save cache
        save_d = {k: feats[k] for k in ['tr', 'va', 'te']}
        _save_feats(cp, save_d['tr'], {'tr': y_tr, 'va': y_va, 'te': y_te},
                    spec, extra=dict(blocks=blocks, g4_mode=g4_mode))
        all_feats[v] = {split: feats[split] for split in ['tr', 'va', 'te']}
        elapsed = time.time() - t0
        dims = {n: feats['tr'][n].shape[1] for n in feats['tr']}
        log_fn(f"  [{ds_tag}/{v}] features: {dims}, total={sum(dims.values())}, {elapsed:.1f}s")

    # ── fit variants ──────────────────────────────────────────────
    results = dict(cached) if cached else dict(dataset=ds_tag, M_global=M_g, M_local=M_l,
                                               seq_len=seq_len, n_cls=n_cls)
    results.setdefault('variants', {})
    results.setdefault('timing', {})

    for v in requested:
        if v in results['variants'] and not force:
            continue
        t0 = time.time()
        cfg = VARIANTS[v]
        block_names = list(cfg['blocks'])
        fv = all_feats[v]
        res, ctx = fit_variant(
            fv['tr'], fv['va'], fv['te'],
            y_tr, y_va, y_te, n_cls, block_names, device, variant=v)
        res['elapsed_s'] = round(time.time() - t0, 1)
        # strip large arrays from stored results
        store = {k: v for k, v in res.items() if k not in ('probs', 'pred')}
        results['variants'][v] = store
        results['timing'][v] = res['elapsed_s']
        log_fn(f"  [{ds_tag}/{v}] MF1={res['test']['macro_f1']:.4f} "
               f"(lam={res['lam']}, dim={res['feature_dim']}, {res['elapsed_s']}s)")

    # save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log_fn(f"  [{ds_tag}] saved -> {out_path}")
    return results


# ── cross-dataset summary ──────────────────────────────────────────
def run_all(device, force=False, M_g=2048, M_l=128, log_fn=log):
    all_results = {}
    for ds in DATASETS:
        log_fn(f"\n=== {ds} ===")
        r = run_dataset(ds, device, force=force, M_g=M_g, M_l=M_l, log_fn=log_fn)
        all_results[ds] = r
    return all_results


# ── tables ─────────────────────────────────────────────────────────
def write_tables(all_results, tags):
    os.makedirs(TABLE_DIR, exist_ok=True)
    import csv

    # Table 2: A0-A7 Macro-F1
    rows = []
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        row = dict(dataset=ds)
        for v in ABLATION_VARIANTS + LOCALITY_VARIANTS:
            row[v] = round(r.get(v, {}).get('test', {}).get('macro_f1', float('nan')), 4)
        rows.append(row)
    with open(os.path.join(TABLE_DIR, '02_ablation_mf1.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Table 3: B1 vs B2
    rows = []
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        b1 = r.get('B1', {}).get('test', {}).get('macro_f1', float('nan'))
        b2 = r.get('B2', {}).get('test', {}).get('macro_f1', float('nan'))
        rows.append(dict(dataset=ds, B1_whole=b1, B2_local=b2,
                         delta=round(b2 - b1, 4) if np.isfinite(b1) and np.isfinite(b2) else None))
    with open(os.path.join(TABLE_DIR, '03_locality_test.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Table 5: feature dimensions + cost
    rows = []
    for ds in tags:
        r = all_results.get(ds, {})
        for v in ABLATION_VARIANTS + LOCALITY_VARIANTS:
            rec = r.get('variants', {}).get(v, {})
            rows.append(dict(dataset=ds, variant=v,
                             feature_dim=rec.get('feature_dim'),
                             block_dims=json.dumps(rec.get('block_dims', {})),
                             lam=rec.get('lam'),
                             elapsed_s=rec.get('elapsed_s')))
    with open(os.path.join(TABLE_DIR, '05_dimensions_cost.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    log(f"  tables -> {TABLE_DIR}")


# ── figures ────────────────────────────────────────────────────────
def write_figures(all_results, tags):
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 2: Ablation bar chart
    variants = ABLATION_VARIANTS
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for idx, ds in enumerate(tags):
        ax = axes[idx]
        vals = [all_results.get(ds, {}).get('variants', {}).get(v, {})
                .get('test', {}).get('macro_f1', 0) for v in variants]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#00BCD4', '#9C27B0',
                  '#795548', '#607D8B', '#E91E63']
        ax.bar(range(len(variants)), vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, fontsize=8)
        ax.set_title(ds, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=vals[0], color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        if idx == 0:
            ax.set_ylabel('Test Macro-F1')
    fig.suptitle('TURS-GLD: Ablation (A7 = Global + Local + Local G4)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'gld_ablation.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: B1 vs B2 locality effect
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tags))
    w = 0.35
    b1_vals = [all_results.get(ds, {}).get('variants', {}).get('B1', {})
               .get('test', {}).get('macro_f1', 0) for ds in tags]
    b2_vals = [all_results.get(ds, {}).get('variants', {}).get('B2', {})
               .get('test', {}).get('macro_f1', 0) for ds in tags]
    ax.bar(x - w/2, b1_vals, w, label='B1 (whole-signal G4)', color='#F44336', edgecolor='black')
    ax.bar(x + w/2, b2_vals, w, label='B2 (local-window G4)', color='#4CAF50', edgecolor='black')
    for i, (b1, b2) in enumerate(zip(b1_vals, b2_vals)):
        d = b2 - b1
        ax.annotate(f'{d:+.3f}', xy=(x[i]+w/2, b2), xytext=(x[i]+w/2, b2+0.03),
                    fontsize=9, ha='center', color='green' if d > 0 else 'red', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel('Test Macro-F1')
    ax.set_title('TURS-GLD: Locality Test (B1 whole-signal vs B2 local-window G4)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'gld_locality_test.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    log(f"  figures -> {FIG_DIR}")


# ── report ─────────────────────────────────────────────────────────
def write_report(all_results, tags, elapsed_s):
    os.makedirs(REPORT_DIR, exist_ok=True)
    a = []
    a.append("# TURS-GLD: Global Bank + Local Lag-Drift — Full Report\n")
    a.append(f"*Runtime {elapsed_s/60:.1f} min · seed {SEED} · M_global=2048 · M_local=128*\n")

    # Executive summary
    a.append("## 1. Executive Summary\n")
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        a0 = r.get('A0', {}).get('test', {}).get('macro_f1')
        a7 = r.get('A7', {}).get('test', {}).get('macro_f1')
        a3 = r.get('A3', {}).get('test', {}).get('macro_f1')
        b1 = r.get('B1', {}).get('test', {}).get('macro_f1')
        b2 = r.get('B2', {}).get('test', {}).get('macro_f1')
        if a0 and a7:
            a.append(f"- **{ds}**: A0={a0:.4f} → A3(+local)={a3:.4f} → A7(+localG4)={a7:.4f}")
        if b1 and b2:
            a.append(f"  - Locality: B1(whole)={b1:.4f} → B2(local)={b2:.4f} (Δ={b2-b1:+.4f})")
    a.append("")

    # Locality test
    a.append("## 6. B1 vs B2 Locality Experiment\n")
    a.append("| Dataset | B1 (whole G4) | B2 (local G4) | Δ(B2−B1) |")
    a.append("|---|---|---|---|")
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        b1 = r.get('B1', {}).get('test', {}).get('macro_f1', float('nan'))
        b2 = r.get('B2', {}).get('test', {}).get('macro_f1', float('nan'))
        d = b2 - b1 if np.isfinite(b1) and np.isfinite(b2) else float('nan')
        a.append(f"| {ds} | {b1:.4f} | {b2:.4f} | {d:+.4f} |")
    a.append("")

    # Ablation table
    a.append("## 8. A0-A7 Ablation Results (Test Macro-F1)\n")
    variants = ABLATION_VARIANTS
    a.append("| Dataset | " + " | ".join(variants) + " |")
    a.append("|---|" + "|".join(["---"] * len(variants)) + "|")
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        cells = [f"{r.get(v, {}).get('test', {}).get('macro_f1', float('nan')):.4f}" for v in variants]
        a.append(f"| {ds} | " + " | ".join(cells) + " |")
    a.append("")

    # Global vs Local sequence
    a.append("## 12. Global vs Local Analysis\n")
    a.append("| Dataset | Global | Global+Local | Global+Local+WholeG4 | Global+Local+LocalG4 |")
    a.append("|---|---|---|---|---|")
    for ds in tags:
        r = all_results.get(ds, {}).get('variants', {})
        a0 = r.get('A0', {}).get('test', {}).get('macro_f1', float('nan'))
        a3 = r.get('A3', {}).get('test', {}).get('macro_f1', float('nan'))
        a6 = r.get('A6', {}).get('test', {}).get('macro_f1', float('nan'))
        a7 = r.get('A7', {}).get('test', {}).get('macro_f1', float('nan'))
        a.append(f"| {ds} | {a0:.4f} | {a3:.4f} | {a6:.4f} | {a7:.4f} |")
    a.append("")

    a.append("## 14. Limitations\n")
    a.append("- Single seed (42); multi-seed robustness not assessed.")
    a.append("- Local G4 lag features are aggregated over time; finer temporal diagnostics possible.")
    a.append("- CWRU may be inherently less sensitive to drift-based transport features.\n")

    a.append("## 15. Final Scientific Conclusion\n")
    a.append("See Section 22 (Final Decision Criteria) in the full implementation spec.\n")

    path = os.path.join(REPORT_DIR, 'TURS_GLD_REPORT.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(a) + '\n')
    log(f"  report -> {path}")
    return path


# ── main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', default=True)
    ap.add_argument('--dataset', type=str, default=None)
    ap.add_argument('--variant', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--M-global', type=int, default=2048)
    ap.add_argument('--M-local', type=int, default=128)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"TURS-GLD pipeline. Device: {device}")

    tags = [args.dataset] if args.dataset else list(DATASETS)

    if not args.report_only:
        for ds in tags:
            log(f"\n=== {ds} ===")
            run_dataset(ds, device, force=args.force,
                        variants=[args.variant] if args.variant else None,
                        M_g=args.M_global, M_l=args.M_local, log_fn=log)

    # reload for reporting
    all_results = {}
    for ds in tags:
        fp = os.path.join(RESULTS, ds, 'full_results.json')
        if os.path.exists(fp):
            all_results[ds] = json.load(open(fp))

    log("\nPHASE: tables / figures / report")
    write_tables(all_results, tags)
    write_figures(all_results, tags)
    write_report(all_results, tags, time.time() - t0)

    # save combined
    dump_path = os.path.join(RESULTS, 'all_results.json')
    with open(dump_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"\nCOMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
