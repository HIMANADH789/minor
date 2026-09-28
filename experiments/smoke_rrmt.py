"""Fast end-to-end smoke of the TURS-RRMT pipeline (sandbox dirs, 2 epochs)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np
import torch

import experiments.run_turs_rrmt_full as RUN
from experiments.turs_rrmt.data import load_split

# sandbox dirs
SB = os.path.join(ROOT, "results", "_smoke_rrmt")
RUN.RESULTS = SB
RUN.CKPT_DIR = os.path.join(SB, "checkpoints")
RUN.TABLE_DIR = os.path.join(SB, "tables")
RUN.FIG_DIR = os.path.join(SB, "figures")
RUN.AUDIT_DIR = os.path.join(SB, "audit")
RUN.REPORT = os.path.join(SB, "TURS_RRMT_REPORT.md")
for d in [SB, RUN.CKPT_DIR, RUN.TABLE_DIR, RUN.FIG_DIR, RUN.AUDIT_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SMOKE: device", device, flush=True)

ds = load_split("ECG5000_UNBAL")
# truncate to 240/120/120
def tr(X, y, n):
    idx = np.linspace(0, len(y) - 1, n).astype(int)
    return X[idx], y[idx]
ds["Xtr"], ds["y_train"] = tr(ds["Xtr"], ds["y_train"], 240)
ds["Xva"], ds["y_val"] = tr(ds["Xva"], ds["y_val"], 120)
ds["Xte"], ds["y_test"] = tr(ds["Xte"], ds["y_test"], 120)
print("SMOKE: shapes", ds["Xtr"].shape, ds["Xva"].shape, ds["Xte"].shape, flush=True)

# 2-epoch training for speed
import experiments.turs_rrmt.train_eval as TE
_orig_train = TE.train_variant
def fast_train(variant, ds, device, log=print, force=False, ckpt_dir=None,
               max_epochs=2, patience=8):
    return _orig_train(variant, ds, device, log=log, force=True,
                       ckpt_dir=RUN.CKPT_DIR, max_epochs=2, patience=8)
TE.train_variant = fast_train
RUN.TE.train_variant = fast_train

print("SMOKE: run_dataset (all variants + interventions + diagnostics)", flush=True)
res, diag = RUN.run_dataset("ECG5000_UNBAL", device, force=True, train_only=False,
                            only_ablation=None, diag_only=False)
v = res["variants"]
print("SMOKE: MF1s:", {k: round(v[k]["test"]["macro_f1"], 4)
                       for k in ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
                       if k in v}, flush=True)
print("SMOKE: A4_vs_A3:", {k: (round(x, 4) if isinstance(x, float) else x)
                           for k, x in res["A4_vs_A3"].items()}, flush=True)
print("SMOKE: diag keys:", sorted(res["diagnostics"].keys()), flush=True)
print("SMOKE: hyp rows:", len(diag["_hyp_rows"]), flush=True)
n_nan = sum(1 for r in diag["_hyp_rows"]
            if isinstance(r.get("estimate"), float) and np.isnan(r["estimate"]))
print(f"SMOKE: NaN estimates: {n_nan}", flush=True)

print("SMOKE: tables + figures + report", flush=True)
RUN.ablation_table({"ECG5000_UNBAL": res})
from experiments.turs_rrmt.plotting import make_figures
make_figures({"ECG5000_UNBAL": res}, RUN.FIG_DIR)
from experiments.turs_rrmt.reporting import write_report
write_report({"ECG5000_UNBAL": res}, [dict(dataset="ECG5000_UNBAL", **r)
                                     for r in diag["_hyp_rows"]],
             RUN.REPORT, elapsed_s=1.0)
print("SMOKE: report exists:", os.path.exists(RUN.REPORT), flush=True)
print("SMOKE PASS", flush=True)
