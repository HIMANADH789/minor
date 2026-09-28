"""Sandboxed smoke for the FRESH full study (benchmark + diagnostics path).

Truncates ECG5000_UNBAL and redirects every output dir to a smoke sandbox so
the real fresh checkpoints / extraction caches are never touched or pre-created.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

from src.diagnostics.turs_stack_final import config as C
from src.diagnostics.turs_stack_final import benchmark as BM
from src.diagnostics.turs_stack_final import diagnostics as DG

SMOKE = os.path.join(ROOT, "results", "_smoke_full_study")
# ---- sandbox every output location BEFORE anything runs ----
C.STUDY_ROOT = SMOKE
C.AUDIT_DIR = os.path.join(SMOKE, "audit")
C.BENCH_DIR = os.path.join(SMOKE, "benchmark")
C.REPLAY_DIR = os.path.join(SMOKE, "replay")
C.EXTRACT_DIR = os.path.join(SMOKE, "extracted")
C.DIAG_DIR = os.path.join(SMOKE, "diagnostics")
C.TABLE_DIR = os.path.join(SMOKE, "tables")
C.FIG_DIR = os.path.join(SMOKE, "figures")
C.CASE_DIR = os.path.join(SMOKE, "cases")
C.CKPT_DIR = os.path.join(SMOKE, "checkpoints")
C.RUBRIC_PATH = os.path.join(SMOKE, "evidence_rubric.json")
C.MASTER_CSV = os.path.join(SMOKE, "master.csv")
for d in [C.AUDIT_DIR, C.BENCH_DIR, C.REPLAY_DIR, C.EXTRACT_DIR, C.DIAG_DIR,
          C.TABLE_DIR, C.FIG_DIR, C.CASE_DIR, C.CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def truncate(ds, n=160):
    def tr(X, y):
        if len(y) <= n:
            return X, y
        # keep class balance: first n samples stratified-ish by taking every kth
        idx = np.linspace(0, len(y) - 1, n).astype(int)
        return X[idx], y[idx]
    ds = dict(ds)
    ds["Xtr"], ds["y_train"] = tr(ds["Xtr"], ds["y_train"])
    ds["Xva"], ds["y_val"] = tr(ds["Xva"], ds["y_val"])
    ds["Xte"], ds["y_test"] = tr(ds["Xte"], ds["y_test"])
    return ds

print("SMOKE: load + truncate", flush=True)
ds = truncate(BM.load_split("ECG5000_UNBAL"))
print(f"  shapes tr/va/te: {ds['Xtr'].shape} {ds['Xva'].shape} {ds['Xte'].shape}", flush=True)

print("SMOKE: fresh canonical training (30 ep on tiny data)", flush=True)
_, meta = BM.train_stack(ds, device, log=lambda m: print(m, flush=True), force=True)
print(f"  trained: params={meta['params']:,} best_val={meta['best_val_soft_mf1']:.4f} "
      f"combiner={meta['selected_combiner']}", flush=True)

print("SMOKE: evaluate fresh + replay gate", flush=True)
result, model, fwd, combos, best_combo = BM.evaluate_fresh("ECG5000_UNBAL", ds, device)
r2, _, _, _, _ = BM.evaluate_fresh("ECG5000_UNBAL", ds, device)
assert abs(r2["final"]["macro_f1"] - result["final"]["macro_f1"]) < 1e-9, "replay mismatch"
print(f"  test MF1={result['final']['macro_f1']:.4f} branches="
      f"{ {k: round(v, 3) for k, v in result['branch_test_mf1'].items()} } replay=OK", flush=True)

print("SMOKE: extraction", flush=True)
ext = DG.extract_all("ECG5000_UNBAL", model, ds, device, force=True)
print(f"  splits={list(ext)} test keys={len(ext['test'])}", flush=True)

print("SMOKE: diagnostics", flush=True)
replay = dict(dataset="ECG5000_UNBAL", all_passed=True)
diag = DG.run_all_diagnostics("ECG5000_UNBAL", model, ds, device, ext, replay)
diag["complexity"] = DG.complexity_analysis("ECG5000_UNBAL", model, ds, device, meta)
from src.diagnostics.turs_stack.case_studies import build_case_studies
diag["case_study_manifest"] = build_case_studies("ECG5000_UNBAL", model, ds, device, ext)
print(f"  hyp rows={len(diag['_hyp_rows'])} "
      f"cx={diag['complexity']['stack_params']:,} params "
      f"({diag['complexity']['params_ratio']:.2f}x lite) cases={len(diag['case_study_manifest'])}",
      flush=True)

# rubric + added-value path (Lite checkpoints exist from the Lite study)
from src.diagnostics.turs_stack_final import reporting as RP
per_nb = {"ECG5000_UNBAL": dict(tag="ECG5000_UNBAL", benchmark=result, diag={
    k: v for k, v in diag.items() if k != "_hyp_rows"}, replay=replay,
    extracted=dict(test=dict(final_pred=ext["test"]["final_pred"],
                             final_correct=ext["test"]["final_correct"])))}
added = RP.added_value_analysis(per_nb, {t: r["_hyp_rows"] for t, r in per_nb.items()})
rubric = RP.build_rubric(per_nb, added)
n_dims = len(rubric["dimensions"]["ECG5000_UNBAL"])
grades = [d["grade"] for d in rubric["dimensions"]["ECG5000_UNBAL"].values()]
print(f"  rubric dims={n_dims} grades={grades}", flush=True)
print(f"  added_value: {added['ECG5000_UNBAL'].get('available')} "
      f"mf1_delta={added['ECG5000_UNBAL'].get('mf1_delta')}", flush=True)

print("SMOKE PASS", flush=True)
