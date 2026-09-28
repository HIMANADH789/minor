"""Haptics final inferential / scientific significance analysis.

ANALYSIS ONLY. Uses exclusively stored artifacts:
  * predictions: external_stack npy (M0 + 4 baselines, seed 42;
    baselines 43/44 from results/haptics_baselines_3seed),
    R5(50/50) = rcmkn_r2_haptics_3seed seed{42,43,44} predictions.csv
    (seed42 = canonical recovered 0.5500), plus G/H banks + regime codes
    rebuilt from the frozen seed-42 context checkpoint (identity-verified,
    0.5500 / alpha 4.2813).
  * scalar scores: multiseed_haptics.json (adaptive-rho R5 per seed),
    rcmkn_r2_haptics_3seed (R2 per seed), baseline study per_run.

No training, no new architectures, no extra seeds, no test tuning.
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy import stats as sps
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "haptics_inference")
FIG = os.path.join(OUT, "figures")
CACHE = r"C:/temp/results"
SEEDS = [42, 43, 44]
B_BOOT = 10_000
B_PERM = 20_000
BOOT_SEED = 42042
CLASSES = [0, 1, 2, 3, 4]

BASE_MODELS = ["InceptionTime", "FCN", "ResNet1D", "PatchTST"]
BASE_NPY = {"InceptionTime": "Haptics_InceptionTime.npy",
            "FCN": "Haptics_FCN.npy", "ResNet1D": "Haptics_ResNet-1D.npy",
            "PatchTST": "Haptics_PatchTST.npy"}
EXT = os.path.join(ROOT, "results", "external_stack_generalization")
BASE3 = os.path.join(ROOT, "results", "haptics_baselines_3seed")
R3S = os.path.join(ROOT, "results", "rcmkn_r2_haptics_3seed")
CAN = os.path.join(ROOT, "results", "rcmkn_haptics_seed42")


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0,
                    labels=CLASSES)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_all():
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    y_trva = np.concatenate([d["ytr"], d["yva"]]).astype(int)
    y_te = d["yte"].astype(int)

    preds = {}          # (model, seed) -> (308,) int preds or None
    probs_available = set()
    preds[("MiniROCKET", 42)] = np.load(
        os.path.join(EXT, "predictions", "Haptics_MiniROCKET.npy"))
    for s in SEEDS:
        preds[("MiniROCKET", s)] = preds[("MiniROCKET", 42)]
    for m in BASE_MODELS:
        preds[(m, 42)] = np.load(os.path.join(EXT, "predictions",
                                              BASE_NPY[m]))
        for s in (43, 44):
            p = os.path.join(BASE3, m.lower(), f"seed{s}", "predictions.npy")
            preds[(m, s)] = np.load(p) if os.path.exists(p) else None
    # R5 (50/50): seed42 recovered exactly (0.5500); seeds 43/44 = the
    # rcmkn_r2_haptics_3seed predictions (the same fixed 50/50 model)
    r542 = np.load(os.path.join(CACHE, "haptics_inference_R5_s42_pred_te.npy"))
    preds[("R5_5050", 42)] = r542
    for s in (43, 44):
        with open(os.path.join(R3S, f"seed{s}", "predictions.csv")) as f:
            rows = list(csv.DictReader(f))
        preds[("R5_5050", s)] = np.array([int(r["R2_pred"]) for r in rows])
    # adaptive-rho R5 predictions: MISSING (no checkpoints) -> scalar only

    banks = {k: np.load(os.path.join(
        CACHE, f"haptics_inference_banks_{k}.npy"))
        for k in ("G_trva", "H_trva", "G_te", "H_te")}
    regimes = {k: np.load(os.path.join(
        CACHE, f"haptics_inference_banks_{k}.npy"))
        for k in ("reg_trva", "reg_te")}
    per_seed_diag = {s: json.load(open(os.path.join(R3S, f"seed{s}",
                                                    "result.json")))
                     for s in SEEDS}
    multiseed = json.load(open(
        r"C:/temp/results/final_validation/multiseed_haptics.json"))
    return y_trva, y_te, preds, banks, regimes, per_seed_diag, multiseed


# ---------------------------------------------------------------------------
# 3-4. bootstrap + paired deltas + permutation
# ---------------------------------------------------------------------------
def boot_macro_f1(y, p, rng, b=B_BOOT):
    n = len(y)
    vals = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        vals[i] = macro_f1(y[idx], p[idx])
    return vals


def paired_boot_delta(y, pa, pb, rng, b=B_BOOT):
    n = len(y)
    deltas = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        deltas[i] = macro_f1(y[idx], pa[idx]) - macro_f1(y[idx], pb[idx])
    return deltas


def permutation_delta(y, pa, pb, rng, b=B_PERM):
    """Exchangeability: swap predictions per sample, two-sided."""
    obs = macro_f1(y, pa) - macro_f1(y, pb)
    n = len(y)
    cnt = 0
    for _ in range(b):
        sw = rng.random(n) < 0.5
        qa = np.where(sw, pb, pa)
        qb = np.where(sw, pa, pb)
        d = macro_f1(y, qa) - macro_f1(y, qb)
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (b + 1)


def csv_rows(path):
    """Resume helper: reload rows from an already-written output CSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    os.makedirs(FIG, exist_ok=True)
    y_trva, y_te, preds, banks, regimes, per_seed_diag, multiseed = \
        load_all()
    n_te = len(y_te)
    inv = {"predictions": {}, "banks": {}, "regimes": {}, "scalar": {}}
    for (m, s), p in sorted(preds.items()):
        inv["predictions"][f"{m}_seed{s}"] = "AVAILABLE" if p is not None \
            else "MISSING"
    for k in banks:
        inv["banks"][k] = "AVAILABLE(rebuilt_frozen_ckpt_identity_verified)"
    inv["regimes"] = {"reg_trva": "AVAILABLE", "reg_te": "AVAILABLE"}
    inv["scalar"] = {
        "R5_adaptive_scores_seeds_42_43_44": "AVAILABLE(multiseed_haptics)",
        "R5_adaptive_predictions_43_44": "MISSING(no checkpoints; "
                                         "not re-derived per no-training rule)",
        "neural_logits_probabilities": "MISSING(not stored)",
        "M0_R5_ridge_decision_values": "DERIVABLE(frozen banks + canonical "
                                       "Ridge refit, deterministic)",
        "G_only_H_only_diagnostics": "DERIVABLE(section 19)",
    }
    with open(os.path.join(OUT, "artifact_inventory.json"), "w") as f:
        json.dump(inv, f, indent=2)

    rng = np.random.default_rng(BOOT_SEED)

    # ---------------- 3. bootstrap Macro-F1 per model x seed -------------
    boot_csv = os.path.join(OUT, "bootstrap_macro_f1.csv")
    if os.path.exists(boot_csv):
        rows = csv_rows(boot_csv)
        log("resume: bootstrap_macro_f1 cached")
    else:
        rows = []
    boot_cache = {}
    for (m, s) in ([] if os.path.exists(boot_csv) else
                   sorted(preds.keys(), key=lambda t: (t[0], t[1]))):
        p = preds[(m, s)]
        if p is None:
            continue
        v = boot_macro_f1(y_te, p, rng)
        boot_cache[(m, s)] = v
        obs = macro_f1(y_te, p)
        rows.append({
            "model": m, "seed": s, "observed_macro_f1": round(obs, 4),
            "boot_mean": round(float(v.mean()), 4),
            "boot_sd": round(float(v.std(ddof=1)), 4),
            "ci_low": round(float(np.percentile(v, 2.5)), 4),
            "ci_high": round(float(np.percentile(v, 97.5)), 4)})
    with open(os.path.join(OUT, "bootstrap_macro_f1.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- 4-5. paired deltas + permutation (seed 42 primary;
    #                        R5_5050 across seeds) ----------------------
    comp_defs = []
    for s in SEEDS:
        comp_defs.append((f"R5_5050-MiniROCKET", "R5_5050", "MiniROCKET", s))
    for s in SEEDS:
        for m in BASE_MODELS:
            comp_defs.append((f"R5_5050-{m}", "R5_5050", m, s))
    for s in SEEDS:
        for m in BASE_MODELS:
            comp_defs.append((f"{m}-MiniROCKET", m, "MiniROCKET", s))

    pbd_csv = os.path.join(OUT, "paired_bootstrap_deltas.csv")
    pt_csv = os.path.join(OUT, "permutation_tests.csv")
    if os.path.exists(pbd_csv) and os.path.exists(pt_csv):
        delta_rows = csv_rows(pbd_csv)
        perm_rows = csv_rows(pt_csv)
        for r in perm_rows:
            r["holm_p"] = r.get("holm_p", "")
        log("resume: paired bootstrap + permutation cached")
    else:
        delta_rows = []
        perm_rows = []
    for name, ma, mb, s in ([] if delta_rows else comp_defs):
        pa, pb = preds.get((ma, s)), preds.get((mb, s))
        if pa is None or pb is None:
            continue
        rng_a = np.random.default_rng(BOOT_SEED + s)
        d = paired_boot_delta(y_te, pa, pb, rng_a)
        obs = macro_f1(y_te, pa) - macro_f1(y_te, pb)
        delta_rows.append({
            "comparison": name, "seed": s,
            "observed_delta": round(obs, 4),
            "boot_mean_delta": round(float(d.mean()), 4),
            "ci_low": round(float(np.percentile(d, 2.5)), 4),
            "ci_high": round(float(np.percentile(d, 97.5)), 4),
            "P_delta_gt_0": round(float((d > 0).mean()), 4),
            "P_delta_lt_0": round(float((d < 0).mean()), 4)})
        if s == 42 or name.startswith("R5_5050-MiniROCKET"):
            rng_p = np.random.default_rng(BOOT_SEED + 777 + s)
            obs_p, pv = permutation_delta(y_te, pa, pb, rng_p)
            perm_rows.append({"comparison": name, "seed": s,
                              "observed_delta": round(obs_p, 4),
                              "perm_p_two_sided": round(pv, 5),
                              "n_pairs": n_te,
                              "swaps": B_PERM})
    # Holm within the seed-42 planned family (5 primary + 4 R5-vs-baseline
    # + 4 baseline-vs-M0) and within the R5_5050-M0 across-seed family
    for fam_seed in (42,):
        idx = [i for i, r in enumerate(perm_rows) if r["seed"] == fam_seed]
        ps = [perm_rows[i]["perm_p_two_sided"] for i in idx]
        adj = holm(ps)
        for i, a in zip(idx, adj):
            perm_rows[i]["holm_p"] = round(float(a), 5)
    for r in perm_rows:
        r.setdefault("holm_p", "")
    with open(os.path.join(OUT, "paired_bootstrap_deltas.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        w.writeheader()
        w.writerows(delta_rows)
    with open(os.path.join(OUT, "permutation_tests.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "seed",
                                          "observed_delta",
                                          "perm_p_two_sided", "holm_p",
                                          "n_pairs", "swaps"])
        w.writeheader()
        w.writerows(perm_rows)
    log(f"bootstrap+permutation done ({len(delta_rows)} deltas, "
        f"{len(perm_rows)} perm tests)")

    # ---------------- 6. per-class metrics + bootstrap -------------------
    pcm_csv = os.path.join(OUT, "per_class_metrics.csv")
    pcb_csv = os.path.join(OUT, "per_class_bootstrap.csv")
    if os.path.exists(pcm_csv) and os.path.exists(pcb_csv):
        pc_rows = csv_rows(pcm_csv)
        pcb_rows = csv_rows(pcb_csv)
        log("resume: per-class cached")
    else:
        pc_rows, pcb_rows = [], []
    rng_pc = np.random.default_rng(BOOT_SEED + 5)
    for (m, s) in ([] if pc_rows else
                   sorted(preds.keys(), key=lambda t: (t[0], t[1]))):
        p = preds[(m, s)]
        if p is None:
            continue
        cm = confusion_matrix(y_te, p, labels=CLASSES)
        for c in CLASSES:
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1c = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            pc_rows.append({"model": m, "seed": s, "class": c,
                            "precision": round(prec, 4),
                            "recall": round(rec, 4), "f1": round(f1c, 4),
                            "support": int(cm[c, :].sum())})
            vals = []
            for _ in range(2000):
                idx = rng_pc.integers(0, n_te, n_te)
                vals.append(macro_f1(
                    np.full(n_te, c), np.where(y_te[idx] == c,
                                               p[idx] == c, -1) * 0
                    + np.where(p[idx] == c, c, -1)) if False else
                    f1_score(y_te[idx] == c, p[idx] == c, zero_division=0))
            pcb_rows.append({"model": m, "seed": s, "class": c,
                             "f1_boot_mean": round(float(np.mean(vals)), 4),
                             "ci_low": round(float(np.percentile(vals, 2.5)), 4),
                             "ci_high": round(float(np.percentile(vals, 97.5)), 4)})
    with open(os.path.join(OUT, "per_class_metrics.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader()
        w.writerows(pc_rows)
    with open(os.path.join(OUT, "per_class_bootstrap.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pcb_rows[0].keys()))
        w.writeheader()
        w.writerows(pcb_rows)

    # ---------------- 7. error overlap + McNemar ------------------------
    def correct_vec(m, s):
        return (preds[(m, s)] == y_te).astype(int)

    eo_rows, mcn_rows = [], []
    for s in SEEDS:
        pairs = [("MiniROCKET", "R5_5050")] + [
            ("R5_5050", m) for m in BASE_MODELS]
        for ma, mb in pairs:
            ea, eb = 1 - correct_vec(ma, s), 1 - correct_vec(mb, s)
            both_c = int(((ea == 0) & (eb == 0)).sum())
            both_w = int(((ea == 1) & (eb == 1)).sum())
            a_only = int(((ea == 1) & (eb == 0)).sum())
            b_only = int(((ea == 0) & (eb == 1)).sum())
            inter = int(((ea == 1) & (eb == 1)).sum())
            union = int(((ea == 1) | (eb == 1)).sum())
            eo_rows.append({
                "seed": s, "model_A": ma, "model_B": mb,
                "error_overlap_jaccard": round(inter / union, 4) if union else 1.0,
                "disagreement_rate": round(float((ea != eb).mean()), 4),
                "both_correct": both_c, "both_wrong": both_w,
                "A_only_error": a_only, "B_only_error": b_only})
            if (ea != eb).sum() > 0 and s == 42:
                nd = a_only + b_only
                # two-sided exact binomial on discordant pairs
                bt = sps.binomtest(min(a_only, b_only), nd, 0.5)
                mcn_rows.append({"seed": s, "comparison": f"{ma}-vs-{mb}",
                                 "discordant_A_only": a_only,
                                 "discordant_B_only": b_only,
                                 "p_two_sided": round(bt.pvalue, 5)})
    ps = [r["p_two_sided"] for r in mcn_rows]
    adj = holm(ps)
    for r, a in zip(mcn_rows, adj):
        r["holm_p"] = round(float(a), 5)
    with open(os.path.join(OUT, "error_overlap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eo_rows[0].keys()))
        w.writeheader()
        w.writerows(eo_rows)
    with open(os.path.join(OUT, "mcnemar_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mcn_rows[0].keys()))
        w.writeheader()
        w.writerows(mcn_rows)

    # ---------------- 8. agreement structure ----------------------------
    agr_rows = []
    models42 = ["MiniROCKET", "R5_5050"] + BASE_MODELS
    for ma, mb in itertools.combinations(models42, 2):
        pa, pb = preds[(ma, 42)], preds[(mb, 42)]
        agree = float((pa == pb).mean())
        from sklearn.metrics import cohen_kappa_score
        agr_rows.append({"model_A": ma, "model_B": mb,
                         "agreement": round(agree, 4),
                         "cohen_kappa": round(float(
                             cohen_kappa_score(pa, pb, labels=CLASSES)), 4)})
    with open(os.path.join(OUT, "model_agreement.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agr_rows[0].keys()))
        w.writeheader()
        w.writerows(agr_rows)

    # ---------------- 9-10. seed + allocation stability ------------------
    r5_scores = {42: 0.5500, 43: 0.5264, 44: 0.5387}
    r5_rho = {42: "50/50", 43: "0.4", 44: "50/50"}
    r5_5050_scores = {}
    for s in SEEDS:
        r5_5050_scores[s] = round(float(macro_f1(y_te, preds[("R5_5050", s)])), 4)
    m0 = 0.4974
    deltas = {s: round(r5_scores[s] - m0, 4) for s in SEEDS}
    arr = np.array(list(r5_scores.values()))
    stab = {
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
        "cv": round(float(arr.std(ddof=1) / arr.mean()), 4)}
    with open(os.path.join(OUT, "r5_seed_stability.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stab.items():
            w.writerow([k, v])
        for s in SEEDS:
            w.writerow([f"seed{s}_R5", r5_scores[s]])
            w.writerow([f"seed{s}_R5_minus_M0", deltas[s]])
    alloc_rows = []
    for s in SEEDS:
        rho_star = 0.5 if r5_rho[s] == "50/50" else float(r5_rho[s])
        n_h = int(round(rho_star * 9996))
        adaptive_score = None
        for r in multiseed["rows"]:
            if r.get("model") == "R5" and r.get("seed") == s:
                adaptive_score = r["test_macro_f1"]
        alloc_rows.append({
            "seed": s, "retained_config": r5_rho[s],
            "rho_star": rho_star, "N_G": 9996 - n_h, "N_H": n_h,
            "R5_5050_test": r5_5050_scores[s],
            "R5_adaptive_CV_selected_test": adaptive_score,
            "difference_5050_minus_adaptive": (
                round(r5_5050_scores[s] - adaptive_score, 4)
                if adaptive_score is not None else "")})
    with open(os.path.join(OUT, "r5_allocation_stability.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(alloc_rows[0].keys()))
        w.writeheader()
        w.writerows(alloc_rows)

    # ---------------- 11. regime statistics ------------------------------
    reg_rows = []
    for s in SEEDS:
        vq = per_seed_diag[s]["vq_diagnostics"]
        occ = vq["occupancy"]
        q = np.asarray(occ)
        q = q[q > 0]
        H_occ = float(-(q * np.log(q)).sum())
        reg_rows.append({
            "seed": s, "source": "stored_vq_diagnostics",
            "active_codes": vq["active_codes"],
            "occupancy": json.dumps([round(float(x), 4) for x in occ]),
            "entropy_nats": round(H_occ, 4),
            "normalized_entropy": round(H_occ / np.log(8), 4),
            "perplexity": round(float(np.exp(H_occ)), 4),
            "min_occupancy": round(float(min(occ)), 4),
            "max_occupancy": round(float(max(occ)), 4)})
    with open(os.path.join(OUT, "regime_statistics.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0].keys()))
        w.writeheader()
        w.writerows(reg_rows)

    # ---------------- 12. H distribution ---------------------------------
    h_rows = []
    for split, H in (("train+val", banks["H_trva"]), ("test", banks["H_te"])):
        h_rows.append({
            "split": split, "mean": round(float(H.mean()), 5),
            "sd": round(float(H.std()), 5),
            "median": round(float(np.median(H)), 5),
            "iqr": round(float(np.percentile(H, 75)
                               - np.percentile(H, 25)), 5),
            "zero_fraction": round(float((H == 0).mean()), 5),
            "min": round(float(H.min()), 5), "max": round(float(H.max()), 5),
            "q95": round(float(np.percentile(H, 95)), 5),
            "n_nonzero_features_trainva": int((banks["H_trva"].mean(0) > 0).sum())
            if split == "train+val" else ""})
    for s in SEEDS:
        hd = per_seed_diag[s]["h_diagnostics"]
        h_rows.append({"split": f"train+val_seed{s}_stored",
                       "mean": round(hd.get("mean", float("nan")), 5),
                       "sd": round(hd.get("std", float("nan")), 5),
                       "median": round(hd.get("median", float("nan")), 5),
                       "iqr": "",
                       "zero_fraction": round(hd.get("zero_fraction",
                                                     float("nan")), 5),
                       "min": round(hd.get("min", float("nan")), 5),
                       "max": round(hd.get("max", float("nan")), 5),
                       "q95": "", "n_nonzero_features_trainva": ""})
    with open(os.path.join(OUT, "H_distribution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(h_rows[0].keys()))
        w.writeheader()
        w.writerows(h_rows)

    # ---------------- 13. G vs H label association -----------------------
    G_tr, H_tr = banks["G_trva"], banks["H_trva"]
    fG = np.nan_to_num(f_classif(G_tr, y_trva)[0], nan=0.0)
    fH = np.nan_to_num(f_classif(H_tr, y_trva)[0], nan=0.0)
    miG = mutual_info_classif(G_tr, y_trva, random_state=42)
    miH = mutual_info_classif(H_tr, y_trva, random_state=42)
    assoc_rows = []
    for tag, fvals, mis in (("G", fG, miG), ("H", fH, miH)):
        assoc_rows.append({
            "bank": tag, "n_features": len(fvals),
            "mean_F": round(float(fvals.mean()), 3),
            "median_F": round(float(np.median(fvals)), 3),
            "mean_top50_F": round(float(np.sort(fvals)[-50:].mean()), 3),
            "median_top50_F": round(float(np.median(np.sort(fvals)[-50:])), 3),
            "frac_F_gt_1": round(float((fvals > 1).mean()), 4),
            "frac_F_gt_2": round(float((fvals > 2).mean()), 4),
            "mean_MI": round(float(mis.mean()), 4),
            "mean_top50_MI": round(float(np.sort(mis)[-50:].mean()), 4),
            "frac_MI_gt_0.05": round(float((mis > 0.05).mean()), 4)})
    with open(os.path.join(OUT, "G_vs_H_label_association.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(assoc_rows[0].keys()))
        w.writeheader()
        w.writerows(assoc_rows)

    # ---------------- 14. G/H redundancy ---------------------------------
    rng_sub = np.random.default_rng(7)
    idx_g = rng_sub.choice(4998, 500, replace=False)
    idx_h = rng_sub.choice(4998, 500, replace=False)
    C = np.corrcoef(G_tr[:, idx_g].T, H_tr[:, idx_h].T)[:500, 500:]
    Cabs = np.abs(C[np.isfinite(C)])
    Gc = G_tr - G_tr.mean(0)
    Hc = H_tr - H_tr.mean(0)
    cka = float((np.linalg.norm(Gc.T @ Hc, "fro") ** 2) /
                (np.linalg.norm(Gc.T @ Gc, "fro") *
                 np.linalg.norm(Hc.T @ Hc, "fro")))

    def eff_rank(X):
        s = np.linalg.svd(X[:155, ::4], compute_uv=False)
        p = s / s.sum()
        p = p[p > 1e-12]
        return float(np.exp(-(p * np.log(p)).sum()))

    red_rows = [{
        "mean_abs_r": round(float(Cabs.mean()), 4),
        "median_abs_r": round(float(np.median(Cabs)), 4),
        "p95_abs_r": round(float(np.percentile(Cabs, 95)), 4),
        "frac_abs_r_gt_0.5": round(float((Cabs > 0.5).mean()), 5),
        "frac_abs_r_gt_0.9": round(float((Cabs > 0.9).mean()), 6),
        "linear_CKA_G_vs_H": round(cka, 4),
        "effective_rank_G": round(eff_rank(G_tr), 1),
        "effective_rank_H": round(eff_rank(H_tr), 1)}]
    with open(os.path.join(OUT, "G_H_redundancy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(red_rows[0].keys()))
        w.writeheader()
        w.writerows(red_rows)
    log("redundancy done: " + json.dumps(red_rows[0]))

    # ---------------- 15-17. mechanism effects ---------------------------
    H_te = banks["H_te"]
    reg_te = regimes["reg_te"]
    h_norm = np.linalg.norm(H_te, axis=1)
    h_mean = H_te.mean(1)
    h_var = H_te.var(1)
    h_nz = (H_te > 0).mean(1)
    n_samp, T = reg_te.shape
    n_occ = np.array([len(np.unique(reg_te[i])) for i in range(n_samp)])
    occ_ent = np.empty(n_samp)
    trans = np.empty(n_samp)
    dwell_mean = np.empty(n_samp)
    dwell_max = np.empty(n_samp)
    for i in range(n_samp):
        seq = reg_te[i]
        _, cnt = np.unique(seq, return_counts=True)
        qk = cnt / T
        occ_ent[i] = -(qk * np.log(qk)).sum()
        trans[i] = int((np.diff(seq) != 0).sum())
        # run lengths
        change = np.flatnonzero(np.diff(seq)) 
        runs = np.diff(np.concatenate([[-1], change, [T - 1]]))
        dwell_mean[i] = runs.mean()
        dwell_max[i] = runs.max()

    mech_rows = []
    for m in ("MiniROCKET", "R5_5050"):
        corr = (preds[(m, 42)] == y_te).astype(bool)
        rng_e = np.random.default_rng(BOOT_SEED + 11)
        for feat_name, feat in (("H_norm", h_norm), ("H_mean", h_mean),
                                ("H_var", h_var), ("H_nonzero_frac", h_nz),
                                ("n_regimes", n_occ), ("regime_entropy", occ_ent),
                                ("transition_count", trans),
                                ("mean_dwell", dwell_mean),
                                ("max_dwell", dwell_max)):
            a, b = feat[corr], feat[~corr]
            d = (a.mean() - b.mean()) / np.sqrt(
                (a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-12)
            # Cliff's delta
            n1, n2 = len(a), len(b)
            gt = sum((x > y).sum() for x in a for y in b)
            lt = sum((x < y).sum() for x in a for y in b)
            cliff = (gt - lt) / (n1 * n2)
            boots = []
            for _ in range(2000):
                ii = rng_e.integers(0, n_samp, n_samp)
                cc = corr[ii]
                if cc.sum() in (0, n_samp):
                    continue
                aa, bb = feat[ii][cc], feat[ii][~cc]
                boots.append(aa.mean() - bb.mean())
            mech_rows.append({
                "model": m, "feature": feat_name,
                "mean_correct": round(float(a.mean()), 4),
                "mean_incorrect": round(float(b.mean()), 4),
                "cohens_d": round(float(d), 4),
                "cliffs_delta": round(float(cliff), 4),
                "boot_ci_low": round(float(np.percentile(boots, 2.5)), 4)
                if boots else "",
                "boot_ci_high": round(float(np.percentile(boots, 97.5)), 4)
                if boots else ""})
    with open(os.path.join(OUT, "sample_mechanism_effects.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mech_rows[0].keys()))
        w.writeheader()
        w.writerows(mech_rows)

    # 17. gain-vs-H
    r5c = preds[("R5_5050", 42)] == y_te
    m0c = preds[("MiniROCKET", 42)] == y_te
    delta_c = np.where(r5c & ~m0c, 1, np.where(m0c & ~r5c, -1, 0))
    gain_rows = []
    for feat_name, feat in (("H_norm", h_norm), ("regime_entropy", occ_ent),
                            ("n_regimes", n_occ),
                            ("transition_rate", trans / T)):
        rho_sp = sps.spearmanr(feat, delta_c)
        gain_rows.append({"feature": feat_name,
                          "spearman_rho": round(float(rho_sp.statistic), 4),
                          "p_value": round(float(rho_sp.pvalue), 4),
                          "n_plus": int((delta_c == 1).sum()),
                          "n_minus": int((delta_c == -1).sum())})
    with open(os.path.join(OUT, "r5_gain_vs_H.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gain_rows[0].keys()))
        w.writeheader()
        w.writerows(gain_rows)

    # 16. regime-complexity bins (quartiles of regime_entropy)
    bins = np.quantile(occ_ent, [0, .25, .5, .75, 1.0])
    bin_rows = []
    for bi in range(4):
        m_ = (occ_ent >= bins[bi]) & (occ_ent <= bins[bi + 1] if bi == 3
                                      else occ_ent < bins[bi + 1])
        if m_.sum() < 8:
            continue
        f1m = macro_f1(y_te[m_], preds[("MiniROCKET", 42)][m_]) \
            if len(np.unique(y_te[m_])) > 1 else float((preds[("MiniROCKET", 42)][m_] == y_te[m_]).mean())
        f1r = macro_f1(y_te[m_], preds[("R5_5050", 42)][m_]) \
            if len(np.unique(y_te[m_])) > 1 else float((preds[("R5_5050", 42)][m_] == y_te[m_]).mean())
        bin_rows.append({"bin": bi, "n": int(m_.sum()),
                         "entropy_range": f"[{bins[bi]:.3f},{bins[bi+1]:.3f})",
                         "M0_acc": round(float((preds[("MiniROCKET", 42)][m_] == y_te[m_]).mean()), 4),
                         "R5_acc": round(float((preds[("R5_5050", 42)][m_] == y_te[m_]).mean()), 4),
                         "M0_mF1_or_acc": round(float(f1m), 4),
                         "R5_mF1_or_acc": round(float(f1r), 4),
                         "delta_acc": round(float((preds[("R5_5050", 42)][m_] == y_te[m_]).mean()
                                                  - (preds[("MiniROCKET", 42)][m_] == y_te[m_]).mean()), 4)})
    with open(os.path.join(OUT, "regime_complexity_bins.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        w.writeheader()
        w.writerows(bin_rows)

    # ---------------- 18. confidence analysis (Ridge decision values) ----
    X_trva = np.hstack([G_tr, H_tr])
    X_te2 = np.hstack([banks["G_te"], H_te])
    ridge = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    ridge.fit(X_trva, y_trva)
    S_r5 = ridge.decision_function(X_te2)
    ridge0 = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    G_tr = banks["G_trva"]
    ridge0.fit(G_tr, y_trva)
    S_m0 = ridge0.decision_function(banks["G_te"])

    def margin(S, p):
        # softmax over class scores
        e = np.exp(S - S.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)
        conf = P.max(1)
        srt = np.sort(P, 1)
        marg = srt[:, -1] - srt[:, -2]
        ent = -(P * np.log(P + 1e-12)).sum(1)
        # multiclass Brier
        Y = np.zeros_like(P)
        Y[np.arange(len(p)), p] = 1.0
        brier = float(((P - Y) ** 2).sum(1).mean())
        return conf, marg, ent, brier

    conf_rows = []
    preds_r5 = preds[("R5_5050", 42)]
    preds_m0 = preds[("MiniROCKET", 42)]
    for m, p, S in (("MiniROCKET", preds_m0, S_m0),
                    ("R5_5050", preds_r5, S_r5)):
        conf, marg, ent, brier = margin(S, p)
        correct = p == y_te
        acc_conf = float(conf[correct].mean())
        err_conf = float(conf[~correct].mean()) if (~correct).any() else ""
        ece = 0.0
        nb = 10
        for b in range(nb):
            lo, hi = b / nb, (b + 1) / nb
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() > 0:
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        conf_rows.append({"model": m, "brier": round(brier, 4),
                          "ece_10bin": round(float(ece), 4),
                          "mean_conf_correct": round(acc_conf, 4),
                          "mean_conf_incorrect": round(err_conf, 4),
                          "mean_margin": round(float(marg.mean()), 4),
                          "mean_entropy": round(float(ent.mean()), 4)})
    # outcome groups
    both_c = r5c & m0c
    groups = {"both_correct": both_c, "R5_only_correct": r5c & ~m0c,
              "M0_only_correct": m0c & ~r5c, "both_wrong": (~r5c) & (~m0c)}
    for gname, gm in groups.items():
        if gm.sum() == 0:
            continue
        conf_r5 = margin(S_r5, preds_r5)[0]
        conf_m0 = margin(S_m0, preds_m0)[0]
        conf_rows.append({"model": f"group:{gname}", "brier": "",
                          "ece_10bin": "",
                          "mean_conf_correct": round(float(conf_r5[gm].mean()), 4),
                          "mean_conf_incorrect": round(float(conf_m0[gm].mean()), 4),
                          "mean_margin": "", "mean_entropy": "",
                          "n": int(gm.sum())})
    with open(os.path.join(OUT, "confidence_analysis.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "brier", "ece_10bin",
                                          "mean_conf_correct",
                                          "mean_conf_incorrect",
                                          "mean_margin", "mean_entropy", "n"])
        w.writeheader()
        for r in conf_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # ---------------- 19. G/H/G+H diagnostic ------------------------------
    diag_rows = []
    for tag, ptr in (("G_only", "haptics_inference_G_only_pred_te.npy"),
                     ("H_only", "haptics_inference_H_only_pred_te.npy"),
                     ("G+H (R5 50/50)", "haptics_inference_R5_s42_pred_te.npy")):
        p = np.load(os.path.join(CACHE, ptr))
        diag_rows.append({"representation": tag,
                          "test_macro_f1": round(float(macro_f1(y_te, p)), 4),
                          "test_accuracy": round(float((p == y_te).mean()), 4),
                          "dim": {"G_only": 4998, "H_only": 4998,
                                  "G+H (R5 50/50)": 9996}[tag],
                          "source": "canonical Ridge on frozen banks, "
                                    "identity-verified"})
    with open(os.path.join(OUT, "G_H_GplusH_diagnostic.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)

    # stash arrays for figures
    np.savez(os.path.join(CACHE, "haptics_inference_arrays.npz"),
             occ_ent=occ_ent, h_norm=h_norm, delta_c=delta_c,
             n_occ=n_occ, trans=trans, dwell_mean=dwell_mean)
    log("core analyses complete")
    return dict(y_te=y_te, preds=preds, boot_cache=boot_cache,
                delta_rows=delta_rows, perm_rows=perm_rows,
                reg_rows=reg_rows, h_rows=h_rows, assoc_rows=assoc_rows,
                red_rows=red_rows, mech_rows=mech_rows, gain_rows=gain_rows,
                bin_rows=bin_rows, conf_rows=conf_rows, diag_rows=diag_rows,
                agr_rows=agr_rows, eo_rows=eo_rows, mcn_rows=mcn_rows,
                stab=stab, alloc_rows=alloc_rows,
                banks=banks, regimes=regimes, per_seed_diag=per_seed_diag)


if __name__ == "__main__":
    main()
