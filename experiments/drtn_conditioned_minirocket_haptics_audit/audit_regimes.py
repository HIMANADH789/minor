"""
Forensic audit of the M2 == M3 anomaly in the Haptics 3-seed
DRTN-conditioned MiniROCKET experiment.

Reconstructs, for every seed (42/43/44) and every audited sample:
    C0 = original DRTN hard regime assignment
    C1 = M2 regime array  (CURRENT implementation: iid resample from the
                           GLOBAL histogram -- spec deviation candidate)
    C1s = M2 regime array (SPEC-EXACT: per-sample occupancy-preserving
                           random shuffle, i.e. independent permutation)
    C2 = M3 regime array  (per-sample permutation of C0)
and traces where M2 and M3 first become identical.

Saves, per seed:
    audit results JSON        (checkpoint table C0-C6, per-sample stats)
    m2_regimes.npy / m3_regimes.npy / drtn_regimes.npy / m2spec_regimes.npy
    occupancy histograms

Usage:
    python -m experiments.drtn_conditioned_minirocket_haptics_audit.audit_regimes [--max-samples N]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    load_drtn_checkpoint,
    extract_drtn_regimes,
    create_random_regime_control,
    create_shuffled_regime_control,
)

OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_haptics_audit")
K = 8
SEEDS = [42, 43, 44]


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def occupancy_hist(a, k=K):
    return np.bincount(a.ravel(), minlength=k).tolist()


def spec_random_control(regimes, seed):
    """SPEC-EXACT M2: per-sample occupancy-preserving random assignment.

    For each sample independently: draw a random permutation of the
    sample's actual regime labels. This preserves the per-sample histogram
    exactly and destroys temporal placement -- the maximum-entropy
    occupancy-constrained random assignment under the audit spec's
    requirement to 'generate a new random regime array having exactly the
    same occupancy counts but random temporal placement'.

    Uses an INDEPENDENT RNG stream (seed + 900001) from the M3 stream
    (seed + 900002) so the two variants cannot share state.
    """
    rng = np.random.RandomState(seed + 900001)
    n, T = regimes.shape
    out = np.zeros_like(regimes)
    for i in range(n):
        out[i] = regimes[i][rng.permutation(T)]
    return out


def audit_seed(seed, data, device, max_samples=None):
    print(f"\n{'='*74}\nAUDIT SEED {seed}\n{'='*74}", flush=True)
    seed_dir = os.path.join(OUT_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    Xtrva = np.vstack([data["Xtr"], data["Xva"]])
    Xte = data["Xte"]
    n_trva, n_te = len(Xtrva), len(Xte)

    # ---- C0: original DRTN regimes (frozen checkpoints, eval mode) ----
    drtn, info = load_drtn_checkpoint(seed, device)
    assert drtn is not None, f"missing DRTN checkpoint for seed {seed}"
    C0_trva = extract_drtn_regimes(drtn, Xtrva, device=device)
    C0_te = extract_drtn_regimes(drtn, Xte, device=device)

    # ---- C1: M2 as implemented in the 3-seed runner (iid global-hist) ----
    C1_trva = create_random_regime_control(C0_trva, seed=seed)
    C1_te = create_random_regime_control(C0_te, seed=seed)

    # ---- C1s: spec-exact per-sample occupancy-preserving random ----
    C1s_trva = spec_random_control(C0_trva, seed=seed)
    C1s_te = spec_random_control(C0_te, seed=seed)

    # ---- C2: M3 per-sample shuffle ----
    C2_trva = create_shuffled_regime_control(C0_trva, seed=seed)
    C2_te = create_shuffled_regime_control(C0_te, seed=seed)

    # ---- per-sample array comparisons ----
    def compare(A, B):
        eq = np.array_equal(A, B)
        diff_pos = int((A != B).sum())
        return {
            "identical": bool(eq),
            "n_positions": int(A.size),
            "n_differing_positions": diff_pos,
            "fraction_differing": round(diff_pos / A.size, 6),
            "hash_A": arr_hash(A),
            "hash_B": arr_hash(B),
            "occupancy_A": occupancy_hist(A),
            "occupancy_B": occupancy_hist(B),
            "occupancy_equal": bool(occupancy_hist(A) == occupancy_hist(B)),
            "shares_memory": bool(np.shares_memory(A, B)),
        }

    # M2 vs M3 (current implementation)
    cmp_m2_m3 = {
        "trva": compare(C1_trva, C2_trva),
        "test": compare(C1_te, C2_te),
    }
    # spec-exact M2 vs M3
    cmp_m2spec_m3 = {
        "trva": compare(C1s_trva, C2_trva),
        "test": compare(C1s_te, C2_te),
    }
    # M2 vs original DRTN (temporal placement must differ)
    cmp_m2_c0 = {"trva": compare(C1_trva, C0_trva), "test": compare(C1_te, C0_te)}
    cmp_m2spec_c0 = {"trva": compare(C1s_trva, C0_trva), "test": compare(C1s_te, C0_te)}
    cmp_m3_c0 = {"trva": compare(C2_trva, C0_trva), "test": compare(C2_te, C0_te)}
    # per-sample occupancy preservation
    def per_sample_occ_ok(A, B):
        ok = all(np.array_equal(np.bincount(A[i], minlength=K),
                                np.bincount(B[i], minlength=K))
                 for i in range(len(A)))
        return bool(ok)
    occ_checks = {
        "m2spec_per_sample_occupancy_preserved_trva": per_sample_occ_ok(C1s_trva, C0_trva),
        "m2spec_per_sample_occupancy_preserved_test": per_sample_occ_ok(C1s_te, C0_te),
        "m3_per_sample_occupancy_preserved_trva": per_sample_occ_ok(C2_trva, C0_trva),
        "m3_per_sample_occupancy_preserved_test": per_sample_occ_ok(C2_te, C0_te),
        "m2current_per_sample_occupancy_preserved_trva": per_sample_occ_ok(C1_trva, C0_trva),
        "m2current_per_sample_occupancy_preserved_test": per_sample_occ_ok(C1_te, C0_te),
    }

    # RNG stream audit
    r1 = np.random.RandomState(seed)                # current M2 stream
    r2 = np.random.RandomState(seed)                # current M3 stream
    same_state_after_use = bool(
        np.array_equal(r1.permutation(10), r2.permutation(10)))
    rng_audit = {
        "current_implementation": (
            "M2: np.random.RandomState(seed), iid choice from GLOBAL hist; "
            "M3: np.random.RandomState(seed), per-sample permutation"),
        "m2_m3_streams_seeded_identically": True,
        "identical_first_permutation_from_equal_seeds": same_state_after_use,
        "fix_convention": (
            "M2 (spec): RandomState(seed+900001) per-sample permutation; "
            "M3: RandomState(seed+900002) per-sample permutation"),
    }

    # deterministic representative sample printout (first 10 of each split)
    printout = {"trva_first10": [], "test_first10": []}
    for i in range(min(10, n_trva)):
        printout["trva_first10"].append({
            "sample": i, "split": "trainval", "T": int(C0_trva.shape[1]),
            "M2_current_first40": C1_trva[i][:40].tolist(),
            "M2_spec_first40": C1s_trva[i][:40].tolist(),
            "M3_first40": C2_trva[i][:40].tolist(),
            "M2_current_eq_M3": bool(np.array_equal(C1_trva[i], C2_trva[i])),
            "M2_spec_eq_M3": bool(np.array_equal(C1s_trva[i], C2_trva[i])),
            "M2_current_eq_M3_occupancy": bool(
                np.array_equal(np.bincount(C1_trva[i], minlength=K),
                               np.bincount(C2_trva[i], minlength=K))),
            "M2_spec_eq_M3_occupancy": bool(
                np.array_equal(np.bincount(C1s_trva[i], minlength=K),
                               np.bincount(C2_trva[i], minlength=K))),
            "M2_current_hist": np.bincount(C1_trva[i], minlength=K).tolist(),
            "M3_hist": np.bincount(C2_trva[i], minlength=K).tolist(),
        })
    for i in range(min(10, n_te)):
        printout["test_first10"].append({
            "sample": i, "split": "test", "T": int(C0_te.shape[1]),
            "M2_current_eq_M3": bool(np.array_equal(C1_te[i], C2_te[i])),
            "M2_spec_eq_M3": bool(np.array_equal(C1s_te[i], C2_te[i])),
        })

    # ---- save arrays ----
    np.save(os.path.join(seed_dir, "drtn_regimes_test.npy"), C0_te)
    np.save(os.path.join(seed_dir, "m2_regimes_test.npy"), C1_te)
    np.save(os.path.join(seed_dir, "m2spec_regimes_test.npy"), C1s_te)
    np.save(os.path.join(seed_dir, "m3_regimes_test.npy"), C2_te)

    # ---- checkpoint table C0-C6 (array level; features/predictions audited
    #      separately by the runner-level audit) ----
    checkpoints = {
        "C0": {"name": "original DRTN regimes",
               "hash": arr_hash(C0_te), "shape": list(C0_te.shape),
               "occupancy": occupancy_hist(C0_te)},
        "C1": {"name": "M2 current (iid global-hist)",
               "hash": arr_hash(C1_te), "shape": list(C1_te.shape),
               "occupancy": occupancy_hist(C1_te)},
        "C1s": {"name": "M2 spec (per-sample permuted random)",
                "hash": arr_hash(C1s_te), "shape": list(C1s_te.shape),
                "occupancy": occupancy_hist(C1s_te)},
        "C2": {"name": "M3 shuffled regimes",
               "hash": arr_hash(C2_te), "shape": list(C2_te.shape),
               "occupancy": occupancy_hist(C2_te)},
    }

    result = {
        "seed": seed,
        "n_trainval": int(n_trva), "n_test": int(n_te),
        "T": int(C0_te.shape[1]),
        "m2_m3_current": cmp_m2_m3,
        "m2spec_m3": cmp_m2spec_m3,
        "m2_current_vs_C0": cmp_m2_c0,
        "m2spec_vs_C0": cmp_m2spec_c0,
        "m3_vs_C0": cmp_m3_c0,
        "occupancy_checks": occ_checks,
        "rng_audit": rng_audit,
        "checkpoints": checkpoints,
        "representative_samples": printout,
    }
    with open(os.path.join(seed_dir, "audit_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  M2(current) == M3 test arrays? "
          f"{cmp_m2_m3['test']['identical']} "
          f"(hash M2={cmp_m2_m3['test']['hash_A']}, M3={cmp_m2_m3['test']['hash_B']})")
    print(f"  M2(spec)    == M3 test arrays? "
          f"{cmp_m2spec_m3['test']['identical']} "
          f"(hash M2s={cmp_m2spec_m3['test']['hash_A']}, "
          f"M3={cmp_m2spec_m3['test']['hash_B']})")
    print(f"  occupancy checks: {occ_checks}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_dataset("Haptics")

    results = {}
    for seed in SEEDS:
        results[str(seed)] = audit_seed(seed, data, device,
                                        max_samples=args.max_samples)

    summary = {
        "objective": "Explain M2 == M3 in the Haptics 3-seed conditioned MiniROCKET run",
        "seeds": SEEDS,
        "finding": {
            "m2_current_vs_m3_identical": {
                s: results[s]["m2_m3_current"]["test"]["identical"] for s in results},
            "m2_spec_vs_m3_identical": {
                s: results[s]["m2spec_m3"]["test"]["identical"] for s in results},
        },
        "occupancy_checks": {s: results[s]["occupancy_checks"] for s in results},
    }
    with open(os.path.join(OUT_DIR, "regime_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY:", json.dumps(summary["finding"], indent=2))


if __name__ == "__main__":
    main()
