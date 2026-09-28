"""TURS-MGB Phase 0: repository audit.

Writes results/turs_mgb/audit/{repository_audit,transport_inventory,
kernel_inventory,protocol_audit,feasibility}.json.

The audit RE-VERIFIES reusable components live (import + shape checks)
rather than trusting documentation.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

AUDIT_DIR = os.path.join(ROOT, "results", "turs_mgb", "audit")


def _dump(name, obj):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(os.path.join(AUDIT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_audit(log=print):
    import torch
    from models.turs_rrmt.model import (build_pattern_bank, TransportFlavors,
                                        QUANT_GRID_COARSE, QUANT_GRID_FINE,
                                        TAIL_WEIGHTS)
    from experiments.turs_rrmt.data import load_split, SEED, DATASETS
    from src.diagnostics import statistics as S
    from src.diagnostics.calibration import ece, adaptive_ece, brier, nll

    # ---------------- repository audit
    repo = dict(
        date=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version.split()[0], torch=torch.__version__,
        cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        reused_modules={
            "kernel_recipe": "models/turs_rrmt/model.py::build_pattern_bank",
            "transport_flavors": "models/turs_rrmt/model.py::TransportFlavors",
            "data_protocol": "experiments/turs_rrmt/data.py::load_split",
            "metrics": "experiments/turs_rrmt/train_eval.py::full_metrics",
            "statistics": "src/diagnostics/statistics.py",
            "calibration": "src/diagnostics/calibration.py",
            "perturbation": "src/diagnostics/perturb.py::apply_spec",
            "ridge_reference": "models/turs_glr/block_ridge.py (BlockRidge, for "
                               "primal/dual cross-validation)",
        },
        explicitly_not_reused=[
            "RRMT router", "GLR gate", "block lambda gating", "old MLP head",
            "learned transport selection"],
    )
    _dump("repository_audit.json", repo)

    # ---------------- transport inventory (live verification)
    inv = {}
    try:
        tf = TransportFlavors(lag_set=(1, 2, 4, 8))
        X = np.random.default_rng(0).standard_normal((16, 1, 140)).astype(np.float32)
        tf.fit_reference(torch.from_numpy(X).float())
        T = tf(torch.from_numpy(X).float(), window=7)
        inv["TransportFlavors"] = dict(
            status="OK", output_shape=list(T.shape),
            flavors=list(tf.FLAVORS), ref_grid=int(tf.ref_grid),
            quant_grid_coarse=int(QUANT_GRID_COARSE.numel()),
            quant_grid_fine=int(QUANT_GRID_FINE.numel()),
            tail_weights=[round(float(x), 3) for x in TAIL_WEIGHTS])
    except Exception as e:                                   # pragma: no cover
        inv["TransportFlavors"] = dict(status="FAIL", error=str(e))
    _dump("transport_inventory.json", inv)

    # ---------------- kernel inventory
    kin = {}
    for M in (2048, 4096, 8192):
        try:
            W, spec = build_pattern_bank(M, (7, 11, 15, 23, 31), 1, 42)
            import hashlib
            kin[f"M={M}"] = dict(
                status="OK", shape=list(W.shape),
                lengths=sorted(set(s["length"] for s in spec)),
                kernel_hash=hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16],
                trainable_params=0)
        except Exception as e:                               # pragma: no cover
            kin[f"M={M}"] = dict(status="FAIL", error=str(e))
    _dump("kernel_inventory.json", kin)

    # ---------------- protocol audit
    prot = dict(seed=SEED, datasets=DATASETS, splits={}, normalization=
                "per-sample z-norm (eps 1e-8), applied inside load_split")
    for tag in DATASETS:
        ds = load_split(tag)
        prot["splits"][tag] = dict(
            L=int(ds["L"]), n_cls=int(ds["n_cls"]),
            train=int(len(ds["y_train"])), val=int(len(ds["y_val"])),
            test=int(len(ds["y_test"])),
            source=ds.get("data_sha256", "")[:16])
    _dump("protocol_audit.json", prot)

    # ---------------- feasibility (timing probe at realistic sizes)
    feas = {}
    try:
        from models.turs_mgb.pattern_bank import SharedKernelBank
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bank = SharedKernelBank(M=4096, seed=42, device=dev)
        X = np.random.default_rng(0).standard_normal((256, 1, 1024)).astype(np.float32)
        t0 = __import__("time").time()
        Z = bank.features(X, chunk=128, stats=("ppv", "max"))
        dt = __import__("time").time() - t0
        feas["shared_bank_M4096"] = dict(
            status="OK", feature_dim=int(Z.shape[1]),
            s_per_256_samples_T1024=round(dt, 2),
            projected_full_split_CWRU_BAL_train_s=round(dt * 2642 / 256, 1),
            device=str(dev))
    except Exception as e:                                   # pragma: no cover
        feas["shared_bank_M4096"] = dict(status="FAIL", error=str(e))
    try:
        from models.turs_mgb.transport_views import TransportGeometryBank
        gb = TransportGeometryBank()
        X = np.random.default_rng(0).standard_normal((256, 1, 1024)).astype(np.float32)
        gb.fit(X)
        t0 = __import__("time").time()
        T = gb.extract(X, window=51, batch=64)
        dt = __import__("time").time() - t0
        feas["transport_views_4x"] = dict(
            status="OK", shape=list(T.shape),
            s_per_256_samples_T1024=round(dt, 2),
            projected_full_split_CWRU_BAL_train_s=round(dt * 2642 / 256, 1))
    except Exception as e:                                   # pragma: no cover
        feas["transport_views_4x"] = dict(status="FAIL", error=str(e))
    _dump("feasibility.json", feas)
    log(f"  audit written -> {os.path.relpath(AUDIT_DIR, ROOT)}")
    return repo, inv, kin, prot, feas
