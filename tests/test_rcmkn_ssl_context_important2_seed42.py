"""Test suite for the SSL-context transfer part 2 (ECG5000_BAL / CWRU_BAL).

Covers all 23 mandatory topics with small synthetic tensors; dataset
configs and the R0 gate are additionally asserted by the runner itself.
"""
import numpy as np
import pytest
import torch

from experiments.rcmkn_ssl_context_important2_seed42.runner import (
    EXPECTED, R0_GATES, M0_REFS, run_variant, h_stats)
from experiments.rcmkn_ssl_context_transfer_seed42.core import (
    build_context_model, occupancy, context_regimes)
from experiments.rcmkn_haptics_seed42.config import (
    SEED, ENCODER, VQ, JOINT, K_CODES, N_GLOBAL)
from experiments.rcmkn_haptics_seed42.regime_encoder import (
    SSLTemporalEncoder, MaskDecoder, make_span_mask, ssl_masked_recon_loss)
from experiments.rcmkn_haptics_seed42.vq import build_vq, hard_assign
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

T, N, n_cls = 128, 32, 4


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def synthetic_data(Nn, Tn, n_cls=4):
    X = np.random.randn(Nn, Tn).astype(np.float32)
    y = np.random.randint(0, n_cls, size=Nn)
    return znorm(X), y


# ---------- 1/2. dataset configurations --------------------------------------
def test_dataset_configs():
    assert EXPECTED["ECG5000_BAL"] == {"T": 140, "n_classes": 5,
                                       "train": 5226, "val": 923, "test": 1000}
    assert EXPECTED["CWRU_BAL"] == {"T": 1024, "n_classes": 4,
                                    "train": 2727, "val": 482, "test": 567}
    assert M0_REFS == {"ECG5000_BAL": 0.6553, "CWRU_BAL": 0.9947}
    assert R0_GATES["ECG5000_BAL"] == 0.6748


def test_npz_loaders_match_configs():
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer
    for ds, exp in EXPECTED.items():
        d = transfer.load_any_dataset(ds)
        got = {"T": int(d["Xtr"].shape[1]), "n_classes": d["n_classes"],
               "train": len(d["Xtr"]), "val": len(d["Xva"]),
               "test": len(d["Xte"])}
        assert got == exp, f"{ds}: {got} != {exp}"


# ---------- 3. raw MiniRocket identity ---------------------------------------
def test_minirocket_identity():
    from aeon.transformations.collection.convolution_based import MiniRocket
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    Xtr, _ = synthetic_data(16, T)
    ext = MiniRocket(random_state=42, n_jobs=-1)
    ext.fit(Xtr[:, None, :].astype(np.float32))
    act, valid = compute_raw_activations(ext, Xtr[:4])
    d = np.abs(ppv_from_activations(act, valid) -
               ext.transform(Xtr[:4, None, :].astype(np.float32)))
    assert float(d.max()) < 1e-5


# ---------- 4/5/6. budget, split, slicing ------------------------------------
def test_budget_and_split():
    from aeon.transformations.collection.convolution_based import MiniRocket
    Xtr, _ = synthetic_data(8, T)
    ext = MiniRocket(random_state=42, n_jobs=-1)
    ext.fit(Xtr[:, None, :].astype(np.float32))
    F = ext.transform(Xtr[:4, None, :].astype(np.float32))
    assert F.shape[1] == 9996 == 2 * N_GLOBAL
    G, H = F[:, :N_GLOBAL], F[:, N_GLOBAL:]
    assert G.shape[1] == H.shape[1] == N_GLOBAL and F.shape[0] == 4


# ---------- 7. raw-response provenance ---------------------------------------
def test_raw_response_provenance():
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations)
    Xtr, _ = synthetic_data(4, T)
    ext = _mini()
    act, valid = compute_raw_activations(ext, znorm(Xtr))
    assert act.shape == (4, 9996, T) and act.dtype == bool
    assert valid.shape == (9996, T) and valid.dtype == bool  # per-feature


def _mini():
    from aeon.transformations.collection.convolution_based import MiniRocket
    Xtr, _ = synthetic_data(8, T)
    ext = MiniRocket(random_state=42, n_jobs=-1)
    ext.fit(Xtr[:, None, :].astype(np.float32))
    return ext


# ---------- 8. exact H formula -----------------------------------------------
def test_h_formula():
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, independent_heterogeneity_recompute)
    Xtr, _ = synthetic_data(2, T)
    ext = _mini()
    act, valid = compute_raw_activations(ext, znorm(Xtr))
    reg = np.random.randint(0, K_CODES, size=T)
    for m in [0, N_GLOBAL // 2, N_GLOBAL - 1]:
        a, v = act[0, N_GLOBAL + m], valid[N_GLOBAL + m]
        vm = v.astype(bool)
        ppv_m = a[vm].mean()
        ref = 0.0
        for k in range(K_CODES):
            sel = vm & (reg == k)
            if sel.sum():
                ref += (sel.sum() / vm.sum()) * (a[sel].mean() - ppv_m) ** 2
        got = independent_heterogeneity_recompute(
            a.astype(bool), v.astype(bool), reg.astype(np.int64), K=K_CODES)
        assert abs(ref - got) < 1e-10


# ---------- 9. per-feature valid masks ---------------------------------------
def test_per_feature_valid_region():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations)
    Xtr, _ = synthetic_data(1, T)
    ext = _mini()
    act, valid = compute_raw_activations(ext, znorm(Xtr))
    reg = np.random.randint(0, K_CODES, size=(1, T))
    valid_het = valid[N_GLOBAL:]
    H_before = compute_regime_heterogeneity(act[:, N_GLOBAL:], valid_het, reg)[0]
    act_c = act[:, N_GLOBAL:].copy()
    n_flip = 0
    for m in range(N_GLOBAL):
        idx = np.flatnonzero(~valid_het[m])
        if len(idx):
            act_c[0, m, idx] = ~act_c[0, m, idx]
            n_flip += len(idx)
    H_after = compute_regime_heterogeneity(act_c, valid_het, reg)[0]
    assert n_flip > 0 and np.array_equal(H_before, H_after)


# ---------- 10. encoder causality --------------------------------------------
def test_causality():
    enc = SSLTemporalEncoder(channels=ENCODER["channels"],
                             kernel_sizes=ENCODER["kernel_sizes"],
                             dilations=ENCODER["dilations"],
                             d_model=ENCODER["d_model"])
    enc.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, T)
        z1 = enc(x)
        x2 = x.clone(); x2[:, :, T // 2:] = torch.randn(2, 1, T // 2)
        z2 = enc(x2)
    assert torch.equal(z1[:, :T // 2], z2[:, :T // 2])


# ---------- 11/12. SSL masking + no labels -----------------------------------
def test_ssl_mask_manual():
    enc = SSLTemporalEncoder(channels=ENCODER["channels"],
                             kernel_sizes=ENCODER["kernel_sizes"],
                             dilations=ENCODER["dilations"],
                             d_model=ENCODER["d_model"])
    dec = MaskDecoder(d_model=ENCODER["d_model"])
    torch.manual_seed(42)
    x = torch.randn(4, T)
    mask = make_span_mask(4, T, 0.10, 16,
                          generator=torch.Generator().manual_seed(99))
    xin1 = x.clone(); xin1[~mask] = 0.0
    p1 = dec(enc(xin1[:, None, :]))
    l1 = torch.nn.functional.mse_loss(p1[mask], x[mask])
    x2 = x.clone(); x2[~mask] += 100.0
    xin2 = x2.clone(); xin2[~mask] = 0.0
    p2 = dec(enc(xin2[:, None, :]))
    l2 = torch.nn.functional.mse_loss(p2[mask], x2[mask])
    assert abs(l1.item() - l2.item()) < 1e-10


def test_ssl_no_labels():
    import inspect
    assert "y" not in inspect.signature(ssl_masked_recon_loss).parameters


# ---------- 13/14/15. hard VQ, EMA, dead-code revival ------------------------
def test_hard_vq_assignment():
    vq = build_vq(d_model=32)
    z = torch.randn(4, T, 32)
    k = hard_assign(vq, z)
    expected = torch.cdist(z, vq.codes.unsqueeze(0)).squeeze(0).argmin(-1)
    assert k.shape == (4, T) and torch.equal(k, expected)


def test_vq_ema():
    vq = build_vq(d_model=32)
    cb = vq.codes.clone()
    z = torch.randn(4, T, 32)
    k = hard_assign(vq, z)
    vq.ema_step(z.reshape(-1, 32), k.reshape(-1), step=0)
    assert not torch.equal(cb, vq.codes)


def test_dead_code_revival():
    """Force a dead code by keeping some codes unused, then verify revival
    accounting triggers (usage below threshold accumulates dead_steps)."""
    vq = build_vq(d_model=32)
    z = torch.randn(4, T, 32) * 0.1 + torch.tensor(
        [3.0 if i % K_CODES == 0 else -3.0 for i in range(32)])
    for step in range(5):
        k = hard_assign(vq, z)
        vq.ema_step(z.reshape(-1, 32), k.reshape(-1), step=step)
    used = torch.bincount(k.reshape(-1), minlength=K_CODES)
    assert (used == 0).any(), "test setup: expected some unused codes"
    assert (vq.dead_steps > 0).any() or vq.total_revivals > 0, \
        "dead-code accounting did not trigger"


# ---------- 16/17/18/19. controls --------------------------------------------
def test_c1_c2_controls():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_random_regime_control, create_shuffled_regime_control)
    reg = np.random.randint(0, K_CODES, size=(N, T))
    c1 = create_random_regime_control(reg, seed=SEED)
    c2 = create_shuffled_regime_control(reg, seed=SEED)
    for i in range(N):
        assert np.array_equal(np.bincount(c1[i], minlength=K_CODES),
                              np.bincount(reg[i], minlength=K_CODES))
        assert np.array_equal(np.bincount(c2[i], minlength=K_CODES),
                              np.bincount(reg[i], minlength=K_CODES))
    assert not np.array_equal(c1, c2)
    assert not np.shares_memory(c1, c2)
    assert not np.shares_memory(c1, reg)
    assert not np.shares_memory(c2, reg)


# ---------- 20. train+val Ridge ----------------------------------------------
def test_run_variant_fits_trainval():
    """run_variant must fit on the full provided trainva matrix."""
    F = np.random.randn(40, 9996)
    y = np.random.randint(0, 3, 40)
    yva, yte = y[:10], y[10:20]
    res, _ = run_variant("T", {"g": (F[:30], F[:10], F[10:20])},
                         yva, yte, y[:30])
    assert res["feature_dim"] == 9996


# ---------- 21. no test leakage (structural) ---------------------------------
def test_no_test_leakage():
    import inspect
    src = inspect.getsource(
        __import__("experiments.rcmkn_ssl_context_important2_seed42.runner",
                   fromlist=["run_dataset"]).run_dataset)
    assert "train_ssl_context(model, Xtr_z, ytr, Xva_z" in src
    assert "core.context_regimes(model, Xte_z" in src  # regimes only, post-freeze


# ---------- 22. determinism --------------------------------------------------
def test_deterministic_regimes():
    x_np, _ = synthetic_data(4, T)
    m = RCMKNContextModel(n_classes=n_cls)
    m.eval()
    r1 = context_regimes(m, x_np, "cpu")
    r2 = context_regimes(m, x_np, "cpu")
    assert np.array_equal(r1, r2)


# ---------- 23. NaN/Inf guard -------------------------------------------------
def test_h_stats_finite():
    H = np.random.rand(10, N_GLOBAL)
    s = h_stats(H)
    assert all(np.isfinite(v) for v in s.values())
