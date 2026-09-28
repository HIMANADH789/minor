"""R3 gate model: one dataset-level scalar gate on the R2 H block.

Structure (spec sections 5-9, frozen by pre-declaration):
    theta : scalar Parameter, init -2.0  ->  w = sigmoid(theta) ~ 0.119
    W, b  : linear classifier on F = [G || w*H]  (C classes)
    L = CE(logits, y) + lambda_clf * ||W||_2^2 + lambda_w * theta^2

The gate is NOT per-kernel/per-regime/per-sample/per-class: it is one
scalar multiplying the entire 4998-dim H block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

THETA_INIT = -2.0            # w_init = sigmoid(-2.0) = 0.11920292
# Classifier training: Adam, mini-batch 64, spec-recommended lr 1e-2,
# lambda_clf 1e-4, lambda_w 1e-3.  EPOCH COUNT: the R3-GH control (a
# differentiable linear head on [G||H], i.e. the R2 representation) must
# reproduce its R2-equivalent Ridge reference; on Haptics it needs ~350+
# epochs to reach/beat Ridge's val Macro-F1 (0.6767 -> 0.7140 at ep349,
# VALIDATION-ONLY probe, test untouched).  1500 epochs / patience 150
# provides that headroom.  Early runs that stopped at 100 epochs were
# discarded as control-invalid (R3-GH far below its reference).
LAMBDA_CLF = 1e-4
LAMBDA_W = 1e-3
LR = 1e-2
MAX_EPOCHS = 1500
PATIENCE = 150
BATCH = 64


class R3GateModel(nn.Module):
    """F = [G || w*H] -> logits = F W^T + b, plus the scalar gate."""

    def __init__(self, n_features_g=4998, n_features_h=4998, n_classes=5):
        super().__init__()
        self.n_g = n_features_g
        self.n_h = n_features_h
        self.theta = nn.Parameter(torch.tensor(float(THETA_INIT)))
        self.W = nn.Parameter(torch.zeros(n_classes,
                                          n_features_g + n_features_h))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def gate(self):
        return torch.sigmoid(self.theta)

    def forward(self, G, H):
        w = self.gate()
        Fmat = torch.cat([G, w * H], dim=1)
        return Fmat @ self.W.t() + self.b

    def gate_penalty(self):
        """lambda_w * theta^2 (the spec's exact form)."""
        return LAMBDA_W * self.theta ** 2

    def weight_penalty(self):
        return LAMBDA_CLF * (self.W ** 2).sum()


def train_model(model, G_tr, H_tr, y_tr, G_va, H_va, y_va,
                variant="R3", lr=LR, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, batch=BATCH, seed=42, device="cpu"):
    """Train one of the three variants (mini-batch, shuffled per epoch).

    variant: 'R3-G'  -> G only (H ignored, theta frozen)
             'R3-GH' -> [G || H] (w forced 1, theta frozen)
             'R3'    -> learned gate (theta trainable)
    """
    torch.manual_seed(seed)
    is_g_only = variant == "R3-G"
    is_gh = variant == "R3-GH"
    model = model.to(device)
    G_tr, H_tr = G_tr.to(device), H_tr.to(device)
    G_va, H_va = G_va.to(device), H_va.to(device)
    y_tr_d, y_va_d = y_tr.to(device), y_va.to(device)

    learn_gate = variant == "R3"
    model.theta.requires_grad_(learn_gate)
    if is_gh:
        # exact w = 1 control: H enters UNSCALED by construction
        model.theta.requires_grad_(False)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    n = G_tr.shape[0]
    traj = []
    best = {"val_mf1": -1.0, "epoch": -1, "state": None}
    if is_g_only:
        n_g = model.n_g

        def fwd(G, H):
            # G-only floor: use the G slice of W (H columns stay zero)
            return G @ model.W[:, :n_g].t() + model.b
    elif is_gh:
        def fwd(G, H):
            # exact w = 1: unscaled [G || H]
            return torch.cat([G, H], dim=1) @ model.W.t() + model.b
    else:
        def fwd(G, H):
            return model(G, H)

    gen = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        ep_loss, nb = 0.0, 0
        for c0 in range(0, n, batch):
            idx = perm[c0:c0 + batch]
            opt.zero_grad()
            logits = fwd(G_tr[idx], H_tr[idx])
            ce = F.cross_entropy(logits, y_tr_d[idx])
            loss = ce + model.weight_penalty()
            if learn_gate:
                loss = loss + model.gate_penalty()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            val_logits = fwd(G_va, H_va)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_mf1 = macro_f1_safe(y_va.numpy(), val_pred)
            val_acc = float((val_pred == y_va.numpy()).mean())
            theta = float(model.theta.detach())
            w = float(torch.sigmoid(model.theta.detach()))
            wnorm = float(model.W.detach().norm())
        traj.append({"epoch": epoch, "train_loss": ep_loss / nb,
                     "val_macro_f1": val_mf1, "val_accuracy": val_acc,
                     "theta": theta, "w": w,
                     "w_norm": wnorm, "lr": lr})
        if val_mf1 > best["val_mf1"]:
            best = {"val_mf1": val_mf1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "theta": theta, "w": w}
        if epoch - best["epoch"] >= patience:
            break
    # final-epoch values
    final = {"theta": traj[-1]["theta"], "w": traj[-1]["w"],
             "val_mf1": traj[-1]["val_macro_f1"]}
    return best, final, traj


def predict(model, G, H, state, variant="R3", device="cpu"):
    """Predict with the stored best state (same variant forwards as training)."""
    model.load_state_dict(state)
    model = model.to(device).eval()
    with torch.no_grad():
        G = G.to(device)
        H = H.to(device)
        if variant == "R3-G":
            logits = G @ model.W[:, :model.n_g].t() + model.b
        elif variant == "R3-GH":
            logits = torch.cat([G, H], dim=1) @ model.W.t() + model.b
        else:
            logits = model(G, H)
    return logits.argmax(dim=1).cpu().numpy()


def macro_f1_safe(y_true, y_pred):
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    return float(macro_f1(y_true, y_pred))
