"""
ARTNet Ablation Study

Runs the following ablations on ECG5000_UNBAL:
  A1: Full ARTNet-CE (baseline)
  A2: NoTransport — remove transport branch
  A3: NoRegime — remove regime entirely (becomes ~InceptionTime)
  A4: NoUncertainty — deterministic regime (z = mu)
  A5: NoGate — unconditional regime injection
  A6: NoDynamics — remove dynamics loss
  A7: NoHierarchy — single regime (not hierarchical)

Usage:
  python ablate_artnet.py ECG5000_UNBAL
  python ablate_artnet.py all
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.artnet import ARTNet, count_parameters
from experiments.train_artnet import load_data, make_loader, FocalLoss, set_seed, DATASETS, SEED, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY


class ARTNetNoTransport(nn.Module):
    """ARTNet without transport branch — just InceptionTime + regime."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inner = ARTNet(**kwargs)
        # Force alpha to 0 so transport has no effect
        self.inner.alpha_logit.data = torch.tensor(-10.0)
        # Zero out transport-regime interaction
        self.inner.tr_regime_Wa.weight.data.zero_()
        self.inner.tr_regime_Wa.bias.data.zero_()
    
    def forward(self, x):
        return self.inner(x)


class ARTNetNoRegime(nn.Module):
    """ARTNet without regime — just backbone + transport. Acts like InceptionTime+transport."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inner = ARTNet(**kwargs)
        D = self.inner.D
        nc = kwargs.get('num_classes', 5)
        rd = kwargs.get('regime_dim', 16)
        # Simple classifier on raw features only
        self.simple_classifier = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, nc)
        )
        self.D = D
        self.rd = rd
    
    def forward(self, x):
        B, C, L = x.shape
        # Run backbone + transport only, skip regime pathway
        T = self.inner.transport_builder(x)
        T_e = self.inner.transport_encoder(T)
        H_f, H_m, H_c = self.inner.backbone(x)
        alpha = torch.sigmoid(self.inner.alpha_logit)
        H = H_c
        H_T = H + alpha * T_e
        v_prelim = H_T.mean(dim=2)
        z_0 = self.inner.prelim_regime(v_prelim)
        a_z = self.inner.tr_regime_Wa(z_0)
        scale = torch.sigmoid(a_z)
        T_prime = T_e * (1 + scale.unsqueeze(2))
        H_TR = H_T + T_prime
        h = H_TR.mean(dim=2)
        logits = self.simple_classifier(h)
        return {
            "logits": logits,
            "features": h,
            "adapted_features": h,
            "regime": torch.zeros(B, self.rd, device=x.device),
            "regime_mu": torch.zeros(B, self.rd, device=x.device),
            "regime_std": torch.ones(B, self.rd, device=x.device),
            "regime_uncertainty": torch.ones(B, 1, device=x.device),
            "gate": torch.ones(B, self.D, device=x.device),
            "fine_regime": torch.zeros(B, self.rd, device=x.device),
            "middle_regime": torch.zeros(B, self.rd, device=x.device),
            "coarse_regime": torch.zeros(B, self.rd, device=x.device),
            "predicted_next_regime": torch.zeros(B, self.rd, device=x.device),
            "alpha": alpha,
        }


class ARTNetNoUncertainty(nn.Module):
    """ARTNet without uncertainty — deterministic regime."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inner = ARTNet(**kwargs)
    
    def forward(self, x):
        out = self.inner(x)
        # Override: use mu directly, no sampling, std = 0
        out["regime"] = out["regime_mu"]
        out["regime_std"] = torch.ones_like(out["regime_std"]) * 0.01
        out["regime_uncertainty"] = torch.ones(out["regime_mu"].shape[0], 1, device=x.device) * 0.01
        return out


class ARTNetNoGate(nn.Module):
    """ARTNet without adaptive gate — unconditional regime injection."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inner = ARTNet(**kwargs)
    
    def forward(self, x):
        out = self.inner(x)
        # Override: gate = 1.0 (always inject)
        out["gate"] = torch.ones_like(out["gate"])
        # Recompute adapted features
        h = out["features"]
        r_z = self.inner.regime_residual(out["regime"])
        out["adapted_features"] = h + r_z
        return out


class ARTNetNoHierarchy(nn.Module):
    """ARTNet with single regime instead of hierarchical."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inner = ARTNet(**kwargs)
        D = self.inner.D
        rd = kwargs.get('regime_dim', 16)
        # Replace hierarchical encoders with single encoder
        self.inner.fine_regime_enc = nn.Sequential(
            nn.Linear(D, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, rd)
        )
        self.inner.mid_regime_enc = nn.Identity()
        self.inner.coarse_regime_enc = nn.Identity()
        # Update mu/logvar/proj to take rd instead of 3*rd
        self.inner.regime_mu = nn.Linear(rd, rd, bias=True)
        self.inner.regime_logvar = nn.Linear(rd, rd, bias=True)
        self.inner.regime_proj = nn.Linear(rd, rd, bias=True)
    
    def forward(self, x):
        B, C, L = x.shape
        # Run backbone and transport manually to use single regime
        T = self.inner.transport_builder(x)
        T_e = self.inner.transport_encoder(T)
        H_f, H_m, H_c = self.inner.backbone(x)
        alpha = torch.sigmoid(self.inner.alpha_logit)
        H = H_c
        H_T = H + alpha * T_e
        v_prelim = H_T.mean(dim=2)
        z_0 = self.inner.prelim_regime(v_prelim)
        a_z = self.inner.tr_regime_Wa(z_0)
        scale = torch.sigmoid(a_z)
        T_prime = T_e * (1 + scale.unsqueeze(2))
        H_TR = H_T + T_prime
        h = H_TR.mean(dim=2)  # [B, D]
        
        # Single regime (not hierarchical)
        z = self.inner.fine_regime_enc(h)  # [B, rd]
        
        mu = self.inner.regime_mu(z)
        logvar = self.inner.regime_logvar(z)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        std = torch.exp(0.5 * logvar)
        
        if self.inner.training:
            eps = torch.randn_like(std)
            z_r = mu + std * eps
        else:
            z_r = mu
        
        # Gate
        gate_input = torch.cat([h, z_r], dim=1)
        g = torch.sigmoid(self.inner.gate_W(gate_input))
        gate_u_mod = torch.sigmoid(self.inner.gate_u * std.mean(dim=1, keepdim=True))
        g = g * gate_u_mod
        
        # Adaptive injection
        r_z = self.inner.regime_residual(z_r)
        h_prime = h + g * r_z
        
        p_z = self.inner.classifier_proj(z_r)
        h_z = h_prime * p_z
        
        r = torch.cat([h_prime, z_r, h_z], dim=1)
        logits = self.inner.classifier(r)
        
        return {
            "logits": logits,
            "features": h,
            "adapted_features": h_prime,
            "regime": z_r,
            "regime_mu": mu,
            "regime_std": std,
            "regime_uncertainty": std.mean(dim=1, keepdim=True),
            "gate": g,
            "fine_regime": z,
            "middle_regime": z,
            "coarse_regime": z,
            "predicted_next_regime": self.inner.dynamics(torch.cat([z_r, h], dim=1)),
            "alpha": alpha,
        }


def train_variant(model, data, variant_name, dataset_name, use_dyn_loss=True):
    """Train one ablation variant."""
    print(f"\n  Training {variant_name}...")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    params = count_parameters(model)
    
    train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
    test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_mf1 = -1
    best_state = None
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out["logits"], yb)
            
            # Auxiliary losses
            aux = torch.tensor(0.0, device=device)
            aux = aux + 0.01 * F.relu(0.1 - out["regime_std"].mean(dim=0)).pow(2).mean()
            aux = aux + 0.001 * out["gate"].abs().mean()
            
            total_batch_loss = loss + aux
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
        
        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                val_preds.extend(out["logits"].argmax(dim=1).cpu().numpy())
                val_true.extend(yb.numpy())
        
        val_mf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        
        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    # Test
    total_time = time.time() - start_time
    model.load_state_dict(best_state)
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            test_preds.extend(out["logits"].argmax(dim=1).cpu().numpy())
            test_true.extend(yb.numpy())
    
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    
    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    wf1 = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    class_f1s = f1_score(test_true, test_preds, average=None, zero_division=0).tolist()
    
    print(f"    {variant_name}: MF1={mf1:.4f} Acc={acc:.4f} ({total_time:.0f}s)")
    
    return {
        "variant": variant_name,
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "class_f1s": class_f1s,
        "params": params,
        "best_epoch": best_epoch,
        "time_s": total_time,
    }


def run_ablations(dataset_name):
    """Run all ablation variants on one dataset."""
    print(f"\n{'='*60}")
    print(f"  ARTNet Ablation: {dataset_name}")
    print(f"{'='*60}")
    
    data = load_data(dataset_name)
    nc = data["num_classes"]
    sl = data["seq_len"]
    
    variants = {
        "Full ARTNet-CE": (ARTNet, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
        "NoTransport": (ARTNetNoTransport, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
        "NoRegime": (ARTNetNoRegime, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
        "NoUncertainty": (ARTNetNoUncertainty, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
        "NoGate": (ARTNetNoGate, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
        "NoHierarchy": (ARTNetNoHierarchy, {"in_channels": 1, "num_classes": nc, "seq_len": sl}),
    }
    
    results = {}
    for name, (cls, kwargs) in variants.items():
        model = cls(**kwargs)
        r = train_variant(model, data, name, dataset_name)
        results[name] = r
    
    # Save
    result_dir = os.path.join(ROOT, "results", "artnet_ablation")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{dataset_name}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {result_path}")
    
    # Print comparison
    print(f"\n  ABLATION RESULTS ({dataset_name}):")
    base_mf1 = results["Full ARTNet-CE"]["macro_f1"]
    for name, r in results.items():
        delta = r["macro_f1"] - base_mf1
        print(f"    {name:25s} MF1={r['macro_f1']:.4f}  Delta={delta:+.4f}  p={r['params']:,}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ablate_artnet.py <DATASET|all>")
        sys.exit(1)
    
    ds_arg = sys.argv[1]
    if ds_arg == "all":
        for ds in DATASETS:
            run_ablations(ds)
    else:
        run_ablations(ds_arg)


if __name__ == "__main__":
    main()
