"""
Fast ARTNet ablation — 15 epochs per variant, saves incrementally.
Usage: python ablate_artnet_fast.py ECG5000_UNBAL <variant_index>
  variant_index: 0=Full, 1=NoTransport, 2=NoRegime, 3=NoUncertainty, 4=NoGate, 5=NoHierarchy
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.artnet import ARTNet, count_parameters
from experiments.train_artnet import load_data, make_loader, FocalLoss, set_seed, DATASETS

SEED = 42
MAX_EPOCHS = 15
PATIENCE = 8
BATCH_SIZE = 64
LR = 1e-3


def get_variant(name, nc, sl):
    kwargs = {"in_channels": 1, "num_classes": nc, "seq_len": sl}
    if name == "Full":
        return ARTNet(**kwargs)
    elif name == "NoTransport":
        m = ARTNet(**kwargs)
        m.alpha_logit.data = torch.tensor(-10.0)
        m.tr_regime_Wa.weight.data.zero_()
        m.tr_regime_Wa.bias.data.zero_()
        return m
    elif name == "NoRegime":
        from experiments.ablate_artnet import ARTNetNoRegime
        return ARTNetNoRegime(**kwargs)
    elif name == "NoUncertainty":
        from experiments.ablate_artnet import ARTNetNoUncertainty
        return ARTNetNoUncertainty(**kwargs)
    elif name == "NoGate":
        from experiments.ablate_artnet import ARTNetNoGate
        return ARTNetNoGate(**kwargs)
    elif name == "NoHierarchy":
        from experiments.ablate_artnet import ARTNetNoHierarchy
        return ARTNetNoHierarchy(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {name}")


def train_and_eval(model, data, variant_name):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
    test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    
    best_val_mf1, best_state, best_epoch, patience_ct = -1, None, 0, 0
    start = time.time()
    
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out["logits"], yb)
            # Light aux
            aux = 0.01 * F.relu(0.1 - out["regime_std"].mean(dim=0)).pow(2).mean() + 0.001 * out["gate"].abs().mean()
            optimizer.zero_grad(); (loss + aux).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                vp.extend(out["logits"].argmax(1).cpu().numpy()); vt.extend(yb.numpy())
        vmf1 = f1_score(vt, vp, average="macro", zero_division=0)
        
        if vmf1 > best_val_mf1:
            best_val_mf1, best_epoch, best_state = vmf1, ep+1, copy.deepcopy(model.state_dict())
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break
    
    model.load_state_dict(best_state); model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            tp.extend(out["logits"].argmax(1).cpu().numpy()); tt.extend(yb.numpy())
    
    tp, tt = np.array(tp), np.array(tt)
    acc = accuracy_score(tt, tp)
    mf1 = f1_score(tt, tp, average="macro", zero_division=0)
    class_f1s = f1_score(tt, tp, average=None, zero_division=0).tolist()
    elapsed = time.time() - start
    
    print(f"    {variant_name:20s} MF1={mf1:.4f} Acc={acc:.4f} best_ep={best_epoch} ({elapsed:.0f}s)")
    return {"variant": variant_name, "accuracy": acc, "macro_f1": mf1, "class_f1s": class_f1s,
            "params": count_parameters(model), "best_epoch": best_epoch, "time_s": elapsed}


def main():
    ds_name = sys.argv[1]
    var_idx = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    data = load_data(ds_name)
    nc, sl = data["num_classes"], data["seq_len"]
    
    variant_names = ["Full", "NoTransport", "NoRegime", "NoUncertainty", "NoGate", "NoHierarchy"]
    
    # Load existing results
    result_path = os.path.join(ROOT, "results", "artnet_ablation", f"{ds_name}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    
    if var_idx >= 0:
        # Run single variant
        name = variant_names[var_idx]
        if name in existing:
            print(f"  {name} already done, skipping")
            return
        model = get_variant(name, nc, sl)
        r = train_and_eval(model, data, name)
        existing[name] = r
        with open(result_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        # Run all missing variants
        for name in variant_names:
            if name in existing:
                print(f"  {name}: MF1={existing[name]['macro_f1']:.4f} (cached)")
                continue
            model = get_variant(name, nc, sl)
            r = train_and_eval(model, data, name)
            existing[name] = r
            with open(result_path, "w") as f:
                json.dump(existing, f, indent=2)
    
    # Print summary
    print(f"\n  ABLATION SUMMARY ({ds_name}):")
    base = existing.get("Full", {}).get("macro_f1", 0)
    for name in variant_names:
        if name in existing:
            d = existing[name]["macro_f1"] - base
            print(f"    {name:20s} MF1={existing[name]['macro_f1']:.4f}  Delta={d:+.4f}")


if __name__ == "__main__":
    main()
