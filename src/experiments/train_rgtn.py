import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.rgtn import RGTN
from src.losses.rgtn_loss import RGTNLoss
from loaders.ecg5000_loader import get_dataloaders
import time

def train_rgtn():
    # Hardware Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Strict ECG5000 split (4000 train, 1000 test if not 4500 test). 
    # Usually standard is 500 train, 4500 test. But user said "Train = 4000, Test = 1000. Same stratified split as baseline."
    # The dataloader handles this.
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=32,
        train_size=4000, # Use the resplit specified
        val_size=1000,
        augment=False,
        balanced=False
    )
    
    # Hardware G: DataLoader pin_memory etc is handled in loader usually, 
    # but we will just pass it if possible, or assume it's in the loader.
    
    model = RGTN(in_features=1, num_classes=5).to(device)
    # Extractor compilation is unsupported on this Windows environment due to missing Triton
    # model.extractor = torch.compile(model.extractor)
    
    criterion = RGTNLoss().to(device)
    
    # Initial Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=150
    )
    
    best_f1 = 0.0
    patience = 25
    epochs_no_improve = 0
    geodesic_initialized = False
    
    print("Starting pure RGTN training (3-Phase Curriculum)...")
    
    start_epoch = 1
    import os
    if os.path.exists('archive/best_rgtn.pt'):
        print("Found checkpoint! Resuming...")
        checkpoint = torch.load('archive/best_rgtn.pt', map_location=device)
        
        # If it's a full state dict with epoch/optimizer
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            print(f"Successfully fully resumed from Epoch {start_epoch-1}")
        else:
            # It's just a raw model state_dict from the old code
            model.load_state_dict(checkpoint)
            start_epoch = 56 # Hardcoded resume as per user instruction
            best_f1 = 0.38 # Approximate from logs
            print(f"Loaded raw model state. Hard-resuming from Epoch {start_epoch-1}")
            
            # Fast-forward scheduler to Epoch 55 end
            for _ in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
    
    for epoch in range(start_epoch, 151):
        # ---------------------------------------------------------
        # Phase Logistics
        # ---------------------------------------------------------
        if epoch <= 30:
            # Phase 1: CE only
            lw = {'w_geom': 0.0, 'w_persist': 0.0, 'w_eps': 0.0, 'w_occ': 0.0, 'w_spec': 0.0, 'w_sep': 0.0}
        elif epoch <= 60:
            # Phase 2: CE + light geometry
            lw = {'w_geom': 0.01, 'w_persist': 0.001, 'w_eps': 0.0, 'w_occ': 0.0, 'w_spec': 0.0, 'w_sep': 0.0}
        else:
            # Phase 3: Geodesic Finetuning
            if not geodesic_initialized:
                # Hot-swap classifier and recreate optimizer with differential LR
                classifier_params = model.initialize_geodesic_finetuning(train_loader, device)
                
                # Unfreeze backbone but use lower LR
                for param in model.parameters():
                    param.requires_grad = True
                    
                optimizer = optim.AdamW([
                    {'params': model.extractor.parameters(), 'lr': 1e-5},
                    {'params': [p for n, p in model.named_parameters() if not n.startswith('extractor.') and not n.startswith('classifier.')], 'lr': 1e-5},
                    {'params': model.classifier.parameters(), 'lr': 1e-4}
                ], weight_decay=1e-2)
                
                # Reset scheduler for the remaining 90 epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
                geodesic_initialized = True
                
            lw = {'w_geom': 0.01, 'w_persist': 0.001, 'w_eps': 0.005, 'w_occ': 0.001, 'w_spec': 0.001, 'w_sep': 0.0005}
            
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch:03d} | Batch {batch_idx}/{len(train_loader)}...")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward
            logits, intermediates = model(inputs, targets=targets)
            
            # Loss
            loss, loss_dict = criterion(
                logits, targets, 
                K_batch=intermediates.get('K_batch'),
                Pi_seq=intermediates.get('Pi_seq'),
                eps_seq=intermediates.get('eps_seq'),
                K_t_seq=intermediates.get('K_t_seq'),
                evals_x=intermediates.get('evals_x'),
                **lw
            )
            
            # Backward
            loss.backward()
            
            # Grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if epoch <= 60:
                scheduler.step()
            
            train_loss += loss.item()
            
        if epoch > 60:
            scheduler.step()
            
        train_loss /= len(train_loader)
        
        # Eval
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, intermediates = model(inputs)
                
                loss, _ = criterion(logits, targets, **lw)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Macro F1 manual calc to avoid sklearn dependency
        classes = list(set(all_targets))
        f1s = []
        for c in classes:
            tp = sum((p == c and t == c) for p, t in zip(all_preds, all_targets))
            fp = sum((p == c and t != c) for p, t in zip(all_preds, all_targets))
            fn = sum((p != c and t == c) for p, t in zip(all_preds, all_targets))
            if tp + fp == 0 or tp + fn == 0:
                f1s.append(0.0)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1s.append(2 * precision * recall / (precision + recall + 1e-8))
        macro_f1 = sum(f1s) / len(classes) if classes else 0
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {macro_f1:.4f} | Time: {epoch_time:.2f}s")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            
            # Save full state dicts for proper resuming!
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_f1': best_f1
            }, 'archive/best_rgtn.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch > 60:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == '__main__':
    train_rgtn()
