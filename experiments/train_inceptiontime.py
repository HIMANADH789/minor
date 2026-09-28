import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

# Hardware Opts
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

console = Console()

class ECGDataset(Dataset):
    def __init__(self, X, y):
        # Apply per-sample standardization
        eps = 1e-8
        mu = np.mean(X, axis=-1, keepdims=True)
        sigma = np.std(X, axis=-1, keepdims=True)
        X = (X - mu) / (sigma + eps)
        
        # 1D channels_last equivalent: standard contiguous tensors, float32
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_inceptiontime(data_file, output_file, checkpoint_dir):
    from models.inceptiontime import InceptionTime
    
    console.print(f"[cyan]Loading dataset from {data_file}...[/cyan]")
    data = np.load(data_file)
    
    train_dataset = ECGDataset(data['X_train'], data['y_train'])
    test_dataset = ECGDataset(data['X_test'], data['y_test'])
    
    # DataLoader setup
    batch_size = 32
    grad_accum = 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = InceptionTime(num_classes=5, num_blocks=6, kernel_sizes=[10, 20, 40])
    model = model.to(device)
    
    console.print("[cyan]Compiling model (max-autotune)...[/cyan]")
    # torch.compile uses Triton backend which is generally unsupported on Windows.
    # Skipping to avoid runtime crashes during the first forward pass.
    console.print("[yellow]Skipped torch.compile (Triton unsupported on Windows)[/yellow]")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
    
    epochs = 150
    steps_per_epoch = len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=epochs, steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_inceptiontime.pt")
    
    console.print("[cyan]Starting Training...[/cyan]")
    
    total_train_time = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True)
        
        epoch_start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y) / grad_accum
                
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
            epoch_loss += loss.item() * grad_accum
            
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        avg_val_loss = val_loss / len(test_loader)
        
        acc = accuracy_score(all_targets, all_preds)
        
        console.print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Time: {epoch_time:.2f}s")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
            
    console.print(f"[green]Training complete. Total time: {total_train_time:.2f}s[/green]")
    
    # Final Evaluation on best model
    console.print("[cyan]Evaluating Best Model...[/cyan]")
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    final_acc = accuracy_score(all_targets, all_preds)
    final_macro_f1 = f1_score(all_targets, all_preds, average='macro')
    final_recalls = recall_score(all_targets, all_preds, average=None)
    cm = confusion_matrix(all_targets, all_preds)
    
    console.print(f"[bold]Final Accuracy: {final_acc:.4f}[/bold]")
    console.print(f"[bold]Final Macro F1: {final_macro_f1:.4f}[/bold]")
    
    results = {
        "model": "InceptionTime",
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "class_recalls": [float(r) for r in final_recalls],
        "confusion_matrix": cm.tolist()
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    console.print(f"[green]Saved results to {output_file}[/green]")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_balanced.npz")
    output_file = os.path.join(base_dir, "results", "balanced", "inceptiontime_balanced.json")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    train_inceptiontime(data_file, output_file, checkpoint_dir)
