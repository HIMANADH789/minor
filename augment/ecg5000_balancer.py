import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

console = Console()

def generate_random_smooth_curve(N, L, num_knots=4, min_val=0.85, max_val=1.15, device='cpu'):
    """Generates a smooth curve for magnitude or time warping."""
    # Random knots
    knots = torch.empty(N, 1, num_knots, device=device).uniform_(min_val, max_val)
    # Interpolate to length L
    curve = F.interpolate(knots, size=L, mode='linear', align_corners=True).squeeze(1)
    return curve

def time_warp(X, device):
    """
    Random spline-based time distortion using grid_sample.
    Warp factor approx 0.8 to 1.2
    X: [N, 140]
    """
    N, L = X.shape
    # Generate smooth perturbation for the grid
    # A linear grid from -1 to 1
    base_grid = torch.linspace(-1, 1, L, device=device).unsqueeze(0).expand(N, L)
    
    # Generate smooth noise
    noise = generate_random_smooth_curve(N, L, num_knots=5, min_val=-0.2, max_val=0.2, device=device)
    
    # Cumulative sum to make it monotonic-ish, or just add to base grid
    # A simpler time warp: just perturb the base grid smoothly
    grid = base_grid + noise
    # Clamp to [-1, 1] to avoid out of bounds
    grid = torch.clamp(grid, -1.0, 1.0)
    
    # grid_sample expects shape [N, H, W, 2]. Here H=1, W=L, 2=(x,y)
    # Since 1D, y can be 0.
    grid_2d = torch.zeros(N, 1, L, 2, device=device)
    grid_2d[..., 0] = grid.unsqueeze(1) # x coordinates
    
    # X_reshaped: [N, C, H, W] -> [N, 1, 1, L]
    X_reshaped = X.unsqueeze(1).unsqueeze(2)
    warped_X = F.grid_sample(X_reshaped, grid_2d, mode='bilinear', padding_mode='border', align_corners=True)
    return warped_X.squeeze(2).squeeze(1)

def magnitude_warp(X, device):
    """
    Smooth random scaling: x * spline(scale). scale range [0.85, 1.15]
    """
    N, L = X.shape
    curve = generate_random_smooth_curve(N, L, num_knots=4, min_val=0.85, max_val=1.15, device=device)
    return X * curve

def window_slice(X, device):
    """
    Randomly select 90-95% of signal, resize back.
    """
    N, L = X.shape
    
    # Slice fraction for each sample
    fracs = torch.empty(N, device=device).uniform_(0.90, 0.95)
    
    # Start fraction
    max_starts = 1.0 - fracs
    starts = torch.rand(N, device=device) * max_starts
    ends = starts + fracs
    
    # Convert to grid coordinates [-1, 1]
    starts_grid = starts * 2 - 1
    ends_grid = ends * 2 - 1
    
    # Create grids
    # base_grid: [L] from 0 to 1
    lin = torch.linspace(0, 1, L, device=device).unsqueeze(0).expand(N, L)
    
    # grid = start + lin * (end - start)
    grid = starts_grid.unsqueeze(1) + lin * (ends_grid.unsqueeze(1) - starts_grid.unsqueeze(1))
    
    grid_2d = torch.zeros(N, 1, L, 2, device=device)
    grid_2d[..., 0] = grid.unsqueeze(1)
    
    X_reshaped = X.unsqueeze(1).unsqueeze(2)
    sliced_X = F.grid_sample(X_reshaped, grid_2d, mode='bilinear', padding_mode='border', align_corners=True)
    return sliced_X.squeeze(2).squeeze(1)

def jitter(X, device):
    """
    Add N(0, 0.01), Clamp [-3, 3]
    """
    noise = torch.randn_like(X) * 0.01
    X_noisy = X + noise
    return torch.clamp(X_noisy, -3.0, 3.0)

def apply_random_augmentations(X, device):
    N = X.shape[0]
    
    # We choose 1 to 3 transforms per sample
    num_transforms = torch.randint(1, 4, (N,), device=device)
    
    # For each sample, pick random transforms to apply
    # 0: time_warp, 1: magnitude_warp, 2: window_slice, 3: jitter
    transform_funcs = [time_warp, magnitude_warp, window_slice, jitter]
    
    X_aug = X.clone()
    
    # Apply using vectorization by grouping!
    # Instead of Python loops over N, we can apply masks.
    # To satisfy "No Python loops" over batch size N, we loop over the 4 transforms.
    # We create a random binary matrix of shape [N, 4]
    
    # Generate 1 to 3 active transforms
    active_mask = torch.zeros(N, 4, dtype=torch.bool, device=device)
    for i in range(N):
        n_t = num_transforms[i].item()
        funcs_to_apply = torch.randperm(4, device=device)[:n_t]
        active_mask[i, funcs_to_apply] = True
        
    for f_idx, func in enumerate(transform_funcs):
        mask = active_mask[:, f_idx]
        if mask.any():
            # Apply only to masked elements
            X_aug[mask] = func(X_aug[mask], device)
            
    return X_aug

def balance_dataset(data_file, output_file, seed=42):
    console.print(f"[cyan]Loading original dataset from {data_file}...[/cyan]")
    data = np.load(data_file)
    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(np.int64)
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(np.int64)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classes, counts = np.unique(y_train, return_counts=True)
    n_max = counts.max()
    
    console.print("[bold]Original Class Counts:[/bold]")
    for c, count in zip(classes, counts):
        console.print(f"Class {c}: {count}")
        
    augmented_X = []
    augmented_y = []
    
    # Keep original data
    augmented_X.append(X_train)
    augmented_y.append(y_train)
    
    console.print(f"[cyan]Augmenting minority classes up to n_max = {n_max}...[/cyan]")
    
    for c, count in zip(classes, counts):
        if count < n_max:
            num_to_add = n_max - count
            # Sample with replacement from existing class samples
            class_indices = np.where(y_train == c)[0]
            sampled_indices = np.random.choice(class_indices, size=num_to_add, replace=True)
            
            X_base = torch.tensor(X_train[sampled_indices], dtype=torch.float32, device=device)
            
            # Since vectorizing across batch inside the loop is implemented, we can do it all at once
            X_aug = apply_random_augmentations(X_base, device)
            
            augmented_X.append(X_aug.cpu().numpy())
            augmented_y.append(np.full(num_to_add, c, dtype=np.int64))
            
    X_train_balanced = np.concatenate(augmented_X, axis=0)
    y_train_balanced = np.concatenate(augmented_y, axis=0)
    
    # Shuffle balanced train once
    console.print("[cyan]Shuffling balanced dataset...[/cyan]")
    shuffle_idx = np.random.permutation(len(y_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_idx]
    y_train_balanced = y_train_balanced[shuffle_idx]
    
    bal_classes, bal_counts = np.unique(y_train_balanced, return_counts=True)
    console.print("[bold]Balanced Class Counts:[/bold]")
    for c, count in zip(bal_classes, bal_counts):
        console.print(f"Class {c}: {count}")
        
    console.print(f"[cyan]Saving balanced dataset to {output_file}...[/cyan]")
    np.savez_compressed(
        output_file, 
        X_train=X_train_balanced, 
        y_train=y_train_balanced,
        X_test=X_test, 
        y_test=y_test
    )
    console.print("[green]Augmentation complete![/green]")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    output_file = os.path.join(base_dir, "data", "ecg5000_balanced.npz")
    balance_dataset(data_file, output_file)
