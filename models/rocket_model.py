import os
import json
import time
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from numba import set_num_threads
from rich.console import Console

console = Console()

def run_minirocket(data_file, output_file):
    # Setup hardware optimizations
    # Numba will use default max threads implicitly
    
    console.print(f"[cyan]Loading dataset from {data_file}...[/cyan]")
    data = np.load(data_file)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Reshape for aeon: (n_cases, n_channels, n_timepoints)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    console.print("[cyan]Initializing MiniRocket (target ~20,000 features)...[/cyan]")
    
    try:
        transformer = MiniRocket(num_features=20000, random_state=42, n_jobs=-1)
    except TypeError:
        # Fallback if num_features is not supported
        console.print("[yellow]Falling back to default MiniRocket params...[/yellow]")
        transformer = MiniRocket(random_state=42, n_jobs=-1)
        
    classifier = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20), class_weight=None)
    
    console.print("[cyan]Transforming train data...[/cyan]")
    t0 = time.time()
    X_train_transform = transformer.fit_transform(X_train)
    t1 = time.time()
    console.print(f"[green]Train transform time: {t1 - t0:.2f} seconds[/green]")
    
    console.print("[cyan]Training Ridge Classifier...[/cyan]")
    t0 = time.time()
    classifier.fit(X_train_transform, y_train)
    t1 = time.time()
    console.print(f"[green]Train classifier time: {t1 - t0:.2f} seconds[/green]")
    
    console.print("[cyan]Transforming test data...[/cyan]")
    t0 = time.time()
    X_test_transform = transformer.transform(X_test)
    t1 = time.time()
    console.print(f"[green]Test transform time: {t1 - t0:.2f} seconds[/green]")
    
    console.print("[cyan]Evaluating...[/cyan]")
    preds = classifier.predict(X_test_transform)
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    recalls = recall_score(y_test, preds, average=None)
    cm = confusion_matrix(y_test, preds)
    
    console.print(f"[bold]Accuracy: {acc:.4f}[/bold]")
    console.print(f"[bold]Macro F1: {macro_f1:.4f}[/bold]")
    
    results = {
        "model": "MiniRocket",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist()
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    console.print(f"[green]Saved results to {output_file}[/green]")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_balanced.npz")
    output_file = os.path.join(base_dir, "results", "balanced", "minirocket_balanced.json")
    
    run_minirocket(data_file, output_file)
