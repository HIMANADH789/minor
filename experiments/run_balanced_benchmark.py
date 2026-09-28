import os
import sys
import json
import subprocess
from rich.console import Console

console = Console()

def run_balanced_benchmark():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    python_exe = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    
    # Ensure results/balanced exists
    results_dir = os.path.join(base_dir, "results", "balanced")
    os.makedirs(results_dir, exist_ok=True)
    
    # If python execution is needed directly (like python -m), we use that:
    # but the paths here are full paths so we can just pass them.
    # Note: swrst uses python -m src.swrst_v10_5.experiments.train_swrst_v10_5 for correct imports
    # so we should use -m for swrst.
    
    scripts = [
        ("SWRST-v10.5", ["-m", "src.swrst_v10_5.experiments.train_swrst_v10_5"]),
        ("MiniRocket", [os.path.join("models", "rocket_model.py")]),
        ("InceptionTime", [os.path.join("experiments", "train_inceptiontime.py")])
    ]
    
    for name, args in scripts:
        console.print(f"[bold cyan]Running {name}...[/bold cyan]")
        try:
            # Run the subprocess, wait for it to complete
            cmd = [python_exe] + args
            result = subprocess.run(cmd, cwd=base_dir, check=True)
            console.print(f"[bold green]{name} completed successfully.[/bold green]\n")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running {name}.[/bold red]")
            # We continue even if one fails, to get partial results
    
    console.print("[bold cyan]Aggregating Results...[/bold cyan]")
    
    summary = {}
    
    # Paths to output json files
    result_files = {
        "SWRST-v10.5": os.path.join(results_dir, "swrst_v10_5_balanced.json"),
        "MiniRocket": os.path.join(results_dir, "minirocket_balanced.json"),
        "InceptionTime": os.path.join(results_dir, "inceptiontime_balanced.json")
    }
    
    for name, filepath in result_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Normalize keys since different models might have slightly different keys
            accuracy = data.get("accuracy", data.get("Test_Accuracy"))
            macro_f1 = data.get("macro_f1", data.get("Test_Macro_F1"))
            per_class_recall = data.get("class_recalls", data.get("Per_Class_Recall"))
            cm = data.get("confusion_matrix", data.get("Confusion_Matrix"))
            
            summary[name] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class_recall": per_class_recall,
                "confusion_matrix": cm
            }
        else:
            summary[name] = "Failed or results missing."
            
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    console.print(f"[bold green]Summary saved to {summary_path}[/bold green]")
    console.print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    run_balanced_benchmark()
