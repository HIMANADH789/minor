import os
import json
from rich.console import Console

console = Console()

def generate_report(results_dir, output_file):
    rocket_file = os.path.join(results_dir, "rocket_results.json")
    inception_file = os.path.join(results_dir, "inceptiontime_results.json")
    
    results = []
    
    if os.path.exists(rocket_file):
        with open(rocket_file, "r") as f:
            results.append(json.load(f))
            
    if os.path.exists(inception_file):
        with open(inception_file, "r") as f:
            results.append(json.load(f))
            
    if not results:
        console.print("[red]No results found to generate report.[/red]")
        return
        
    report = "# ECG5000 Baseline Benchmark Report\n\n"
    report += "| Model | Accuracy | Macro F1 | Class Recalls (0-4) |\n"
    report += "|-------|----------|----------|---------------------|\n"
    
    for res in results:
        model = res["model"]
        acc = f"{res['accuracy']:.4f}"
        f1 = f"{res['macro_f1']:.4f}"
        recalls = ", ".join([f"{r:.4f}" for r in res["class_recalls"]])
        
        report += f"| {model} | {acc} | {f1} | {recalls} |\n"
        
    with open(output_file, "w") as f:
        f.write(report)
        
    console.print(f"[green]Report generated at {output_file}[/green]")
    console.print("\n" + report)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "results")
    output_file = os.path.join(results_dir, "benchmark_report.md")
    
    generate_report(results_dir, output_file)
