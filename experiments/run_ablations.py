import os
import sys
import torch
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swrst.config import SWRSTConfig
from experiments.train_swrst_ce import train_and_evaluate, verify_dataset
from swrst.diagnostics.benchmark import HardwareProfiler

def main():
    # Example minimal data load
    train_x = torch.randn(100, 1, 140)
    train_y = torch.randint(0, 2, (100,))
    test_x = torch.randn(20, 1, 140)
    test_y = torch.randint(0, 2, (20,))
    
    verify_dataset(train_x, train_y, test_x, test_y)
    
    ablations = [
        {"mode": 0, "name": "Full SWRST-CE"},
        {"mode": 1, "name": "Fixed Locality (No adaptive controller)"},
        {"mode": 2, "name": "Without Novelty Weighting"},
        {"mode": 3, "name": "Without Confidence Weighting"},
        {"mode": 4, "name": "Frequency only (No Time/Energy/Gradient)"}
    ]
    
    results = []
    
    print("Starting SWRST-CE Ablation Study...")
    
    for ab in ablations:
        print(f"\n=============================================")
        print(f"Running Ablation: {ab['name']}")
        print(f"=============================================")
        
        config = SWRSTConfig()
        config.ablation_mode = ab["mode"]
        
        # Override atoms based on mode 4
        if ab["mode"] == 4:
            config.use_time = False
            config.use_energy = False
            config.use_gradient = False
            
        profiler = HardwareProfiler()
        
        try:
            acc = train_and_evaluate(train_x, train_y, test_x, test_y, config, profiler)
            print(f"Result for {ab['name']}: {acc*100:.2f}% Accuracy")
            results.append({"Ablation": ab["name"], "Accuracy": acc})
        except Exception as e:
            print(f"Failed {ab['name']} due to error: {e}")
            results.append({"Ablation": ab["name"], "Accuracy": "ERROR"})
            
    # Export to CSV
    with open("ablation.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Ablation", "Accuracy"])
        writer.writeheader()
        writer.writerows(results)
        
    print("\nAblation study completed. Results exported to ablation.csv")

if __name__ == "__main__":
    main()
