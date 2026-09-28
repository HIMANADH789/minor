import os
import sys
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swrst.config import SWRSTConfig
from experiments.train_swrst_ce import train_and_evaluate, verify_dataset
from swrst.diagnostics.benchmark import HardwareProfiler

def main():
    print("Running Scientific Audit for SWRST-CE...")
    
    # Example minimal data load
    train_x = torch.randn(50, 1, 140)
    train_y = torch.randint(0, 2, (50,))
    test_x = torch.randn(10, 1, 140)
    test_y = torch.randint(0, 2, (10,))
    
    verify_dataset(train_x, train_y, test_x, test_y)
    
    config = SWRSTConfig()
    profiler = HardwareProfiler()
    
    train_and_evaluate(train_x, train_y, test_x, test_y, config, profiler)
    
    print("Scientific Audit complete.")
    print("Please review representation_flow.json for stage-by-stage information preservation metrics.")
    
if __name__ == "__main__":
    main()
