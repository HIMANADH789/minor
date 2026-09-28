import os
import sys
import torch
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.ecg5000_loader import get_dataloaders
from swrst.config import SWRSTConfig
from swrst.model import SWRSTModel
from swrst.kernel.builder import build_kernel_matrix
from swrst.diagnostics.benchmark import HardwareProfiler

def profile_run():
    print("Loading ECG5000 dataset for profiling (100 samples)...")
    train_loader, _, _ = get_dataloaders(batch_size=32)
    
    # Take just 100 samples
    train_x = train_loader.dataset.tensors[0][:100]
    train_y = train_loader.dataset.tensors[1][:100]
    
    config = SWRSTConfig()
    model = SWRSTModel(config).cuda() if torch.cuda.is_available() else SWRSTModel(config)
    profiler = HardwareProfiler()
    
    # Manual coarse profiling loops to get fine-grained metrics
    # Note: HardwareProfiler only tracks what we explicitly wrap in start/stop
    
    print("Running Profiling...")
    
    profiler.start("total_runtime")
    
    profiler.start("representation_and_transport")
    B_train = train_x.shape[0]
    train_seqs = []
    batch_size = config.batch_size
    
    for i in range(0, B_train, batch_size):
        x_batch = train_x[i:i+batch_size].cuda() if torch.cuda.is_available() else train_x[i:i+batch_size]
        seqs = model(x_batch)
        train_seqs.extend(seqs)
        
    profiler.stop("representation_and_transport")
    
    profiler.start("kernel_construction")
    K_train = build_kernel_matrix(train_seqs, config)
    profiler.stop("kernel_construction")
    
    profiler.stop("total_runtime")
    
    # We will export to profiling.json
    profiler.export("profiling.json")
    print("Profiling complete. Saved to profiling.json")

if __name__ == "__main__":
    profile_run()
