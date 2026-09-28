import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from swrst.config import SWRSTConfig
from swrst.kernel.builder import build_kernel_matrix

class DiskBackedList:
    def __init__(self, chunk_paths, chunk_lengths, max_L, max_F):
        self.chunk_paths = chunk_paths
        self.chunk_lengths = chunk_lengths
        self.max_L = max_L
        self.max_F = max_F
        self._len = sum(chunk_lengths)
        self._current_chunk_idx = -1
        self._current_chunk = []
        self._chunk_starts = []
        c = 0
        for l in chunk_lengths:
            self._chunk_starts.append(c)
            c += l
            
    def __len__(self):
        return self._len
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._len)
            return [self[i] for i in range(start, stop, step)]
        
        chunk_idx = sum(1 for s in self._chunk_starts if s <= idx) - 1
        if chunk_idx != self._current_chunk_idx:
            self._current_chunk = torch.load(self.chunk_paths[chunk_idx], map_location='cpu', weights_only=False)
            self._current_chunk_idx = chunk_idx
        return self._current_chunk[idx - self._chunk_starts[chunk_idx]]
        
    def __iter__(self):
        for p in self.chunk_paths:
            chunk = torch.load(p, map_location='cpu', weights_only=False)
            for item in chunk:
                yield item

def main():
    print("Loading chunks...")
    chunk_paths = [f"c:/temp/ECG_Benchmark/cache/train_seqs_chunk_{i}.pt" for i in range(8)]
    # 512 for chunks 0-6, 416 for chunk 7
    chunk_lengths = [512] * 7 + [416]
    max_L = 15
    max_F = 612
    
    train_seqs = DiskBackedList(chunk_paths, chunk_lengths, max_L, max_F)
    
    print("Extracting final_obs...")
    final_obs = []
    for chunk_path in chunk_paths:
        print(f"Loading {chunk_path}...")
        chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
        for seq in chunk:
            if seq:
                final_obs.append(seq[-1]['transport_plan'].flatten().float())
                
    print(f"Stacking final_obs ({len(final_obs)} elements)...")
    try:
        T_features = torch.stack(final_obs)
        print(f"T_features shape: {T_features.shape}")
    except Exception as e:
        print(f"Error stacking: {e}")
        
    print("Building kernel matrix...")
    config = SWRSTConfig()
    K_train = build_kernel_matrix(train_seqs, config)
    print(f"K_train shape: {K_train.shape}")

if __name__ == "__main__":
    main()
