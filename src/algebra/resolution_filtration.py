import torch
import torch.nn as nn

class ResolutionFiltration(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresholds = [0.7, 0.4, 0.0]
        
    def forward(self, delta_t_seq):
        """
        delta_t_seq: Spectral gaps across time, shape (B, T)
        
        Returns:
        features: 7D abstract representation, shape (B, 7)
        """
        B, T = delta_t_seq.shape
        device = delta_t_seq.device
        
        features = []
        
        for thresh in self.thresholds:
            # Build A_x(Delta)
            # Mask is 1 where delta_t > thresh
            mask = (delta_t_seq > thresh).float() # (B, T)
            
            # Vectorized block decomposition (Union-Find along 1D time sequence)
            # Find starts of blocks
            shifted = torch.cat([torch.zeros(B, 1, device=device), mask[:, :-1]], dim=1)
            starts = (mask - shifted) > 0 # (B, T)
            num_blocks = starts.sum(dim=1) # (B,)
            
            # Max block size
            # We can compute sizes of blocks using cumsum and difference at ends
            ends = (shifted - mask) > 0
            # To vectorize max block size without loops, we can use a cumulative sum approach
            # that resets at 0s, but that requires scan.
            # A trick for short sequences (T=16 or 32): 
            # B x T x T upper triangular matrix to find block lengths
            # Actually, since T is small, we can just do a fast convolution or cumsum
            # Let's do a simple scan via matrix mult:
            # L_ij = 1 if i <= j and all mask[i:j+1] == 1
            idx = torch.arange(T, device=device)
            i_idx = idx.unsqueeze(0).unsqueeze(-1) # 1 x T x 1
            j_idx = idx.unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
            
            # mask cumulative product: prod_{k=i}^j mask[k]
            # using log sum trick or just repeated cumprod since T is small
            # Actually, faster:
            # We can find the length of 1s directly
            mask_pad = torch.cat([torch.zeros(B, 1, device=device), mask, torch.zeros(B, 1, device=device)], dim=1)
            diff = mask_pad[:, 1:] - mask_pad[:, :-1]
            # diff has 1 at start, -1 at end
            
            max_block_sizes = []
            for b in range(B):
                st = torch.where(diff[b] == 1)[0]
                ed = torch.where(diff[b] == -1)[0]
                if len(st) > 0:
                    max_len = torch.max(ed - st)
                else:
                    max_len = torch.tensor(0, device=device)
                max_block_sizes.append(max_len)
                
            max_block_size = torch.stack(max_block_sizes).float() # (B,)
            
            features.append(num_blocks.float())
            features.append(max_block_size)
            
        # Algebraic contrast ratio
        # Ratio between high threshold blocks and mid threshold blocks?
        # Let's use ratio of active mask areas: sum(mask_0.7) / (sum(mask_0.4) + 1e-8)
        mask_high = (delta_t_seq > self.thresholds[0]).float().sum(dim=1)
        mask_mid = (delta_t_seq > self.thresholds[1]).float().sum(dim=1)
        contrast = mask_high / (mask_mid + 1e-8)
        features.append(contrast)
        
        # Total: 3 * 2 + 1 = 7 dimensions
        features_tensor = torch.stack(features, dim=1) # (B, 7)
        
        return features_tensor
