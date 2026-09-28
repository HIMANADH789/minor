import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.net(x))

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        # x: (B, C, L)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class PhaseCanonicalizer(nn.Module):
    def __init__(self, in_features=1, dim=256, out_dim=64):
        super().__init__()
        
        # We assume input is (B, 140) or (B, 1, 140).
        # Projects 1 feature to 256 for each timestep.
        self.proj = nn.Sequential(
            nn.Linear(in_features, dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(dim) for _ in range(4)])
        
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2)
        )
        
        self.kernels = [3, 5, 9, 17, 33]
        self.convs = nn.ModuleList([
            CausalConv1d(dim, out_dim, k) for k in self.kernels
        ])
        
        self.attn_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, len(self.kernels))
        )
        
    def forward(self, x):
        # x: (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, 140, 1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2) # (B, 140, 1)
            
        # x is now (B, L, 1)
        # Linear projection
        h = self.proj(x) # (B, L, 256)
        
        # Residual Blocks
        for block in self.res_blocks:
            h = block(h)
            
        # FiLM Conditioning
        # Pool across time to generate conditioning
        h_pool = h.mean(dim=1) # (B, 256)
        film_params = self.film_gen(h_pool) # (B, 512)
        gamma, beta = film_params.chunk(2, dim=-1) # (B, 256) each
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        h = h * (1 + gamma) + beta # FiLM
        
        # Attention weighting
        attn_logits = self.attn_net(h_pool) # (B, 5)
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, 5)
        
        # Parallel Causal Convs
        h_conv_in = h.transpose(1, 2) # (B, 256, L)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(h_conv_in)) # each is (B, 64, L)
            
        conv_outputs = torch.stack(conv_outputs, dim=1) # (B, 5, 64, L)
        
        # Apply attention
        attn_weights = attn_weights.view(-1, 5, 1, 1)
        z = (conv_outputs * attn_weights).sum(dim=1) # (B, 64, L)
        
        return z.transpose(1, 2) # (B, L, 64)
