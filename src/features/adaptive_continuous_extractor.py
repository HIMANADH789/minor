import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        
    def forward(self, x):
        # x is (B, C, L)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class FiLMResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3)
        self.gelu = nn.GELU()
        
        # Self-conditioning FiLM
        self.film_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels * 2)
        )
        
    def forward(self, x):
        res = x
        
        # Inner activations
        out = self.conv1(x)
        out = self.gelu(out)
        
        # FiLM generation
        film_params = self.film_gen(out) # (B, 2C)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        
        # Apply FiLM
        out = self.conv2(out)
        out = out * gamma + beta
        out = self.gelu(out + res)
        
        return out

class AdaptiveContinuousExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Causal Conv Stem
        self.stem = nn.Sequential(
            CausalConv1d(1, 32, kernel_size=7, dilation=1),
            nn.GELU(),
            CausalConv1d(32, 64, kernel_size=5, dilation=2),
            nn.GELU(),
            CausalConv1d(64, 128, kernel_size=3, dilation=4),
            nn.GELU()
        )
        
        self.proj = nn.Linear(128, 256)
        
        self.res_blocks = nn.ModuleList([
            FiLMResidualBlock(256) for _ in range(4)
        ])
        
        self.kernels = [3, 5, 9, 17, 33]
        self.parallel_convs = nn.ModuleList([
            CausalConv1d(256, 256, kernel_size=k) for k in self.kernels
        ])
        
        # Attention generation for receptive spectrum
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, len(self.kernels)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (B, 140)
        x = x.unsqueeze(1) # (B, 1, 140)
        
        # Stem
        x = self.stem(x) # (B, 128, 140)
        
        # Proj
        x = x.transpose(1, 2) # (B, 140, 128)
        x = self.proj(x)
        x = F.gelu(x)
        x = x.transpose(1, 2) # (B, 256, 140)
        
        # Res blocks with FiLM
        for block in self.res_blocks:
            x = block(x)
            
        # Parallel causal convolutions
        conv_outs = [conv(x).unsqueeze(1) for conv in self.parallel_convs] # List of (B, 1, 256, 140)
        conv_outs = torch.cat(conv_outs, dim=1) # (B, 5, 256, 140)
        
        # Adaptive attention
        attn = self.attention_net(x) # (B, 5)
        attn = attn.view(-1, 5, 1, 1) # (B, 5, 1, 1)
        
        # Weighted combination
        Z = (conv_outs * attn).sum(dim=1) # (B, 256, 140)
        
        return Z
