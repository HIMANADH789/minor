"""
External Baselines: ResNet-1D, FCN, PatchTST-Cls
Standard architectures for time-series classification comparison.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. ResNet-1D
# ============================================================

class ResBlock1D(nn.Module):
    """Pre-activation ResNet block for 1D time series."""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same', bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class ResNet1D(nn.Module):
    """
    ResNet-1D for time-series classification.
    
    Architecture:
        Input -> [Conv+BN+ReLU] ->
        Stage1: 3x ResBlock1D(64, k=8) + AvgPool ->
        Stage2: 3x ResBlock1D(128, k=5) + AvgPool ->
        Stage3: 3x ResBlock1D(128, k=3) + AvgPool ->
        GAP -> FC -> Classes
    
    ~150K params for 1-channel input.
    """
    def __init__(self, in_channels=1, num_classes=5, channels=(32, 64, 64),
                 blocks=(2, 2, 2), kernels=(8, 5, 3), dropout=0.1):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, channels[0], 7, padding='same', bias=False)
        self.bn_in = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        # Residual stages
        self.stages = nn.ModuleList()
        self.pool = nn.ModuleList()
        ci = channels[0]
        for ch, n_blocks, k in zip(channels, blocks, kernels):
            stage = nn.Sequential(*[ResBlock1D(ch, k, dropout) for _ in range(n_blocks)])
            self.stages.append(stage)
            # Pool between stages (not after last)
            if ch != ci:
                self.pool.append(nn.Sequential(
                    nn.Conv1d(ci, ch, 1, bias=False),
                    nn.BatchNorm1d(ch)
                ))
            else:
                self.pool.append(nn.Identity())
            ci = ch
        
        # After last stage
        self.bn_out = nn.BatchNorm1d(channels[-1])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        
        # Weight init
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (B, C_in, L)
        out = self.relu(self.bn_in(self.conv_in(x)))
        
        for stage, proj in zip(self.stages, self.pool):
            out = proj(out)  # project channels if needed
            out = stage(out)
        
        out = self.relu(self.bn_out(out))
        out = self.gap(out).squeeze(-1)
        return self.fc(out)


# ============================================================
# 2. FCN (Fully Convolutional Network)
# ============================================================

class FCN(nn.Module):
    """
    Fully Convolutional Network for time-series classification.
    
    Architecture (Zhang et al., 2017):
        Input ->
        Conv1D(128, k=8) + BN + ReLU ->
        Conv1D(256, k=5) + BN + ReLU ->
        Conv1D(128, k=3) + BN + ReLU ->
        GAP -> FC -> Classes
    
    ~150K params for 1-channel input.
    """
    def __init__(self, in_channels=1, num_classes=5,
                 filters=(64, 128, 64), kernels=(8, 5, 3), dropout=0.0):
        super().__init__()
        
        layers = []
        ci = in_channels
        for ch, k in zip(filters, kernels):
            layers.extend([
                nn.Conv1d(ci, ch, k, padding='same', bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            ci = ch
        
        self.conv_block = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1], num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        out = self.conv_block(x)
        out = self.gap(out).squeeze(-1)
        out = self.dropout(out)
        return self.fc(out)


# ============================================================
# 3. PatchTST-Cls
# ============================================================

class PatchEmbedding(nn.Module):
    """Extract patches from 1D time series and project to embedding dimension."""
    def __init__(self, in_channels, d_model, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Conv1d to extract and embed patches simultaneously
        self.proj = nn.Conv1d(in_channels, d_model, patch_len, stride=stride, bias=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, C, L) -> (B, d_model, n_patches)
        x = self.proj(x)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)
        x = self.norm(x)
        return x


class PositionalEncoding1D(nn.Module):
    """Learnable + sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Sinusoidal (fixed)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        # Learnable component
        self.learnable = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len] + self.learnable[:, :seq_len])


class PatchTSTCls(nn.Module):
    """
    PatchTST for Classification.
    
    Architecture:
        Input (B, 1, L) ->
        Patch Embedding (patch_len, stride) ->
        Positional Encoding ->
        Transformer Encoder (d_model, n_heads, n_layers, ffn_dim) ->
        Global Average Pooling over patches ->
        Classification Head -> Classes
    
    Config (compact):
        patch_len=16, stride=8, d_model=64, heads=4,
        layers=2, ffn=128, dropout=0.1
    """
    def __init__(self, in_channels=1, num_classes=5,
                 patch_len=16, stride=8,
                 d_model=64, n_heads=4, n_layers=2,
                 ffn_dim=128, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_len, stride)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding1D(d_model, dropout=dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (B, 1, L)
        B = x.size(0)
        
        # Patch embedding: (B, n_patches, d_model)
        patches = self.patch_embed(x)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls, patches], dim=1)
        
        # Positional encoding
        x = self.pos_enc(patches)
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        x = self.norm(x[:, 0])
        
        return self.head(x)


# ============================================================
# Utility
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def verify_model(model, name, in_channels=1, seq_len=140, num_classes=5):
    """Verify model forward/backward pass, shapes, and parameter count."""
    model.eval()
    x = torch.randn(4, in_channels, seq_len)
    
    # Forward
    with torch.no_grad():
        out = model(x)
    
    assert out.shape == (4, num_classes), f"{name}: output shape {out.shape} != (4, {num_classes})"
    assert not torch.isnan(out).any(), f"{name}: NaN in output"
    assert not torch.isinf(out).any(), f"{name}: Inf in output"
    
    # Backward
    model.train()
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # Check all params have gradients
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"{name}: params without gradient: {no_grad}"
    
    params = count_params(model)
    print(f"  {name}: OK  output={out.shape}  params={params:,}")
    return params


if __name__ == "__main__":
    print("Verifying external baselines...")
    
    # ECG5000: L=140, 5 classes
    print("\n--- ECG5000 (L=140, C=5) ---")
    verify_model(ResNet1D(1, 5), "ResNet1D", 1, 140, 5)
    verify_model(FCN(1, 5), "FCN", 1, 140, 5)
    verify_model(PatchTSTCls(1, 5, patch_len=16, stride=8), "PatchTST-Cls", 1, 140, 5)
    
    # CWRU: L=1024, 4 classes
    print("\n--- CWRU (L=1024, C=4) ---")
    verify_model(ResNet1D(1, 4), "ResNet1D", 1, 1024, 4)
    verify_model(FCN(1, 4), "FCN", 1, 1024, 4)
    verify_model(PatchTSTCls(1, 4, patch_len=16, stride=8), "PatchTST-Cls", 1, 1024, 4)
    
    print("\nAll models verified successfully!")
