"""
Gated Transport Network (GTN)

Novel architecture: transport features generate gates that modulate CNN features.
This preserves CNN performance on majority classes while injecting transport awareness
for minority classes.

Key innovations:
  1. Transport-gated CNN: transport features generate soft gates that scale CNN feature maps
  2. Multi-scale transport pyramid: transport features at L, 2L, 4L window sizes
  3. Contrastive transport pre-training: self-supervised learning on transport features
  4. Adaptive class reweighting: dynamic loss based on transport-distance statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Multi-Scale Transport Feature Extractor
# ============================================================

class TransportPyramid(nn.Module):
    """
    Computes transport features (quantile-like) at multiple scales.
    Uses learned projections instead of fixed quantile functions.
    """
    def __init__(self, input_len=140, K=32, num_scales=3):
        super().__init__()
        self.scales = nn.ModuleList()
        for i in range(num_scales):
            scale_len = input_len // (2 ** i)
            self.scales.append(nn.Sequential(
                nn.Linear(scale_len, K),
                nn.LayerNorm(K),
                nn.ReLU(),
            ))
        self.output_dim = K * num_scales

    def forward(self, x):
        B, _, L = x.shape
        features = []
        for i, scale in enumerate(self.scales):
            target_len = max(L // (2 ** i), 4)
            x_scaled = F.adaptive_avg_pool1d(x, target_len)
            x_flat = x_scaled.squeeze(1)
            features.append(scale(x_flat))
        return torch.cat(features, dim=1)


# ============================================================
# Inception Block (proven architecture)
# ============================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=32, kernels=[10, 20, 40]):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        in_conv = bottleneck if in_ch > 1 else 1
        self.convs = nn.ModuleList([
            nn.Conv1d(in_conv, out_ch, k, padding='same', bias=False) for k in kernels
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch * len(kernels) + out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        bx = self.bottleneck(x)
        conv_outs = [c(bx) for c in self.convs]
        pool_out = self.pool_conv(self.maxpool(x))
        return self.relu(self.bn(torch.cat(conv_outs + [pool_out], dim=1)))


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch != out_ch:
            self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            self.bn = nn.BatchNorm1d(out_ch)
        else:
            self.conv = nn.Identity()
            self.bn = nn.Identity()
    def forward(self, x):
        return self.bn(self.conv(x))


# ============================================================
# Transport-Gated CNN
# ============================================================

class TransportGatedCNN(nn.Module):
    """
    CNN where transport features generate gates that modulate feature maps.
    This is the core novel mechanism: transport features control what the CNN sees.
    """
    def __init__(self, in_channels=1, out_channels=32, bottleneck=32,
                 kernels=[10, 20, 40], num_blocks=3, transport_dim=96):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.gates = nn.ModuleList()

        current_in = in_channels
        block_out_dim = out_channels * len(kernels) + out_channels

        for i in range(num_blocks):
            self.blocks.append(InceptionBlock(current_in, out_channels, bottleneck, kernels))

            # Gate: transport features -> sigmoid -> scale CNN features
            self.gates.append(nn.Sequential(
                nn.Linear(transport_dim, block_out_dim),
                nn.Sigmoid(),
            ))

            current_in = block_out_dim
            if i % 3 == 2:
                shortcut_in = in_channels if i == 2 else block_out_dim
                self.shortcuts.append(Shortcut(shortcut_in, current_in))

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, transport_feat):
        """
        x: (B, 1, L) raw signal
        transport_feat: (B, transport_dim) transport features
        """
        shortcut_input = x
        shortcut_idx = 0

        for i, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            # CNN features
            cnn_out = block(x)

            # Transport-gated modulation
            gate_vals = gate(transport_feat).unsqueeze(-1)  # (B, channels, 1)
            gated_out = cnn_out * gate_vals  # element-wise gating

            # Residual connection
            if i % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](shortcut_input)
                gated_out = torch.relu(gated_out + shortcut)
                shortcut_input = gated_out
                shortcut_idx += 1

            x = gated_out

        return self.gap(x).squeeze(-1)  # (B, block_out_dim)


# ============================================================
# Contrastive Transport Loss
# ============================================================

class ContrastiveTransportLoss(nn.Module):
    """
    Self-supervised contrastive loss on transport features.
    Encourages the transport encoder to learn meaningful representations
    by pulling together augmented views of the same signal and pushing apart
    different signals.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (B, D) embeddings from two augmented views
        """
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Similarity matrix
        sim = torch.mm(z1, z2.T) / self.temperature  # (B, B)

        # Labels: diagonal is positive
        labels = torch.arange(B, device=z1.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss


# ============================================================
# Full GTN Model
# ============================================================

class GatedTransportNetwork(nn.Module):
    """
    Gated Transport Network (GTN).

    Architecture:
      Input (B, 1, 140)
        |
        +---> TransportPyramid ---> transport_feat (B, transport_dim)
        |
        +---> TransportGatedCNN(x, transport_feat) ---> cnn_feat (B, cnn_dim)
        |
        +---> ClassificationHead(concat(transport_feat, cnn_feat)) ---> logits (B, C)

    Training:
      Phase 1: Contrastive pre-training on transport features (self-supervised)
      Phase 2: End-to-end fine-tuning with classification + contrastive loss
    """
    def __init__(self, num_classes=5, input_length=140, K=32,
                 num_scales=3, conv_hidden=64, num_conv_blocks=3):
        super().__init__()

        # Transport pyramid
        self.transport = TransportPyramid(input_length, K, num_scales)
        transport_dim = self.transport.output_dim

        # Transport encoder (for contrastive pre-training)
        self.transport_encoder = nn.Sequential(
            nn.Linear(transport_dim, conv_hidden),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.Linear(conv_hidden, conv_hidden),
        )

        # Transport-gated CNN
        self.gated_cnn = TransportGatedCNN(
            in_channels=1, out_channels=32, bottleneck=32,
            kernels=[10, 20, 40], num_blocks=num_conv_blocks,
            transport_dim=transport_dim,
        )

        # Compute CNN output dimension
        cnn_out_dim = 32 * 3 + 32  # out_channels * len(kernels) + out_channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + conv_hidden, conv_hidden),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(conv_hidden, num_classes),
        )

        # Contrastive loss
        self.contrastive_loss = ContrastiveTransportLoss()

    def forward(self, x, return_transport_only=False):
        """
        x: (B, 1, 140)
        """
        # Transport features
        transport_feat = self.transport(x)  # (B, transport_dim)

        # Transport encoder (for contrastive learning)
        z = self.transport_encoder(transport_feat)  # (B, conv_hidden)

        if return_transport_only:
            return z, transport_feat

        # Transport-gated CNN
        cnn_feat = self.gated_cnn(x, transport_feat)  # (B, cnn_out_dim)

        # Classification
        combined = torch.cat([cnn_feat, z], dim=1)
        logits = self.classifier(combined)

        return logits, {
            "transport_feat": transport_feat,
            "transport_embedding": z,
            "cnn_feat": cnn_feat,
        }

    def contrastive_forward(self, x1, x2):
        """For contrastive pre-training: encode two augmented views."""
        z1, _ = self.forward(x1, return_transport_only=True)
        z2, _ = self.forward(x2, return_transport_only=True)
        return z1, z2
