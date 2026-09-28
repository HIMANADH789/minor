import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=[10, 20, 40]):
        super().__init__()
        # Bottleneck reduces dimensionality
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        ) if in_channels > 1 else nn.Identity()
        
        # Parallel convolutions
        in_conv = bottleneck_channels if in_channels > 1 else 1
        self.convs = nn.ModuleList([
            nn.Conv1d(in_conv, out_channels, kernel_size=k, padding="same", bias=False)
            for k in kernel_sizes
        ])
        
        # Max pool parallel branch
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes) + out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck_x = self.bottleneck(x)
        
        conv_outputs = [conv(bottleneck_x) for conv in self.convs]
        
        pool_x = self.max_pool(x)
        pool_output = self.pool_conv(pool_x)
        
        concatenated = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.relu(self.bn(concatenated))

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.conv = nn.Identity()
            self.bn = nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

class InceptionTime(nn.Module):
    def __init__(self, num_classes=5, num_blocks=6, in_channels=1, out_channels=32, bottleneck_channels=32, kernel_sizes=[10, 20, 40]):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        current_in_channels = in_channels
        
        for i in range(num_blocks):
            self.blocks.append(
                InceptionModule(
                    current_in_channels, out_channels, bottleneck_channels, kernel_sizes
                )
            )
            current_in_channels = out_channels * len(kernel_sizes) + out_channels
            
            # Residual connection every 3 blocks
            if i % 3 == 2:
                shortcut_in = in_channels if i == 2 else (out_channels * len(kernel_sizes) + out_channels)
                self.shortcuts.append(Shortcut(shortcut_in, current_in_channels))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_in_channels, num_classes)

    def forward(self, x):
        shortcut_input = x
        shortcut_idx = 0
        
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            
            if i % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](shortcut_input)
                x = torch.relu(x + shortcut)
                shortcut_input = x
                shortcut_idx += 1
                
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
