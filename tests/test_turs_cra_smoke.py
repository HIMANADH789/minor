import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.turs_cra.model import TURSCRA, ema_within_window


def test_turs_cra_forward_smoke():
    torch.manual_seed(42)
    model = TURSCRA(in_channels=1, num_classes=5, regime_dim=16, branch_dim=16, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98))
    x = torch.randn(4, 1, 140)
    logits, aux = model(x, return_aux=True)
    assert logits.shape == (4, 5)
    assert aux['beta'].shape[-1] == 3
    assert aux['uncertainty'].shape[-1] == 1


def test_turs_cra_ema_manual():
    z = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).view(1, 2, 3)
    zbar, v = ema_within_window(z, torch.tensor(0.5))
    assert torch.allclose(zbar[:, :, 0], z[:, :, 0])
    assert torch.allclose(v[:, :, 0], torch.zeros_like(v[:, :, 0]))


def test_turs_cra_gradient_and_shape():
    torch.manual_seed(42)
    model = TURSCRA(in_channels=1, num_classes=5, regime_dim=16, branch_dim=16, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98))
    x = torch.randn(4, 1, 140)
    y = torch.randint(0, 5, (4,))
    logits, aux = model(x, return_aux=True)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()
