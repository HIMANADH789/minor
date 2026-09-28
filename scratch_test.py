import torch

x = torch.tensor([[1.0, 2.0], [float('nan'), 3.0]])
print(f"max: {x.max()}")
print(f"min: {x.min()}")
