import torch
import numpy as np

# Simulate a good kernel (block diagonal)
N = 500
K_good = torch.zeros((N, N))
for i in range(N):
    for j in range(N):
        if (i < N//2 and j < N//2) or (i >= N//2 and j >= N//2):
            K_good[i, j] = 1.0
        else:
            K_good[i, j] = 0.36

# Simulate a bad kernel (uniform)
K_bad_uniform = torch.full((N, N), 0.36)
K_bad_uniform.fill_diagonal_(1.0)

# Simulate a bad kernel (spiky outliers)
K_bad_spiky = torch.full((N, N), 0.36)
K_bad_spiky.fill_diagonal_(1.0)
for i in range(10):
    K_bad_spiky[i, :] = 0.0
    K_bad_spiky[:, i] = 0.0
    K_bad_spiky[i, i] = 1.0

def get_h(K_mat):
    K_sum = torch.sum(K_mat, dim=1, keepdim=True) + 1e-8
    K_tilde = K_mat / K_sum
    h = -torch.sum(K_tilde * torch.log(K_tilde + 1e-8), dim=1)
    return h.mean().item()

print(f"Good Kernel Entropy: {get_h(K_good)}")
print(f"Bad Uniform Kernel Entropy: {get_h(K_bad_uniform)}")
print(f"Bad Spiky Kernel Entropy: {get_h(K_bad_spiky)}")
