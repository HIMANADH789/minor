import torch


def rbf_kernel_batched(Z1: torch.Tensor, Z2: torch.Tensor,
                       sigma_sq: float, chunk: int = 256) -> torch.Tensor:
    """RBF kernel K(i,j) = exp(-||z1_i - z2_j||^2 / sigma_sq)."""
    N1 = Z1.shape[0]
    K = torch.empty(N1, Z2.shape[0], device=Z1.device, dtype=torch.float32)
    sq2 = (Z2 ** 2).sum(-1, keepdim=True).T                    # (1, N2)
    for i in range(0, N1, chunk):
        z = Z1[i:i + chunk]
        sq1 = (z ** 2).sum(-1, keepdim=True)                    # (chunk, 1)
        dist_sq = sq1 + sq2 - 2.0 * (z @ Z2.T)
        K[i:i + chunk] = torch.exp(-dist_sq / sigma_sq)
    return K


def compute_hellinger_sigma(NU_train: torch.Tensor) -> float:
    """Median heuristic on Hellinger squared distances."""
    sqrt_nu = torch.sqrt(NU_train.clamp(min=1e-10))
    dists = torch.pdist(sqrt_nu)                                # L2 between sqrt histograms
    return dists.pow(2).median().item()


def hellinger_kernel(nu1: torch.Tensor, nu2: torch.Tensor,
                     sigma_sq_global: float) -> torch.Tensor:
    sqrt1 = torch.sqrt(nu1.clamp(min=1e-10))
    sqrt2 = torch.sqrt(nu2.clamp(min=1e-10))
    bc = (sqrt1 @ sqrt2.T).clamp(max=1.0)
    hell_sq = (2.0 - 2.0 * bc).clamp(min=0.0)
    return torch.exp(-hell_sq / sigma_sq_global)


def learn_fusion_weights_grad(K_list, K_y_w, n_steps=300, lr=0.5):
    """
    Replaces grid search (infeasible for >4-5 kernels) with softmax-parameterized
    gradient ascent on the same alignment objective. theta has one entry per
    kernel; alpha = softmax(theta) always lies on the simplex.
    Uses Centered Kernel Alignment (CKA) to prevent all-ones degenerate kernels
    from receiving high unearned alignment scores.
    """
    K_y_c = K_y_w - K_y_w.mean(0, keepdim=True) - K_y_w.mean(1, keepdim=True) + K_y_w.mean()
    Ky_norm = K_y_c.norm()
    
    logs = torch.stack([torch.log(K.clamp(min=1e-30)) for K in K_list])  # (k, N, N)
    theta = torch.zeros(len(K_list), requires_grad=True, device=K_y_w.device)
    opt = torch.optim.Adam([theta], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        alpha = torch.softmax(theta, dim=0)
        log_K = (alpha.view(-1, 1, 1) * logs).sum(0)
        K_fused = torch.exp(torch.clamp(log_K, min=-30.0))
        K_f_c = K_fused - K_fused.mean(0, keepdim=True) - K_fused.mean(1, keepdim=True) + K_fused.mean()
        score = (K_f_c * K_y_c).sum() / (K_f_c.norm() * Ky_norm + 1e-12)
        (-score).backward()
        opt.step()

    return torch.softmax(theta.detach(), dim=0)

def fuse_kernels_v2(Kp: torch.Tensor, Kc: torch.Tensor, Kg: torch.Tensor, Km: torch.Tensor) -> torch.Tensor:
    log_K = (0.50 * torch.log(Kp.clamp(min=1e-30)) +
             0.25 * torch.log(Kc.clamp(min=1e-30)) +
             0.10 * torch.log(Kg.clamp(min=1e-30)) +
             0.15 * torch.log(Km.clamp(min=1e-30)))
    return torch.exp(torch.clamp(log_K, min=-30.0))

def fuse_kernels(Kp: torch.Tensor, Kc: torch.Tensor,
                 Kg: torch.Tensor) -> torch.Tensor:
    """Log-domain weighted fusion."""
    log_K = (0.6 * torch.log(Kp.clamp(min=1e-30))
             + 0.3 * torch.log(Kc.clamp(min=1e-30))
             + 0.1 * torch.log(Kg.clamp(min=1e-30)))
    return torch.exp(log_K.clamp(min=-30.0))

def compute_train_kernel(Z_tr: torch.Tensor, PHI_tr: torch.Tensor,
                         NU_tr: torch.Tensor, sigmas: dict) -> torch.Tensor:
    """Build fused 4000×4000 training kernel."""
    device = Z_tr.device
    use_cuda = device.type == 'cuda'

    if use_cuda:
        # H11: parallel streams
        s1, s2, s3 = (torch.cuda.Stream() for _ in range(3))
        with torch.cuda.stream(s1):
            Kp = rbf_kernel_batched(Z_tr, Z_tr, sigmas['proj'])
        with torch.cuda.stream(s2):
            Kc = rbf_kernel_batched(PHI_tr, PHI_tr, sigmas['cross'])
        with torch.cuda.stream(s3):
            Kg = hellinger_kernel(NU_tr, NU_tr, sigmas['global'])
        torch.cuda.synchronize()
    else:
        Kp = rbf_kernel_batched(Z_tr, Z_tr, sigmas['proj'])
        Kc = rbf_kernel_batched(PHI_tr, PHI_tr, sigmas['cross'])
        Kg = hellinger_kernel(NU_tr, NU_tr, sigmas['global'])

    K = fuse_kernels(Kp, Kc, Kg)
    K = (K + K.T) / 2.0                                          # H12: symmetrise

    # Ensure PSD: add minimal diagonal jitter
    # This is mathematically equivalent to a tiny RBF bandwidth perturbation
    # and preserves the kernel structure unlike eigendecomposition projection
    eigvals_min = torch.linalg.eigvalsh(K)[0].item()
    if eigvals_min < 0:
        jitter = abs(eigvals_min) + 1e-6
        K = K + jitter * torch.eye(K.shape[0], device=K.device)
        # Re-normalise diagonal to ~1.0
        d = torch.sqrt(torch.diag(K).clamp(min=1e-8))
        K = K / (d.unsqueeze(1) * d.unsqueeze(0))

    return K


def compute_test_kernel(Z_te, Z_tr, PHI_te, PHI_tr,
                        NU_te, NU_tr, sigmas, chunk: int = 128) -> torch.Tensor:
    """Build 1000×4000 test kernel in chunks (H14)."""
    N_test = Z_te.shape[0]
    parts = []
    for i in range(0, N_test, chunk):
        Kp = rbf_kernel_batched(Z_te[i:i + chunk], Z_tr, sigmas['proj'])
        Kc = rbf_kernel_batched(PHI_te[i:i + chunk], PHI_tr, sigmas['cross'])
        Kg = hellinger_kernel(NU_te[i:i + chunk], NU_tr, sigmas['global'])
        parts.append(fuse_kernels(Kp, Kc, Kg))
    return torch.cat(parts, dim=0)

def compute_test_kernel_v2(Z_test, Z_train, PHI_test, PHI_train,
                             NU_test, NU_train, MORPH_test, MORPH_train,
                             sigmas, chunk=128):
    N_test = Z_test.shape[0]
    K_chunks = []
    for i in range(0, N_test, chunk):
        sl = slice(i, i+chunk)
        Kp = rbf_kernel_batched(Z_test[sl], Z_train, sigmas['proj'])
        Kc = rbf_kernel_batched(PHI_test[sl], PHI_train, sigmas['cross'])
        Kg = hellinger_kernel(NU_test[sl], NU_train, sigmas['global'])
        Km = rbf_kernel_batched(MORPH_test[sl], MORPH_train, sigmas['morph'])
        K_chunks.append(fuse_kernels_v2(Kp, Kc, Kg, Km))
    return torch.cat(K_chunks, dim=0)
