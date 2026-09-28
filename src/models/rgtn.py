import torch
import torch.nn as nn
from src.utils.safe_eigh import safe_eigh

from src.features.phase_canonicalizer import PhaseCanonicalizer
from src.features.path_tensor import PathTensor
from src.transport.adaptive_cost import AdaptiveCostGeometry
from src.transport.entropic_resolution_flow import EntropicResolutionFlow
from src.transport.continuous_resolution_integral import ContinuousResolutionIntegral
from src.geometry.spectral_gap import SpectralGapResolutionState
from src.manifolds.signal_barycenter import SignalBarycenter
from src.manifolds.residual_manifold import ResidualManifold
from src.operators.adaptive_ptco import AdaptivePTCO
from src.operators.resolution_curvature import ResolutionCurvature
from src.operators.persistence_energy import PersistenceEnergy
from src.algebra.resolution_filtration import ResolutionFiltration
from src.operators.cp_abstraction_flow import CPAbstractionFlow
from src.classifiers.geodesic_prototype_classifier import GeodesicPrototypeClassifier
from src.kernels.product_kernel import ProductKernel

class RGTN(nn.Module):
    def __init__(self, in_features=1, num_classes=5):
        super().__init__()
        
        self.extractor = PhaseCanonicalizer(in_features=in_features)
        
        self.erf = EntropicResolutionFlow()
        self.cost_geom = AdaptiveCostGeometry()
        self.cri = ContinuousResolutionIntegral()
        
        self.spec_gap = SpectralGapResolutionState()
        self.sig_barycenter = SignalBarycenter()
        self.res_manifold = ResidualManifold()
        
        self.ptco = AdaptivePTCO()
        self.curvature = ResolutionCurvature()
        self.persistence = PersistenceEnergy()
        
        self.algebra = ResolutionFiltration()
        self.cp_flow = CPAbstractionFlow()
        
        self.path_tensor = PathTensor()
        self.classifier = nn.Linear(208, num_classes) # Pretraining Head
        self.kernel = ProductKernel()
        
        # Hyperparameters for sliding window
        self.window_size = 20
        self.stride = 8
        
    def initialize_geodesic_finetuning(self, dataloader, device):
        """
        Phase 4: Freeze backbone, swap to GeodesicPrototypeClassifier.
        Initializes prototypes M_c from the class-conditional mean of G_x.
        """
        print("Initializing Geodesic Finetuning...")
        
        # Freeze backbone
        for param in self.parameters():
            param.requires_grad = False
            
        # Hot-swap classifier
        num_classes = self.classifier.out_features
        self.classifier = GeodesicPrototypeClassifier(in_features=208, num_classes=num_classes).to(device)
        
        # We need to compute M_c
        self.eval()
        class_log_sums = {c: 0 for c in range(num_classes)}
        class_counts = {c: 0 for c in range(num_classes)}
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, intermediates = self.forward(inputs)
                
                phi = intermediates['phi']
                # G_x = phi phi^T + ...
                D = phi.size(-1)
                G_x = phi.unsqueeze(-1) @ phi.unsqueeze(-2) + \
                      (phi.unsqueeze(-1) @ phi.unsqueeze(-2)).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / D * torch.eye(D, device=device)
                
                for i in range(targets.size(0)):
                    c = targets[i].item()
                    
                    # Normalize each SPD to remove magnitude bias
                    G_curr = G_x[i]
                    tr_G = torch.trace(G_curr)
                    tau = 0.02
                    G_tilde = G_curr / (tr_G + tau) + tau * torch.eye(D, device=device)
                    
                    # Log mapping via safe_eigh
                    evals, evecs = safe_eigh(G_tilde.float())
                    log_evals = torch.log(evals.clamp(min=1e-8)).unsqueeze(-1)
                    log_G = torch.matmul(evecs * log_evals.transpose(-2, -1), evecs.transpose(-2, -1)).to(G_curr.dtype)
                    
                    class_log_sums[c] = class_log_sums[c] + log_G
                    class_counts[c] += 1
                    
        # Set M_c using expm of the log mean
        for c in range(num_classes):
            if class_counts[c] > 0:
                mean_log = class_log_sums[c] / class_counts[c]
                evals, evecs = safe_eigh(mean_log.float())
                exp_evals = torch.exp(evals).unsqueeze(-1)
                M_c_new = torch.matmul(evecs * exp_evals.transpose(-2, -1), evecs.transpose(-2, -1)).to(mean_log.dtype)
                self.classifier.M_c.data[c] = M_c_new
                
        self.train()
        print("Geodesic Finetuning initialized.")
        return self.classifier.parameters()

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, targets=None):
        B = x.size(0)
        device = x.device
        
        # 1. Feature Extraction with Checkpointing (Use AMP here for speed)
        with torch.amp.autocast('cuda', enabled=True):
            if self.training:
                import torch.utils.checkpoint as checkpoint
                Z = checkpoint.checkpoint(self.extractor, x, use_reentrant=False)
            else:
                Z = self.extractor(x)
                
        # Cast Z to float32 for all downstream geometric operations!
        Z = Z.float()
        
        # Sliding window over Z
        T = (Z.size(1) - self.window_size) // self.stride + 1
        windows = []
        for t in range(T):
            w = Z[:, t*self.stride : t*self.stride + self.window_size, :]
            windows.append(w)
        windows = torch.stack(windows, dim=1) # (B, T, K, 64)
        
        K_win = windows.size(2)
        D_feat = windows.size(3)
        windows_flat = windows.view(B * T, K_win, D_feat)
        
        # ==========================================
        # PASS 1: Base Transport (eps_0)
        # ==========================================
        eps_0_flat = torch.full((B * T, 1), self.erf.eps_0, device=device)
        eps_k_0_flat = eps_0_flat.view(B * T, 1, 1, 1) + torch.tensor([-0.02, 0.0, 0.02], device=device).view(1, 3, 1, 1)
        
        C_k_flat = self.cost_geom(windows_flat, eps_k_0_flat)
        S_bar_0, _, evals_0, _, log_S_c_0, _, _ = self.cri(C_k_flat, eps_k_0_flat)
        
        evals_0_seq = evals_0.view(B, T, -1)
        
        # Compute H_t (Entropy of base transport)
        eval_probs_0 = evals_0_seq / (evals_0_seq.sum(dim=-1, keepdim=True) + 1e-8)
        H_t_seq = -torch.sum(eval_probs_0 * torch.log(eval_probs_0.clamp(min=1e-8)), dim=-1) # (B, T)
        H_t_seq = torch.nan_to_num(H_t_seq, nan=0.0)
        
        # Compute Spectral Gap & Curvature for Pass 1
        delta_seq = []
        d1, d2 = None, None
        for t in range(T):
            d, _, _ = self.spec_gap(evals_0_seq[:, t], d1, d2)
            d2 = d1
            d1 = d
            delta_seq.append(d)
        delta_seq = torch.stack(delta_seq, dim=1).squeeze(-1) # (B, T)
        
        # Simple delta derivative for dynamic temperature
        d_delta_seq = torch.zeros_like(delta_seq)
        d_delta_seq[:, 1:] = delta_seq[:, 1:] - delta_seq[:, :-1]
        
        # Dynamic Temperature S_t
        # S_t = H_t + lambda |Pi_t| (approx) + gamma kappa_t
        # For pure parallel, we use H_t and delta_seq variations
        S_t_seq = H_t_seq + 0.5 * torch.abs(d_delta_seq)
        
        # ==========================================
        # PASS 2: Dynamic Transport (eps_t)
        # ==========================================
        # Patch A: Sigmoid ERF
        eps_t_seq = self.erf.eps_min + (self.erf.eps_0 - self.erf.eps_min) * torch.sigmoid(-self.erf.alpha * S_t_seq)
        # Patch E: Sinkhorn gradient stop on eps path
        eps_t_seq = eps_t_seq.detach()
        
        eps_t_flat = eps_t_seq.view(B * T, 1)
        eps_k_t_flat = eps_t_flat.view(B * T, 1, 1, 1) + torch.tensor([-0.02, 0.0, 0.02], device=device).view(1, 3, 1, 1)
        
        # H_t passed into CRI for entropy-adaptive weights
        H_t_flat = H_t_seq.view(B * T, 1)
        S_bar_flat, _, evals_flat, evecs_flat, _, log_S_bar_flat, P_c_flat = self.cri(C_k_flat, eps_k_t_flat, H_t=H_t_flat)
        
        K_dim_spd = S_bar_flat.size(-1)
        S_bar_seq = S_bar_flat.view(B, T, K_dim_spd, K_dim_spd)
        log_S_bar_seq = log_S_bar_flat.view(B, T, K_dim_spd, K_dim_spd)
        central_evals_seq = evals_flat.view(B, T, K_dim_spd)
        central_evecs_seq = evecs_flat.view(B, T, K_dim_spd, K_dim_spd)
        P_t_seq = P_c_flat.view(B, T, K_win, K_win)
        
        eps_seq = eps_t_seq # (B, T)
        
        # Re-compute exact delta derivatives for downstream operators
        d_delta_seq_final, d2_delta_seq_final = [], []
        d1, d2 = None, None
        for t in range(T):
            _, dd1, dd2 = self.spec_gap(central_evals_seq[:, t], d1, d2)
            d2 = d1
            d1 = delta_seq[:, t].unsqueeze(-1)
            d_delta_seq_final.append(dd1)
            d2_delta_seq_final.append(dd2)
            
        d_delta_seq = torch.stack(d_delta_seq_final, dim=1).squeeze(-1)
        d2_delta_seq = torch.stack(d2_delta_seq_final, dim=1).squeeze(-1)
        
        # Curvature
        K_t_seq = self.curvature(delta_seq, 
                                 torch.cat([torch.zeros(B, 1, device=device), delta_seq[:, :-1]], dim=1),
                                 torch.cat([torch.zeros(B, 2, device=device), delta_seq[:, :-2]], dim=1))
        
        # Signal Barycenter
        S_bar_x, log_S_bar_x, evals_x, evecs_x = self.sig_barycenter(log_S_bar_seq, eps_seq.unsqueeze(-1))
        
        # Residual Manifold
        R_hat_seq = self.res_manifold(log_S_bar_seq, log_S_bar_x)
        
        # PTCO
        M_t_seq, _, p_t_seq = self.ptco(S_bar_x, R_hat_seq, central_evals_seq, K_dim_spd, evals_x, evecs_x, delta_seq=delta_seq)
        
        # Persistence Energy
        E_seq, Pi_seq = self.persistence(M_t_seq, P_t_seq=P_t_seq)
        
        # Filtration
        algebra_feats = self.algebra(delta_seq)
        
        # CP Flow
        eps_0_batch = torch.full((B, T, 1, 1), self.cost_geom.eps_0, device=device)
        C_t_seq = self.cost_geom(windows.view(-1, K_win, D_feat), eps_0_batch.view(-1, 1, 1)).view(B, T, K_win, K_win)
        qtcf_feats, cp_feats = self.cp_flow(C_t_seq, central_evals_seq, central_evecs_seq, H_t_seq)
        
        # Path Tensor
        phi = self.path_tensor(eps_seq, delta_seq, d_delta_seq, d2_delta_seq, H_t_seq,
                               algebra_feats, Pi_seq, E_seq, p_t_seq, cp_feats, qtcf_feats,
                               evals_x, K_t_seq)
                               
        if not torch.isfinite(phi).all():
            raise RuntimeError("Inf/NaN in phi components")
                               
        # Classifier
        logits = self.classifier(phi)
        
        if not torch.isfinite(logits).all():
            raise RuntimeError("Inf/NaN in logits but not in phi")
        
        # Product Kernel (for geometric target alignment loss)
        D = phi.size(-1)
        G_x = phi.unsqueeze(-1) @ phi.unsqueeze(-2) + \
              (phi.unsqueeze(-1) @ phi.unsqueeze(-2)).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / D * torch.eye(D, device=device)
              
        K_batch = self.kernel(
            torch.cat([eps_seq, delta_seq, d_delta_seq, d2_delta_seq, H_t_seq], dim=-1),
            G_x,
            M_t_seq,
            qtcf_feats
        )
        
        return logits, {
            'phi': phi,
            'evals_x': evals_x,
            'K_batch': K_batch,
            'Pi_seq': Pi_seq,
            'eps_seq': eps_seq,
            'K_t_seq': K_t_seq
        }
