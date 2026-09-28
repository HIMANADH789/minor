import torch
from typing import List, Dict, Any
from ..config import SWRSTConfig
from ..representation.locality import LocalityContext
from ..representation.measure import IncrementalMeasureBuilder
from .controller import ObservationController
from ..transport.sinkhorn import batched_log_domain_sinkhorn
from ..transport.confidence import compute_transport_confidence

class EvolutionController:
    def __init__(self, config: SWRSTConfig):
        self.config = config
        self.controller = ObservationController(config)
        self.measure_builder = IncrementalMeasureBuilder(config)
        
    def run_evolution(self, shared_stft: torch.Tensor, x: torch.Tensor) -> List[List[Dict[str, Any]]]:
        B, F, T_max = shared_stft.shape
        device = shared_stft.device
        
        sequences: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        
        # 1. Precompute master Geometry table for all B sequences up to T_max
        # master_geom: (B, F, T_max, D)
        master_geom, master_mass = self.measure_builder.build_master_coordinates(shared_stft)
        
        # We process in groups of active sequences. Initially all B are active at Locality = min_locality
        # But instead of grouping by arbitrary counts, we group by `current_loc`.
        # Initially, all sequences are at `loc = min_locality`.
        active_indices = torch.arange(B, device=device)
        current_locs = torch.full((B,), self.config.min_locality, dtype=torch.long, device=device)
        
        # Persistent Cache
        # P_cache[b] holds the transport plan from the previous locality
        P_cache = [None] * B
        f_cache = [None] * B
        g_cache = [None] * B
        
        step_count = 0
        
        # While there are active sequences
        while len(active_indices) > 0 and step_count < self.config.max_locality:
            step_count += 1
            
            # Group sequences by their current locality
            # We find unique localities among active sequences
            unique_locs = torch.unique(current_locs[active_indices])
            
            next_active_indices = []
            
            for loc_val in unique_locs:
                loc = min(loc_val.item(), T_max)
                # Find indices in the original batch that are at this locality
                group_mask = (current_locs[active_indices] == loc_val.item())
                group_idx = active_indices[group_mask] # Indices of sequences in this group
                
                N_atoms = F * loc
                
                # Slices for this group
                # Geometry: (GroupSize, F, loc, D) -> (GroupSize, N_atoms, D)
                geom_loc = master_geom[group_idx, :, :loc, :].reshape(len(group_idx), N_atoms, -1)
                mass_loc = master_mass[group_idx, :, :loc].reshape(len(group_idx), N_atoms)
                
                prob = mass_loc / (mass_loc.sum(dim=1, keepdim=True) + 1e-8)
                
                # Lazy Cost construction for this group
                # (GroupSize, N, 1, D) - (GroupSize, 1, N, D) -> (GroupSize, N, N)
                C = torch.sum((geom_loc.unsqueeze(2) - geom_loc.unsqueeze(1))**2, dim=-1)
                
                uniform_ref = torch.ones_like(prob) / N_atoms
                
                # Gather initial duals from cache
                # Since dimension changed (or this is first step), we might not be able to reuse f, g exactly 
                # unless we pad them. For now, strict mathematically exact zero initialization if size changed.
                # However, Phase 6/15 says "Warm-start every locality". 
                # We can pad f and g with zeros for new atoms.
                f_init_group = []
                g_init_group = []
                prev_P_group = []
                
                for b in group_idx.tolist():
                    f_old = f_cache[b]
                    g_old = g_cache[b]
                    P_old = P_cache[b]
                    
                    if f_old is not None:
                        # Pad from old N to current N
                        N_old = f_old.shape[0]
                        f_pad = torch.zeros(N_atoms, device=device)
                        g_pad = torch.zeros(N_atoms, device=device)
                        f_pad[:N_old] = f_old
                        g_pad[:N_old] = g_old
                        
                        f_init_group.append(f_pad)
                        g_init_group.append(g_pad)
                        
                        # Pad P_old
                        P_pad = torch.zeros((N_atoms, N_atoms), device=device)
                        P_pad[:N_old, :N_old] = P_old
                        prev_P_group.append(P_pad)
                    else:
                        f_init_group.append(torch.zeros(N_atoms, device=device))
                        g_init_group.append(torch.zeros(N_atoms, device=device))
                        prev_P_group.append(torch.zeros((N_atoms, N_atoms), device=device))
                        
                f_init = torch.stack(f_init_group)
                g_init = torch.stack(g_init_group)
                prev_P = torch.stack(prev_P_group)
                
                # Batched Sinkhorn! (GroupSize, N, N)
                P, f_out, g_out, metrics = batched_log_domain_sinkhorn(
                    uniform_ref, prob, C,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                    tol=self.config.sinkhorn_tol,
                    f_init=f_init, g_init=g_init,
                    prev_P=prev_P,
                    incremental_threshold=self.config.incremental_sinkhorn_threshold
                )
                
                # Compute Novelty and Confidence for the group
                P_uniform = torch.ones_like(P) / (N_atoms * N_atoms)
                # Batched Hellinger distance
                novelty = torch.sqrt(torch.clamp(1.0 - torch.sum(torch.sqrt(P * P_uniform), dim=(1,2)), min=0.0))
                
                confidence = []
                for i in range(len(group_idx)):
                    conf = compute_transport_confidence({
                        'iterations': metrics['iterations'][i] if isinstance(metrics['iterations'], list) else metrics['iterations'],
                        'marginal_error': metrics['marginal_error'][i] if isinstance(metrics['marginal_error'], list) else metrics['marginal_error']
                    }, self.config.sinkhorn_max_iter)
                    confidence.append(conf)
                confidence = torch.tensor(confidence, device=device)
                
                # Evaluate controller for each item
                for i, b in enumerate(group_idx.tolist()):
                    nov = novelty[i].item()
                    conf = confidence[i].item()
                    
                    # Update cache
                    f_cache[b] = f_out[i]
                    g_cache[b] = g_out[i]
                    P_cache[b] = P[i]
                    
                    # Controller step mapping (Phase 6: Non-linear adaptive step)
                    step = self.controller.get_adaptive_step(nov, conf)
                    decision = self.controller.evaluate_event(nov, conf, step, loc)
                    
                    obs = {
                        'transport_plan': P[i].detach().cpu().to(torch.float16),
                        'locality': loc,
                        'information_gain': decision['information_gain'],
                        'confidence': conf,
                        'novelty': nov,
                        'locality_efficiency': decision['locality_efficiency'],
                        'iterations': metrics['iterations'][i] if isinstance(metrics['iterations'], list) else metrics['iterations'],
                        'reused': metrics['reused'][i] if isinstance(metrics['reused'], list) else metrics['reused']
                    }
                    sequences[b].append(obs)
                    
                    if not self.config.get_weights()['adaptive_locality']:
                        pass # Done
                    elif not decision['should_stop']:
                        current_locs[b] += decision['step_next']
                        next_active_indices.append(b)
                        
            active_indices = torch.tensor(next_active_indices, dtype=torch.long, device=device)
            
        return sequences
