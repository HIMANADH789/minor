import torch

def compute_geometry_cost(geometry: torch.Tensor) -> torch.Tensor:
    """
    Computes the geometric cost matrix from the structured atom coordinates.
    geometry: (N, D) where N is number of atoms, D is number of channels (F, T, E, G).
    Since each coordinate channel is already normalized to [0,1], 
    the cost is simply the squared Euclidean distance in the D-dimensional atom space.
    C = \Delta F^2 + \Delta T^2 + \Delta E^2 + \Delta G^2
    Returns: C (N, N)
    """
    # geometry: (N, D)
    # Pairwise squared Euclidean distance
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2(x \cdot y)
    
    sq_norm = (geometry ** 2).sum(dim=1) # (N,)
    
    # x \cdot y -> (N, N)
    dot_prod = torch.mm(geometry, geometry.T)
    
    C = sq_norm.unsqueeze(1) + sq_norm.unsqueeze(0) - 2 * dot_prod
    
    # Clamp to prevent negative values due to numerical instability
    C = torch.clamp(C, min=0.0)
    
    return C
