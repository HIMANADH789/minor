import torch
import math
from typing import Dict

def compute_transport_confidence(metrics: Dict[str, float], max_iter: int) -> float:
    """
    Computes a continuous confidence score in [0, 1] based on Sinkhorn convergence metrics.
    Confidence = 0.4 * Marginal + 0.3 * Dual Stability + 0.3 * Iteration Stability
    """
    if metrics.get('reused', False):
        return 1.0
        
    marginal_error = metrics.get('marginal_error', 1.0)
    dual_residual = metrics.get('dual_residual', 1.0)
    iterations = metrics.get('iterations', max_iter)
    
    # Scale errors to [0,1] confidence scores
    conf_marginal = math.exp(-50.0 * marginal_error)
    conf_dual = math.exp(-50.0 * dual_residual)
    conf_iter = 1.0 - (iterations / max_iter)
    
    # Final confidence is the weighted sum
    confidence = 0.4 * conf_marginal + 0.3 * conf_dual + 0.3 * conf_iter
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))
