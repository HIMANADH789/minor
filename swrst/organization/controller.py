import math
from typing import Tuple

class ObservationController:
    """
    Decides where observations are needed. 
    It computes Information Gain and dictates whether to Stop or Continue.
    """
    def __init__(self, config: 'SWRSTConfig'):
        self.min_loc = config.min_locality
        self.max_loc = config.max_locality
        self.base_step = config.locality_base_step
        self.max_step = config.locality_max_step
        
    def get_adaptive_step(self, novelty: float, confidence: float) -> int:
        ig = novelty * confidence
        
        # Adaptive ODE sequence
        # We start from base step, but the progression should follow ODE-style leaps 
        # based on IG. For simplicity, we just use the scale: 1, 2, 4, 6, 9, 13...
        # We map IG to an index in the sequence
        sequence = [1, 2, 4, 6, 9, 13, 18, 25, 34]
        
        if ig > 0.05:
            # High IG -> small step (high resolution needed)
            idx = 0
        elif ig > 0.02:
            idx = 1
        elif ig > 0.01:
            idx = 2
        elif ig > 0.005:
            idx = 3
        elif ig > 0.001:
            idx = 5
        else:
            idx = 7
            
        step_next = sequence[idx]
        step_next = min(step_next, self.max_step)
        step_next = max(1, step_next)
        return step_next
        
    def evaluate_event(self, novelty: float, confidence: float, current_step_size: int, current_loc: int) -> dict:
        locality_efficiency = novelty / max(1.0, float(current_step_size))
        ig = novelty * confidence * locality_efficiency
        
        # We determine the NEXT step for the sequence
        step_next = self.get_adaptive_step(novelty, confidence)
        
        should_stop = (ig < 1e-4) or (current_loc >= self.max_loc)
        
        return {
            'information_gain': ig,
            'locality_efficiency': locality_efficiency,
            'step_next': step_next,
            'should_stop': should_stop
        }
