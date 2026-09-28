import json
import os
from collections import defaultdict
from typing import List, Dict, Any

class OrganizationAuditor:
    def __init__(self):
        self.stats = defaultdict(list)
        
    def log_sequences(self, sequences: List[List[Dict[str, Any]]]):
        for seq in sequences:
            if not seq:
                continue
            
            final_loc = seq[-1]['locality']
            avg_nov = sum(obs['information_gain'] for obs in seq) / len(seq)
            avg_conf = sum(obs['confidence'] for obs in seq) / len(seq)
            
            self.stats['final_localities'].append(final_loc)
            self.stats['avg_novelties'].append(avg_nov)
            self.stats['avg_confidences'].append(avg_conf)
            
    def export(self, filepath: str = "observation_statistics.json"):
        if not self.stats['final_localities']:
            return
            
        avg_loc = sum(self.stats['final_localities']) / len(self.stats['final_localities'])
        avg_nov = sum(self.stats['avg_novelties']) / len(self.stats['avg_novelties'])
        avg_conf = sum(self.stats['avg_confidences']) / len(self.stats['avg_confidences'])
        
        data = {
            'average_locality_depth': avg_loc,
            'average_novelty': avg_nov,
            'average_confidence': avg_conf,
            'histograms': {
                'localities': self.stats['final_localities']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
