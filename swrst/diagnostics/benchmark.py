import torch
import time
import json
from typing import Dict

class HardwareProfiler:
    def __init__(self):
        self.events = {}
        self.start_times = {}
        self.vram_peaks = {}
        
    def start(self, event_name: str):
        self.start_times[event_name] = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def stop(self, event_name: str):
        if event_name in self.start_times:
            elapsed = time.time() - self.start_times[event_name]
            self.events[event_name] = self.events.get(event_name, 0.0) + elapsed
            
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.vram_peaks[event_name] = max(self.vram_peaks.get(event_name, 0.0), peak_mb)
                
    def export(self, filepath: str = "hardware_profile.json"):
        data = {
            'timers_seconds': self.events,
            'vram_peak_mb': self.vram_peaks
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
