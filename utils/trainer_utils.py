from typing import Tuple, Union

import torch
import numpy as np


def _move_tensors_to_device(batch: Tuple[torch.Tensor], device: Union[str,torch.device]):
    batch = [x.to(device) for x in batch]
    return tuple(batch)

class MetricTracker(object):
    def __init__(self) -> None:
        self.history = []
    
    def add(self, x):
        self.history.append(x)
    
    def reset(self):
        self.history = []
    
    def mean(self):
        return np.mean(self.history)