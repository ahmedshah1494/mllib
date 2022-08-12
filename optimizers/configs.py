from typing import NamedTuple, Type, List
import attrs
from attrs import define, field
import torch
from mllib.param import BaseParameters, Parameterized

from mllib.utils.config import ConfigBase

@define(slots=False)
class AbstractOptimizerConfig:
    _cls: Type[torch.optim.Optimizer] = None
    lr: float = 0.1
    weight_decay: float = 0.

    def asdict(self):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')))

@define(slots=False)
class AdamOptimizerConfig(AbstractOptimizerConfig):
    _cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float = 1e-3

@define(slots=False)
class SGDOptimizerConfig(AbstractOptimizerConfig):
    _cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    momentum: float = 0.
    nesterov: bool = False
    

@define(slots=False)
class AbstractSchedulerConfig:
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = None
    
    def asdict(self):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')))

@define(slots=False)
class ReduceLROnPlateauConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau
    patience: int = 5
    mode: str = 'min'
    factor: float = 0.5

@define(slots=False)
class CosineAnnealingWarmRestartsConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    T_0: int = 100
    T_mult: int = 1
    eta_min: float = 0

class _SequentialLRWrapper(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self, schedulers: List[AbstractSchedulerConfig], milestones: List[int], last_epoch: int = ...) -> None:
        schedulers = [p.cls(**(p.asdict())) for p in schedulers]
        super().__init__(schedulers, milestones, last_epoch)

@define(slots=False)
class SequentialLRConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = _SequentialLRWrapper
    schedulers: List[AbstractSchedulerConfig] = field(factory=list)
    milestones: List[int] = field(factory=list)