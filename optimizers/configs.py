from typing import Callable, NamedTuple, Type, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
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
    
    def asdict(self, **kwargs):
        return attrs.asdict(self, filter=lambda attr, value: (not attr.name.startswith('_')), **kwargs)

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
    last_epoch: int = -1
    verbose: bool = False

@define(slots=False)
class LinearLRConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.LinearLR
    start_factor: float = 1/3
    end_factor: float = 1.
    total_iters: int = 5
    last_epoch: int = -1
    verbose:bool = False

@define(slots=False)
class CyclicLRConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CyclicLR
    base_lr: float = None
    max_lr: float = None
    step_size_up: int = 2000
    step_size_down: int = None
    mode: str = 'triangular'
    gamma: float = 1.0
    scale_fn: Callable = None
    scale_mode: str = 'cycle'
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9
    last_epoch: int = - 1
    verbose: bool = False

@define(slots=False)
class OneCycleLRConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.OneCycleLR
    max_lr: float = None
    total_steps: int = None
    epochs: int = None
    steps_per_epoch: int = None
    pct_start: float = 0.3
    anneal_strategy: Literal['cos', 'linear'] = 'cos'
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    three_phase: bool = False
    last_epoch: int = - 1
    verbose=False

class _SequentialLRWrapper(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self, optimizer, schedulers: List[AbstractSchedulerConfig], milestones: List[int], last_epoch: int = -1) -> None:
        schedulers = [p._cls(optimizer, **(p.asdict())) for p in schedulers]
        super().__init__(optimizer, schedulers, milestones, last_epoch)

@define(slots=False)
class SequentialLRConfig(AbstractSchedulerConfig):
    _cls: Type[torch.optim.lr_scheduler._LRScheduler] = _SequentialLRWrapper
    schedulers: List[AbstractSchedulerConfig] = field(factory=list)
    milestones: List[int] = field(factory=list)

    def asdict(self, **kwargs):
        return super().asdict(recurse=False, **kwargs)

