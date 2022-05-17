from typing import NamedTuple, Type
import attrs
from attrs import define
import torch
from param import BaseParameters, Parameterized

from utils.config import ConfigBase


class AbstractOptimizerConfig(Parameterized):
    @define(slots=False)
    class OptimizerArgs:
        lr: float = 0.1
        weight_decay: float = 0.

        def asdict(self):
            return attrs.asdict(self)

    cls: Type[torch.optim.Optimizer] = None

    @classmethod
    def get_params(cls):
        return cls.OptimizerArgs()

class AdamOptimizerConfig(AbstractOptimizerConfig):
    cls = torch.optim.Adam

    class OptimizerArgs(AbstractOptimizerConfig.OptimizerArgs):
        lr: float = 1e-3

class SGDOptimizerConfig(AbstractOptimizerConfig):
    cls = torch.optim.SGD

    class OptimizerArgs(AbstractOptimizerConfig.OptimizerArgs):
        momentum: float = 0.
        nesterov: bool = False
    

class AbstractSchedulerConfig(Parameterized):
    cls: Type[torch.optim.lr_scheduler._LRScheduler] = None
    
    @define(slots=False)
    class SchedulerArgs:
        def asdict(self):
            return attrs.asdict(self)

    @classmethod
    def get_params(cls):
        return cls.SchedulerArgs()

class ReduceLROnPlateauConfig(AbstractSchedulerConfig):
    cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    class SchedulerArgs(AbstractSchedulerConfig.SchedulerArgs):
        patience: int = 5
        mode: str = 'min'
        factor: float = 0.5