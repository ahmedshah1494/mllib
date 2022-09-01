
from typing import Type
from attrs import define, field
from mllib.param import BaseParameters
from mllib.utils.config import ConfigBase
from mllib.optimizers.configs import AbstractOptimizerConfig, AbstractSchedulerConfig

@define(slots=False)
class TrainingParams:
    logdir: str = 'logs/'
    nepochs: int = 100
    early_stop_patience: int = 5
    tracked_metric: str = 'train_loss'
    tracking_mode: str = 'min'
    schduler_step_after_epoch: bool = True
    debug:bool = False

@define(slots=False)
class BaseExperimentConfig:
    trainer_params: BaseParameters = None
    optimizer_config: AbstractOptimizerConfig = None
    scheduler_config: AbstractSchedulerConfig = None
    batch_size: int = 128
    logdir: str = 'logs/'
    exp_name: str = ''
    num_trainings: int = 1
    