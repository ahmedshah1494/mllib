
from attrs import define
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
    debug:bool = False

@define(slots=False)
class BaseExperimentConfig:
    optimizer_config: AbstractOptimizerConfig = None
    scheduler_config: AbstractSchedulerConfig = None
    batch_size: int = 128
    logdir: str = 'logs/'
    exp_name: str = ''
    training_params: TrainingParams = TrainingParams()

    