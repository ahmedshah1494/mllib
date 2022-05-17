import numpy as np
import torch
from datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from models.base_models import MLPClassifier
from optimizers.configs import AdamOptimizerConfig, ReduceLROnPlateauConfig
from param import BaseParameters
from runners.configs import BaseExperimentConfig

class AbstractTask(object):
    def get_dataset_params(self) -> BaseParameters:
        raise NotImplementedError
    
    def get_model_params(self) -> BaseParameters:
        raise NotImplementedError
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        raise NotImplementedError
    
class MNISTMLP(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.MNIST
        p.datafolder = '.'
        p.max_num_train = 1000
        p.max_num_test = 1000
        return p
    
    def get_model_params(self) -> BaseParameters:
        p = MLPClassifier.get_params()
        p.widths = [32, 64]
        p.input_size = 28*28
        p.output_size = 10
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        p = BaseExperimentConfig()
        p.batch_size = 128
        p.optimizer_config = AdamOptimizerConfig()
        p.scheduler_config = ReduceLROnPlateauConfig()
        p.training_params.nepochs = 2
        p.logdir = 'logs'
        return p