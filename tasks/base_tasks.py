import pickle
import numpy as np
import torch
import types
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier
from mllib.optimizers.configs import AdamOptimizerConfig, ReduceLROnPlateauConfig
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig

class AbstractTask(object):
    def get_dataset_params(self) -> BaseParameters:
        raise NotImplementedError
    
    def get_model_params(self) -> BaseParameters:
        raise NotImplementedError
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        raise NotImplementedError
    
    def save_task(self, outpath):
        task_dict = {
            'dataset_params': self.get_dataset_params(),
            'model_params': self.get_model_params(),
            'experiment_params': self.get_experiment_params()
        }
        with open(outpath, 'wb') as f:
            pickle.dump(task_dict, f)
    
    @classmethod
    def load_task(cls, filepath):
        with open(filepath, 'rb') as f:
            task_dict = pickle.load(f)
        t = cls()
        t.get_dataset_params = types.MethodType(lambda self: task_dict['dataset_params'], t)
        t.get_model_params = types.MethodType(lambda self: task_dict['model_params'], t)
        t.get_experiment_params = types.MethodType(lambda self: task_dict['experiment_params'], t)
        return t

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