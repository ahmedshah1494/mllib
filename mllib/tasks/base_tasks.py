import pickle
from typing import Callable
import numpy as np
import torch
import types
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig
from attrs import define

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

@define
class AbstractTaskV2:
    dataset_params: BaseParameters
    model_params: BaseParameters
    experiment_params: BaseParameters

    def get_dataset_params(self) -> BaseParameters:
        return self.dataset_params()
    
    def get_model_params(self) -> BaseParameters:
        return self.model_params()
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        return self.experiment_params()

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