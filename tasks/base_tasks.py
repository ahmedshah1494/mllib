from utils.config import ConfigBase


class BaseTask(object):
    def get_dataset_params(self) -> ConfigBase:
        raise NotImplementedError
    
    def get_model_params(self) -> ConfigBase:
        raise NotImplementedError
    
    def get_training_parameters(self) -> ConfigBase:
        raise NotImplementedError
    
    def get_evaluation_parameters(self) -> ConfigBase:
        raise NotImplementedError
    
