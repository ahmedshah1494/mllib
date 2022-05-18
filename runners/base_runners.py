import os
from sched import scheduler
from typing import Tuple
from sklearn import datasets
import torch
from mllib.datasets.dataset_factory import ImageDatasetFactory
from mllib.tasks.base_tasks import AbstractTask
from mllib.trainers.base_trainers import Trainer
from mllib.utils.config import ConfigBase


class AbstractRunner(object):
    def create_datasets(self) -> Tuple[torch.utils.data.Dataset]:
        raise NotImplementedError

    def create_dataloaders(self) -> Tuple[torch.utils.data.DataLoader]:
        raise NotImplementedError

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError
    
    def create_trainer(self) -> Trainer:
        raise NotImplementedError

class BaseRunner(AbstractRunner):
    def __init__(self, task: AbstractTask) -> None:
        super().__init__()
        self.task = task
        self.trainer: Trainer = None
    
    def create_datasets(self) -> Tuple[torch.utils.data.Dataset]:
        p = self.task.get_dataset_params()
        train_dataset, val_dataset, test_dataset, nclasses = p.cls.get_image_dataset(p)
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self) -> Tuple[torch.utils.data.DataLoader]:
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        p = self.task.get_experiment_params()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def create_model(self) -> torch.nn.Module:
        p = self.task.get_model_params()
        model = p.cls(p)
        print(model)
        return model

    def get_experiment_dir(self, logdir, model_name, exp_name):
        exp_params = self.task.get_experiment_params()
        exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
        logdir = os.path.join(exp_params.logdir, model_name+exp_name)
        exp_num = len(os.listdir(logdir)) if os.path.exists(logdir) else 0
        logdir = os.path.join(logdir, str(exp_num))
        return logdir

    def create_trainer(self) -> Trainer:
        model = self.create_model()
        exp_params = self.task.get_experiment_params()
        opt_params = exp_params.optimizer_config
        optimizer = opt_params._cls(model.parameters(), **(opt_params.asdict()))
        sch_params = exp_params.scheduler_config
        scheduler = sch_params._cls(optimizer, **(sch_params.asdict()))

        train_loader, val_loader, test_loader = self.create_dataloaders()

        device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

        trainer_params = Trainer.get_params()
        trainer_params.model = model
        trainer_params.train_loader = train_loader
        trainer_params.val_loader = val_loader
        trainer_params.test_loader = test_loader
        trainer_params.optimizer = optimizer
        trainer_params.scheduler = scheduler
        trainer_params.device = device

        exp_params.training_params.logdir = self.get_experiment_dir(exp_params.logdir,
                                                                        model.name,
                                                                        exp_params.exp_name)
        trainer_params.training_params = exp_params.training_params

        self.trainer = trainer_params.cls(trainer_params)
    
    def train(self):
        self.trainer.train()
        self.trainer.logger.flush()
    
    def test(self):
        self.trainer.test()
        self.trainer.logger.flush()