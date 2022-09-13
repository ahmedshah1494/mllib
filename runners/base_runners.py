import os
import pickle
from sched import scheduler
import shutil
from typing import Tuple
import torch
from mllib.datasets.dataset_factory import ImageDatasetFactory
from mllib.tasks.base_tasks import AbstractTask
from mllib.trainers.base_trainers import Trainer
from mllib.utils.config import ConfigBase

from mllib.utils.common_utils import is_file_in_dir


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
    def __init__(self, task: AbstractTask, ckp_pth: str=None, load_model_from_ckp: bool=False) -> None:
        super().__init__()
        self.task = task
        self.trainer: Trainer = None
        self.ckp_pth = ckp_pth
        self.load_model_from_ckp = load_model_from_ckp
    
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

    def load_model(self):
        model = torch.load(self.ckp_pth)
        return model
    
    def create_model(self) -> torch.nn.Module:
        if self.load_model_from_ckp:
            model = self.load_model()
        else:
            p = self.task.get_model_params()
            model = p.cls(p)
        return model

    def get_experiment_dir(self, logdir, exp_name):
        def is_exp_complete(i):
            return os.path.exists(os.path.join(logdir, str(i), 'task.pkl'))
        # if self.load_model_from_ckp:
        #     return self.ckp_dir
        exp_params = self.task.get_experiment_params()
        exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
        task_name = type(self.task).__name__
        logdir = os.path.join(exp_params.logdir, task_name+exp_name)
        exp_num = len(os.listdir(logdir)) if os.path.exists(logdir) else 0
        exp_num = 0
        while is_exp_complete(exp_num):
            exp_num += 1
        logdir = os.path.join(logdir, str(exp_num))
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        print(f'writing logs to {logdir}')
        return logdir

    def create_optimizer(self, parameters):
        exp_params = self.task.get_experiment_params()
        opt_params = exp_params.optimizer_config
        optimizer = opt_params._cls(parameters, **(opt_params.asdict()))
        return optimizer

    def create_scheduler(self, optimizer):
        exp_params = self.task.get_experiment_params()
        sch_params = exp_params.scheduler_config
        scheduler = sch_params._cls(optimizer, **(sch_params.asdict()))
        return scheduler

    def create_trainer_params(self) -> Trainer.TrainerParams:
        exp_params = self.task.get_experiment_params()
        trainer_params = exp_params.trainer_params
        trainer_params.training_params.logdir = self.get_experiment_dir(exp_params.logdir, exp_params.exp_name)
        return trainer_params

    def create_trainer(self) -> Trainer:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer_params = self.create_trainer_params()
        model = self.create_model().to(device)
        optimizer = self.create_optimizer(model.parameters())
        scheduler = self.create_scheduler(optimizer)
        train_loader, val_loader, test_loader = self.create_dataloaders()
        self.trainer = trainer_params.cls(trainer_params, model, train_loader, val_loader, test_loader, optimizer, scheduler, device=device)
    
    def save_task(self):
        self.task.save_task(os.path.join(self.trainer.logdir, 'task.pkl'))
    
    def train(self):
        self.trainer.train()
        self.trainer.logger.flush()
        self.save_task()
    
    def test(self):
        self.trainer.test()
        self.trainer.logger.flush()
        self.save_task()
    
    def run(self):
        ntrains = self.task.get_experiment_params().num_trainings
        for _ in range(ntrains):
            self.create_trainer()
            self.train()
            self.test()