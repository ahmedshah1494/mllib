from copy import deepcopy
import os
import pickle
from attrs import define, asdict
import shutil
from typing import List, Optional, Tuple, Union
import torch
from mllib.datasets.dataset_factory import ImageDatasetFactory
from mllib.tasks.base_tasks import AbstractTask
from mllib.trainers.base_trainers import Trainer
from mllib.utils.config import ConfigBase

from mllib.utils.common_utils import is_file_in_dir

import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.plugins import PLUGIN_INPUT

import webdataset as wd

class LightningLiteTrainerWrapper(LightningLite):
    def __init__(self, trainer, accelerator: Optional[Union[str, Accelerator]] = None, strategy: Optional[Union[str, Strategy]] = None, 
                    devices: Optional[Union[List[int], str, int]] = None, num_nodes: int = 1, precision: Union[int, str] = 32, 
                    plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None, gpus: Optional[Union[List[int], str, int]] = None, 
                    tpu_cores: Optional[Union[List[int], str, int]] = None) -> None:
        super().__init__(accelerator, strategy, devices, num_nodes, precision, plugins, gpus, tpu_cores)
        self.trainer = trainer
    
    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except:
            return getattr(self.trainer, __name)

    def setup_loaders(self, *loaders):
        _loaders = []
        for loader in loaders:
            if not isinstance(loader, wd.WebLoader):
                loader = self.setup_dataloaders(loader)
            _loaders.append(loader)
        return tuple(loaders)

    def run(self, train):
        device = self.trainer.device
        train_loader = self.trainer.train_loader
        val_loader = self.trainer.val_loader
        test_loader = self.trainer.test_loader

        self.trainer.lightning_lite_instance = self
        self.trainer.is_rank_zero = self.is_global_zero
        self.trainer.model, self.trainer.optimizer = self.setup(self.trainer.model, self.trainer.optimizer)
        self.trainer.train_loader, self.trainer.val_loader, self.trainer.test_loader = self.setup_loaders(self.trainer.train_loader, self.trainer.val_loader, self.trainer.test_loader)
        if train:
            self.trainer.train()
        else:
            self.trainer.test()
        
        if isinstance(self.trainer.model, pl.lite.wrappers._LiteModule):
            self.trainer.model = self.trainer.model.module
        self.trainer.optimizer = self.trainer.optimizer.optimizer
        self.trainer.train_loader = train_loader
        self.trainer.val_loader = val_loader
        self.trainer.test_loader = test_loader
        self.trainer.lightning_lite_instance = None
        self.trainer.is_rank_zero = True
        self.trainer.device = device

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
    def __init__(self, task: AbstractTask, num_trainings: int=1, ckp_pth: str=None, load_model_from_ckp: bool=False, wrap_trainer_with_lightning: bool=False, lightning_kwargs={}) -> None:
        super().__init__()
        self.task = task
        self.trainer: Trainer = None
        self.num_trainings = num_trainings
        self.ckp_pth = ckp_pth
        self.load_model_from_ckp = load_model_from_ckp
        self.wrap_trainer_with_lightning = wrap_trainer_with_lightning
        self.lightning_kwargs = lightning_kwargs
    
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
        if self.wrap_trainer_with_lightning:
            LightningLiteTrainerWrapper(self.trainer, **(self.lightning_kwargs)).run(True)
        elif isinstance(self.trainer, pl.LightningModule):
            if not hasattr(self, 'plTrainer'):
                if self.ckp_pth is not None:
                    self.lightning_kwargs['resume_from_checkpoint'] = self.ckp_pth
                self.plTrainer = pl.Trainer(accelerator='auto', devices='auto', strategy='ddp' if torch.cuda.device_count() > 1 else None, logger=self.trainer.mloggers,
                                            max_epochs=self.trainer.nepochs, **(self.lightning_kwargs), log_every_n_steps=10)
            if self.lightning_kwargs.get('auto_lr_find', False):
                print('tuning_lr')
                lr_finder = self.plTrainer.tuner.lr_find(self.trainer, train_dataloaders=self.trainer.train_loader)
                lr_finder.plot().savefig('lr_tuning_curve.png')
                exit()
            self.plTrainer.fit(self.trainer, train_dataloaders=self.trainer.train_loader, val_dataloaders=self.trainer.val_loader)
        else:
            self.trainer.train()
        self.save_task()
    
    def test(self):
        if self.wrap_trainer_with_lightning:
            LightningLiteTrainerWrapper(self.trainer, **(self.lightning_kwargs)).run(False)
        elif isinstance(self.trainer, pl.LightningModule):
            if not hasattr(self, 'plTrainer'):
                self.plTrainer = pl.Trainer(accelerator='auto', devices=1, logger=self.trainer.mloggers, 
                                            max_epochs=1, **(self.lightning_kwargs))
            self.plTrainer.test(self.trainer, dataloaders=self.trainer.test_loader)
        else:
            self.trainer.test()
        self.save_task()
    
    def run(self):
        # ntrains = self.task.get_experiment_params().num_trainings
        for _ in range(self.num_trainings):
            self.create_trainer()
            self.train()
            self.test()