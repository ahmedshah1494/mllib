from typing import Any, Callable
from attrs import define
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
from tqdm import tqdm
import os
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters, Parameterized
from mllib.runners.configs import BaseExperimentConfig, TrainingParams

from mllib.utils.metric_utils import compute_accuracy

from mllib.utils.trainer_utils import _move_tensors_to_device

class AbstractTrainer(Parameterized):
    def __init__(self, params) -> None:
        pass

    def train_step(self, batch, batch_idx):
        raise NotImplementedError
    def val_step(self, batch, batch_idx):        
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    def _batch_loop(self, func, loader, epoch_idx):
        raise NotImplementedError
    def train_loop(self, epoch_idx, post_loop_fn=None):
        raise NotImplementedError
    def val_loop(self, epoch_idx, post_loop_fn=None):
        raise NotImplementedError
    def test_loop(self, post_loop_fn=None):
        raise NotImplementedError
    def create_or_clear_cpdir(self, metric, epoch_idx):
        raise NotImplementedError
    def checkpoint(self, metric, epoch_idx, comparator):
        raise NotImplementedError
    def check_early_stop(self):
        raise NotImplementedError
    def epoch_end(self, epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics):
        raise NotImplementedError
    def train_epoch_end(self, outputs, metrics, epoch_idx):
        raise NotImplementedError
    def val_epoch_end(self, outputs, metrics, epoch_idx):
        raise NotImplementedError
    def test_epoch_end(self, outputs, metrics):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def test(self):
        raise NotImplementedError

class Trainer(AbstractTrainer):
    @define(slots=False)
    class TrainerParams(BaseParameters):
        model: AbstractModel = None
        train_loader: torch.utils.data.DataLoader = None
        val_loader: torch.utils.data.DataLoader = None
        test_loader: torch.utils.data.DataLoader = None
        optimizer: torch.optim.Optimizer = None
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
        device: torch.device = torch.device('cpu')
        training_params: TrainingParams = None        

    @classmethod
    def get_params(cls):
        return cls.TrainerParams(cls)

    def __init__(self, params: TrainerParams):
        super(Trainer, self).__init__(params)
        self.params = params
        self.model = params.model
        self.train_loader = params.train_loader
        self.val_loader = params.val_loader
        self.test_loader = params.test_loader
        self.nepochs = params.training_params.nepochs
        self.optimizer = params.optimizer
        self.scheduler = params.scheduler
        self.device = params.device
        self.logdir = params.training_params.logdir
        self.logger = SummaryWriter(log_dir=params.training_params.logdir, flush_secs=60)
        self.early_stop_patience = params.training_params.early_stop_patience
        self.tracked_metric = params.training_params.tracked_metric
        self.tracking_mode = params.training_params.tracking_mode
        self.schduler_step_after_epoch = self.params.training_params.schduler_step_after_epoch
        self.schduler_step_after_epoch = self.params.training_params.schduler_step_after_epoch
        self.debug = params.training_params.debug

        self.track_metric()

    def track_metric(self):
        if self.tracking_mode == 'min':
            self.best_metric = float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x<y
        else:
            self.best_metric = -float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x>y

    def _log(self, logs, step):
        for k,v in logs.items():
            self.logger.add_scalar(k, v, global_step=step)

    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            output, logs = func(*args, **kwargs)
            output['loss'].backward()
            self.optimizer.step()
            if not self.schduler_step_after_epoch:
                self._scheduler_step(logs)
            return output, logs
        return wrapper    

    def _get_outputs_and_loss(self, x, y):
        return self.model.compute_loss(x, y)
        
    def train_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self._get_outputs_and_loss(x, y)
        acc, correct = compute_accuracy(logits.detach().cpu(), y.detach().cpu())
        
        loss = loss.mean()

        return {'loss':loss}, {'train_accuracy': acc,
                             'train_loss': float(loss.detach().cpu())}
    
    def val_step(self, batch, batch_idx):        
        output, logs = self.train_step(batch, batch_idx)
        output['loss'] = output['loss'].detach().cpu()
        val_logs = {'lr':self.scheduler.optimizer.param_groups[0]['lr']}
        for k,v in logs.items():
            val_logs[k.replace('train', 'val')] = v
        return output, val_logs
    
    def test_step(self, batch, batch_idx):
        output, logs = self.train_step(batch, batch_idx)
        output['loss'] = output['loss'].detach().cpu()
        test_logs = {}
        for k,v in logs.items():            
            test_logs[k.replace('train', 'test')] = v
        return output, test_logs

    def _batch_loop(self, func, loader, epoch_idx, logging=True):
        t = tqdm(enumerate(loader))
        t.set_description('epoch %d' % epoch_idx)
        all_outputs = []
        for i, batch in t:
            batch = _move_tensors_to_device(batch, self.device)
            outputs, logs = func(batch, i)
            all_outputs.append(outputs)
            if logging:
                self._log(logs, i + epoch_idx*len(loader))
            if 'metrics' not in locals():
                metrics = {k:0 for k in logs.keys()}
            for k,v in logs.items():
                metrics[k] = (i*metrics[k] + v)/(i+1)
            t.set_postfix(**metrics, best_metric=self.best_metric)
            if self.debug and (i == 5):
                break
        return all_outputs, metrics

    def train_loop(self, epoch_idx, post_loop_fn=None):
        self.model = self.model.train()
        outputs, metrics = self._batch_loop(self._optimization_wrapper(self.train_step),
                                    self.train_loader, epoch_idx)
        if post_loop_fn is not None:                                    
            outputs = post_loop_fn(outputs, metrics, epoch_idx)
        return outputs, metrics

    def val_loop(self, epoch_idx, post_loop_fn=None):
        self.model = self.model.eval()
        outputs, metrics = self._batch_loop(self.val_step, self.val_loader, epoch_idx)
        if post_loop_fn is not None:                                    
            outputs = post_loop_fn(outputs, metrics, epoch_idx)
        return outputs, metrics

    def test_loop(self, post_loop_fn=None):
        self.model = self.model.eval()
        outputs, metrics = self._batch_loop(self.test_step, self.test_loader, 0)
        if post_loop_fn is not None:                                    
            outputs, metrics = post_loop_fn(outputs, metrics)
        return outputs, metrics

    def create_or_clear_cpdir(self, metric, epoch_idx):
        outdir = os.path.join(self.logdir, 'checkpoints')
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            for fn in os.listdir(outdir):
                os.remove(os.path.join(outdir, fn))
        outfile = os.path.join(outdir,
                                "model_checkpoint.pt")
        return outfile

    def checkpoint(self, metric, epoch_idx, comparator):
        if comparator(metric, self.best_metric):
            self.best_metric = metric
            self.epochs_since_best = 0

            outfile = self.create_or_clear_cpdir(metric, epoch_idx)
            torch.save(self.model, outfile)
            self.best_checkpoint = outfile
        else:
            self.epochs_since_best += 1

    def check_early_stop(self):
        return self.epochs_since_best > self.early_stop_patience
    
    def _scheduler_step(self, metrics):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics[self.tracked_metric])
        else:
            self.scheduler.step()

    def epoch_end(self, epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics):
        metrics = train_metrics
        metrics.update(val_metrics)

        self.checkpoint(metrics[self.tracked_metric], epoch_idx, self.metric_comparator)
        
        if self.schduler_step_after_epoch:
            self._scheduler_step(metrics)
    
    def train_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics
    
    def val_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics

    def test_epoch_end(self, outputs, metrics):
        return outputs, metrics

    def train(self):
        for i in range(self.nepochs):
            train_output, train_metrics = self.train_loop(i, post_loop_fn=self.train_epoch_end)
            val_output, val_metrics = self.val_loop(i, post_loop_fn=self.val_epoch_end)
            self.epoch_end(i, train_output, val_output, train_metrics, val_metrics)

                       
            if self.check_early_stop():
                break

        metrics = train_metrics
        metrics.update(val_metrics, post_loop_fn=self.test_epoch_end)
        self.model = torch.load(self.best_checkpoint)
        return metrics

    def test(self):        
        _, test_metrics = self.test_loop(post_loop_fn=self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)
        return test_metrics

class MixedPrecisionTrainerMixin(object):
    def __init__(self, *args, **kwargs) -> None:
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            with torch.autocast():
                output, logs = func(*args, **kwargs)
            self.scaler.scale(output['loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return output, logs
        return wrapper   