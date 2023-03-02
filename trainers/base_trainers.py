from time import time
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

from mllib.utils.metric_utils import compute_accuracy, get_preds_from_logits

from mllib.utils.trainer_utils import _move_tensors_to_device

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.lite import LightningLite
import torchmetrics
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm

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
        training_params: TrainingParams = None        

    @classmethod
    def get_params(cls):
        return cls.TrainerParams(cls)

    def __init__(self, params: TrainerParams, model: AbstractModel, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, *args,
                    device: torch.device = torch.device('cpu'), lightning_lite_instance: LightningLite=None, **kwargs):
        super(Trainer, self).__init__(params)
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.nepochs = params.training_params.nepochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logdir = params.training_params.logdir
        self.early_stop_patience = params.training_params.early_stop_patience
        self.tracked_metric = params.training_params.tracked_metric
        self.tracking_mode = params.training_params.tracking_mode
        print(params)
        self.scheduler_step_after_epoch = params.training_params.scheduler_step_after_epoch
        self.debug = params.training_params.debug
        self.lightning_lite_instance = lightning_lite_instance
        self.is_rank_zero = lightning_lite_instance.is_global_zero if lightning_lite_instance is not None else True

        self.track_metric()
    
    def _maybe_initialize_logger(self):
        if self.is_rank_zero and (not hasattr(self,'logger')):
            self.logger = SummaryWriter(log_dir=self.params.training_params.logdir, flush_secs=60)
            self.global_step = 0
    
    def _maybe_gather_all(self, d):
        if self.lightning_lite_instance is not None:
            all_d = self.lightning_lite_instance.all_gather(d)
            if isinstance(all_d, dict):
                d = {k: v.mean() for k,v in all_d.items()}
        return d

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
        if self.is_rank_zero:
            for k,v in logs.items():
                self.logger.add_scalar(k, v, global_step=step)

    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            output, logs = func(*args, **kwargs)
            loss = output['loss']
            if self.lightning_lite_instance is None:
                loss.backward()
            else:
                self.lightning_lite_instance.backward(loss)
            self.optimizer.step()
            if not self.scheduler_step_after_epoch:
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
        self._maybe_initialize_logger()
        t = tqdm(enumerate(loader))
        t.set_description(f'{"/".join(self.logdir.split("/")[-2:])} epoch {epoch_idx}')
        all_outputs = []
        metrics = {}
        for i, batch in t:
            batch = _move_tensors_to_device(batch, self.model.parameters().__next__().device)
            outputs, logs = func(batch, i)
            all_outputs.append(outputs)
            if logging and self.is_rank_zero:
                self._log(logs, self.global_step)
            if metrics == {}:
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
        metrics = self._maybe_gather_all(metrics)
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
        if self.is_rank_zero:
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
            if self.is_rank_zero:
                if self.lightning_lite_instance is not None:
                    model = self.model.module
                else:
                    model = self.model
                torch.save(model, outfile)
            if self.lightning_lite_instance is not None:
                self.lightning_lite_instance.barrier()
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
        
        if self.scheduler_step_after_epoch:
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
        if self.lightning_lite_instance is not None:
            all_metrics = self.lightning_lite_instance.all_gather(test_metrics)
            test_metrics = {k: v.mean() for k, v in all_metrics}
        print('test metrics:')
        print(test_metrics)
        return test_metrics

class CustomTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self) -> Tqdm:
        bar = super().init_train_tqdm()
        bar.ncols=10
        bar.mininterval = 1.
        return bar

class PytorchLightningTrainer(pl.LightningModule):
    @define(slots=False)
    class TrainerParams(BaseParameters):
        training_params: TrainingParams = None        

    @classmethod
    def get_params(cls):
        return cls.TrainerParams(cls)

    def __init__(self, params: TrainerParams, model: AbstractModel, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, *args,
                    **kwargs):
        super().__init__()
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.nepochs = params.training_params.nepochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logdir = params.training_params.logdir
        self.early_stop_patience = params.training_params.early_stop_patience
        self.tracked_metric = params.training_params.tracked_metric
        self.tracking_mode = params.training_params.tracking_mode
        self.scheduler_step_after_epoch = params.training_params.scheduler_step_after_epoch
        print(params)
        self.accuracy_tracker = torchmetrics.MeanMetric()
        self.loss_tracker = torchmetrics.MeanMetric()
        self.mloggers = [
            TensorBoardLogger(self.logdir),
            CSVLogger(self.logdir)
        ]
        self.lr = optimizer.param_groups[0]['lr']
        self.t0 = time()

    def track_metric(self):
        if self.tracking_mode == 'min':
            self.best_metric = float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x<y
        else:
            self.best_metric = -float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x>y
    
    def create_or_clear_cpdir(self):
        outdir = os.path.join(self.logdir, 'checkpoints')
        if self.is_rank_zero:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        outfile = os.path.join(outdir, "model_checkpoint.pt")
        return outfile

    def checkpoint(self, metrics):
        metric = metrics[self.tracked_metric]
        if self.metric_comparator(metric, self.best_metric):
            self.best_metric = metric
            self.epochs_since_best = 0

            outfile = self.create_or_clear_cpdir()
            if self.is_rank_zero:
                if self.lightning_lite_instance is not None:
                    model = self.model.module
                else:
                    model = self.model
                torch.save(model, outfile)
            if self.lightning_lite_instance is not None:
                self.lightning_lite_instance.barrier()
            self.best_checkpoint = outfile
        else:
            self.epochs_since_best += 1
    
    def _epoch_end(self, outputs) -> None:
        metrics = {}
        for o in outputs:
            for m,v in o['logs'].items():
                metrics[m] = metrics.get(m, 0) + v
        for m, v in metrics.items():
            metrics[m] = v / len(outputs)
        self.checkpoint(metrics)

    def _maybe_average_dict(self, d):
        if isinstance(d, dict):
            d = {k: v.mean() if (isinstance(v, torch.Tensor) and v.dim() == 1) else v for k,v in d.items()}
        return d

    def _get_outputs_and_loss(self, x, y):
        return self.model.compute_loss(x, y)

    def forward_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self._get_outputs_and_loss(x, y)
        acc, _ = compute_accuracy(logits.detach(), y.detach())
        
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        loss = loss.mean()
        t = time() - self.t0
        logs = {'time': t, 'lr': lr, 'accuracy': acc, 'loss': loss.detach()}
        return {'loss':loss, 'logs':logs}
    
    def on_train_start(self) -> None:
        self.t0 = time()
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        output = self.forward_step(batch, batch_idx)
        logs = output.pop('logs')
        _logs = {}
        for k,v in logs.items():
            _logs['train_'+k] = v
        output['logs'] = _logs
        self.log_dict(_logs, on_step=True, prog_bar=True, rank_zero_only=True)
        return output
    
    # def training_step_end(self, output):
    #     logs = output.pop('logs')
    #     output = self._maybe_average_dict(output)
    #     logs = self._maybe_average_dict(logs)
    #     output['logs'] = logs
    #     self.log_dict(logs, on_step=True, prog_bar=True, rank_zero_only=True)
    #     return output
    
    # def training_epoch_end(self, outputs) -> None:
    #     self._epoch_end(outputs)

    # def on_validation_start(self) -> None:
    #     self.accuracy_tracker.reset()
    #     self.loss_tracker.reset()

    def validation_step(self, batch, batch_idx):        
        output = self.forward_step(batch, batch_idx)
        return output
    
    def validation_step_end(self, output):
        logs = output.pop('logs')
        output['loss'] = output['loss'].detach()

        self.accuracy_tracker.update(logs['accuracy'])
        self.loss_tracker.update(logs['loss'])

        logs['accuracy'] = self.accuracy_tracker#.compute()
        logs['loss'] = self.loss_tracker#.compute()
        _logs = {}
        for k,v in logs.items():
            _logs['val_'+k] = v
        output['logs'] = _logs
        self.log_dict(_logs, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return output
    
    # def validation_step_end(self, output):
    #     logs = output.pop('logs')
    #     output = self._maybe_average_dict(output)
    #     logs = self._maybe_average_dict(logs)
    #     output['logs'] = logs
    #     self.log_dict(logs, on_step=True, prog_bar=True, rank_zero_only=True)
    #     return output

    # def validation_epoch_end(self, outputs) -> None:
    #     self._epoch_end(outputs)
    
    def on_test_start(self) -> None:
        self.accuracy_tracker.reset()
        self.loss_tracker.reset()
    
    def test_step(self, batch, batch_idx):
        output = self.forward_step(batch, batch_idx)
        logs = output.pop('logs')
        output['loss'] = output['loss'].detach()

        self.accuracy_tracker.update(logs['accuracy'])
        self.loss_tracker.update(logs['loss'])

        _logs = {}
        for k,v in logs.items():
            _logs['test_'+k] = v
        output['logs'] = _logs
        return output

    def test_step_end(self, output):
        logs = output.pop('logs')
        output = self._maybe_average_dict(output)
        logs = self._maybe_average_dict(logs)
        output['logs'] = logs
        self.log_dict(logs, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return output

    def configure_optimizers(self):
        scheduler_config = {
            'scheduler': self.scheduler,
            'interval': 'epoch' if self.scheduler_step_after_epoch else 'step'
        }
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_config['monitor'] = self.tracked_metric
        self.optimizer.param_groups[0]['lr'] = self.lr
        return [self.optimizer], [scheduler_config]

    # def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
    #     print(torch.norm(torch.nn.utils.parameters_to_vector(self.model.parameters())))
    #     return super().on_before_optimizer_step(optimizer, optimizer_idx)
    
    # def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False) -> None:
    #     out = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
    #     print(torch.norm(torch.nn.utils.parameters_to_vector(self.model.parameters())))
    #     return out

    def configure_callbacks(self):
        early_stop =  EarlyStopping(monitor=self.tracked_metric, mode=self.tracking_mode, 
                                                patience=self.early_stop_patience, check_on_train_epoch_end=False)
        ckpdir = os.path.join(self.logdir, 'checkpoints')
        checkpoint = ModelCheckpoint(monitor=self.tracked_metric, mode=self.tracking_mode, dirpath=ckpdir,
                                                    save_last=True, save_on_train_epoch_end=False, every_n_epochs=1,
                                                    verbose=True)
        checkpoint.CHECKPOINT_NAME_LAST = 'model_checkpoint'
        checkpoint.FILE_EXTENSION = '.pt'
        return [early_stop, checkpoint]

class MixedPrecisionTrainerMixin(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, logs = func(*args, **kwargs)
            self.scaler.scale(output['loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not self.scheduler_step_after_epoch:
                self._scheduler_step(logs)
            return output, logs
        return wrapper   