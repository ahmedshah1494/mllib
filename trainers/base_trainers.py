from collections import namedtuple
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
import torchvision
from tqdm import tqdm
import argparse
import os
import functools
from copy import deepcopy
import itertools
import typing

from utils.metric_utils import compute_accuracy

class AbstractTrainer(object):

    @classmethod
    def get_params(cls):
        class Parameters(object):
            pass
        return Parameters

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
    @classmethod
    def get_params(cls):
        param_cls = super().get_params()
        class Parameters(param_cls):
            model = None
            model = model
            train_loader = None
            val_loader = None
            test_loader = None
            args = None
            optimizer = None
            scheduler = None
            device = None

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(Trainer, self).__init__()
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = SummaryWriter(log_dir=args.logdir, flush_secs=60)        
        self.early_stop = False
        self.tracked_metric = 'train_loss'
        self.metric_comparator = lambda x,y: x<y

    def track_metric(self, metric_name, direction):
        self.tracked_metric = metric_name
        if direction == 'min':
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
            return output, logs
        return wrapper

    def criterion(self, logits, y):
        if self.args.loss_type == 'xent':
            loss = 1.0 * nn.functional.cross_entropy(logits, y, reduction='none')
        else:
            raise NotImplementedError(self.args.loss_type)
        return loss    

    def train_step(self, batch, batch_idx):
        x,y = batch        
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)
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

    def _batch_loop(self, func, loader, epoch_idx):
        t = tqdm(enumerate(loader))
        t.set_description('epoch %d' % epoch_idx)
        all_outputs = []
        for i, batch in t:            
            outputs, logs = func(batch, i)
            all_outputs.append(outputs)
            self._log(logs, i + epoch_idx*len(loader))
            if 'metrics' not in locals():
                metrics = {k:0 for k in logs.keys()}
            for k,v in logs.items():
                metrics[k] = (i*metrics[k] + v)/(i+1)
            t.set_postfix(**metrics, best_metric=self.best_metric)
            if self.args.debug and (i == 5):
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
        outdir = os.path.join(self.args.logdir, 'checkpoints')
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            for fn in os.listdir(outdir):
                os.remove(os.path.join(outdir, fn))
        outfile = os.path.join(outdir,
                                "metric=%.2f-epoch=%d.pt" % (metric, epoch_idx))
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
        if self.epochs_since_best > 3*self.args.patience:
            self.early_stop = True
    
    def epoch_end(self, epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics):
        metrics = train_metrics
        metrics.update(val_metrics)

        self.checkpoint(metrics[self.tracked_metric], epoch_idx, self.metric_comparator)
        self.check_early_stop()
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics[self.tracked_metric])
        else:
            self.scheduler.step()
    
    def train_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics
    
    def val_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics

    def test_epoch_end(self, outputs, metrics):
        return outputs, metrics

    def train(self):
        torch.save(self.args, os.path.join(self.args.logdir, 'args.pkl'))

        for i in range(self.args.nepochs):
            train_output, train_metrics = self.train_loop(i, post_loop_fn=self.train_epoch_end)
            val_output, val_metrics = self.val_loop(i, post_loop_fn=self.val_epoch_end)
            self.epoch_end(i, train_output, val_output, train_metrics, val_metrics)
            
            if self.early_stop:
                break

        metrics = train_metrics
        metrics.update(val_metrics, post_loop_fn=self.test_epoch_end)
        self.logger.add_hparams(dict(vars(self.args)), {self.tracked_metric: metrics[self.tracked_metric]})
        self.model = torch.load(self.best_checkpoint)

    def test(self):        
        _, test_metrics = self.test_loop(post_loop_fn=self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)