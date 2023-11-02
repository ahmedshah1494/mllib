from mllib.runners.base_runners import BaseRunner
from mllib.tasks.base_tasks import AbstractTask
from mllib.param import BaseParameters

from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.trainers.base_trainers import PytorchLightningTrainer
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier
from mllib.optimizers.configs import AdamOptimizerConfig, OneCycleLRConfig

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
        return BaseExperimentConfig(
            PytorchLightningTrainer.TrainerParams(PytorchLightningTrainer,
                TrainingParams(logdir='test-logs/', nepochs=1, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(lr=0.2, weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=1, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir='test-logs/', batch_size=128
        )

task = MNISTMLP()
runner = BaseRunner(task)
runner.create_trainer()
runner.train()
runner.test()