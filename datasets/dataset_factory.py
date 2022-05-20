from enum import Enum, auto
import os
from typing import List
from attrs import define
import numpy as np
import torch
import torchvision
from mllib.param import BaseParameters, Parameterized

from mllib.utils.image_dataset_utils import filter_dataset_by_target, make_val_dataset

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

class SupportedDatasets(AutoName):
    CIFAR10 = auto()
    CIFAR100 = auto()
    TINY_IMAGENET = auto()
    MNIST = auto()

class AbstractDatasetFactory(Parameterized):
    @classmethod
    def get_image_dataset(cls, params: BaseParameters) -> torch.utils.data.Dataset:
        raise NotImplementedError

class ImageDatasetFactory(AbstractDatasetFactory):
    class DatasetConfig(object):
        def __init__(self, dataset_class, nclasses, min_train_class_counts, max_val_counts) -> None:
            self.dataset_class = dataset_class
            self.nclasses = nclasses
            self.min_train_class_counts = min_train_class_counts
            self.max_val_counts = max_val_counts
    
    class ImageDatasetParams(BaseParameters):
        dataset: SupportedDatasets = None
        datafolder: str = ''
        class_idxs: List[int] = None
        custom_transforms: list = None
        max_num_train: int = np.inf
        max_num_test: int = np.inf


    dataset_config = {
        SupportedDatasets.MNIST : DatasetConfig(
                                        torchvision.datasets.MNIST,
                                        10, 4500, 5000
                                    ),
        SupportedDatasets.CIFAR10 : DatasetConfig(
                                        torchvision.datasets.CIFAR10,
                                        10, 4500, 5000
                                    ),
        SupportedDatasets.CIFAR100 : DatasetConfig(
                                        torchvision.datasets.CIFAR100,
                                        100, 450, 5000
                                    ),
        SupportedDatasets.TINY_IMAGENET : DatasetConfig(
                                        torchvision.datasets.ImageFolder,
                                        200, 475, 5000
                                    )
    }

    @classmethod
    def get_params(cls):
        return cls.ImageDatasetParams(cls)
    
    @classmethod
    def get_transforms(cls, dataset: SupportedDatasets):
        train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),        
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),        
        ])
        if dataset != SupportedDatasets.MNIST:
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                train_transform,        
            ])
        return train_transform, test_transform

    @classmethod
    def get_image_dataset(cls, params: ImageDatasetParams):
        cfg = cls.dataset_config[params.dataset]
        dataset_class = cfg.dataset_class
        nclasses = cfg.nclasses if params.class_idxs is None else len(params.class_idxs)
        train_class_counts = params.max_num_train // nclasses
        train_class_counts = min(cfg.min_train_class_counts, train_class_counts)
        test_class_counts = params.max_num_test // nclasses
        max_val_counts = cfg.max_val_counts

        if params.custom_transforms is not None:
            train_transform, test_transform = params.custom_transforms
        else:
            train_transform, test_transform = cls.get_transforms(params.dataset)
        print(train_transform)
        print(test_transform)
        
        if params.dataset in [SupportedDatasets.CIFAR10, SupportedDatasets.CIFAR100, SupportedDatasets.MNIST]:
            train_dataset = dataset_class('%s/'%params.datafolder, transform=train_transform, download=True)        
            test_dataset = dataset_class('%s/'%params.datafolder, train=False, transform=test_transform, download=True)
        elif params.dataset == SupportedDatasets.TINY_IMAGENET:
            train_dataset = dataset_class(os.path.join(params.datafolder, 'train'), transform=train_transform)
            test_dataset = dataset_class(os.path.join(params.datafolder, 'val'), transform=test_transform)

        filter_dataset_by_target(train_dataset, params.class_idxs)
        train_idxs, val_idxs = make_val_dataset(train_dataset, nclasses, train_class_counts, class_idxs=params.class_idxs)
        if len(val_idxs) > max_val_counts:
            val_idxs = np.random.choice(val_idxs, max_val_counts, replace=False)
        val_dataset = torch.utils.data.Subset(train_dataset, val_idxs)
        train_dataset = torch.utils.data.Subset(train_dataset, train_idxs)

        filter_dataset_by_target(test_dataset, params.class_idxs)
        test_idxs, _ = make_val_dataset(test_dataset, nclasses, test_class_counts, class_idxs=params.class_idxs)
        test_dataset = torch.utils.data.Subset(test_dataset, test_idxs)
        return train_dataset, val_dataset, test_dataset, nclasses
