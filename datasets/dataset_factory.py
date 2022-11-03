from enum import Enum, auto
import os
from typing import List
from attrs import define
import numpy as np
import torch
import torchvision
from mllib.param import BaseParameters, Parameterized

from mllib.utils.image_dataset_utils import filter_dataset_by_target, make_val_dataset

from mllib.datasets.tiny_imagenet_dataset import TinyImagenetNPZDataset

from mllib.datasets.imagenet_filelist_dataset import ImagenetFileListDataset, get_webdataset
from webdataset import WebDataset
from mllib.datasets.torchaudio_datasets import SpeechCommandDatasetWrapper

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

class SupportedDatasets(AutoName):
    CIFAR10 = auto()
    CIFAR100 = auto()
    TINY_IMAGENET = auto()
    IMAGENET = auto()
    ECOSET = auto()
    ECOSET_FOLDER = auto()
    ECOSET10 = auto()
    ECOSET10_FOLDER = auto()
    ECOSET100_FOLDER = auto()
    ECOSET100 = auto()
    IMAGENET_FOLDER = auto()
    IMAGENET10 = auto()
    IMAGENET100_64 = auto()
    IMAGENET100_81 = auto()
    IMAGENET75_64 = auto()
    IMAGENET100 = auto()
    MNIST = auto()
    FMNIST = auto()
    SVHN = auto()
    SPEECHCOMMANDS = auto()

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
    
    @define(slots=True)
    class ImageDatasetParams(BaseParameters):
        dataset: SupportedDatasets = None
        datafolder: str = ''
        class_idxs: List[int] = None
        custom_transforms: list = None
        max_num_train: int = np.inf
        max_num_test: int = np.inf


    dataset_config = {
        SupportedDatasets.SPEECHCOMMANDS: DatasetConfig(
                                        SpeechCommandDatasetWrapper,
                                        10, 4000, 4000
                                    ),
        SupportedDatasets.MNIST : DatasetConfig(
                                        torchvision.datasets.MNIST,
                                        10, 4500, 5000
                                    ),
        SupportedDatasets.FMNIST : DatasetConfig(
                                        torchvision.datasets.FashionMNIST,
                                        10, 4500, 5000
                                    ),
        SupportedDatasets.SVHN : DatasetConfig(
                                        torchvision.datasets.SVHN,
                                        10, 7000, 3500
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
                                        TinyImagenetNPZDataset,
                                        200, 475, 5000
                                    ),
        SupportedDatasets.IMAGENET10 : DatasetConfig(
                                        TinyImagenetNPZDataset,
                                        10, 1275, 250
                                    ),
        SupportedDatasets.IMAGENET100_64 : DatasetConfig(
                                        TinyImagenetNPZDataset,
                                        100, 1275, 2500
                                    ),
        SupportedDatasets.IMAGENET100_81 : DatasetConfig(
                                        TinyImagenetNPZDataset,
                                        100, 1275, 2500
                                    ),
        SupportedDatasets.IMAGENET75_64 : DatasetConfig(
                                        TinyImagenetNPZDataset,
                                        75, 1275, 2500
                                    ),
        SupportedDatasets.IMAGENET : DatasetConfig(
                                        get_webdataset,
                                        1000, 128, 8
                                    ),
        SupportedDatasets.ECOSET : DatasetConfig(
                                        get_webdataset,
                                        565, 176, 8
                                    ),
        SupportedDatasets.ECOSET_FOLDER : DatasetConfig(
                                        torchvision.datasets.ImageFolder,
                                        565, 5000, 0
                                    ),
        SupportedDatasets.ECOSET10 : DatasetConfig(
                                        TinyImagenetNPZDataset,
                                        10, 4800, 859
                                    ),
        SupportedDatasets.ECOSET10_FOLDER : DatasetConfig(
                                        torchvision.datasets.ImageFolder,
                                        10, 4800, 859
                                    ),
        SupportedDatasets.ECOSET100_FOLDER : DatasetConfig(
                                        torchvision.datasets.ImageFolder,
                                        100, 5000, 0
                                    ),
        SupportedDatasets.ECOSET100 : DatasetConfig(
                                        get_webdataset,
                                        100, 40, 4
                                    ),
        SupportedDatasets.IMAGENET_FOLDER : DatasetConfig(
                                        torchvision.datasets.ImageFolder,
                                        1000, 1271, 10
                                    ),
        # SupportedDatasets.IMAGENET100 : DatasetConfig(
        #                                 get_imagenet_webdataset,
        #                                 100, 25, 1
        #                             ),
        SupportedDatasets.IMAGENET100 : DatasetConfig(
                                        ImagenetFileListDataset,
                                        100, 1275, 2500
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
        print(params, train_class_counts, max_val_counts, test_class_counts)

        if params.custom_transforms is not None:
            train_transform, test_transform = params.custom_transforms
        else:
            train_transform, test_transform = cls.get_transforms(params.dataset)
        print(train_transform)
        print(test_transform)
        
        if params.dataset in [SupportedDatasets.CIFAR10, SupportedDatasets.CIFAR100, SupportedDatasets.MNIST, SupportedDatasets.FMNIST]:
            train_dataset = dataset_class('%s/'%params.datafolder, transform=train_transform, download=True)        
            test_dataset = dataset_class('%s/'%params.datafolder, train=False, transform=test_transform, download=True)
        elif params.dataset == SupportedDatasets.SVHN:
            train_dataset = dataset_class('%s/'%params.datafolder, transform=train_transform, download=True)        
            test_dataset = dataset_class('%s/'%params.datafolder, split='test', transform=test_transform, download=True)

            setattr(train_dataset, 'targets', train_dataset.labels)
            setattr(test_dataset, 'targets', test_dataset.labels)
        elif cfg.dataset_class in [TinyImagenetNPZDataset, ImagenetFileListDataset]: #params.dataset in [SupportedDatasets.TINY_IMAGENET, SupportedDatasets.IMAGENET10, SupportedDatasets.IMAGENET100_64]:
            train_dataset = dataset_class(params.datafolder, split='train', transform=train_transform)
            if os.path.exists(os.path.join(params.datafolder, 'val.pkl.npz')):
                val_dataset = dataset_class(params.datafolder, split='val', transform=train_transform)
            test_dataset = dataset_class(params.datafolder, split='test', transform=test_transform)
        elif (cfg.dataset_class == WebDataset) or (params.dataset in [SupportedDatasets.IMAGENET, SupportedDatasets.ECOSET, SupportedDatasets.ECOSET100]):#, SupportedDatasets.IMAGENET100]:
            num_train_shards = min(params.max_num_train, cfg.min_train_class_counts)
            num_val_shards = cfg.max_val_counts
            num_test_shards = params.max_num_test if params.max_num_test < np.inf else None
            len_shard = 10_000 if params.dataset == SupportedDatasets.IMAGENET else 5_000
            train_dataset = get_webdataset(params.datafolder, params.dataset.value.lower(), nshards=num_train_shards, split='train', transform=train_transform, len_shard=len_shard)
            val_dataset = get_webdataset(params.datafolder, params.dataset.value.lower(), nshards=num_val_shards, split='val', transform=train_transform, len_shard=len_shard)
            test_dataset = get_webdataset(params.datafolder, params.dataset.value.lower(), nshards=num_test_shards, split='test', transform=test_transform, len_shard=len_shard)
            return train_dataset, val_dataset, test_dataset, nclasses
        elif cfg.dataset_class == torchvision.datasets.ImageFolder:
            train_dataset = torchvision.datasets.ImageFolder(os.path.join(params.datafolder, 'train'), transform=train_transform)
            if os.path.exists(os.path.join(params.datafolder, 'test')):
                test_dataset = torchvision.datasets.ImageFolder(os.path.join(params.datafolder, 'test'), transform=test_transform)
                val_dataset = torchvision.datasets.ImageFolder(os.path.join(params.datafolder, 'val'), transform=test_transform)
            else:
                test_dataset = torchvision.datasets.ImageFolder(os.path.join(params.datafolder, 'val'), transform=test_transform)
        elif params.dataset == SupportedDatasets.SPEECHCOMMANDS:
            train_dataset = SpeechCommandDatasetWrapper(params.datafolder, subset='training', download=True)
            val_dataset = SpeechCommandDatasetWrapper(params.datafolder, subset='validation', download=True)
            test_dataset = SpeechCommandDatasetWrapper(params.datafolder, subset='testing', download=True)

        filter_dataset_by_target(train_dataset, params.class_idxs)
        train_idxs, val_idxs = make_val_dataset(train_dataset, nclasses, train_class_counts, class_idxs=params.class_idxs)
        if len(val_idxs) > max_val_counts:
            val_idxs = np.random.choice(val_idxs, max_val_counts, replace=False)
        if 'val_dataset' not in locals():
            val_dataset = torch.utils.data.Subset(train_dataset, val_idxs)
        train_dataset = torch.utils.data.Subset(train_dataset, train_idxs)

        filter_dataset_by_target(test_dataset, params.class_idxs)
        test_idxs, _ = make_val_dataset(test_dataset, nclasses, test_class_counts, class_idxs=params.class_idxs)
        test_dataset = torch.utils.data.Subset(test_dataset, test_idxs)
        print(f'train_dataset_len: {len(train_dataset)}, val_dataset_len: {len(val_dataset)}, test_dataset_len: {len(test_dataset)}')
        return train_dataset, val_dataset, test_dataset, nclasses
