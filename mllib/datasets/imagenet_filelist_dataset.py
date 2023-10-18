import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image

META_FILE = "meta.bin"

def readlines_and_strip(fp):
    with open(fp) as f:
        lines = [l.strip() for l in f.readlines()]
    return lines

def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)
    return torch.load(file)

class ImagenetFileListDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, split='train', transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        filelist_path = os.path.join(root, f'{split}_paths.txt')
        pth2cls = lambda pth: pth.split('/')[-2]
        filelist = readlines_and_strip(filelist_path)
        filelist = sorted(filelist, key=pth2cls)
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        for pth in filelist:
            self.samples.append(os.path.join(root, pth))
            cls = pth2cls(pth)
            self.targets.append(self.class_to_idx.setdefault(cls, len(self.class_to_idx)))
        self.classes = sorted(list(self.class_to_idx.keys()), key=lambda x: self.class_to_idx[x])
        
        root = self.root = os.path.expanduser(root)
        if os.path.exists(os.path.join(self.root, 'meta.bin')):
            wnid_to_classes = load_meta_file(self.root)[0]
            self.root = root

            self.wnids = self.classes
            self.wnid_to_idx = self.class_to_idx
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
            self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

    def __getitem__(self, index: int) -> Any:
        try:
            pth = self.samples[index]
            with open(pth, 'rb') as f:
                x = Image.open(f).convert("RGB")
            y = self.targets[index]
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
        except:
            print(f'Error loading {self.samples[index]}')
            x, y = self.__getitem__(np.random.randint(0, len(self.samples)))

        return x, y
    
    def __len__(self) -> int:
        return len(self.samples)

def identity(x):
    return x

def toFloatTensor(x):
    return torch.FloatTensor(x)

def get_webdataset(root, dataset_name, first_shard_idx=0, nshards=None, split='train', transform=None, shuffle=10_000, len_shard=10_000):
    if split =='train':
        if nshards is None:
            nshards = 127
        urls = dataset_name+"-train-{}.tar"
    elif split =='val':
        if nshards is None:
            nshards = 8
        urls = dataset_name+"-trainval-{}.tar"
    elif split == 'test':
        shuffle = 0
        if nshards is None:
            nshards = 7
        urls = dataset_name+"-val-{}.tar"
    else:
        raise ValueError(f'split must be one of train, val, or test, but got {split}')
    if nshards > 1:
        shard_str = f'{{{first_shard_idx:06d}..{first_shard_idx+nshards-1:06d}}}'
    else:
        shard_str = f'{first_shard_idx:06d}'
    urls = os.path.join(root, urls.format(shard_str))
    print(nshards, urls)
    dataset = wds.WebDataset(urls, shardshuffle=(split=='train'), nodesplitter=wds.split_by_node)
    if split == 'train':
        dataset = dataset.shuffle(shuffle, initial=shuffle//2)
    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(transform, identity)
        # .with_length(nshards*len_shard)
    )
    return dataset

def get_clickme_webdataset(root, dataset_name, first_shard_idx=0, nshards=None, split='train', transform=None, shuffle=10_000, len_shard=10_000):
    if split =='train':
        urls = dataset_name+"-train-{}.tar"
    elif split =='val':
        urls = dataset_name+"-trainval-{}.tar"
    elif split == 'test':
        shuffle = 0
        urls = dataset_name+"-val-{}.tar"
    else:
        raise ValueError(f'split must be one of train, val, or test, but got {split}')
    if nshards > 1:
        shard_str = f'{{{first_shard_idx:06d}..{first_shard_idx+nshards-1:06d}}}'
    else:
        shard_str = f'{first_shard_idx:06d}'
    urls = os.path.join(root, urls.format(shard_str))
    print(nshards, urls)
    dataset = wds.WebDataset(urls, shardshuffle=(split=='train'), nodesplitter=wds.split_by_node)
    if split == 'train':
        dataset = dataset.shuffle(shuffle, initial=shuffle//2)
    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls heatmap.npy")
        .map_tuple(transform, identity, toFloatTensor)
        # .with_length(nshards*len_shard)
    )
    return dataset
        