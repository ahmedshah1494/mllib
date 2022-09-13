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
    def __init__(self, root: str, train=True, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if train:
            filelist_path = os.path.join(root, 'train_paths.txt')
        else:
            filelist_path = os.path.join(root, 'val_paths.txt')
        filelist = readlines_and_strip(filelist_path)
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        for pth in filelist:
            self.samples.append(os.path.join(root, pth))
            cls = pth.split('/')[-2]
            self.targets.append(self.class_to_idx.setdefault(cls, len(self.class_to_idx)))
        self.classes = sorted(list(self.class_to_idx.keys()), key=lambda x: self.class_to_idx[x])
        
        root = self.root = os.path.expanduser(root)
        wnid_to_classes = load_meta_file(self.root)[0]
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

    def __getitem__(self, index: int) -> Any:
        pth = self.samples[index]
        with open(pth, 'rb') as f:
            x = Image.open(f).convert("RGB")
        y = self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
    
    def __len__(self) -> int:
        return len(self.samples)

def identity(x):
    return x

def get_imagenet_webdataset(root, first_shard_idx=0, nshards=None, train=True, transform=None, shuffle=10_000, len_shard=10_000):
    if train:
        if nshards is None:
            nshards = 1282
        urls = "imagenet-train-{}.tar"
    else:
        shuffle = 0
        if nshards is None:
            nshards = 7
        urls = "imagenet-val-{}.tar"
    if nshards > 1:
        shard_str = f'{{{first_shard_idx:06d}..{first_shard_idx+nshards-1:06d}}}'
    else:
        shard_str = f'{first_shard_idx:06d}'
    urls = os.path.join(root, urls.format(shard_str))
    print(nshards, urls)
    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(shuffle, initial=shuffle//2)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(transform, identity)
        .with_length(nshards*len_shard)
    )
    return dataset
        