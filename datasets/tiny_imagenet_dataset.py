import json
import os
from typing import Callable, Optional
import numpy as np
import torchvision
from PIL import Image

class TinyImagenetNPZDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, train=True, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if train:
            npz = np.load(os.path.join(root, 'train.pkl.npz'))
        else:
            npz = np.load(os.path.join(root, 'test.pkl.npz'))
        with open(os.path.join(root, 'idx2word.json')) as f:
            idx2word = json.load(f)
        self.class_to_idx = {v:k for k,v in idx2word.items()}
        self.samples = [Image.fromarray(x) for x in npz['X']]
        self.targets = npz['Y']
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        x = self.samples[index]
        y = self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        return len(self.samples)
    
