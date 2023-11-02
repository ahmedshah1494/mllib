import torchvision
import torch
import numpy as np
from typing import Literal
from mllib.datasets.imagenet_filelist_dataset import ImagenetFileListDataset

class FixationPointDataset(ImagenetFileListDataset):
    def __init__(self, root: str, split, fixation_map_root: str, fixation_target: Literal['logit', 'correct', 'prob'] = 'prob', 
                    transforms = None, transform = None, target_transform = None, fmap_transform=None) -> None:
        super().__init__(root, split, transforms, transform, target_transform)
        self.fixation_map_root = f'{fixation_map_root}/{split}'
        self.fixation_target = fixation_target
        print('fmap_transform:', fmap_transform)
        self.fmap_transform = fmap_transform
    
    def load_fixation_maps(self, path):
        a = np.load(path)
        if self.fixation_target == 'logit':
            m = a['fixation_logits']
        elif self.fixation_target == 'correct':
            m = a['fixation_correct']
        elif self.fixation_target == 'prob':
            m = a['fixation_probs']
        m = torch.FloatTensor(m)
        if self.fmap_transform is not None:
            m = self.fmap_transform(m)
        return m

    def __getitem__(self, index: int):
        ipth = self.samples[index]
        [cdir, fname] = ipth.split('/')[-2:]
        mpth = f'{self.fixation_map_root}/{cdir}/{fname.split(".")[0]}.npz'
        try:
            m = self.load_fixation_maps(mpth)
            x, y = super().__getitem__(index)
        except Exception as exp:
            # print(f'failed to load {mpth}')
            print(exp)
            x, y, m = self.__getitem__(np.random.randint(0, len(self.samples)))
        return x, y, m