import torchvision
import os
import torch
import numpy as np

class ImageFolderWithAnnotations(torchvision.datasets.ImageFolder):
    def __init__(self, annotation_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_dir = annotation_dir

    def __getitem__(self, index: int):
        c, fp = self.samples[index][0].split('/')[-2:]
        ann_pth = f'{self.annotation_dir}/{c}/{fp}'
        if os.path.exists(ann_pth):
            ann = torch.FloatTensor(np.loadtxt(ann_pth, usecols=[0,1,2,3]))
            if ann.dim() > 1:
                ann = ann[0]
        else:
            ann = torch.zeros((4,)).float() + 0.5
        x, y = super().__getitem__(index)
        return x, y, ann