import torchvision
import torch
from torch import nn
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def get_datasets(datafolder):
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: x[0])
                ])
    train_dataset = torchvision.datasets.ImageFolder(datafolder, 
                                                        transform=transform,
                                                        is_valid_file=lambda x: int(os.path.basename(x).split('.')[0]) < 3000)
    val_dataset = torchvision.datasets.ImageFolder(datafolder, 
                                                        transform=transform,
                                                        is_valid_file=lambda x: int(os.path.basename(x).split('.')[0]) in np.arange(3000, 4000))
    test_dataset = torchvision.datasets.ImageFolder(datafolder, 
                                                        transform=transform,
                                                        is_valid_file=lambda x: int(os.path.basename(x).split('.')[0]) >= 4000)
    num_classes = len(os.listdir(datafolder))
    return train_dataset, val_dataset, test_dataset, num_classes

def make_val_dataset(dataset, num_classes, train_class_counts, class_idxs=None):
    train_idxs = []        
    val_idxs = []
    class_counts = {i:0 for i in range(num_classes)}

    for i,y in tqdm(enumerate(dataset.targets)):
        # y = sample[1]
        if class_counts[int(y)] < train_class_counts:
            train_idxs.append(i)
            class_counts[int(y)] += 1
        else:
            val_idxs.append(i)
    print(len(train_idxs), len(val_idxs))
    return train_idxs, val_idxs

def filter_dataset_by_target(dataset, class_idxs):
    if class_idxs is not None:
        dataset.targets = np.array([class_idxs.index(t) if t in class_idxs else -1 for t in dataset.targets])
        if hasattr(dataset, 'data'):
            dataset.data = dataset.data[dataset.targets != -1]
        elif hasattr(dataset, 'samples'):
            dataset.samples = [s for s in dataset.samples if s[1] != -1]
        else:
            raise NotImplementedError(f"{type(dataset)} does not have an attribute data or samples")
        dataset.targets = dataset.targets[dataset.targets != -1]

def get_mnist_dataset(datafolder):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),        
    ])
    nclasses = 10
    train_class_counts = 5000
    train_dataset = torchvision.datasets.MNIST('%s/'%datafolder, download=True, transform=transform)
    train_idxs, val_idxs = make_val_dataset(train_dataset, nclasses, train_class_counts)
    val_dataset = torch.utils.data.Subset(train_dataset, val_idxs)
    train_dataset = torch.utils.data.Subset(train_dataset, train_idxs)   
    test_dataset = torchvision.datasets.MNIST('%s/'%datafolder, train=False, download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset, nclasses

def add_occlusion(img, num_occlusions, min_occlusion_len, max_occlusion_len, fill_noise=False):
        b = img.size(0)
        c = img.size(1)
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((b, 1, h, w), np.float32)
        mask_coo = []
        for n in range(num_occlusions):
            y = np.random.randint(h, size=b)
            x = np.random.randint(w, size=b)
            x_length = np.random.randint(min_occlusion_len, max_occlusion_len, b)
            y_length = np.random.randint(min_occlusion_len, max_occlusion_len, b)

            y1 = np.clip(y - y_length // 2, 0, h)
            y2 = np.clip(y + y_length // 2, 0, h)
            x1 = np.clip(x - x_length // 2, 0, w)
            x2 = np.clip(x + x_length // 2, 0, w)
            mask_coo.append([y1, y2, x1, x2])
        for y1, y2, x1, x2 in mask_coo:
            for i in range(b):
                mask[i, :, y1[i]: y2[i], x1[i]: x2[i]] = 0.
        noise = np.random.uniform(0, 1, img.shape)
        masked_noise = noise * (1-mask)

        mask = torch.from_numpy(mask).to(img.device)
        masked_noise = torch.from_numpy(masked_noise).to(img.device)
        img = img * mask
        if fill_noise:
            img += masked_noise

        return img.float(), mask.float()