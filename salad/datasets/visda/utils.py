import os
import os.path as osp
import sys
import argparse

import torch
from torch import nn

from salad import solver, datasets, optim

# from augment import AffineTransformer
# from augment2 import ImageAugmentation

from torch.nn import functional as F
from salad.layers import AccuracyScore, MeanAccuracyScore

import torch
from torch import nn
from torch.utils.data import DataLoader

from salad import solver, optim, layers
from torchvision.models import resnet

import salad
import salad.datasets

from torchvision import datasets, transforms
import os.path as osp

from torch.utils.data.sampler import WeightedRandomSampler

class Augmentation():
    
    def __init__(self, n_samples=1):
        self.transformer = ImageAugmentation(
            affine_std=0.1,
            gaussian_noise_std=0.1,
            hflip=False,
            intens_flip=False,
            intens_offset_range_lower=-.5, intens_offset_range_upper=.5,
            intens_scale_range_lower=0.25, intens_scale_range_upper=1.5,
            xlat_range=2.0
        )
        
        self.n_samples = n_samples
        
    def __call__(self, x):
        
        X = torch.stack([x.clone() for _ in range(self.n_samples)], dim=0)
        X = self.transformer.augment(X.numpy())
        
        outp = [torch.from_numpy(x).float() for x in X]

        if len(outp) == 1:
            return outp[0]
                
        return outp

class MultiTransform():
    
    def __init__(self, transforms, n_samples=1):
        self.transforms = transforms
        self.n_samples  = n_samples
        
    def __call__(self, x):

        outp = [ self.transforms(x) for i in range(self.n_samples) ]
        
        if len(outp) == 1:
            return outp[0]
                
        return outp

def get_class_counts(data):
    import pandas as pd
    import numpy as np
    
    lbl_counts = {l : 0 for l in data.class_to_idx.values()}

    df = pd.DataFrame(data.samples, columns=['fname', 'label'])
    counts = df.label.value_counts(sort=False).to_dict() #.values
    
    lbl_counts.update(counts)

    return np.array([lbl_counts[i] for i in range(len(lbl_counts))])

def get_balanced_loader(data, **kwargs):

    counts = get_class_counts(data)
    weights = 1. / (1e-5 + counts)
    weights[counts == 0] = 0.
    weights = torch.from_numpy(weights / weights.sum()).float()

    print('Class Counts', counts)
    print('Weights', weights)

    sampler = WeightedRandomSampler(weights, kwargs.get('batch_size'))
    loader = DataLoader(data, sampler = sampler, **kwargs)

    return loader

def get_unbalanced_loader(data, **kwargs):

    loader = DataLoader(data, drop_last = True, **kwargs)

    return loader


def visda_data_loader(path, batch_size, n_src = 1, n_tgt=1):
    T = lambda i : transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(.1, .8, .75, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
        Augmentation(i)
    ])
    train      = datasets.ImageFolder(osp.join(path, 'train/classes'), transform=T(n_src))
    validation = datasets.ImageFolder(osp.join(path, 'validation/classes'), transform=T(n_tgt))

    loader = salad.datasets.JointLoader(
                get_balanced_loader(train, shuffle=True, batch_size=batch_size, num_workers=24),
                get_balanced_loader(validation, shuffle=True, batch_size=batch_size, num_workers=24)
    )

    return loader

#def visda_data_loader_full(path, batch_size, n_src = 1, n_tgt=1):
#    T = lambda i : transforms.Compose([
#        transforms.Resize(256),
#        transforms.RandomResizedCrop(224),
#        #transforms.RandomCrop(224),
#        transforms.ColorJitter(.1, .8, .75, 0),
#        transforms.ToTensor(),
#        transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                             std  = [0.229, 0.224, 0.225]),
#        Augmentation(i)
#    ])
#    train      = datasets.ImageFolder(osp.join(path, 'train'), transform=T(n_src))
#    validation = datasets.ImageFolder(osp.join(path, 'validation'), transform=T(n_tgt))
#
#    loader = salad.datasets.JointLoader(
#                get_balanced_loader(train, shuffle=True, batch_size=batch_size, num_workers=12),
#                get_balanced_loader(validation, shuffle=True, batch_size=batch_size, num_workers=12)
#    )
#
#    return loader

def visda_data_loader_pseudo(path, batch_size, n_aug = 1):
    T = lambda i : transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(.1, .8, .75, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225]),
        Augmentation(i)
    ])
    data = datasets.ImageFolder(osp.join(path, 'pseudo-5'), transform=T(n_aug))

    return get_balanced_loader(data, shuffle=True, batch_size=batch_size, num_workers=12)

def visda_data_loader_full(path, batch_size, n_src = 1, n_tgt=1):
#def new_visda_data_loader(path, batch_size, n_aug = 1, which='train'):
    T = lambda i : MultiTransform(
                transforms.Compose([
                transforms.RandomAffine(15, translate=(0,.1), scale=None, shear=10, resample=False, fillcolor=0),
                transforms.Resize(224),
                transforms.RandomResizedCrop(160, scale=(0.4, 1.0)),
                transforms.RandomGrayscale(p=.5),
                transforms.RandomHorizontalFlip(.5),
                transforms.ColorJitter(.1, .5, .5, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std  = [0.229, 0.224, 0.225]),
            ]), n_samples = i
        )

    train      = datasets.ImageFolder(osp.join(path, 'train'), transform=T(n_src))
    validation = datasets.ImageFolder(osp.join(path, 'validation'), transform=T(n_tgt))

    loader = salad.datasets.JointLoader(
                get_unbalanced_loader(train, shuffle=True, batch_size=batch_size, num_workers=12),
                get_unbalanced_loader(validation, shuffle=True, batch_size=batch_size, num_workers=12)
    )

    return loader