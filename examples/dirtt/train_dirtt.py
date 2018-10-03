""" Training script for DIRT-T and VADA models
"""

import os
import os.path as osp
import sys
import argparse

import torch
from torch import nn

from salad import solver, models, datasets

import salad.datasets.transforms
from salad.datasets.transforms.ensembling import ImageAugmentation

from torch.nn import functional as F
from salad.layers import AccuracyScore
from salad.utils import config

import salad.models.digits.dirtt as model

class Augmentation():
    
    def __init__(self, dataset, n_samples=1):
        self.transformer = ImageAugmentation(
            affine_std=0.1,
            gaussian_noise_std=0.1,
            hflip=False,
            intens_flip=True,
            intens_offset_range_lower=-.5, intens_offset_range_upper=.5,
            intens_scale_range_lower=0.25, intens_scale_range_upper=1.5,
            xlat_range=2.0
        )
        
        self.dataset = dataset
        self.n_samples = n_samples
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        X = torch.stack([x.clone() for _ in range(self.n_samples)], dim=0)
        X = self.transformer.augment(X.numpy())
        outp = [torch.from_numpy(x).float() for x in X] + [y,]
        return outp
        
if __name__ == '__main__':

    parser = config.DomainAdaptConfig("DIRT-T Solver")
    args   = parser.parse_args()

    model   = model.SVHN_MNIST_Model(n_domains=1)
    disc    = nn.Linear(128, 1)

    # Dataset
    data = datasets.da.load_dataset2(path="/tmp/data", train=True)

    train_loader = torch.utils.data.DataLoader(
        Augmentation(data[args.source]), batch_size=args.sourcebatch,
        shuffle=True, num_workers=args.njobs)
    val_loader   = torch.utils.data.DataLoader(
        Augmentation(data[args.target]), batch_size=args.targetbatch,
        shuffle=True, num_workers=args.njobs)

    loader = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment
    experiment = solver.da.VADASolver(model, disc, loader,
                               n_epochs=args.epochs,
                               savedir=args.log,
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)

    experiment.optimize()

    