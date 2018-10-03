""" Minimal Training Script for Associative Domain Adaptation
"""

import os
import os.path as osp
import sys
import argparse

import torch
from torch import nn

from salad import solver, datasets, optim

from torch.nn import functional as F
from salad.layers import AccuracyScore

from salad import solver
from salad.datasets.transforms.ensembling import ImageAugmentation
from salad.utils import config

import salad.models.digits.ensemble as models

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

    parser = config.DomainAdaptConfig("Ensembling Solver")
    args   = parser.parse_args()

    # Network
    
    # Optimize the student on the supervised loss. Transfer weights to the teacher
    # Vary the weight sharing between teacher and student and gradually let the teacher
    # take over the dominant role
    #
    # student params <- CE
    # student bn src <- CE
    # student bn tgt <- Prox Label
    # teacher params <- EMA
    # teacher bn     <- student src * a + student tgt * (1 - a)
    model   = models.SVHN_MNIST_Model(n_domains=2)
    teacher = models.SVHN_MNIST_Model(n_domains=1)
    
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Dataset
    from torchvision import transforms
    from salad.datasets.transforms.noise import SaltAndPepper, Gaussian
    from salad.datasets.da.digits import NoiseLoader

    loader = NoiseLoader('/tmp/data', 'svhn', collate = 'stack',
                            noisemodels=[lambda x : x, SaltAndPepper(0.05)],
                            batch_size = 32, shuffle = True

                            )
    source = loader.datasets[0].dataset
    target = loader.datasets[1].dataset

    train_loader = torch.utils.data.DataLoader(
        Augmentation(source), batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        Augmentation(target, 2), batch_size=args.targetbatch,
        shuffle=True, num_workers=4)

    loader = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment
    experiment = solver.SelfEnsemblingSolver(model, teacher, loader,
                               n_epochs=args.epochs,
                               savedir=args.log,
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)

    experiment.optimize()

    
