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

from salad.datasets.transforms import Augmentation

if __name__ == '__main__':
    parser = config.DomainAdaptConfig("Ensembling Solver")
    args = parser.parse_args()
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
    model = models.SVHN_MNIST_Model(n_domains=2)
    teacher = models.SVHN_MNIST_Model(n_domains=1)

    for param in teacher.parameters():
    param.requires_grad_(False)
    # Dataset
    data = datasets.da.load_dataset2(path="data", train=True)
    train_loader = torch.utils.data.DataLoader(
        Augmentation(data[args.source]), batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        Augmentation(data[args.target], 2), batch_size=args.targetbatch,
        shuffle=True, num_workers=4)
    loader = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment
    experiment = solver.SelfEnsemblingSolver(model, teacher, loader,
                                             n_epochs=args.epochs,
                                             savedir=args.log,
                                             dryrun=args.dryrun,
                                             learningrate=args.learningrate,
                                             gpu=args.gpu if not args.cpu else None)
    experiment.optimize()
