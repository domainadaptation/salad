""" Minimal Training Script for Associative Domain Adaptation
"""

import sys
import os
import os.path as osp

import torch
from torch import nn

from salad import solver, datasets
import salad.models.digits.assoc as models

from salad.utils import config 

class AssociationConfig(config.DomainAdaptConfig):
    def _init(self):
        super()._init()
        # Associative DA Hyperparams
        self.add_argument('--visit', default=0.1, type=float, help="Visit weight")
        self.add_argument('--walker', default=1.0, type=float, help="Walker weight")

if __name__ == '__main__':

    parser = AssociationConfig('Associative Domain Adaptation')
    args   = parser.parse_args()

    # Network
    if osp.exists(args.checkpoint):
        print("Resume from checkpoint file at {}".format(args.checkpoint))
        model = torch.load(args.checkpoint)
    else:
        model = models.SVHNmodel()

    # Dataset
    data = datasets.da.load_dataset2(path="/tmp/data", train=True)

    train_loader = torch.utils.data.DataLoader(
        data[args.source], batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        data[args.target], batch_size=args.targetbatch,
        shuffle=True, num_workers=4)

    dataset = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment

    experiment = solver.AssociativeSolver(model, dataset,
                               n_epochs=args.epochs,
                               savedir=args.log,
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)

    experiment.optimize()
