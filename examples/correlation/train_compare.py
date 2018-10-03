import sys
import os
import os.path as osp
sys.path.append(osp.dirname(__file__))

import torch
from torch import nn

from salad import solver, datasets
from salad.utils import config 

import salad.models.digits.corr as models

from salad.layers import coral

class CorrelationAlignmentConfig(config.DomainAdaptConfig):

    funcs = {
        'coral'    : coral.CoralLoss,
        'logcoral' : coral.LogCoralLoss,
        'stein'    : coral.SteinDivergence,
        'jeffrey'  : coral.JeffreyDivergence,
        'affine'   : coral.AffineInvariantDivergence
    }
    
    def _init(self):
        super()._init()
        self.add_argument('--dist', default="coral", choices=self.funcs.keys(),
                            help="Distance Function for Correlation Alignment")
    
    def get_func(self, key):
        return self.funcs[key]()

if __name__ == '__main__':

    parser = CorrelationAlignmentConfig('Correlation Distance Comparision')
    args   = parser.parse_args()

    # Network
    if osp.exists(args.checkpoint):
        print("Resume from checkpoint file at {}".format(args.checkpoint))
        model = torch.load(args.checkpoint)
    else:
        model = models.SVHNmodel()

    # Dataset
    data = datasets.da.load_dataset2(path="data", train=True)

    train_loader = torch.utils.data.DataLoader(
        data[args.source], batch_size=args.sourcebatch,
        shuffle=True, num_workers=args.njobs)
    val_loader   = torch.utils.data.DataLoader(
        data[args.target], batch_size=args.targetbatch,
        shuffle=True, num_workers=args.njobs)

    dataset = datasets.JointLoader(train_loader, val_loader)

    dist = parser.get_func(args.dist)

    # Initialize the solver for this experiment
    experiment = solver.da.CorrelationDistanceSolver(model, dataset,
                            corr_dist = dist,
                            n_epochs=args.epochs,
                            savedir=args.log,
                            dryrun = args.dryrun,
                            learningrate = args.learningrate,
                            gpu=args.gpu if not args.cpu else None)

    experiment.optimize()