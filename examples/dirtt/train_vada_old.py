""" Minimal Training Script for Associative Domain Adaptation
"""

import os
import os.path as osp
import sys
import argparse

import torch

from salad import solver, models, datasets


def build_parser():

    parser = argparse.ArgumentParser(description='Associative Domain Adaptation')

    # General setup
    parser.add_argument('--gpu', default=0, help='Specify GPU', type=int)
    parser.add_argument('--cpu', action='store_true', help="Use CPU Training")
    parser.add_argument('--log', default="./log/log2", help="Log directory. Will be created if non-existing")
    parser.add_argument('--epochs', default="1000", help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
    parser.add_argument('--checkpoint', default="", help="Checkpoint path")
    parser.add_argument('--learningrate', default=3e-4, type=float, help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
    parser.add_argument('--dryrun', action='store_true', help="Perform a test run, without actually training a network. Usefule for debugging.")

    # Domain Adaptation Args
    parser.add_argument('--source', default="svhn", choices=['mnist', 'svhn'], help="Source Dataset. Choose mnist or svhn")
    parser.add_argument('--target', default="mnist", choices=['mnist', 'svhn'], help="Target Dataset. Choose mnist or svhn")

    parser.add_argument('--sourcebatch', default=128, type=int, help="Batch size of Source")
    parser.add_argument('--targetbatch', default=128, type=int, help="Batch size of Target")

    # Associative DA Hyperparams
    parser.add_argument('--dirtt', action='store_true', help='Start DIRT-T')

    return parser

if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()

    # Network
    if osp.exists(args.checkpoint):
        print("Resume from checkpoint file at {}".format(args.checkpoint))
        model = torch.load(args.checkpoint)
    else:
        model   = models.FrenchModel()

    # Dataset
    data = datasets.load_dataset(path="data", train=True)

    train_loader = torch.utils.data.DataLoader(
        data[args.source], batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        data[args.target], batch_size=args.targetbatch,
        shuffle=True, num_workers=4)

    loader = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment

    if args.dirtt:
        teacher = models.FrenchModel()
        experiment = solver.DIRTTSolver(model, teacher, val_loader,
                                n_epochs=args.epochs,
                                savedir=args.log,
                                dryrun = args.dryrun,
                                learningrate = args.learningrate,
                                gpu=args.gpu if not args.cpu else None)
    else:
        experiment = solver.VADASolver(model, loader,
                                n_epochs=args.epochs,
                                savedir=args.log,
                                dryrun = args.dryrun,
                                learningrate = args.learningrate,
                                gpu=args.gpu if not args.cpu else None)

    experiment.optimize()
