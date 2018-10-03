""" Minimal Training Script for Associative Domain Adaptation
"""

import os
import os.path as osp
import sys
import argparse

import torch

from salad import solver, datasets, models

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
from collections import OrderedDict

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return x, y

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)

def svhn(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['svhn'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


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

    parser.add_argument('--sourcebatch', default=64, type=int, help="Batch size of Source")
    parser.add_argument('--targetbatch', default=64, type=int, help="Batch size of Target")

    # Associative DA Hyperparams
    parser.add_argument('--visit', default=0.1, type=float, help="Visit weight")
    parser.add_argument('--walker', default=1.0, type=float, help="Walker weight")

    return parser

if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()

    # Network
    if osp.exists(args.checkpoint):
        print("Resume from checkpoint file at {}".format(args.checkpoint))
        model = torch.load(args.checkpoint)
    else:
        model   = models.SVHNmodel()

    # Dataset
    data = datasets.load_dataset(path="data", train=True, img_size = 32)

    train_loader = torch.utils.data.DataLoader(
        data[args.source], batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        data[args.target], batch_size=args.targetbatch,
        shuffle=True, num_workers=4)

    dataset = datasets.JointLoader(train_loader, val_loader)

    # Initialize the solver for this experiment
    experiment = solver.DANNSolver(model, dataset,
                               n_epochs=args.epochs,
                               savedir=args.log,
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)


    experiment.optimize()
