# !/bin/env python3
# encoding: utf-8

""" Comparison Study of Recent Unsupervised Domain Adaptation Approaches on Digit Benchmark
"""

import torch
from torch import nn
import numpy as np

from salad.datasets.da import toy
from salad.utils import config
from salad.datasets.da import DigitsLoader
from salad.models.digits import DigitModel

from salad.datasets.transforms import Augmentation


class ComparisonConfig(config.DomainAdaptConfig):
    """ Configuration for the comparison study
    """

    _algorithms = ['adv', 'vada', 'dann', 'assoc', 'coral', 'teach']
    _algo_names = [
        "Adversarial Domain Regularization",
        "Virtual Adversarial Domain Adaptation",
        "Domain Adversarial Training",
        "Associative Domain Adaptation",
        "Deep Correlation Alignment",
        "Self-Ensembling"
    ]

    def _init(self):
        super()._init()
        self.add_argument('--seed', default=None, type=int, help="Random Seed")
        self.add_argument('--print', action='store_true')
        self.add_argument('--null', action='store_true')

        for arg, name in zip(self._algorithms, self._algo_names):
            self.add_argument('--{}'.format(arg), action='store_true', help="Train a model with {}".format(name))

def print_experiments():
    import itertools

    datasets = ['mnist', 'synth', 'svhn']
    algos = ComparisonConfig._algorithms

    s = []

    for algo, source, target in itertools.product(algos, datasets, datasets):

        if source == target:
            continue

        print('python3 {} --{} --source {} --target {} {}'.format(
            __file__,
            algo,
            source,
            target,
            '--epochs 10'
        ))

def experiment_setup(args):
    """ Set default params and construct models for various experiments
    """

    model = DigitModel()
    teacher = DigitModel()
    disc = nn.Linear(128, 1)

    kwargs = {
        'n_epochs': args.epochs,
        'multiclass': True,
        'learningrate': 3e-4,
        'gpu': None if args.cpu else args.gpu,
        'savedir': 'log/new14-{}-{}'.format(args.source, args.target),
        'dryrun': args.dryrun
    }

    return model, teacher, disc, kwargs


if __name__ == '__main__':

    from torch.utils.data import TensorDataset, DataLoader
    from salad.datasets import DigitsLoader
    from salad import solver
    import sys

    parser = ComparisonConfig('Domain Adaptation Comparision and Reproduction Study')
    args = parser.parse_args()

    if args.print:
        print_experiments()
        sys.exit(0)

    parser.print_config()

    dataset_names = [args.source, args.target]

    loader_plain   = DigitsLoader('/tmp/data', dataset_names, download=True, shuffle=True, batch_size=128, normalize=True, num_workers=4)
    loader_augment = DigitsLoader('/tmp/data', dataset_names, download=True, shuffle=True, batch_size=128, num_workers=4,
                                  normalize=True, augment={args.target: 2})

    if args.adv:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.AdversarialDropoutSolver(model, loader_plain, **kwargs)
        experiment.optimize()

    if args.vada:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.VADASolver(model, disc, loader_plain, **kwargs)
        experiment.optimize()

    if args.dann:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.DANNSolver(model, disc, loader_plain, **kwargs)
        experiment.optimize()

    if args.assoc:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.AssociativeSolver(model, loader_plain, **kwargs)
        experiment.optimize()

    if args.coral:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.DeepCoralSolver(model, loader_plain, use_nullspace = args.null, **kwargs)
        experiment.optimize()

    if args.teach:
        model, teacher, disc, kwargs = experiment_setup(args)
        experiment = solver.SelfEnsemblingSolver(model, teacher, loader_augment, **kwargs)
        experiment.optimize()
