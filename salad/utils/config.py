""" Experiment Configurations for ``salad``

This file contains classes to easily configure experiments for different solvers
available in ``salad``.
"""


import sys
import argparse

class BaseConfig(argparse.ArgumentParser):

    """ Basic configuration with arguments for most deep learning experiments
    """
    
    def __init__(self, description, log = './log'):
        super().__init__(description=description)

        self.log = log
        self._init()

    def _init(self):

        self.add_argument('--gpu', default=0,
            help='Specify GPU', type=int)
        self.add_argument('--cpu', action='store_true',
            help="Use CPU Training")
        self.add_argument('--njobs', default=4,
            help='Number of processes per dataloader', type=int)
        self.add_argument('--log', default=self.log,
            help="Log directory. Will be created if non-existing")
        self.add_argument('--epochs', default="100",
            help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
        self.add_argument('--checkpoint', default="",
            help="Checkpoint path")
        self.add_argument('--learningrate', default=1e-3, type=float,
            help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
        self.add_argument('--dryrun', action='store_true',
            help="Perform a test run, without actually training a network.")

    def print_config(self):
        print("Start Experiments")


class DomainAdaptConfig(BaseConfig):
    """ Base Configuration for Unsupervised Domain Adaptation Experiments
    """

    def _init(self):
        super()._init()

        self.add_argument('--source', default="svhn", choices=['mnist', 'svhn', 'usps', 'synth', 'synth-small'],
                            help="Source Dataset. Choose mnist or svhn")
        self.add_argument('--target', default="mnist", choices=['mnist', 'svhn', 'usps', 'synth', 'synth-small'],
                            help="Target Dataset. Choose mnist or svhn")

        self.add_argument('--sourcebatch', default=128, type=int,
                            help="Batch size of Source")
        self.add_argument('--targetbatch', default=128, type=int,
                            help="Batch size of Target")