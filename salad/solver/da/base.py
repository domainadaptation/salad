""" Solver classes for domain adaptation experiments 
"""

__author__ = "Steffen Schneider"
__email__  = "steffen.schneider@tum.de"

import os, time
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn

from .. import Solver, BaseClassSolver
from ... import layers, optim

import itertools

class DABaseSolver(BaseClassSolver):
    
    """ Base Class for Unsupervised Domain Adaptation Approaches

    """

    def __init__(self, *args, **kwargs):
        super(DABaseSolver, self).__init__(*args, **kwargs)
    
    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(layers.AccuracyScore(), name = 'acc_s', weight = None)
        self.register_loss(layers.AccuracyScore(), name = 'acc_t', weight = None)

class DATeacher(Solver):
    
    """ Base Class for Unsupervised Domain Adaptation Approaches using a teacher model
    """

    def __init__(self, model, teacher, dataset, *args, **kwargs):
        super().__init__(model, dataset, *args, **kwargs)
        self.teacher = teacher
        
    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)
        self.register_model(self.teacher, 'teacher')

class DABaselineLoss(object):

    def __init__(self, solver):
        self.solver = solver

    def _predict(self, x, y):

        _ , y_ = self.solver.model(x)
        if not self.solver.multiclass:
            y_ = y_.squeeze()
            y  = y.float()

        return y_, y

    def __call__(self, batch):
        losses = {}
        (x, y) = batch[0]

        losses['acc_s'] = losses['ce'] = self._predict(x,y)

        with torch.no_grad():
            x,y = batch[1]
            losses['acc_t'] =  self._predict(x,y)

        return losses

class BaselineDASolver(DABaseSolver):
    """ A domain adaptation solver that actually does not run any adaptation algorithm

    This is useful to establish baseline results for the case of no adaptation, for measurement
    of the domain shift between datasets.
    """

    def _init_optims(self, lr = 3e-4, **kwargs):
        super()._init_optims(**kwargs)

        self.register_optimizer(torch.optim.Adam(self.model.parameters(),
                                lr=lr, amsgrad=True),
                                DABaselineLoss(self))