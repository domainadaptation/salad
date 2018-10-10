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
    
    r""" Base Class for Unsupervised Domain Adaptation Approaches

    Unsupervised DA assumes the presence of a single source domain :math:`\mathcal S`
    along with a target domain :math:`\mathcal T` known at training time.
    Given a labeled sample of points drawn from :math:`\mathcal S`, :math:`\{x^s_i, y^s_i\}_{i}^{N_s}`,
    and an unlabeled sample drawn from :math:`\mathcal T`, :math:`\{x^t_i\}_{i}^{N_t}`, unsupervised
    adaptation aims at minimizing the 
    
    .. math::
        \min_\theta \mathcal{R}^l_{\mathcal S} (\theta) +  \lambda \mathcal{R}^u_{\mathcal {S \times T}} (\theta), 

    leveraging an unsupervised risk term :math:`\mathcal{R}^u_{\mathcal {S \times T}} (\theta)` that depends on
    feature representations :math:`f_\theta(x^s,s)` and :math:`f_\theta(x^t,t)`, 
    classifier labels :math:`h_\theta(x^s,s), h_\theta(x^t,t)` as well as source labels :math:`y^s`.
    The full model :math:`h = g \circ f` is a composition of a feature extractor :math:`f` and classifier :math:`g`, both of which
    can possibly depend on the domain label :math:`s` or :math:`t` for domain-specific computations.

    Notes
    -----
    This solver adds two accuracies with keys ``acc_s`` and ``acc_t`` for the source and target domain, respectively.
    Make sure to include derivation of these accuracy in your loss computation.

    """

    def __init__(self, *args, **kwargs):
        super(DABaseSolver, self).__init__(*args, **kwargs)
    
    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(layers.AccuracyScore(), name = 'acc_s', weight = None)
        self.register_loss(layers.AccuracyScore(), name = 'acc_t', weight = None)


class DGBaseSolver(BaseClassSolver):

    r""" Base Class for Domain Generalization Approaches
    
    Domain generalization assumes the presence of multiple source domains alongside
    a target domain unknown at training time.
    Following \cite{Shankar2018}, this setting requires a dataset of training examples
    :math:`\{x_i, y_i, d_i\}_{i}^{N}` with class and domain labels.
    Importantly, the domains present at training time should reflect the kind of variability
    that can be expected during inference.
    The ERM problem is then approached as

    .. math::

        \min_\theta \sum_d \mathcal{R}^l_{\mathcal S_d} (\theta) 
        = \sum_d \lambda_d \mathbb{E}_{x,y \sim \mathcal S_d }[\ell ( f_\theta(x), h_\theta(x), y, d) ].

    In contrast to the unsupervised setting, samples are now presented in a single batch
    comprised of inputs :math:`x`, labels :math:`y` and domains :math:`d`.
    In a addition to a feature extractor :math:`f_\theta` and classifier :math:`g_\theta`, models should
    also provide a feature extractor :math:`f^d_\theta` to derive domain features along with a domain
    classifier :math:`g^d_\theta`, with possibly shared parameters.

    In contrast to unsupervised DA, this training setting leverages information from multiple labeled
    source domains with the goal of generalizing well on data from a previously unseen domain during
    test time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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