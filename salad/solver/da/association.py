""" Associative Domain Adaptation

[Hausser et al., CVPR 2017](#)
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
from .base import DABaseSolver
from ... import layers, optim

import itertools

class AssociationLoss(object):
    """ Loss function for associative domain adaptation

    Given a model, derive a function that computes arguments for the association loss.
    """

    def __init__(self, model):

        self.model = model

    def __call__(self, batch):

        (xs, ys), (xt, yt) = batch

        phi_s, yp   = self.model(xs)
        phi_t, ypt  = self.model(xt)

        yp  = yp.clone()
        ypt = ypt.clone()

        losses = {}
        losses['ce']      = (yp, ys)
        losses['assoc']   = (phi_s, phi_t, ys)
        losses['acc_s']   = (yp, ys)
        losses['acc_t']   = (ypt, yt)

        return losses

class AssociativeSolver(DABaseSolver):

    r""" Implementation of "Associative Domain Adaptation"

    Associative Domain Adaptation [1] leverages a random walk based on feature similarity as a distance between source and 
    target feature correlations.
    The algorithm is based on two loss functions that are added to the standard cross entropy loss on the source domain.

    Given features for source and target domain, a kernel function is used to measure similiarity between both domains.
    The original implementation uses the scalar product between feature vectors, scaled by an exponential,

    .. math::
        
        K_{ij} = k(x^s_i, x^t_j) = \exp(\langle x^s_i, x^t_j \rangle)

    This kernel is then used to compute transition probabilities

    .. math::
        p(x^t_j | x^s_i) = \frac{K_{ij}}{\sum_{l} K_{lj}}

    and 

    .. math::
        
        p(x^s_k | x^t_j) = \frac{K_{jk}}{\sum_{l} K_{kl}}

    to compute the roundtrip

    .. math::
        
        p(x^s_k | x^s_i) =  \sum_{j} p(x^s_k | x^t_j) p(x^t_j | x^s_i)

    It is then required that

    1. `WalkerLoss` The roundtrip ends at a sample with the same class label, i.e., $y^s_i = y^s_k$
    2. `VisitLoss`  Each target sample is visited with a certain probability 

    As one possible modification, different kernel functions could be used to measure similarity between
    the domains.
    With this solver, it is advised to use large sample sizes for the target domain and ensure that a
    sufficient number of source samples is available for each batch.

    TODO: Possibly in the solver class, implement a functionality to aggregate batches to avoid memory issues.

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model to be trained by association

    dataset : StackedDataset
        A dataset suitable for an unsupervised solver
    
    learningrate : int
        TODO

    References
    ----------

    [1] Associative Domain Adaptation, Häusser et al., CVPR 2017, https://arxiv.org/abs/1708.00938 
    """

    def __init__(self, model, dataset, learningrate,
                    walker_weight = 1.,
                    visit_weight = .1,
                    *args, **kwargs):

        super(AssociativeSolver, self).__init__(model, dataset,*args, **kwargs)

    def _init_losses(self, walker_weight = 1., visit_weight = .1, **kwargs):
        super()._init_losses(**kwargs)
        loss = layers.AssociativeLoss(walker_weight=walker_weight,
                                      visit_weight=visit_weight)
        self.register_loss(loss, 1, 'assoc')

    def _init_optims(self, lr=3e-4, **kwargs):
        super()._init_optims(**kwargs)
        self.register_optimizer(torch.optim.Adam(self.model.parameters(),
                                lr=lr),
                                AssociationLoss(self.model))