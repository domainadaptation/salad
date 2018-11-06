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
from .base import DABaseSolver

import itertools

class AdversarialLoss(object):

    def __init__(self, G, D, train_G = True):

        self.D = D
        self.G = G
        self.train_G = train_G

    def __call__(self, batch):

        (src_x, src_y), (trg_x, trg_y) = batch

        src_e, src_p = self.G(src_x)
        trg_e, trg_p = self.G(trg_x)

        # Compute outputs
        src_logit   = self.D(src_e)
        trg_logit   = self.D(trg_e)

        if self.train_G:
            return {
                'ce'        : (src_p, src_y),
                'CL_src'    : (src_logit, torch.zeros_like(src_logit)),
                'CL_tgt'    : (trg_logit, torch.ones_like(trg_logit)),
                'acc_s'     : (src_p, src_y),
                'acc_t'     : (trg_p, trg_y)
        }
        else:
            return {
                'D_src'    : (src_logit, torch.ones_like(src_logit)),
                'D_tgt'    : (trg_logit, torch.zeros_like(trg_logit))
            }

class DANNSolver(DABaseSolver):

    """ Domain Adversarial Neural Networks Solver

    This builds upon the normal classification solver that uses CrossEntropy or
    BinaryCrossEntropy for optimizing neural networks.

    Parameters
    ----------

    model : nn.Module
        The model to train
    discriminator : nn.Module
        The domain discriminator. Feature dimension should match the values returned
        by `model`
    dataset : Dataset
        A multi-domain dataset
    lr_G : float
        Model learning rate
    lr_D : float
        Discriminator learning rate
    cl_weight : float
        Classifier weight
    d_weight : float
        Discriminator weight
    """

    def __init__(self, model, discriminator, dataset, *args, **kwargs):
        self.discriminator = discriminator
        
        super().__init__(model, dataset, *args, **kwargs)


    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)
        self.register_model(self.discriminator, 'discriminator')

    def _init_losses(self, cl_weight=1., d_weight=1., **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(nn.BCEWithLogitsLoss(), cl_weight * .5, 'CL_src')
        self.register_loss(nn.BCEWithLogitsLoss(), cl_weight * .5, 'CL_tgt')
        self.register_loss(nn.BCEWithLogitsLoss(), d_weight  * .5, 'D_src' )
        self.register_loss(nn.BCEWithLogitsLoss(), d_weight  * .5, 'D_tgt' )


    def _init_optims(self, lr_G = 3e-4, lr_D = 3e-4, **kwargs):
        super()._init_optims(**kwargs)

        self.register_optimizer(torch.optim.Adam(self.model.parameters(),
                                                lr=lr_G),
                                AdversarialLoss(self.model, self.discriminator, True),
                                False)

        self.register_optimizer(torch.optim.Adam(self.discriminator.parameters(),
                                                lr=lr_D),
                                AdversarialLoss(self.model, self.discriminator, False),
                                False)