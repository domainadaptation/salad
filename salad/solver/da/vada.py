__author__ = "Steffen Schneider"
__email__  = "steffen.schneider@tum.de"

import os, time
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn

from ... import layers, optim
from .. import Solver, BaseClassSolver
from .base import DABaseSolver
from .dann import AdversarialLoss, DANNSolver 

import itertools


class VADA(AdversarialLoss):

    def __call__(self, batch):
        # TODO improve instead of overriding
        
        (src_x, src_y), (trg_x, trg_y___) = batch

        src_e, src_p = self.G(src_x)
        trg_e, trg_p = self.G(trg_x)

        # Compute outputs
        real_logit   = self.D(src_e)
        fake_logit   = self.D(trg_e)

        if self.train_G:
            return {
            'ce'           : (src_p, src_y),
            'CL_src'       : (real_logit, torch.zeros_like(real_logit)),
            'CL_tgt'       : (fake_logit, torch.ones_like(fake_logit)),
            'VAT_src'      : (src_x, src_p),
            'VAT_tgt'      : (trg_x, trg_p),
            'H_tgt'        : (trg_p,),
            'acc_s'        : (src_p, src_y),
            'acc_t'        : (trg_p, trg_y___)
            }
        else:
            return {
            'D_src'    : (real_logit, torch.ones_like(real_logit)),
            'D_tgt'    : (fake_logit, torch.zeros_like(fake_logit))
            }



class VADASolver(DANNSolver):

    """ Virtual Adversarial Domain Adaptation

    This is the first step described in the DIRT-T paper, applicable to the 
    unsupervised domain adaptation scenario.

    References
    ----------
    
        ..[1] Shu et al., A DIRT-T approach to unsupervised
              domain adaptation. ICLR 2018
    """

    def __init__(self, model, discriminator, dataset, *args, **kwargs):
        super(VADASolver, self).__init__(model, discriminator, dataset, *args, **kwargs)

    def _init_optims(self, **kwargs):
        # override original call, but call init of higher class
        DABaseSolver._init_optims(self)

        opt_stud_src  = torch.optim.Adam(self.model.parameters(0), lr=3e-4)
        opt = optim.JointOptimizer(opt_stud_src)
        
        loss_model = VADA(self.model, self.discriminator, train_G = True)
        loss_disc  = VADA(self.model, self.discriminator, train_G = False)

        self.register_optimizer(opt, loss_model)
        self.register_optimizer(torch.optim.Adam(
                                    self.discriminator.parameters(),
                                    lr=3e-4),
                                loss_disc)

    def _init_losses(self, **kwargs):

        super()._init_losses(cl_weight=1e-2)

        self.register_loss(layers.VATLoss(self.model),  1, "VAT_src")
        self.register_loss(layers.VATLoss(self.model),  1e-2, "VAT_tgt")
        self.register_loss(layers.ConditionalEntropy(), 1e-2, "H_tgt")