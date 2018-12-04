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

class DIRTT():

    def __init__(self, model, teacher):

        self.model   = model
        self.teacher = teacher

    def __call__(self, batch):
        
        (trg_x, trg_y___) = batch

        losses_student = {}

        _, trg_y     = self.teacher(trg_x)
        _, trg_p     = self.model(trg_x)

        losses_student.update({
            'DIRT_tgt'  : (trg_p, trg_y),
            'VAT_tgt'   : (trg_x, trg_p),
            'H_tgt'     : (trg_p,),
            'acc_t'     : (trg_p, trg_y___)
        })

        return losses_student

class VADASolver(DANNSolver):

    """ Virtual Adversarial Domain Adaptation
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
        
class DIRTTSolver(Solver):

    """ DIRT-T Finetuning on the Target Domain
    """

    def __init__(self, model, teacher, dataset, *args, **kwargs):
        super().__init__(model, dataset, *args, **kwargs)
        
        self.model = model
        self.teacher = teacher

    def _init_models(self):
        """ Register student, teacher and discriminator model
        """
        self.register_model(self.model, 'Target model')
        self.register_model(self.teacher, 'Teacher')
        

    def _init_optims(self):
        opt_stud_src  = torch.optim.Adam(self.model.parameters(0), lr=3e-4)
        opt = optim.JointOptimizer(opt_stud_src)
        
        loss_model = VADA(self.model, self.discriminator, train_G = True)

        self.register_optimizer(opt, loss_model)
        self.register_optimizer(torch.optim.Adam(
                                    self.discriminator.parameters(),
                                    lr=3e-4),
                                loss_disc)

    def _init_losses(self):

        super()._init_losses(cl_weight=1e-2)

        self.register_loss(layers.VATLoss(self.model),  1e-2, "VAT_tgt")
        self.register_loss(layers.ConditionalEntropy(), 1e-2, "H_tgt")


# class DIRTTSolver(Solver):
#     """ Train a Model using DIRT-T

#     Reference:
#     Shu et al (ICLR 2018).
#     A DIRT-T approach to unsupervised domain adaptation.
#     """

#     def __init__(self, model, teacher, dataset,
#                  learning_rate = 3e-4, teacher_alpha = .1,
#                  *args, **kwargs):

#         super(DIRTTSolver, self).__init__(dataset, *args, **kwargs)

#         # Add the teacher model with Weight EMA training
#         # Teacher uses gradient-free optimization
#         self.model = model
#         self.teacher = teacher
#         student_params = list(self.model.parameters())
#         teacher_params = list(self.teacher.parameters())
#         for param in teacher_params:
#             param.requires_grad = False

#         self.register_model(self.teacher,
#                             optim.DelayedWeight(teacher_params, student_params)
#                             )
#         self.register_model(self.model,
#                             torch.optim.Adam(self.model.parameters(), 3e-4)
#                             )



#         self.register_loss(layers.VATLoss(self.model), 1, "VAT_tgt")
#         self.register_loss(layers.ConditionalEntropy(), 1, "H_tgt")
#         self.register_loss(layers.KLDivWithLogits(), 1, "DIRT_tgt")

