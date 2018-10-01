""" Routines for open set classification """

import torch 
from torch import nn

from .da import AdversarialLoss, BaseClassSolver
from .classification import BaseClassSolver

class VADAOpenset(AdversarialLoss):

    def __call__(self, batch):
        # TODO improve instead of overriding
        
        (src_x, src_y, src_set), (trg_x, trg_y___, trg_set___) = batch

        src_e, src_p, src_t = self.G(src_x)
        trg_e, trg_p, trg_t = self.G(trg_x)

        # Compute outputs
        real_logit   = self.D(src_e)
        fake_logit   = self.D(trg_e)

        if self.train_G:
            return {
            'ce'           : (src_p, src_y),
            'ce_set'       : (src_t, src_set),

            'CL_src'       : (real_logit, torch.zeros_like(real_logit)),
            'CL_tgt'       : (fake_logit, torch.ones_like(fake_logit)),
            'VAT_src'      : (src_x, src_p),
            'VAT_tgt'      : (trg_x, trg_p),

            'H_tgt'        : (trg_p,),
            'H_set'        : (trg_t,),

            'acc_s'        : (src_p, src_y),
            'acc_t'        : (trg_p, trg_y___),
            'acc_s_set'    : (trg_p, src_set),
            'acc_t_set'    : (trg_p, trg_y___)
            }
        else:
            return {
            'D_src'    : (real_logit, torch.ones_like(real_logit)),
            'D_tgt'    : (fake_logit, torch.zeros_like(fake_logit))
            }

class BaseOpensetSolver(BaseClassSolver):

    def __init__(self, model, dataset, multiclass = True, *args, **kwargs):

        super().__init__(dataset=dataset, *args, **kwargs)

        self.model      = model
        self.multiclass = multiclass

        # Specify the loss functions
        self.register_loss(nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss(),
                           weight = 1,
                           name   = 'ce')        

        # Specify the optimizer
        self.register_model(self.model, "classifier")