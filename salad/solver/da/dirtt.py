""" DIRT-T solver for finetuning on an unsupervised target domain
"""

import torch
from torch import nn

from .base import DABaseSolver
from ... import layers
from ...layers import WeightedCE, AccuracyScore, MeanAccuracyScore
from ...optim import WeightEMA
from ... import optim


class DIRTT(object):

    def __init__(self, model, teacher):

        self.model   = model
        self.teacher = teacher

    def __call__(self, batch):        
        (x_stud_xt,x_teach_xt,yt) = batch
        
        _, stud_yt  = self.model(x_stud_xt)
        with torch.no_grad():
            _, teach_yt = self.teacher(x_teach_xt)

        losses = {        
            'ensemble'     : (stud_yt, teach_yt.max(dim=-1)[1]),        
            'VAT_tgt'      : (stud_yt, trg_p),
            'H_tgt'        : (stud_yt,),
            'acc_t'        : (stud_yt, yt)
        }

        return losses


class DIRTTSolver(Solver):

    """ DIRT-T Finetuning on the Target Domain

    References
    ----------
    
        ..[1] Shu et al., A DIRT-T approach to unsupervised
              domain adaptation. ICLR 2018
    """

    def __init__(self, model, teacher, dataset, *args, **kwargs):
        super().__init__(model, dataset, *args, **kwargs)
        
        self.model = model
        self.teacher = teacher

    def _init_models(self, **kwargs):
        self.register_model(self.model, 'Target model')
        self.register_model(self.teacher, 'Teacher')
        

    def _init_optims(self, teacher_alpha = 0.98, **kwargs):
        opt_stud_src  = torch.optim.Adam(self.model.parameters(0), lr=3e-4)
        opt_teach = WeightEMA(self.teacher.parameters(),
                              self.model.parameters(),
                              alpha=teacher_alpha)

        opt = optim.JointOptimizer(opt_stud_src)
        
        loss_model = VADA(self.model, self.discriminator, train_G = True)

    def _init_losses(self, cl_weight = 1e-2, vat_weight = 1e-2, entropy_weight = 1e-2):

        self.register_loss(WeightedCE(), cl_weight, "ensemble")
        self.register_loss(layers.VATLoss(self.model),  vat_weight, "VAT_tgt")
        self.register_loss(layers.ConditionalEntropy(), entropy_weight, "H_tgt")
        
        self.register_loss(layers.AccuracyScore(), None, "acc_t")