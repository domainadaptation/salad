""" Self Ensembling for Visual Domain Adaptation
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
        (x_stud_xt,yt) = batch
        
        _, stud_yt  = self.model(x_stud_xt)
        with torch.no_grad():
            _, teach_yt = self.teacher(x_teach_xt)

        losses = {        
            'ce'           : (stud_yt, teach_yt.max(dim=-1)[1]),        
            'VAT_tgt'      : (trg_x, trg_p),
            'H_tgt'        : (trg_p,),
            'acc_t'        : (trg_p, yt)
        }

        return losses

class DIRTTSolver(DABaseSolver):

    def __init__(self, model, teacher, dataset, learningrate, *args, **kwargs):
        super().__init__(model, dataset, *args, **kwargs)

        teacher_alpha = 0.98
        
        self.register_model(teacher, "teacher")
        self.teacher = teacher
        
        opt_stud_src  = torch.optim.Adam(model.parameters(), lr=learningrate)
        opt_teach = WeightEMA(teacher.parameters(),
                              model.parameters(),
                              alpha=teacher_alpha)
        
        opt = optim.JointOptimizer(opt_stud_src, opt_teach)
        

        self.register_optimizer(opt, EnsemblingLoss(self.model, self.teacher),
                               name='Joint Optimizer')
        self.register_loss(WeightedCE(), 3, 'ensemble')
        self.register_loss(AccuracyScore(), None, 'acc_teacher')