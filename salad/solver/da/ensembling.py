""" Self Ensembling for Visual Domain Adaptation
"""
import torch
from torch import nn

from .base import DABaseSolver
from ... import layers
from ...layers import WeightedCE, AccuracyScore, MeanAccuracyScore
from ...optim import WeightEMA
from ... import optim

class EnsemblingLoss(object):

    def __init__(self, model, teacher):

        self.model   = model
        self.teacher = teacher

    def __call__(self, batch):        
        (x_stud_xs, ys), (x_stud_xt,x_teach_xt,yt) = batch
        
        # TODO check if model is able to track domain
        # TODO maybe write a wrapper to modify networks for this
        _, stud_ys  = self.model(x_stud_xs, 0)
        _, stud_yt  = self.model(x_stud_xt, 1)
        with torch.no_grad():
            _, teach_yt = self.teacher(x_teach_xt, 0)

        losses = {}
        losses['ce']         = (stud_ys, ys)
        losses['ensemble']   = (stud_yt, teach_yt)
        
        losses['acc_s']       = (stud_ys, ys)
        losses['acc_t']       = (stud_yt, yt)
        losses['acc_teacher'] = (teach_yt, yt)

        return losses

class SelfEnsemblingSolver(DABaseSolver):
    """ Self-Ensembling for Visual Domain Adaptation

    A solver for self-ensembling techniques, using the implementation 
    proposed in [1]_.
    Note that the default hyperparameters are tuned to the small digit
    benchmarks used by [1]_, and should be adapted when the solver
    is used for new problem settings.

    Parameters
    ----------

    model : nn.Module
        The student model
    teacher : nn.Module
        The teacher model, should be equivalent to the student model
        in terms of parameters
    learningrate : float
        Learningrate for the student model, for use with an Adam
        optimizer
    ensemble : float
        Weight for the ensembling loss (default: 3)
    confidence_threshold : float
        Confidence threshold for the teacher model, between 0 and 1.
        Values close to 1 are recommended (default: 0.96837722)
    teacher_alpha : float
        Decay parameter for the exponential moving average optimizer
        used to determine the teacher weights. Values close to 1 are
        desired

    References
    ----------

    ..[1] French, Geoff, Michal Mackiewicz, and Mark Fisher. "Self-ensembling for visual domain adaptation." (2018).
           https://arxiv.org/abs/1706.05208
    """

    def __init__(self, model, teacher, dataset, *args, **kwargs):
        self.teacher = teacher
        super(SelfEnsemblingSolver, self).__init__(model, dataset, *args, **kwargs)

    def _init_optims(self, teacher_alpha = 0.99,
                        learningrate = 3e-4, **kwargs):
        super()._init_optims(**kwargs)
        
        opt_stud_src  = torch.optim.Adam(self.model.parameters(0),
                                        lr=learningrate)
        opt_teach = WeightEMA(self.teacher.parameters(0),
                              self.model.parameters(0),
                              alpha=teacher_alpha)
        
        opt = optim.JointOptimizer(opt_stud_src, opt_teach)

        self.register_optimizer(opt,
            EnsemblingLoss(self.model, self.teacher),
            name='Joint Optimizer')

    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)
        self.register_model(self.teacher, "teacher")

    def _init_losses(self, ensemble = 3,
                        confidence_threshold = 0.96837722,
                        **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(WeightedCE(), ensemble, 'ensemble')
        self.register_loss(AccuracyScore(), None, 'acc_teacher')