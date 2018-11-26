""" Losses for Correlatin Alignment 

Deep CORAL: Correlation Alignment for Deep Domain Adaptation
Paper: https://arxiv.org/pdf/1607.01719.pdf

Minimal Entropy Correlation Alignment for Unsupervised Domain Adaptation
Paper: https://openreview.net/pdf?id=rJWechg0Z
"""

import torch 
from torch import nn

from .base import DABaseSolver
from ...layers import CoralLoss, LogCoralLoss, CorrelationDistance

class CentroidLoss(object):

    def __init__(self, model):
        self.model = model
        self.n_classes = 10

    def __call__(self, batch):
        (src_x, src_y), (trg_x, trg_y___) = batch

        src_e, src_p = self.model(src_x)
        trg_e, trg_p = self.model(trg_x)
        
        with torch.no_grad():
            _, trg_y = trg_p.max(dim = 1)

        centroid_loss = []
        for i in range(self.n_classes):
            src_idc = torch.eq(src_y, i)
            trg_idc = torch.eq(trg_y, i)
            src_mu = src_e[src_idc].mean(axis=0)
            trg_mu = trg_e[trg_idc].mean(axis=0)
            centroid_loss.append((src_mu, trg_mu))

        return {
            'ce'           : (src_p, src_y),
            'corr'         : (src_e, trg_e),
            'centroid'     : centroid_loss,
            'acc_s'        : (src_p, src_y),
            'acc_t'        : (trg_p, trg_y___)
        }

class CorrelationDistanceLoss(object):

    def __init__(self, model, n_steps_recompute = 10, nullspace = False):
        self.model = model

        self.nullspace         = nullspace
        self.last_transform    = n_steps_recompute
        self.n_steps_recompute = n_steps_recompute
        self.proj              = None

    def _estimate_nullspace(self):

        if self.last_transform < self.n_steps_recompute:
            self.last_transform += 1
            return self.proj

        with torch.no_grad():
            U,S,Vh = torch.svd(self.model.classifier.weight, some=True)
            self.proj = Vh.mm(Vh.transpose(1,0))

        self.last_transform = 0
        return self.proj

    def __call__(self, batch):
        (src_x, src_y), (trg_x, trg_y___) = batch

        src_e, src_p = self.model(src_x)
        trg_e, trg_p = self.model(trg_x)

        if self.nullspace:
            N = self._estimate_nullspace()
            src_e = src_e.mm(N)
            trg_e = trg_e.mm(N)

        return {
            'ce'           : (src_p, src_y),
            'corr'         : (src_e, trg_e),
            'acc_s'        : (src_p, src_y),
            'acc_t'        : (trg_p, trg_y___)
        }

####################################################################

class CorrelationDistanceSolver(DABaseSolver):

    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(model, dataset, *args, **kwargs)

    def _init_losses(self, corr_weight = 1., corr_dist = None, **kwargs):
        super()._init_losses(**kwargs)
        self.register_loss(corr_dist, corr_weight, 'corr')

    def _init_optims(self, lr = 3e-4, use_nullspace = False, **kwargs):
        super()._init_optims(**kwargs)
        self.register_optimizer(torch.optim.Adam(self.model.parameters(),
                                                lr=lr),
                                CorrelationDistanceLoss(self.model, nullspace = use_nullspace))

class DeepCoralSolver(CorrelationDistanceSolver):
    r"""
    Deep CORAL: Correlation Alignment for Deep Domain Adaptation
    Paper: [https://arxiv.org/pdf/1607.01719.pdf](https://arxiv.org/pdf/1607.01719.pdf)

    Loss Functions:

    .. math::
        
        \mathcal{L}(x^s, x^t) = \frac{1}{4d^2} \| C_s - C_t \|

    """

    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(model, dataset, corr_dist = CoralLoss(), *args, **kwargs)

class DeepLogCoralSolver(CorrelationDistanceSolver):
    """
    Minimal Entropy Correlation Alignment for Unsupervised Domain Adaptation
    Paper: https://openreview.net/pdf?id=rJWechg0Z
    """

    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(model, dataset, corr_dist = LogCoralLoss(), *args, **kwargs)

class CorrDistanceSolver(CorrelationDistanceSolver):
    """
    Minimal Entropy Correlation Alignment for Unsupervised Domain Adaptation
    Paper: https://openreview.net/pdf?id=rJWechg0Z
    """

    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(model, dataset, corr_dist = CorrelationDistance(), *args, **kwargs)

class CentroidDistanceLossSolver(CorrelationDistanceSolver):
    """
    Notes
    -----

    Needs work.
    """

    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(model, dataset, corr_dist = CentroidLoss(), *args, **kwargs)

    def _init_losses(self, centroid = 1., **kwargs):
        super()._init_losses(**kwargs)

        def loss(*args):
            L = sum( ((x-y)**2).sum() for x,y in args )

        self.register_loss(loss, centroid, name='centroid')