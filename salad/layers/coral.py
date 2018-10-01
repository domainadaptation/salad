import torch 
from torch import nn

from . import mat

class CorrelationDistance(nn.Module):

    def __init__(self, distance = mat.euclid):
        super().__init__()

        self.dist = distance


    def forward(self, xs, xt):
        
        Cs = mat.cov(xs) 
        Ct = mat.cov(xt)

        d = self.dist(Cs, Ct)

        return d

class CoralLoss(CorrelationDistance):
    """ Deep CORAL loss from paper: https://arxiv.org/pdf/1607.01719.pdf
    """
    def __init__(self):
        super().__init__(mat.euclid)

class LogCoralLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xs, xt):
        
        d = mat.logeuclid(xs, xt)

        return d
#class LogCoralLoss(CorrelationDistance):
#    """ Log Coral Loss
#    """
#    def __init__(self):
#        super().__init__(mat.logeuclid)

class SteinDivergence(CorrelationDistance):
    """ Log Coral Loss
    """
    def __init__(self):
        super().__init__(mat.stein)

class JeffreyDivergence(CorrelationDistance):
    """ Log Coral Loss
    """
    def __init__(self):
        super().__init__(mat.jeffrey)

class AffineInvariantDivergence(CorrelationDistance):
    def __init__(self):
        super().__init__(distance=mat.affineinvariant)