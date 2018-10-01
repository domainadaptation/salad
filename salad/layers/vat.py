import h5py
import torch
from torch import nn

import torch
import torch.nn.functional as F

from . import KLDivWithLogits

def normalize_perturbation(d):
    d_ = d.view(d.size()[0], -1)
    eps = d.new_tensor(1e-12)
    output = d / torch.sqrt(torch.max((d_**2).sum(dim = -1), eps)[0] )
    return output

class VATLoss(nn.Module):

    """ Virtual Adversarial Training Loss function

    Reference:
    TODO
    """

    def __init__(self, model, radius=1):

        super(VATLoss, self).__init__()
        self.model  = model
        self.radius = 1

        self.loss_func_nll = KLDivWithLogits()

    def forward(self, x, p):

        x_adv    = self._pertub(x, p)
        _, p_adv = self.model(x_adv)
        loss     = self.loss_func_nll(p_adv, p.detach())

        return loss

    def _pertub(self, x, p):
        eps = (torch.randn(size=x.size())).type(x.type())

        eps = 1e-6 * normalize_perturbation(eps)
        eps.requires_grad = True

        eps_p = self.model(x + eps)[1]

        loss  = self.loss_func_nll(eps_p, p.detach())
        loss.backward()
        eps_adv = eps.grad

        eps_adv = normalize_perturbation(eps_adv)
        x_adv = x + self.radius * eps_adv

        return x_adv.detach()

class ConditionalEntropy(nn.Module):

    """ estimates the conditional cross entropy of the input

    $$
    \frac{1}{n} \sum_i \sum_c p(y_i = c | x_i) \log p(y_i = c | x_i)
    $$

    By default, will assume that samples are across the first and class probabilities
    across the second dimension.
    """

    def forward(self, input):
        p     = F.softmax(input, dim=1)
        log_p = F.log_softmax(input, dim=1)

        H = - (p * log_p).sum(dim=1).mean(dim=0)

        return H