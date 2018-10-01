import salad

import torch
from torch import nn

from salad.solver.da import DABaseSolver
from salad.layers import KLDivWithLogits
from salad.optim import JointOptimizer

def pack(*args):
    return torch.cat(args, 0)

def unpack(arg, n_tensors):

    shape = arg.size()
    return list(arg.view(n_tensors, shape[0] // n_tensors, *shape[1:]) )


class SymmetricKL(nn.Module):

    def __init__(self):
        super().__init__()
        self.kl = KLDivWithLogits()

    def forward(self, x, y):
        return .5 * (self.kl(x,y.detach()) + self.kl(y, x.detach()))

class AdversarialDropoutLoss():
    """ Loss Derivation for Adversarial Dropout Regularization

    See also
    --------

    salad.solver.AdversarialDropoutSolver
    """

    def __init__(self, model, step = 1):

        self.model = model
        self.step  = step


    def step1(self, batch):

        (xs, ys), (xt, yt) = batch

        x = pack(xs, xt)
        _, p = self.model(x)
        ps, _ = unpack(p, 2)

        return {
            "ce"    : (ps, ys),
            "acc_s" : (ps, ys)
        }

    def step2(self, batch):

        (xs, ys), (xt, yt) = batch

        with torch.no_grad():
            x = pack(xs, xt)
            z  = self.model.features(x)
            zs, zt = unpack(z, 2)

        ps  = self.model.classifier(zs)
        pt1 = self.model.classifier(zt)
        pt2 = self.model.classifier(zt)
    
        return {
            "ce_C"      : (ps,  ys),
            "adv_C"   : (pt1, pt2),
            "acc_t"   : (pt1,  yt)
        }

    def step3(self, batch):

        (xt, yt) = batch[1]
        
        zt1 = self.model.features(xt)
        zt2 = self.model.features(xt)

        #with torch.no_grad():
        pt1 = self.model.classifier(zt1)
        pt2 = self.model.classifier(zt2)

        return {"adv_G" : (pt1, pt2)}

    def __call__(self, batch):
       if self.step == 1: 
           return self.step1(batch)
       elif self.step == 2: 
           return self.step2(batch)
       elif self.step == 3:
           return self.step3(batch)


class AdversarialDropoutSolver(DABaseSolver):
    r""" Implementation of "Adversarial Dropout Regularization"

    Adversarial Dropout Regulariation [1]_ estimates uncertainties about the classification process
    by sampling different models using dropout.
    On the source domain, a standard cross entropy loss is employed.
    On the target domain, two predictions are sampled from the model.

    Both network parts are jointly trained on the source domain using the standard cross entropy loss,

    ..math::

        \min_{C, G} - \sum_k p^s_k \log y^s_k

    The classifier part of the network is trained to maximize the symmetric KL distance between
    two predictions. This distance is one option for measuring uncertainty in a network. In other
    words, the classifier aims at maximizing uncertainty given two noisy estimates of the current
    feature vector.

    ..math::

        \min_{C} - \sum_k p^s_k \log y^s_k + \frac{p^t_k - q^t_k}{2} \log \frac{p^t_k}{q^t_k} 


    In contrast, the feature extrator aims at minimizing the uncertainty between two noisy samples
    given a fixed target classifier.

    ..math::

        \min_{G} \frac{p^t_k - q^t_k}{2} \log \frac{p^t_k}{q^t_k}


    References
    ----------
    
    [1] Adversarial Dropout Regularization, Saito et al., ICLR 2018

    """

    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)

    def _init_optims(self, lr_GC=0.0002, lr_C=0.0002, lr_G=0.0002, **kwargs):
        super()._init_optims(**kwargs)

        G_params   = list(self.model.features.parameters())
        C_params  = list(self.model.classifier.parameters())
        GC_params = G_params + C_params

        # NOTE: Changing from three to two optimizers (As in the reference implementation)
        # made a huge difference!

        #opt_GC = torch.optim.Adam(GC_params, lr=lr_GC, weight_decay = 0.00005)
        opt_G  = torch.optim.Adam(G_params, lr=lr_G, weight_decay = 0.0005)
        opt_C  = torch.optim.Adam(C_params, lr=lr_C, weight_decay = 0.0005)
        opt_GC = JointOptimizer(opt_C, opt_G)

        self.register_optimizer(opt_GC, AdversarialDropoutLoss(self.model, step=1))
        self.register_optimizer(opt_C,  AdversarialDropoutLoss(self.model, step=2))
        self.register_optimizer(opt_G,  AdversarialDropoutLoss(self.model, step=3), n_steps = 4)

    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)

    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(nn.CrossEntropyLoss(), 1., "ce_C")
        self.register_loss(SymmetricKL(), -1., "adv_C")
        self.register_loss(SymmetricKL(), 1., "adv_G")
