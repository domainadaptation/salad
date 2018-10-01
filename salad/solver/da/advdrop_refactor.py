import salad

import torch
from torch import nn

from salad.solver.da import DABaseSolver
from salad.layers import KLDivWithLogits

class SymmetricKL(nn.Module):

    def __init__(self):
        super().__init__()
        self.kl = KLDivWithLogits()

    def forward(self, x, y):
        return .5 * (self.kl(x,y.detach()) + self.kl(y, x.detach()))

class AdversarialDropoutLoss():

    def __init__(self, model, step):

        self.model = model
        self.step = step


    def step1(self, batch):

        (xs, ys), (xt, yt) = batch

        zs = self.model.features(xs)
        ps = self.model.classifier(xs)

        return {
            "ce"    : (ps, ys),
            "acc_s" : (ps, ys)
        }

    def step2(self, batch):

        xs, ys = batch[0]

        _, ps = self.model(xs)

        return {
            "ce"    : (ps, ys),
            "acc_s" : (ps, ys)
        }

    def step2(self, batch):

        (xs, ys), (xt, yt) = batch

        with torch.no_grad():
            zs  = self.model.features(xs)
            zt1 = self.model.features(xt)
            zt2 = self.model.features(xt)

        ps  = self.model.classifier(zs.detach())
        pt1 = self.model.classifier(zt1.detach())
        pt2 = self.model.classifier(zt2.detach())
    
        return {
            "ce"      : (ps,  ys),
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

    Adversarial Dropout Regulariation [1] estimates uncertainties about the classification process
    by sampling different models using dropout.
    On the source domain, a standard cross entropy loss is employed.
    On the target domain, two predictions are sampled from the model.

    Both network parts are jointly trained on the source domain using the standard cross entropy loss,

    $$
    \min_{C, G} - \sum_k p^s_k \log y^s_k
    $$

    The classifier part of the network is trained to maximize the symmetric KL distance between
    two predictions. This distance is one option for measuring uncertainty in a network. In other
    words, the classifier aims at maximizing uncertainty given two noisy estimates of the current
    feature vector.

    $$
    \min_{C} - \sum_k p^s_k \log y^s_k + \frac{p^t_k - q^t_k}{2} \log \frac{p^t_k}{q^t_k} 
    $$


    In contrast, the feature extrator aims at minimizing the uncertainty between two noisy samples
    given a fixed target classifier.

    $$
    \min_{G} \frac{p^t_k - q^t_k}{2} \log \frac{p^t_k}{q^t_k}
    $$


    References
    ----------
    
    [1] Adversarial Dropout Regularization, Saito et al., ICLR 2018

    """

    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)

    def _init_optims(self, lr_GC=3e-4, lr_C=3e-4, lr_G=3e-4, **kwargs):
        super()._init_optims(**kwargs)

        G_params  = list(self.model.features.parameters())
        C_params  = list(self.model.classifier.parameters())
        GC_params = G_params + C_params

        opt_GC = torch.optim.Adam(GC_params, lr=lr_GC)
        opt_G  = torch.optim.Adam(G_params, lr=lr_G)
        opt_C  = torch.optim.Adam(C_params, lr=lr_C)

        self.register_optimizer(opt_GC, AdversarialDropoutLoss(self.model, step=1))
        self.register_optimizer(opt_C,  AdversarialDropoutLoss(self.model, step=2))
        self.register_optimizer(opt_G,  AdversarialDropoutLoss(self.model, step=3))

    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)

    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(SymmetricKL(), 1, "adv_C")
        self.register_loss(SymmetricKL(), 1, "adv_G")
