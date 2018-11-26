""" Tools for training Generative Adversarial Networks (GANs)

The class is primarily used to train conditional networks (CGANs).

Notes
-----

Contributions for extensions wanted!
"""

from .base import Solver

import torch
from torch import nn

from ..models import ConditionalGAN, Discriminator

class GANSolver(Solver):

    # TODO: implement a more general super class
    pass


class CGANLoss():
    """ Loss Derivation for a Conditional GAN
    """

    def __init__(self, solver, G, Ds, train_G):

        self.solver = solver
        self.cuda = self.solver.cuda

        self.G = G
        self.Ds = Ds
        self.train_G = train_G

    def _derive_D(self, batch):
        losses = {}

        real_, fake_, y_, c_, x_, x_gen = batch

        reals = [real_, y_, c_]
        fakes = [fake_, fake_.long(), fake_.long()]

        return {
            '{} real'.format(name) : (D(x_).squeeze(), real),
            '{} fake'.format(name) : (D(x_gen).squeeze(), fake)
        }
        
        return losses

    def _derive_G(self, batch):
        real_, y_, c_, x_gen = batch

        lbl = [real_, y_, c_]
        losses = {}
        for y, (name, D) in zip(lbl, self.Ds):
            losses['G '+name] = (D(x_gen).squeeze(), y)
        return losses


    def derive_losses(self, batch):
        
        self.train_G = not self.train_G

        (x_clean, y_clean), (x_noise, y_noise) = batch

        # compose the full batches
        x_ =     torch.cat([x_clean, x_noise], dim=0)
        y_ = 1 + torch.cat([y_clean, y_noise], dim=0).long()
        c_ = 1 + torch.cat([torch.zeros(x_clean.size()[0]),
                            torch.ones (x_noise.size()[0])], dim=0).long()
        c_ = self.cuda(c_)

        mini_batch = x_.size()[0]

        # TODO needs to get a pre-computed batch for efficiency

        fake_ = self.cuda(torch.zeros(mini_batch))
        real_ = self.cuda(torch.ones(mini_batch))

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = self.cuda(z_)
        x_gen = self.G(z_, y_ - 1, c_ - 1)

        if self.train_G:
            batch = real_, y_, c_, x_gen
            return self._derive_G(batch)
        else:
            batch = real_, fake_, y_, c_, x_, x_gen
            return self._derive_D(batch)

class ConditionalGANSolver(Solver):

    """ Train a class conditional GAN model


    """

    names     = ['D_GAN', 'D_CL', 'D_CON']
    n_classes = [1, 11, 3]

    def __init__(self, model, dataset, learningrate=0.0002, *args, **kwargs):

        super(ConditionalGANSolver, self).__init__(dataset=dataset, *args, **kwargs)

        self.model    = model
        self.train_G  = True

        self._init_models() 
        self._init_losses()
        self._init_optims()

    def _init_losses(self, **kwargs):
        for cl, name in zip(self.n_classes, self.names):
            loss_func = nn.BCEWithLogitsLoss() if cl == 1 else nn.CrossEntropyLoss()

            # D loss
            self.register_loss(loss_func, weight=1, name='{} real'.format(name))
            self.register_loss(loss_func, weight=1, name='{} fake'.format(name))
            
            # G loss
            self.register_loss(loss_func,weight = 1, name = 'G '+ name)

    def _init_models(self, **kwargs):
        self.register_model(self.model, 'generator')
        self.discriminators = []
        for cl, name in zip(self.n_classes, self.names):
            D = Discriminator(128, n_classes=cl)
            D.weight_init(mean=0.0, std=0.02)
            self.register_model(D, name)

    def _init_optims(self, lr = 2e-4, beta1 = .5, beta2 = .999, **kwargs):
        opt = torch.optim.Adam(self.model.parameters(),lr = lr, betas = (beta1, beta2))
        self.register_optimizer(opt, CGANLoss(), name="Generator")

        for D in self.discriminators:
            optim = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, beta2))
            self.register_optim(optim, CGANLoss())
    
    def format_train_report(self, losses):
                
        if len(losses) < 2:
            return ""
        
        l = dict(losses[-2])
        l.update(losses[-1])
        
        return Solver.format_train_report(self, [l])