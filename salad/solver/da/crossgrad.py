""" Cross Gradient Training

ICLR 2018
"""

import torch
from torch import nn 

from ...optim import JointOptimizer
from ...solver import Solver
from ...layers import AccuracyScore, MeanAccuracyScore

from .base import DGBaseSolver

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]
    if act: layers += [nn.ELU()]
    return nn.Sequential(
        *layers
    )

def features(inp):
    return nn.Sequential(
        conv2d(inp,  32, 3),
        conv2d(32, 32, 3),
        conv2d(32, 32, 3),
        nn.MaxPool2d(2, 2, padding=0),
        conv2d(32, 64, 3),
        conv2d(64, 64, 3),
        conv2d(64, 64, 3),
        nn.MaxPool2d(2, 2, padding=0),
        conv2d(64, 128, 3),
        conv2d(128, 128, 3),
        conv2d(128, 128, 3),
    )

class MultiDomainModule(nn.Module):

    def __init__(self, n_domains):

        super().__init__()

        # TODO

    def parameters_domain(self):
        for p in self.feats_domain.parameters():
            yield p 
        for p in self.domain.parameters():
            yield p

    def parameters_classifier(self):
        for p in self.feats_class.parameters():
            yield p 
        for p in self.classifier.parameters():
            yield p

    def forward(self, x):
        zd, d = self.forward_domain(x)
        x_ = concat(x, zd.detach())
        zy = self.feats_class(x_)
        zy = self.pool(zy).view(zy.size(0), zy.size(1))
        y = self.classifier(zy)

        return d, y

    def forward_domain(self, x):
        zd = self.feats_domain(x)
        zd = self.pool(zd).view(zd.size(0), zd.size(1))
        d  = self.domain(zd)
        return zd, d

class Model(nn.Module):

    def __init__(self, n_classes, n_domains):

        super().__init__()

        self.feats_domain = features(1)
        self.feats_class  = features(1 + 128)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes)
        )
        self.domain = nn.Sequential(
            nn.Linear(128, n_domains)
        )

    def parameters_domain(self):
        for p in self.feats_domain.parameters():
            yield p 
        for p in self.domain.parameters():
            yield p

    def parameters_classifier(self):
        for p in self.feats_class.parameters():
            yield p 
        for p in self.classifier.parameters():
            yield p

    def forward(self, x):
        zd, d = self.forward_domain(x)
        x_ = concat(x, zd.detach())
        zy = self.feats_class(x_)
        zy = self.pool(zy).view(zy.size(0), zy.size(1))
        y = self.classifier(zy)

        return d, y

    def forward_domain(self, x):
        zd = self.feats_domain(x)
        zd = self.pool(zd).view(zd.size(0), zd.size(1))
        d  = self.domain(zd)
        return zd, d

def concat(x, z):

    """ Concat 4D tensor with expanded 2D tensor
    """

    _,_,h,w = x.size()
    n,d     = z.size()
    z_ = z.view(n,d,1,1).expand(-1,-1,h,w)

    return torch.cat([x, z_], dim=1)

class CrossGradLoss():

    """ Cross Gradient Training

    References
    ----------

    ..[1]: http://arxiv.org/abs/1804.10745
    """

    def __init__(self, solver):

        super().__init__()

        self.loss    = solver.compute_loss_dict
        self.model   = solver.model

    def pertub(self, x, loss, eps = 1e-5):
        loss.backward(retain_graph = True)
        with torch.no_grad():
            dx = x.grad
            xd = x + eps * dx 

        return xd

    def __call__(self, batch):
        x, y, d = batch

        x.requires_grad_(True)
        d_, y_ = self.model(x)
        
        losses = self.loss({
            'ce_y' : (y_, y),
            'ce_d' : (d_, d)
        })
        losses['acc_y'] = losses['meanacc_y'] = (y_, y)
        losses['acc_d'] = losses['meanacc_d'] = (d_, d)

        x_d = self.pertub(x, losses['ce_y'])
        x_y = self.pertub(x, losses['ce_d'])

        _, d_ = self.model.forward_domain(x_d)
        _, y_ = self.model(x_y)

        losses.update({
            'cross_y' : (y_, y), 
            'cross_d' : (d_, d)
        })
        return losses

class CrossGradSolver(DGBaseSolver):

    r""" Cross Gradient Optimizer

    A domain generalization solver based on Cross Gradient Training [1]_.

    ..math:
        p(y | x) = \int_d p(y|x,d) p(d|x) dd

    ..math:
        x_d = x + \eps \Nabla_y L(y) \\
        x_y = x + \eps \Nabla_d L(d)

    References
    ----------

    .. [1] Shankar et al., Generalizing Across Domains via Cross-Gradient Training, ICLR 2018

    """

    def __init__(self, model, *args, **kwargs):

        self.model = model
        
        super().__init__(*args, **kwargs)

    def _init_models(self, **kwargs):
        super()._init_models(**kwargs)

        self.register_model(self.model, 'Model')

    def _init_optims(self, **kwargs):
        super()._init_optims(**kwargs)

        optim = torch.optim.Adam(self.model.parameters(), lr = 3e-4, amsgrad = True)

        # TODO this might still be a bug?
        #optim = JointOptimizer(
        #    torch.optim.Adam(self.model.parameters_classifier(), lr = 3e-4, amsgrad = True),
        #    torch.optim.Adam(self.model.parameters_domain(), lr = 3e-4, amsgrad = True)
        #)

        self.register_optimizer(optim, CrossGradLoss(self), name = "optimizer")

    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        for name in ['ce_y', 'ce_d', 'cross_y', 'cross_d']:
            self.register_loss(nn.CrossEntropyLoss(), weight = 1., name = name)
        
        self.register_loss(AccuracyScore(), weight = None, name = 'acc_y')
        self.register_loss(AccuracyScore(), weight = None, name = 'acc_d')
        self.register_loss(MeanAccuracyScore(), weight = None, name = 'meanacc_y')
        self.register_loss(MeanAccuracyScore(), weight = None, name = 'meanacc_d')

def get_dataset(noisemodels, batch_size, shuffle = True, num_workers = 0, which='train'):
    from torchvision import transforms

    data = []

    noisemodels = [
        transforms.RandomRotation([-1,1]),
        transforms.RandomRotation([10,11]),
        transforms.RandomRotation([20,21]),
        transforms.RandomRotation([30,31]),
    ]
    
    for N in noisemodels:
    
        transform = transforms.Compose([
                transforms.ToTensor(),
                N,
                transforms.Normalize(mean=(0.43768448, 0.4437684,  0.4728041 ),
                                    std= (0.19803017, 0.20101567, 0.19703583))
        ])
        mnist = datasets.mnist('/tmp/data', train=True, download=True, transform=transform)

        data.append(torch.utils.data.DataLoader(
            svhn, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers))

    loader = salad.datasets.JointLoader(*data)
    return data, loader