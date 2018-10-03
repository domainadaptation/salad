""" Comparison of Cross-Domain Adaptation Approaches
"""

import torch
from torch import nn
import numpy as np

from salad.datasets.da import toy
from salad.utils import config 
from salad.datasets.da import ToyDatasetLoader
from salad.layers import concat

class MultiDomainModule(nn.Module):

    def __init__(self):
        super().__init__()

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

    def forward_domain(self, x):
        zd = self.feats_domain(x)
        # zd = self.pool(zd).view(zd.size(0), zd.size(1))
        d  = self.domain(zd)
        return zd, d

class MultiDomainModel(MultiDomainModule):

    def __init__(self, n_classes, n_domains, track_stats):

        super().__init__()

        self.track_stats = track_stats

        self.feats_domain = self._features(2)
        self.feats_class  = self._features(2 + 64)

        self.classifier = self._classifier(n_classes)
        self.domain     = self._classifier(n_domains)

    def forward(self, x):
        zd, d = self.forward_domain(x)
        x_ = torch.cat([x, zd.detach()], dim = 1)
        zy = self.feats_class(x_)
        y = self.classifier(zy)

        return d, y

    def _features(self, inp):
        return nn.Sequential(
            nn.Linear(inp, 32),
            nn.BatchNorm1d(32, track_running_stats = self.track_stats),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.BatchNorm1d(64, track_running_stats = self.track_stats),
            nn.ReLU()
        )

    def _classifier(self, n_classes):
        return nn.Linear(64, n_classes)

def experiment_setup(args):
    model   = MultiDomainModel(10, 8, True)
    
    kwargs = {
        'n_epochs'      : args.epochs,
        'multiclass'    : True,
        'learningrate'  : 3e-4,
        'gpu'           : None if args.cpu else args.gpu,
        'savedir'       : args.log
    }
    
    return model, kwargs

class ComparisonConfig(config.DomainAdaptConfig):

    def _init(self):
        super()._init()
        self.add_argument('--seed', default=None, type=int, help="Random Seed")


class NoiseModel(object):
    
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias
        
    def __call__(self, x):
        return x.dot(self.scale) + self.bias


def get_dataset():
    """ A Dataset hard to classify with a traditional ML method, but solveable by CrossGrad
    """ 
    shifts = [0, 1, 2, 3]
    scales = np.array([[[ 1.13792079,  0.31566232],
            [-0.09395214, -0.91731429]],

        [[ 1.77883215,  0.13303441],
            [ 0.84975969,  0.50246399]],

        [[ 1.96448436,  0.11892903],
            [ 0.12573305,  1.79736142]],

        [[-0.10556193, -0.75719817],
            [ 1.27046679,  1.44430405]]])

    noisemodels = [NoiseModel(M, b*2) for M, b in zip(scales, shifts)]
    domains = ToyDatasetLoader(n_domains = len(noisemodels), batch_size = 256,
                                noisemodels=noisemodels, collate='cat')

    return domains
        
if __name__ == '__main__':

    from torch.utils.data import TensorDataset, DataLoader
    from salad.datasets import JointLoader
    from salad import solver

    parser = ComparisonConfig('Domain Adapt Comparison')
    args = parser.parse_args()

    loader = get_dataset()

    model, kwargs = experiment_setup(args)
    experiment = solver.CrossGradSolver(model, loader, **kwargs)
    experiment.optimize()

    ## TODO: implement multi domain solver