""" Comparison Study of Recent Unsupervised Domain Adaptation Approaches on Toy Data
"""

import torch
from torch import nn
import numpy as np

from salad.datasets.da import toy
from salad.utils import config 
from salad.datasets.da import ToyDatasetLoader

class SmallModel(nn.Module):
    """ Model for Toy Dataset
    """

    def __init__(self, track_stats = True):
        
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64, track_running_stats = track_stats),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,64),
            nn.BatchNorm1d(64, track_running_stats = track_stats),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2)
        self._weight_init()
        
    def parameters(self, d = 0):
        return super().parameters()
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
    def forward(self, x, d = None):
        
        z = self.features(x)
        y = self.classifier(z)
        
        return z, y

class ComparisonConfig(config.DomainAdaptConfig):

    def _init(self):
        super()._init()
        self.add_argument('--seed', default=None, type=int, help="Random Seed")
        
def experiment_setup(args):
    model   = SmallModel()
    teacher = SmallModel()
    disc    = nn.Linear(64, 1)
    
    kwargs = {
        'n_epochs' : args.epochs,
        'multiclass' : True,
        'learningrate' : 3e-4,
        'gpu' : None if args.cpu else args.gpu,
        'savedir' : args.log
    }
    
    return model, teacher, disc, kwargs
    
if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    from salad.datasets import JointLoader
    from salad import solver

    parser = ComparisonConfig('Domain Adapt Comparison')
    args = parser.parse_args()

    parser.print_config()

    loader_plain   = ToyDatasetLoader(augment = False, collate='stack', batch_size = 256, seed=1301)
    loader_cross   = ToyDatasetLoader(augment = False, collate='cat', batch_size = 256, seed=1301)
    loader_augment = ToyDatasetLoader(augment = True, collate='stack', batch_size = 256, seed=1301)
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.BaselineDASolver(model, loader_plain, **kwargs)
    experiment.optimize()