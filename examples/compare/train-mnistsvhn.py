""" Comparison Study of Recent Domain Adaptation Approaches
"""

import numpy as np
import torch
from torch import nn

from salad.datasets.da import toy

class SmallModel(nn.Module):
    
    def __init__(self, track_stats = True):
        
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32, track_running_stats = track_stats),
            nn.ReLU(),
            nn.Linear(32,64),
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

from salad.utils import config 

class ComparisonConfig(config.DomainAdaptConfig):

    def _init(self):
        super()._init()
        self.add_argument('--seed', default=None, type=int, help="Random Seed")
        
def get_dataloader(Xt, yt, Xv, yv, augment = False, merge=False):
    def collate(batch):

        X = torch.cat([b[0] for b in batch], dim=0)
        Y = torch.cat([b[1] for b in batch], dim=0)
        D = torch.cat([torch.zeros(b[0].size(0)).long() + n for n,b in enumerate(batch)], dim=0)

        return X,Y,D
    
    train = DataLoader(TensorDataset(Xt, yt.long()), batch_size= 256, shuffle=True)
    if augment:
        val   = DataLoader(TensorDataset(Xv, Xv + 0.01 * torch.randn(Xv.size()), yv.long()),
                           batch_size= 256, shuffle=True)
    else:
        val   = DataLoader(TensorDataset(Xv, yv.long()),
            batch_size= 256, shuffle=True)
        
    joint = JointLoader(train, val, collate_fn = collate if merge else None)
    
    return joint

def experiment_setup(args):
    model   = SmallModel()
    teacher = SmallModel()
    disc    = nn.Linear(64, 1)
    
    kwargs = {
        'n_epochs' : args.epochs,
        'multiclass' : True,
        'learningrate' : 3e-4,
        'gpu' : 0,
        'savedir' : args.log
    }
    
    return model, teacher, disc, kwargs
    
if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    from salad.datasets import JointLoader
    from salad import solver

    parser = ComparisonConfig('Domain Adapt Comparison')
    args = parser.parse_args()

    Xt, yt, Xv, yv = toy.make_data(seed=args.seed)
    loader_plain   = get_dataloader(Xt, yt, Xv, yv, augment = False)
    loader_cross   = get_dataloader(Xt, yt, Xv, yv, augment = False, merge=True)
    loader_augment = get_dataloader(Xt, yt, Xv, yv, augment = True)
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.CrossGradSolver(model, loader_cross, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.VADASolver(model, disc, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.DANNSolver(model, disc, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.AssociativeSolver(model, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.DeepCoralSolver(model, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.DeepLogCoralSolver(model, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.CorrDistanceSolver(model, loader_plain, **kwargs)
    experiment.optimize()
    
    model, teacher, disc, kwargs = experiment_setup(args)
    experiment = solver.SelfEnsemblingSolver(model, teacher, loader_augment, **kwargs)
    experiment.optimize()