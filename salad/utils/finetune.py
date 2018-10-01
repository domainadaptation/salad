import torch
from torch import nn

import pandas as pd
import numpy as np
from salad.solver import BaseClassSolver
from salad.solver.classification import MultidomainBCESolver
from salad.datasets import load_dataset

class Loss():
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, batch):
        
        x,y = batch
        _,y_   = self.model(x, 0)
        
        return {'ce' : (y_,y)}

class FinetuneSolver(BaseClassSolver):
    
    def __init__(self, *args, **kwargs):

        super(FinetuneSolver, self).__init__(*args, **kwargs)
        
        def parameters():
            for p in self.model.features.modulelist[-2].parameters():
                yield p
            for p in self.model.classifier.parameters():
                yield p

        optim = torch.optim.Adam(parameters(), lr = 3e-4)

        self.register_optimizer(optim, Loss(self.model),
                                name='adam')

        
if __name__ == '__main__':

    model = torch.load('log/log2/20180725-131833_MultidomainBCESolver/20180725-131833-checkpoint-ep300.pth')
    model.cuda()

    batch_size = 64

    data = load_dataset('/tmp/data', train=True)
    loader   = torch.utils.data.DataLoader(
        data['mnist'], batch_size=batch_size,
        shuffle=True, num_workers=4)

    solver = FinetuneSolver(model, loader, savedir='/tmp/log', gpu = 0, n_epochs=100)

    solver.optimize()