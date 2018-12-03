
import numpy as np

import torch
from torch import nn
from torchvision.models import resnet
from torchvision import datasets

class ParamCollect():
    def __init__(self, ltype):
        self.ltype = ltype
        self.reset()
    
    def reset(self):
        self.means = []
        self.stds  = []
    
    def __call__(self, x):
        if isinstance(x, self.ltype):
            with torch.no_grad():
                self.means.append(x.running_mean.cpu().numpy())
                self.stds.append(x.running_var.cpu().numpy())

def adapt_stats(model, stages, dataloader):  
    
    def reset(layer):
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean.zero_()
            layer.running_var.fill_(1)
        
    
    def ema_fn(n):
        
        def apply(layer):
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = float(1/(n+1))
                
        return apply
    
    with torch.no_grad():
        for stage in tqdm.tqdm(stages, position=0):
            stage.apply(reset)
            stage.train(True)
            for n, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                stage.apply(ema_fn(n))
                x = batch[0].cuda()
                stage(x).cpu()
            stage.eval()
            Y = []
            for batch in tqdm.tqdm(dataloader):
                x = batch[0].cuda()
                Y.append(stage(x).cpu().detach())
                #break
            Y = torch.cat(Y, dim=0)
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Y),
                                                     shuffle=True,
                                                     batch_size=dataloader.batch_size)
    return x