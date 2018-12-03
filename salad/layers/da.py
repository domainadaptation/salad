import torch
from torch import nn
from torch.nn import functional as F

class FeatureAwareNormalization(nn.Module):

    def __init__(self):
        pass

class AutoAlign2d(nn.BatchNorm2d):

    def __init__(self):
        pass

    def forward(self):
        pass


class AdaIN(nn.Module):
    
    def __init__(self, n_channels, eps=1e-05):
        
        super().__init__()
        
        self.n_channels = n_channels
        self.eps = eps

    def forward(self, x, y):
        N = y.size()[0]
        
        sd = y.view(N, self.n_channels, -1).std(dim=-1)
        mu  = y.view(N, self.n_channels, -1).mean(dim=-1)

        x_ = F.instance_norm(x, running_mean=None, running_var=None, weight=None, #(self.eps + var)**.5,
                                            bias=None, use_input_stats=True, momentum=0.,
                                            eps=self.eps)
        
        x_ = x_ * sd.unsqueeze(-1).unsqueeze(-1) + mu.unsqueeze(-1).unsqueeze(-1)
        
        return x_