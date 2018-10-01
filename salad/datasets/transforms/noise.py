from torchvision import datasets, transforms

import os
import os.path as osp

import numpy as np

import torch

from salad import solver, models
import salad.datasets

class DomainConfusion():
    
    """ Given x and a set of possible transforms, applies a random
    transform on x and returns a pair (x, d)
    """
    
    def __init__(self, transform_list, intermediate):
        
        self.transforms = transform_list
        self.n_domains = len(transform_list)
        self.intermediate = intermediate
        
    def __call__(self, x):
        
        idt = np.random.randint(self.n_domains)
        t = self.transforms[idt]
        x = t(x)
        
        for t in self.intermediate:
            x = t(x)
                
        d = torch.zeros(self.n_domains)
        d[idt] = 1
        
        return x, d
    
class DomainLabel():
    """ concats a domain label to the dataset
    """
    
    def __init__(self, domain, n_domains):
        
        self.domain = domain
        self.n_domains = n_domains
        
    def __call__(self, x):
        d = torch.zeros(self.n_domains)
        d[self.domain] = 1
        
        return x, d

class Uniform():
    """ Add uniform noise
    """
    
    def __init__(self, p=.05):
        
        self.prob = float(p)
    
    def __call__(self, x):
        
        N = np.random.uniform(0,1,size=x.shape)
        X = x.numpy()

        return torch.from_numpy(X + N).float()
    
class Gaussian():
    """ Add gaussian noise
    """
    
    def __init__(self, mu = 0., sigma = 0.1):
        
        self.mu    =  mu
        self.sigma =  sigma
    
    def __call__(self, x):
        
        X = x.numpy()
        
        N = np.random.normal(self.mu, self.sigma, size=X.shape)

        X = x.numpy()

        return torch.from_numpy(X + N).float()
    
class SaltAndPepper():
    """ Adds salt and pepper noise with probability *p* to a given image
    or batch of images
    """
    
    def __init__(self, p=.05):
        
        self.prob = float(p)
    
    def __call__(self, x):
        
        N = np.random.uniform(0,1,size=x.shape)

        X = x.numpy()

        X[N < self.prob]   = 1
        X[N < self.prob/2] = 0

        return torch.from_numpy(X)

class InvertContrast():
    
    def __call__(self, x):
        return 1 - x

    
class Shift():
    
    def __init__(self, w = 5, h = 5):
        self.dw = w
        self.dh = h
    
    def __call__(self, x):
        x[:,self.dh:,self.dw:] = x.clone()[:,:-self.dw,:-self.dh]
        
        x[:,:self.dh,:] = 0
        x[:,:,:self.dw] = 0

        return x