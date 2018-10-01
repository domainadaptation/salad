""" Toy Datasets for domain adaptation experiments
"""

# TODO needs work
from .base import MultiDomainLoader

import numpy as np
import torch
from torch import nn

from sklearn.datasets import make_moons
from torch.utils.data import Dataset, DataLoader, TensorDataset

def noise_augment(x):
    scale = x.new_tensor(torch.from_numpy(np.eye(2) + np.random.normal(0,.01,size=(2,2))))
    bias  = x.new_tensor(torch.from_numpy(np.random.normal(0,.1,size=(2))))
    
    return x.mm(scale) + bias

def make_data(n_samples = 50000, n_domains = 2, plot=False, noisemodels = None, seed = None):
    if noisemodels is None:
        noisemodels = []

        angles = np.linspace(0,np.pi/5,n_domains)
        for _ in range(n_domains):
            #scale = np.eye(2) + np.random.normal(0,.1,size=(2,2))
            a = angles[_]
            scale = np.array([[ np.cos(a), np.sin(a)],
                              [-np.sin(a), np.cos(a)]])

            bias  = 0 #np.random.normal(0,.5,size=(1,2))
        noisemodels.append(lambda x : x.dot(scale) + bias)

    if seed is not None:
        np.random.seed(seed)

    n_total = n_samples * n_domains
    X, y = make_moons(n_samples=n_total, shuffle=True, noise=.1)
    X = X.reshape(n_domains, n_samples, 2)
    y = y.reshape(n_domains, n_samples)

    for domain, noise in enumerate(noisemodels):
        X[domain] = noise(X[domain])

    Xs = torch.from_numpy(X).float()
    ys = torch.from_numpy(y).float()

    return [ (X, y) for (X, y) in zip(Xs, ys)]

class ToyDatasetLoader(MultiDomainLoader):
    """ Digits dataset

    Four domains available: SVHN, MNIST, SYNTH, USPS
    """ 
    
    def __init__(self, seed = None, augment = False,
                 n_domains = 2, download=True, noisemodels = None,
                 collate = 'stack', **kwargs):


        domains = make_data(n_domains = n_domains, seed=seed, noisemodels = noisemodels)

        Xt, yt = domains[0]

        loaders = []

        loaders.append(DataLoader(TensorDataset(Xt, yt.long()), **kwargs))

        for (Xv, yv) in domains[1:]:

            if augment:
                noise = 0.01 * torch.randn(Xv.size())
                loaders.append( DataLoader(TensorDataset(Xv, Xv + noise, yv.long()), **kwargs) )
            else:
                loaders.append( DataLoader(TensorDataset(Xv, yv.long()),**kwargs) )
            
        super().__init__(*loaders, collate = collate)