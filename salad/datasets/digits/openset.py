import numpy as np
import torch

import sys

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

class OpenSetDataset(object):

    """ Dataset wrapper for openset classification

    Works with any classification datasets that outputs a tuple (x, y) when calling
    the `getitem` method.
    Given two sets of label for known and unknown classes, maps unknown class labels
    to zero.
    """
    
    def __init__(self, dataset, known, unknown, labels=None):
        
        self.dataset = dataset
        self.known = known
        self.unknown = unknown
        
        self.labels = labels
        
        self.idx = []
        
        if self.labels is not None:
            self._scan_labels()
        else:
            self._scan_dataset()
        
    def _scan_dataset(self):
        
        for i, (x, y) in enumerate(self.dataset):
            if y in self.known or y in self.unknown:
                self.idx.append(i)
                
    def _scan_labels(self):
        
        for i, y in enumerate(self.labels):
            if y in self.known or y in self.unknown:
                self.idx.append(i)
                
    def __len__(self):
        
        return len(self.idx)
    
    def __getitem__(self, index):
        
        index = self.idx[index]
        
        x,y   = self.dataset[index]
        y     = y if y in self.known else torch.zeros_like(y)
        
        return x, y

def get_data(train=True, batch_size=128):
    
    label_common   = [0,1,2,3]
    openset_source = [4,6,8]
    openset_target = [5,7,9]

    data1 = datasets.MNIST('/tmp/datasets', train=train, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ]))
    data2 = datasets.MNIST('/tmp/datasets', train=train, download=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ]))

    d1 = DataLoader(OpenSetDataset(data1, known=label_common, unknown=openset_source, labels=data1.train_labels if train else data1.test_labels), batch_size=batch_size, shuffle=True)
    d2 = DataLoader(OpenSetDataset(data2, known=label_common, unknown=openset_target, labels=data2.train_labels if train else data1.test_labels), batch_size=batch_size, shuffle=True)
    
    return d1, d2

#if __name__ == '__main__':
#    
#    d1, d2 = get_data(train=True)
#
#    model = FrenchModel()
#    solver = BCESolver(model, d2, n_epochs=5, gpu=0)
#
#    solver.optimize()