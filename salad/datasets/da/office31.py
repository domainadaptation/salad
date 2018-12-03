from torch import nn
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
from . import MultiDomainLoader

class OfficeDataset(Dataset):
    
    """ Office-31 dataset
    
    The Office-31 dataset consists of three domains: Amazon product images,
    real-world photos by a DSLR, and done with a lower resolution webcam.

    Notes
    -----

    Download the data from [1]_ and use the archive content as-is.

    References
    ----------

        [1].. https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
    """
    
    names = ["Amazon", "DSLR", "Webcam"]

    def __init__(self, path, transform = None, target_transform = None):
        
        self.amazon = ImageFolder(os.path.join(path, 'amazon/images/'), transform=transform, target_transform=target_transform)
        self.dslr   = ImageFolder(os.path.join(path, 'dslr/images/'), transform=transform, target_transform=target_transform)
        self.webcam = ImageFolder(os.path.join(path, 'webcam/images/'), transform=transform, target_transform=target_transform)
        
        self.datasets = [self.amazon, self.dslr, self.webcam]
        
        self._check()
        
    @property
    def class_to_idx(self):
        return self.amazon.class_to_idx
        
    def _check(self):
        
        for i in self.datasets:
            for j in self.datasets:
                assert(i.class_to_idx == j.class_to_idx)
    
    def __repr__(self):
        
        return  "\n\n".join(["OfficeDataset", "Amazon"+repr(self.amazon), "DSLR"+repr(self.dslr), "Webcam"+repr(self.webcam)])
        
        
class OfficeDataLoader(MultiDomainLoader):
    
    names = ["Amazon", "DSLR", "Webcam"]
    _keys = ["amazon", "dslr", "webcam"]
    
    def __init__(self, path, normalize = False, **kwargs):
        
        dataset = OfficeDataset(path, transform = self.get_transform(normalize = normalize))
        super().__init__(*[DataLoader(d, **kwargs) for d in dataset.datasets], collate="stack")
    
    def get_loader(self, key):
        
        i = self._keys.index(key)
        return self.datasets[i]
        

    def get_transform(self, normalize):

        T = [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ]
            
        if normalize is True:
            T += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
            
        return transforms.Compose(T)
            
    
