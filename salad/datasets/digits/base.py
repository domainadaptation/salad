import gzip
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets.utils import download_url

import torch

from scipy.io import loadmat

class _BaseDataset(Dataset):

    urls          = None
    training_file = None
    test_file     = None
    
    def __init__(self, root, split = 'train', transform = None,
                 label_transform = None, download=True):

        super().__init__()
        
        self.root = root
        self.which = split 
        
        self.transform = transform
        self.label_transform = label_transform

        if download:
            self.download()

        self.get_data(self.which)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        x = Image.fromarray(self.images[index])
        y = int(self.labels[index])
        
        if self.transform is not None:
            x = self.transform(x)

        if self.label_transform is not None:
            y = self.label_tranform(y)
            
        return x, y

    def get_data(self, name):
        """Utility for convenient data loading."""
        if name in ['train', 'unlabeled']:
            self.extract_images_labels(os.path.join(self.root, self.training_file))
        elif name == 'test':
            self.extract_images_labels(os.path.join(self.root, self.test_file))

    def extract_images_labels(self, filename):
        raise NotImplementedError

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok = True)

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root,
                         filename=filename, md5=None)
        print('Done!')