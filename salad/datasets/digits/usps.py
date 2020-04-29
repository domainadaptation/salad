import gzip
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets.utils import download_url

import torch

from .base import _BaseDataset

class USPS(_BaseDataset):
    """
    
    [USPS](http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html) Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.


    Download USPS dataset from [1]_ or use the expliclict links [2]_ for training and [3]_
    for testing.
    Code for loading the dataset partly adapted from [4]_ (Apache License 2.0).

    References: 
        
        .. [1] http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
        .. [2] Training Dataset http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz
        .. [3] Test Dataset http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz
        .. [4] https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = [
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'
    ]
    training_file = 'zip.train.gz'
    test_file = 'zip.test.gz'
    
    def extract_images_labels(self, filename):
        import gzip

        print('Extracting', filename)
        with gzip.open(filename, 'rb') as f:
            raw_data = f.read().split()
        data = np.asarray([raw_data[start:start + 257]
                           for start in range(0, len(raw_data), 257)],
                          dtype=np.float32)
        images_vec = data[:, 1:]
        self.images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
        self.labels = data[:, 0].astype(int)
        self.images = ((self.images + 1)*128).astype('uint8')
