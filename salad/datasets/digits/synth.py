import gzip
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets.utils import download_url

import torch

from scipy.io import loadmat

from .base import _BaseDataset

class SynthSmall(_BaseDataset):

    """ Synthetic images dataset
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32_small.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32_small.mat?raw=true"
    }
    training_file = 'synth_train_32x32_small.mat?raw=true'
    test_file = 'synth_test_32x32.mat_small?raw=true'
    
    def extract_images_labels(self, filename):
        print('Extracting', filename)

        mat = loadmat(filename)

        self.images = mat['X'].transpose((3,0,1,2))
        self.labels = mat['y'].squeeze()

class Synth(_BaseDataset):
    """ Synthetic images dataset
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32.mat?raw=true"
    }
    training_file = 'synth_train_32x32.mat?raw=true'
    test_file = 'synth_test_32x32.mat?raw=true'
    
    def extract_images_labels(self, filename):
        print('Extracting', filename)

        mat = loadmat(filename)

        self.images = mat['X'].transpose((3,0,1,2))
        self.labels = mat['y'].squeeze()