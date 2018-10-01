""" Standard Transformations for Digit datasets
"""

from torchvision import transforms
from torch import tensor

def default_normalization(key):

    d = {
    'mnist': (      tensor([ 0.1309,  0.1309,  0.1309]),
                    tensor([ 0.2890,  0.2890,  0.2890])),
    'usps': (       tensor([ 0.1576,  0.1576,  0.1576]),
                    tensor([ 0.2327,  0.2327,  0.2327])),
    'synth':       (tensor([ 0.4717,  0.4729,  0.4749]),
                    tensor([ 0.3002,  0.2990,  0.3008])),
    'synth-small': (tensor([ 0.4717,  0.4729,  0.4749]),
                    tensor([ 0.3002,  0.2990,  0.3008])),
    'svhn':        (tensor([ 0.4377,  0.4438,  0.4728]),
                    tensor([ 0.1923,  0.1953,  0.1904]))
    }

    return d[key]

def default_transforms(key):

    d = {

        'mnist' : transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.expand(3,-1,-1).clone())
        ]),

        'svhn' : transforms.Compose([
            transforms.ToTensor(),
        ]),

        'usps' : transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.expand(3,-1,-1).clone())
        ]),

        'synth' : transforms.Compose([
            transforms.ToTensor(),
        ]),

        'synth-small' : transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    return d[key]