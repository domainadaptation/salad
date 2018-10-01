""" Dataset loader for digit experiments

Digit datasets (MNIST, USPS, SVHN, Synth Digits) are standard benchmarks for unsupervised domain adaptation.
In addition to access to these datasets, this module provides a collection of other datasets useful for DA
based on digit datasets.

Datasets are collections of single datasets and are subclasses of the `MultiDomainLoader`.
"""

from ..digits import MNIST, SVHN, USPS, SynthSmall, Synth
from torch.utils.data import Dataset, DataLoader

from .base import JointLoader, MultiDomainLoader

from ..transforms.noise import Gaussian, SaltAndPepper
from ..transforms.digits import default_transforms, default_normalization

import torch
from torchvision import transforms

from salad.datasets.transforms import Augmentation


class DigitsLoader(MultiDomainLoader):
    r""" Digits dataset

    Four domains available: SVHN, MNIST, SYNTH, USPS

    Parameters
    ----------

    root : str
        Root directory where dataset is available or should be downloaded to
    keys : list of str
        pass

    See Also
    --------
    ``torch.utils.data.DataLoader``
    """

    _constructors = {
        'mnist': MNIST,
        'svhn': SVHN,
        'synth': Synth,
        'synth-small': SynthSmall,
        'usps': USPS
    }

    def __init__(self, root, keys,
                 split='train', download=True,
                 collate='stack', normalize=False,
                 augment={}, augment_func = Augmentation, batch_size=1,
                 **kwargs):

        assert split in ['train', 'test']

        self.datasets = {}
        for key in keys:
            T = default_transforms(key)
            if normalize:
                print('Normalize data')
                T.transforms.append(transforms.Normalize(*default_normalization(key)))
            func = self._constructors[key]

            self.datasets[key] = func(root=root, split=split, download=download, transform=T)

            if key in augment.keys():
                self.datasets[key] = augment_func(self.datasets[key], augment[key])

        if isinstance(batch_size, int):
            batch_size = [batch_size] * len(keys)

        super().__init__(*[DataLoader(self.datasets[k], batch_size=b, **kwargs) for k, b in zip(keys, batch_size)],
                         collate=collate
                         )


class AugmentationLoader(MultiDomainLoader):

    _constructors = {
        'mnist': MNIST,
        'svhn': SVHN,
        'synth': Synth,
        'synth-small': SynthSmall,
        'usps': USPS
    }

    def __init__(self, root, dataset_name, transforms, split='train', augment={}, download=True, collate='cat', **kwargs):

        data = []
        for i, T in enumerate(transforms):
            func = self._constructors[dataset_name]
            ds = func(root, split=split, download=download, transform=T)
            if i in augment.keys():
                ds = Augmentation(ds, augment[i])
            data.append(DataLoader(ds, **kwargs))

        super().__init__(*data, collate=collate)


class NoiseLoader(AugmentationLoader):

    eps = 1.

    def __init__(self, root, key,
                 noisemodels=[], normalize=True,
                 **kwargs):


        self.noisemodels = []
        for noisemodel in noisemodels:
            T = transforms.Compose([
                transforms.ToTensor(),
                noisemodel,
            ])
 
            if normalize:
                print('Normalize data')
                T.transforms.append(transforms.Normalize(*default_normalization(key)))

            self.noisemodels.append(T)

        super().__init__(root, key, self.noisemodels, **kwargs)



class RotationLoader(AugmentationLoader):

    eps = 1.

    def __init__(self, root, dataset_name,
                 angles=list(range(0, 90, 15)),
                 normalize = False,
                 **kwargs):

        self.noisemodels = []

        for i in angles:
            T = transforms.Compose([
                transforms.RandomRotation([i-self.eps, i+self.eps]),
                transforms.ToTensor(),
            ])
 
            if normalize:
                print('Normalize data')
                T.transforms.append(transforms.Normalize(*default_normalization(key)))

            self.noisemodels.append(T)

        super().__init__(root, dataset_name, self.noisemodels, **kwargs)


class LowToHighGaussian():
    noisemodels = [.001, .025, .05, .075, .1, .15, .2, .25, .3]


class HighToLowGaussian():
    noisemodels = [.3, .25, .2, .15, .1, .075, .05, .025, .001]


class LowToHighSaltPepper():
    noisemodels = [
        Gaussian(0, .001),
        SaltAndPepper(.0025),
        SaltAndPepper(.01),
        SaltAndPepper(.05),
        SaltAndPepper(.1),
        SaltAndPepper(.15),
        SaltAndPepper(.2),
        SaltAndPepper(.25),
        SaltAndPepper(.36),
    ]


class HighToLowSaltPepper():
    noisemodels = [
        SaltAndPepper(.36),
        SaltAndPepper(.25),
        SaltAndPepper(.2),
        SaltAndPepper(.15),
        SaltAndPepper(.1),
        SaltAndPepper(.05),
        SaltAndPepper(.01),
        SaltAndPepper(.0025),
        Gaussian(0, .001)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
