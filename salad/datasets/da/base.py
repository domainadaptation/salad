from torch.utils.data import Dataset

from torchvision import datasets, transforms

import h5py
import torch
from torch import nn

import torch.nn.functional as F

import torch.utils.data

class JointLoader:

    def __init__(self, *datasets, collate_fn = None):

        self.datasets  = datasets
        self.iterators = [None] * len(datasets)
        self.collate_fn = collate_fn

    def __len__(self):

        return min([len(d) for d in self.datasets])

    def __iter__(self):
        for i, dataset in enumerate(self.datasets):
            self.iterators[i] = dataset.__iter__()
        return self

    def __next__(self):
        try:
            items = []
            for dataset in self.iterators:
                items.append(dataset.__next__())
        except StopIteration:
            raise StopIteration

        if self.collate_fn is not None:
            items = self.collate_fn(items)

        return items


class JointDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        super(JointDataset, self).__init__()

        self.datasets = datasets

    def __len__(self):

        return min([len(d) for d in self.datasets])

    def __getitem__(self, index):

        return [ds[index] for ds in self.datasets]


class AugmentationDataset(Dataset):

    def __init__(self, dataset, transforms, n_samples=2):

        super(AugmentationDataset, self).__init__()

        self.dataset = dataset
        self.n_samples = n_samples
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        x,y = self.dataset[index]

        xs = [self.transforms(x) for _ in range(self.n_samples)]

        return xs + [y,]

def concat_collate(batch):

    X = torch.cat([b[0] for b in batch], dim=0)
    Y = torch.cat([b[1] for b in batch], dim=0)
    D = torch.cat([torch.zeros(b[0].size(0)).long() + n for n,b in enumerate(batch)], dim=0)

    return X,Y,D

class MultiDomainLoader(JointLoader):

    """ Wrapper around Joint Loader for multi domain training
    """

    def __init__(self, *args, collate = 'stack'):#, **kwargs):
        assert collate in ['stack', 'cat']

        if collate == 'stack':
            collate_fn = None
        elif collate == 'cat':
            collate_fn = concat_collate
        else:
            raise NotImplementedError

        super().__init__(*args, collate_fn = collate_fn) #, **kwargs)

### loader functions ###
def load_dataset(path, train=True, img_size = 32, expand = True):

    """
    .. deprecated

        Deprecated
    """

    transform = [
            transforms.Resize(img_size),
            transforms.ColorJitter(.1, 1, .75, 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)),
    ]
    if expand: transform.append(transforms.Lambda(lambda x : x.expand([3,-1,-1])))
    transform = transforms.Compose(transform)
    mnist = datasets.MNIST(path, train=train, download=True, transform=transform)

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.188508,    0.19058265,  0.18615675))
    ])
    svhn = datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)

    return {'mnist' : mnist, 'svhn' : svhn}


def load_dataset2(path, train=True, img_size = 32, expand = True):

    """
    .. deprecated

        Deprecated
    """

    transform = [
            transforms.Resize(img_size),
            #transforms.ColorJitter(.1, 1, .75, 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.13089988, 0.13089988, 0.13089988),
                                 std =(0.28928825, 0.28928825, 0.28928825)),
    ]
    if expand: transform.append(transforms.Lambda(lambda x : x.expand([3,-1,-1])))
    transform = transforms.Compose(transform)
    mnist = datasets.MNIST(path, train=train, download=True, transform=transform)

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.43768448, 0.4437684,  0.4728041 ),
                                 std= (0.19803017, 0.20101567, 0.19703583))
    ])
    svhn = datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)

    return {'mnist' : mnist, 'svhn' : svhn}