import torch
import numpy as np

from torchvision import datasets, models, transforms

def transform(x):

    c,w,h = x.size()
    add  = torch.from_numpy(np.random.normal(0,.2,size=(1,w,h))).float()
    mult = torch.from_numpy(np.random.uniform(.3,1,size=(1,w,h))).float()

    return torch.clamp(x * mult + add, 0, 1)

def noisy_transform(img_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ColorJitter(.1, 1, .75, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)),
        transforms.Lambda(lambda x : x.expand([3,-1,-1]))
    ])

    return transform

def clean_transform(img_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)),
        transforms.Lambda(lambda x : x.expand([3,-1,-1]))
    ])

    return transform


def load_noisy_mnist(path, train=True):

    clean = datasets.MNIST(path, train=train, download=True, transform=clean_transform())
    noisy = datasets.MNIST(path, train=train, download=True, transform=noisy_transform())

    return {'clean' : clean, 'noisy' : noisy}