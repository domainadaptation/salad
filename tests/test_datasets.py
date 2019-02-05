from salad.datasets import MNIST, USPS, SVHN, Synth, SynthSmall
import torch
from torchvision import transforms

def test_digits():

    T = transforms.ToTensor()

    data = [
        MNIST('/tmp/data', transform = T, download=True),
        SVHN('/tmp/data',  transform = T, download=True),
        SynthSmall('/tmp/data',  transform = T, download=True),
        USPS('/tmp/data',  transform = T, download=True),
    ]

    for ds in data:

        for x, y in ds:
            assert isinstance(y, int) or (isinstance(y, torch.Tensor) and y.dim() == 0), (y, type(y))
            assert x.dim() >= 3
            break
