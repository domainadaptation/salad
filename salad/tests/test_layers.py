import torch
from torch import nn
from torch.nn import functional as F

from salad.layers.da import AdaIN


def test_adain():

    from_std, to_std = 5, 3
    from_mean, to_mean = 1, 2.5

    x = torch.randn(10,3,32,32) * from_std + from_mean
    y = torch.randn(10,3,32,32) * to_std + to_mean

    layer = AdaIN(3)

    x_ = layer(x, y)

    x_.mean(), x_.std(), y.mean(), y.std()