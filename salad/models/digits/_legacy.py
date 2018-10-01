import h5py
import torch
from torch import nn

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
from collections import OrderedDict

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return x, y

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)

def svhn(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['svhn'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]

    if act: layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )

class MNISTModel(nn.Module):

    def __init__(self, n_channel):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        y = self.fc2(x)
        return x, y

class SVHNmodel(nn.Module):

    """
    Model for application on SVHN data (32x32x3)
    Architecture identical to https://github.com/haeusser/learning_by_association
    """

    def __init__(self):

        super(SVHNmodel, self).__init__()

        self.features = nn.Sequential(
            nn.InstanceNorm2d(3),
            conv2d(3,  32, 3),
            conv2d(32, 32, 3),
            conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(32, 64, 3),
            conv2d(64, 64, 3),
            conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(64, 128, 3),
            conv2d(128, 128, 3),
            conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 10)
        )

    def forward(self, x):

        phi  = self.features(x)
        phi_mean = phi.view(-1, 128, 16).mean(dim=-1)
        phi = phi.view(-1,128*4*4)
        y = self.classifier(phi)

        return phi_mean, y


class FrenchModel(nn.Module):

    """
    Model used in "Self-Ensembling for Visual Domain Adaptation"
    by French et al.
    """

    def __init__(self):

        super(FrenchModel, self).__init__()

        def conv2d_3x3(inp,outp,pad=1):
            return nn.Sequential(
                nn.Conv2d(inp,outp,kernel_size=3,padding=pad),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def conv2d_1x1(inp,outp):
            return nn.Sequential(
                nn.Conv2d(inp,outp,kernel_size=1,padding=0),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def block(inp,outp):
            return nn.Sequential(
                conv2d_3x3(inp,outp),
                conv2d_3x3(outp,outp),
                conv2d_3x3(outp,outp)
            )

        self.features = nn.Sequential(
            block(3,128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128,256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d_3x3(256, 256, pad=0),
            conv2d_1x1(256, 128),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):

        phi  = self.features(x)
        phi = phi.view(-1,128)
        # print(x.size(), phi.size())
        y = self.classifier(phi)

        return phi, y


