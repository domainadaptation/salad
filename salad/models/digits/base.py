import os
import os.path as osp
import sys
import argparse

import torch
from torch import nn

from ... import solver, models, datasets

from salad.datasets.transforms.ensembling import ImageAugmentation

from torch.nn import functional as F
from salad.layers import AccuracyScore, FeatureRotation

class _ConditionalModel(nn.Module):

    def __init__(self, n_domains):
        super().__init__()
        self.n_domains = n_domains

        self.conditional_layers  = []

    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.conditional_params(d):
                yield p
        
    def parameters(self, d=0, yield_shared=True, yield_conditional=True):

        if yield_shared:
            for param in super().parameters():
                yield param

        if yield_conditional:
            for param in self.conditional_params(d):
                yield param

    def __call__(self, x, d=0):
        return self.forward(x, d)

    def _batch_norm(self, *args, **kwargs):
        layer = ConditionalBatchNorm(*args, n_domains=self.n_domains, **kwargs)
        self.conditional_layers.append(layer)  
        return layer

class ConditionalBatchNorm(_ConditionalModel):
    
    def __init__(self, *args, n_domains = 1, bn_func = nn.BatchNorm2d, **kwargs):
        
        super().__init__(n_domains)
        self.conditional_layers    += [bn_func(*args, **kwargs) for i in range(n_domains)]
        
    def _apply(self, fn): 
        super()._apply(fn)
        for layer in self.conditional_layers:
            layer._apply(fn)

    def conditional_params(self, d):
        return self.conditional_layers[d].parameters()

    def parameters(self, d=0):
        pass
        
    def forward(self, x, d):
        layer = self.conditional_layers[d]
        return layer(x) 

class NoisyDigitFeatures(_ConditionalModel):

    def __init__(self, n_domains):

        super().__init__(n_domains)

        self.norm = nn.InstanceNorm2d(3, affine=False,
                momentum=0,
                track_running_stats=False)

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = self._batch_norm(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = self._batch_norm(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = self._batch_norm(128)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = self._batch_norm(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = self._batch_norm(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = self._batch_norm(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = self._batch_norm(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = self._batch_norm(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = self._batch_norm(128)

        self.drop1_1 = nn.Dropout2d(p=0.25)
        self.drop1_2 = nn.Dropout2d(p=0.25)
        self.drop1_3 = nn.Dropout2d(p=0.25)

        self.drop2_1 = FeatureRotation(256, p=0.25)
        self.drop2_2 = FeatureRotation(256, p=0.25)
        self.drop2_3 = nn.Dropout(p=0.5)

        self.drop3_1 = FeatureRotation(512, p=0.5)
        self.drop3_2 = FeatureRotation(256, p=0.5)
        self.drop3_3 = nn.Dropout(p=0.5)

    def forward(self, x, d):

        x = self.norm(x)

        x = F.relu(self.conv1_1_bn(self.conv1_1(x), d))
        x = self.drop1_1(x)
        x = F.relu(self.conv1_2_bn(self.conv1_2(x), d))
        x = self.drop1_2(x)
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x), d)))
        x = self.drop1_3(x)


        x = F.relu(self.conv2_1_bn(self.conv2_1(x), d))
        x = self.drop2_1(x)
        x = F.relu(self.conv2_2_bn(self.conv2_2(x), d))
        x = self.drop2_2(x)
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x), d)))
        x = self.drop2_3(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x), d))
        x = self.drop3_1(x)
        x = F.relu(self.nin3_2_bn(self.nin3_2(x), d))
        x = self.drop3_2(x)
        x = F.relu(self.nin3_3_bn(self.nin3_3(x), d))
        x = self.drop3_3(x)

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        return x

class DigitFeatures(_ConditionalModel):

    def __init__(self, n_domains):

        super().__init__(n_domains)

        self.norm = nn.InstanceNorm2d(3, affine=False,
                momentum=0,
                track_running_stats=False)

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = self._batch_norm(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = self._batch_norm(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = self._batch_norm(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = self._batch_norm(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = self._batch_norm(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = self._batch_norm(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = self._batch_norm(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = self._batch_norm(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = self._batch_norm(128)

    def forward(self, x, d):

        x = self.norm(x)

        x = F.relu(self.conv1_1_bn(self.conv1_1(x), d))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x), d))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x), d)))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x), d))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x), d))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x), d)))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x), d))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x), d))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x), d))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        return x

class NullspaceDigitFeatures(DigitFeatures):

     def __init__(self, linear):
        super().__init__()
        self.W = linear.weight

     def _compute_nullspace(self):
        with torch.no_grad():
            u,s,vh = torch.svd(self.W)
            N = vh.mm(vh.transpose(1,0))
        return N

     def forward(self, x, d):
        x = super().forward(x)
        N = self._compute_nullspace()
        x_ = x.mm(N)
        return x_

class DigitModel(_ConditionalModel):
    
    def __init__(self, n_classes=10, n_domains=2, noisy=False, nullspace = False):
        super().__init__(n_domains)
        
        self.n_domains = n_domains
        if noisy:
            self.features   = NoisyDigitFeatures(n_domains = n_domains)
        elif nullspace:
            self.features = NullspaceDigitFeatures(self.classifier)
        else:
            self.features   = DigitFeatures(n_domains = n_domains)

        self.classifier = nn.Linear(128, n_classes)

        self.conditional_layers.append(self.features)  
    
    def forward(self, x, d=0):
        z = self.features(x)
        y = self.classifier(z)

        return z, y

