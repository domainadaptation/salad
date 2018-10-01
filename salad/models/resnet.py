import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from .digits.fan import ConditionalLayer


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def convert_state_dict(model, statedict):
    
    import re

    statedict_ = {}
    replace = re.compile(r'conditionals\.[0-9]+\.')

    n_domains = 2

    for key in model.state_dict().keys():

        if "conditionals" in key:

            for domain in range(n_domains):
                tgt = replace.sub('', key)
                #print(key, tgt)
                statedict_[key] = statedict[tgt]

        else:
            statedict_[key] = nn.Parameter(statedict[key].clone())
            
    return statedict_

class ConditionalSequential(nn.Sequential):

    def __init__(self, *modules):

        super(ConditionalSequential, self).__init__(*modules)

    def forward(self, x, *args):

        for module in self._modules.values():
            
            if isinstance(module, ConditionalLayer):
                x = module(x, *args)
            else:
                x = module(x)
        
        return x

class ConditionalBatchNorm(ConditionalLayer):
    
    def __init__(self, *args, n_domains = 1, bn_func = nn.BatchNorm2d, **kwargs):
        
        super(ConditionalBatchNorm, self).__init__()
        
        self.n_domains = n_domains
        self.conditionals    = nn.ModuleList([bn_func(*args, **kwargs) for i in range(n_domains)])
        
    def _apply(self, fn): 
        super(ConditionalBatchNorm, self)._apply(fn)
        for layer in self.conditionals:
            layer._apply(fn)
        
    def parameters(self, d=0):
        return self.conditionals[d].parameters()
        
    def forward(self, x, d):
                
        layer = self.conditionals[d]
        return layer(x) 

class ConditionalParamModel(ConditionalLayer):
    
    def __init__(self):
        
        super(ConditionalParamModel, self).__init__()
        
        self.conditional_layers = []
        
    def _make_batch_norm(self, *args, **kwargs):

        bn = ConditionalBatchNorm(*args, n_domains=2, bn_func=nn.BatchNorm2d,**kwargs)
        self.conditional_layers.append(bn)

        return bn
        
    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.parameters(d):
                yield p

    def parameters(self, d=0):
        if d == 0:
            return super(ConditionalParamModel, self).parameters()
        else:
            return self.conditional_params(d)
        
        

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(ConditionalParamModel):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = self._make_batch_norm(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = self._make_batch_norm(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x, d = 0):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, d)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, d)

        if self.downsample is not None:
            residual = self.downsample(x, d)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(ConditionalParamModel):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self._make_batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self._make_batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = self._make_batch_norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, d):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, d)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, d)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, d)

        if self.downsample is not None:
            residual = self.downsample(x, d)

        out += residual
        out = self.relu(out)

        return out


class ResNet(ConditionalParamModel):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._make_batch_norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)#, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConditionalSequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._make_batch_norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return ConditionalSequential(*layers)

    def forward(self, x, d = 0):
        x = self.conv1(x)
        x = self.bn1(x, d)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, d)
        x = self.layer2(x, d)
        x = self.layer3(x, d)
        x = self.layer4(x, d)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def load_pretrained(self, statedict):
        
        statedict_ = convert_state_dict(self, statedict)
        self.load_state_dict(statedict_)
        
        def freeze(module):
            for param in module.parameters():
                param.requires_grad_(False)

        def enable(module):
            if isinstance(module, ConditionalBatchNorm):
                for param in module.conditionals.parameters():
                    param.requires_grad_(True)

        self.apply(freeze)
        self.apply(enable)
        

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_pretrained(model_zoo.load_url(model_urls['resnet152']))
    return model