""" Helper functions for structural learning
"""

from torch import nn
import torch
from torchvision.models import resnet

## General Helper Functions

def bn2linear(bn):
    scale, shift = get_affine(bn)
    
    W_ = scale.view(-1,1,1,1)
    b_ = shift
    
    n_channels = W_.size()[0]
    
    conv = nn.Conv2d(n_channels,n_channels,kernel_size=1,groups=n_channels)
    with torch.no_grad():
        conv.weight.set_(W_.float())
        conv.bias.set_(b_.float())
    return conv

def replace_bns(module):
    for name, layer in module._modules.items(): 
        if isinstance(layer, nn.BatchNorm2d):
            layer_ = bn2linear(layer)
            module._modules[name] = layer_

def reinit_bns(module):
    """ 
    """

    for name, layer in module._modules.items(): 
        if isinstance(layer, nn.BatchNorm2d):
            with torch.no_grad():
                scale, shift = get_affine(layer)

                layer_ = nn.BatchNorm2d(layer.num_features, eps=layer.eps, momentum=layer.momentum, affine=True, track_running_stats=True)

                layer_.weight.set_(scale.float())
                layer_.bias.set_(shift.float())

                module._modules[name] = layer_
            
def get_affine(layer):
    mu  = layer.running_mean.double()
    var = layer.running_var.double()
    W   = layer.weight.double()
    b   = layer.bias.double()
    eps = layer.eps
    
    inv_std = 1./(var + eps)**.5
    
    scale = inv_std * W
    shift = -mu * scale + b
    
    return scale, shift

def convert_conv_bn(layer, bn):
    W = layer.weight.double()

    scale, shift = get_affine(bn)
    out, inp,_,_ = W.size()
    W_ = (W * scale.view(-1,1,1,1))
    b_ = scale

    layer_ = nn.Conv2d(inp, out, kernel_size=layer.kernel_size, stride=layer.stride, padding = layer.padding)
    with torch.no_grad():
        layer_.weight.set_(W_.float())
        layer_.bias.set_(b_.float())

    return layer_

## Models

class FixedBottleneck(nn.Module):
    
    def __init__(self, conv, downsample):
        
        super().__init__()
        
        self.conv       = conv
        self.downsample = downsample 
        self.relu       = nn.ReLU()
        
    def forward(self, x):
        
        if self.downsample:
            return self.relu(self.conv(x) + self.downsample(x))
        else:
            return self.relu(self.conv(x) + x)

def FixedResnet(backbone):

    """ ResNet Variant where each batch norm layer is replaced by a linear transformation
    """

    backbone.double()
    backbone.apply(replace_bns)   
    return backbone

class CompressedResnet(nn.Module):

    """ ResNet Variant where the batch norm statistics are merged into the transformation
    matrices
    """
    
    def __init__(self, backbone):
        
        super().__init__()
        
        self.preprocessing = nn.Sequential(
            convert_conv_bn(backbone.conv1, backbone.bn1),
            nn.ReLU(),
            backbone.maxpool
        )
        
        self.layer1 = self._convert_layer(backbone.layer1)
        self.layer2 = self._convert_layer(backbone.layer2)
        self.layer3 = self._convert_layer(backbone.layer3)
        self.layer4 = self._convert_layer(backbone.layer4)
        
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
    
    def forward(self, x):
        
        x = self.preprocessing(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _convert_layer(self, layer):
        
        return nn.Sequential(
            *list(self._convert_bottleneck(b) for b in layer.children())
        )
        
    def _convert_bottleneck(self, layer):
        conv1 = convert_conv_bn(layer.conv1, layer.bn1)
        conv2 = convert_conv_bn(layer.conv2, layer.bn2)
        conv3 = convert_conv_bn(layer.conv3, layer.bn3)
        down  = None
        
        if ('downsample' in list(i[0] for i in layer.named_children())):
            down  = convert_conv_bn(layer.downsample[0], layer.downsample[1])
        
        convs = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3
        )
        
        bottleneck = FixedBottleneck(convs, down)
        
        return bottleneck
