""" Models for image registration
"""

__author__ = "Steffen Schneider"
__email__  = "steffen.schneider@tum.de"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels=[16, 16, 32, 32]):

        super(UNet, self).__init__()
        
        self.channels = channels

        self.inc = inconv(n_channels, self.channels[0])
        
        layers_down = []
        layers_up   = []
        for src, tgt in zip(channels, self.channels[1:]):
            layers_down.append(down(src, tgt))
            layers_up.insert(0, up(tgt+src, src))
            
        self.down = nn.ModuleList(layers_down)
        self.up   = nn.ModuleList(layers_up)
    
        self.outc = outconv(self.channels[0], n_classes)

    def get_n_params(self):
        
        return sum(p.view(-1).size()[0] for p in self.parameters())
        
    def forward(self, x):
        h = []
        
        h.append(self.inc(x))
        
        for down in self.down:
            h.append(down(h[-1]))

        # print('\n'.join([str(i.size()) for i in h]))
    
        x = h.pop()
        for up in self.up:
            y = h.pop()
            x = up(x, y)
        x = self.outc(x)
        return x
