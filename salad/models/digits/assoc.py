import torch
from torch import nn

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
            conv2d_3x3(256, 512, pad=0),
            conv2d_1x1(512, 256),
            conv2d_1x1(256, 128),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):

        phi  = self.features(x)
        phi = phi.view(-1,128)
        y = self.classifier(phi)

        return phi, y

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]

    if act: layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )

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