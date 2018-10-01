import torch
from torch import nn

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features   = None
        self.classifier = None 

    def forward(self):
        pass

class ConditionalAdaptive(nn.Module):

    def __init__(self):
        super().__init__()

        self.features   = None
        self.classifier = None 

    def forward(self):
        pass