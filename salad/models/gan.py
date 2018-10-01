import torch
from torch import nn

import torch.nn.functional as F

def to_one_hot(y, n_dims=None):
    y_tensor = y.long().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    #y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).type(y.type()).scatter_(1, y_tensor, 1)
    y_one_hot = y.new(y_tensor.size()[0], n_dims).zero_().scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def cat2d(x, *args):

    w = x.size()[2]
    h = x.size()[3]

    args = torch.cat(args, dim = 1)
    if args.dim() == 2:
        args = args.float().unsqueeze(2).unsqueeze(3)

    return torch.cat([x, args.expand([-1,-1,w,h])], dim = 1)

class ConditionalGAN(nn.Module):
    # initializers
    def __init__(self, d=128, n_classes = 10, n_conditions = 2, n_outputs = 3):
        super(ConditionalGAN, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100 + n_classes + n_conditions, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8 + n_classes + n_conditions, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4 + n_classes + n_conditions, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2 + n_classes + n_conditions, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d + n_classes + n_conditions, n_outputs, 4, 2, 1)

        self.n_classes = n_classes
        self.n_conditions = n_conditions

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label, condition):

        # class is a (N x n_classes) tensor, 1-hot coding
        # condition is a (N x n_conditions) tensor, 1-hot coding

        label     = to_one_hot(label, self.n_classes)
        condition = to_one_hot(condition, self.n_conditions)

        #print(input.size(), label.size(), condition.size())

        x = input
        x = cat2d(x,label,condition)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))

        x = cat2d(x,label,condition)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))

        x = cat2d(x,label,condition)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = cat2d(x,label,condition)
        x = F.relu(self.deconv4_bn(self.deconv4(x)))

        x = cat2d(x,label,condition)
        x = 3 * F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, n_classes=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, n_classes, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
