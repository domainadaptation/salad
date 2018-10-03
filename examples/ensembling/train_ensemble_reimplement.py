""" Minimal Training Script for Associative Domain Adaptation
"""

import os
import os.path as osp
import sys
import argparse

import torch
from torch import nn

from salad import solver, models, datasets

from augment import AffineTransformer
from augment2 import ImageAugmentation

from torch.nn import functional as F
from salad.layers import AccuracyScore

class SVHN_MNIST_Model(nn.Module):
    
    def __init__(self, n_classes):
        super(SVHN_MNIST_Model, self).__init__()
        
        self.norm = nn.InstanceNorm2d(3, affine=False,
                momentum=0,
                track_running_stats=False)
        
        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.norm(x)
        
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        z = x = x.view(-1, 128)

        x = self.fc4(x)
        return z, x


def build_parser():

    parser = argparse.ArgumentParser(description='Associative Domain Adaptation')

    # General setup
    parser.add_argument('--gpu', default=0,
        help='Specify GPU', type=int)
    parser.add_argument('--cpu', action='store_true',
        help="Use CPU Training")
    parser.add_argument('--log', default="./log/testruns",
        help="Log directory. Will be created if non-existing")
    parser.add_argument('--epochs', default="1000",
        help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
    parser.add_argument('--checkpoint', default="",
        help="Checkpoint path")
    parser.add_argument('--learningrate', default=1e-3, type=float,
        help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
    parser.add_argument('--dryrun', action='store_true',
        help="Perform a test run, without actually training a network.")

    # Domain Adaptation Args
    parser.add_argument('--source', default="svhn", choices=['mnist', 'svhn'],
                        help="Source Dataset. Choose mnist or svhn")
    parser.add_argument('--target', default="mnist", choices=['mnist', 'svhn'],
                        help="Target Dataset. Choose mnist or svhn")

    parser.add_argument('--sourcebatch', default=128, type=int,
                        help="Batch size of Source")
    parser.add_argument('--targetbatch', default=128, type=int,
                        help="Batch size of Target")
    
    return parser

from salad import solver

class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model

    Taken from https://github.com/Britefury/self-ensemble-visual-domain-adapt
    """
    def __init__(self, params, src_params, alpha=0.999):

        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

    def zero_grad(self):
        pass

class EnsemblingLoss(object):

    def __init__(self, model, teacher):

        self.model   = model
        self.teacher = teacher

    def __call__(self, batch):        
        (x_stud_xs, ys), (x_stud_xt,x_teach_xt,yt) = batch
        
        _, stud_ys  = self.model(x_stud_xs)
        _, stud_yt  = self.model(x_stud_xt)
        _, teach_yt = self.teacher(x_teach_xt)

        losses = {}
        losses['ce']         = (stud_ys, ys)
        losses['ensemble']   = (stud_yt, teach_yt.detach())
        
        losses['acc_s']       = (stud_ys, ys)
        losses['acc_t']       = (stud_yt, yt)
        losses['acc_teacher'] = (teach_yt, yt)

        return losses
    
class WeightedCE(nn.Module):
    
    def __init__(self):
        
        super(WeightedCE, self).__init__()
        
        self.threshold  =  0.96837722
        self.softmax    = nn.Softmax(dim=-1)
        
    def robust_binary_crossentropy(self, pred, tgt):
        inv_tgt = -tgt + 1.0
        inv_pred = -pred + 1.0 + 1e-6
        return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))
        
    def forward(self, logits, logits_target):
        
        p_         = self.softmax(logits)
        p          = self.softmax(logits_target)
        conf, _    = p.max(dim=-1)
        mask       = torch.gt(conf, self.threshold)
        
        if mask.sum() > 0:
            loss = self.robust_binary_crossentropy(p_[mask], p[mask]).mean()
            return loss
        
        loss = self.robust_binary_crossentropy(p_[:1], p[:1]).mean()
        
        return 0 * loss

    
class JointOptimizer(torch.optim.Optimizer):
    
    def __init__(self, adam, ema):
        
        self.adam = adam
        self.ema  = ema
        
    def zero_grad(self):
        self.adam.zero_grad()
    
    def step(self):
        self.adam.step()
        self.ema.step()
    
class SelfEnsemblingSolver(solver.da.DABaseSolver):

    def __init__(self, model, teacher, dataset, learningrate, *args, **kwargs):
        super(SelfEnsemblingSolver, self).__init__(model, dataset, *args, **kwargs)

        teacher_alpha = 0.99
        
        self.register_model(teacher, "teacher")
        self.teacher = teacher
        
        opt_stud  = torch.optim.Adam(model.parameters(), lr=learningrate)
        opt_teach = WeightEMA(teacher.parameters(),
                              model.parameters(),
                              alpha=teacher_alpha)
        
        optim = JointOptimizer(opt_stud, opt_teach)
        

        self.register_optimizer(optim, EnsemblingLoss(self.model, self.teacher),
                               name='Joint Optimizer')
        self.register_loss(WeightedCE(), 3, 'ensemble')
        self.register_loss(AccuracyScore(), None, 'acc_teacher')

        
class Augmentation():
    
    def __init__(self, dataset, n_samples=1):
        self.transformer = ImageAugmentation(
            affine_std=0.1,
            gaussian_noise_std=0.1,
            hflip=False,
            intens_flip=True,
            intens_offset_range_lower=-.5, intens_offset_range_upper=.5,
            intens_scale_range_lower=0.25, intens_scale_range_upper=1.5,
            xlat_range=2.0
        )
        
        self.dataset = dataset
        self.n_samples = n_samples
        
    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        x, y = self.dataset[index]
        
        X = torch.stack([x.clone() for _ in range(self.n_samples)], dim=0)
        X = self.transformer.augment(X.numpy())
        
        outp = [torch.from_numpy(x).float() for x in X] + [y,]
        
        return outp
        
if __name__ == '__main__':

    parser = build_parser()
    args   = parser.parse_args()

    # Network
    model   = SVHN_MNIST_Model(10)
    teacher = SVHN_MNIST_Model(10)
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Dataset
    data = datasets.da.load_dataset2(path="data", train=True)

    train_loader = torch.utils.data.DataLoader(
        Augmentation(data[args.source]), batch_size=args.sourcebatch,
        shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        Augmentation(data[args.target], 2), batch_size=args.targetbatch,
        shuffle=True, num_workers=4)

    loader = datasets.JointLoader(train_loader, val_loader)
    
    # Initialize the solver for this experiment
    experiment = SelfEnsemblingSolver(model, teacher, loader,
                               n_epochs=args.epochs,
                               savedir=args.log,
                               dryrun = args.dryrun,
                               learningrate = args.learningrate,
                               gpu=args.gpu if not args.cpu else None)

    experiment.optimize()
