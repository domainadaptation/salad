from .base import Solver

import os, time
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn

from salad.layers import MeanAccuracyScore

import itertools

from .. import layers, optim

class ClassificationLoss(object):

    def __init__(self, solver):
        self.solver = solver

    def __call__(self, batch):
        losses = {}
        x, y = batch

        _ , y_ = self.solver.model(x)
        if not self.solver.multiclass:
            y_ = y_.squeeze()
            y  = y.float()

        losses['acc'] = losses['mean_acc'] = losses['ce'] = (y_, y)

        return losses

class MultiDomainClassificationLoss(object):

    def __init__(self, solver, domain):

        self.domain     = domain
        self.solver     = solver

    def __call__(self, batch):
        losses = {}

        (x, y)  = batch[self.domain]

        _ , y_ = self.solver.model(x, self.domain)

        if not self.solver.multiclass:
            y_ = y_.squeeze()
            y  = y.float()
        
        losses['CE_{}'.format(self.domain)] = (y_, y)
        losses['ACC_{}'.format(self.domain)] = (y_, y)

        return losses

class BaseClassSolver(Solver):

    """ Base Solver for classification experiments

    Parameters
    ----------

    model : nn.Module
        A model to train on a classification target
    dataset : torch.utils.data.Dataset
        The dataset providing training samples
    multiclass : bool
        If True, ``CrossEntropyLoss`` is used, ``BCEWithLogitsLoss`` otherwise.
    """

    def __init__(self, model, dataset, multiclass = True, *args, **kwargs):

        self.model      = model
        self.multiclass = multiclass
        
        super().__init__(dataset=dataset, *args, **kwargs)

    def _init_losses(self, **kwargs):
        self.register_loss(nn.CrossEntropyLoss() if self.multiclass else nn.BCEWithLogitsLoss(),
                           weight = 1,
                           name   = 'ce')        

    def _init_models(self, **kwargs):
        self.register_model(self.model, "classifier")

class FinetuneSolver(BaseClassSolver):
    """ Finetune a pre-trained deep learning models

    Given a model with separable feature extractor and classifier, use different learning
    rates and regularization settings. Useful for fine-tuning pre-trained ImageNet models
    or finetuning of saved model checkpoints

    Parameters
    ----------
    model : nn.Module
        Module with two separate parts
    dataset : Dataset
        The dataset used for training
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _init_optims(self, **kwargs):
        optim_finetune  = torch.optim.Adam(self.model.features.parameters(),
                            lr=1e-5, amsgrad=True)
        optim_classifier = torch.optim.Adam(self.model.classifier.parameters(),
                            lr=1e-4, amsgrad=True)
        optim_joint = optim.JointOptimizer(optim_finetune, optim_classifier)

        self.register_optimizer(optim_joint, ClassificationLoss(self),
                                    "Joint Optimizer")


class BCESolver(BaseClassSolver):

    """ Solver for a classification experiment
    """

    def __init__(self, *args, **kwargs):

        super(BCESolver, self).__init__(*args, **kwargs)

    def _init_optims(self, lr = 3e-4, **kwargs):
        super()._init_optims(**kwargs)

        self.register_optimizer(torch.optim.Adam(self.model.parameters(),
                                lr=lr, amsgrad=True),
                                ClassificationLoss(self))

    def _init_losses(self, **kwargs):
        super()._init_losses(**kwargs)

        self.register_loss(layers.AccuracyScore(), None, 'acc')
        self.register_loss(layers.MeanAccuracyScore(), None, 'mean_acc')


class MultidomainBCESolver(Solver):

    def __init__(self, model, dataset, learningrate, multiclass = True,
                 loss_weights = None, *args, **kwargs):

        super(MultidomainBCESolver, self).__init__(dataset=dataset,
                                                   *args, **kwargs)

        self.model      = model
        self.multiclass = multiclass
        self.n_domains  = model.n_domains
        weights         = self.cuda(torch.tensor(loss_weights).float())

        self.register_model(self.model, "domain")

        for d in range(self.n_domains):

            loss_func = nn.CrossEntropyLoss(weight=weights) if multiclass else nn.BCEWithLogitsLoss()
            self.register_loss( loss_func,
                                weight = 1,
                                name   = 'CE_{}'.format(d))
            self.register_loss( MeanAccuracyScore(),
                    weight = None,
                    name   = 'ACC_{}'.format(d))

            # Specify the optimizer
            self.register_optimizer(torch.optim.Adam(self.model.parameters(d),
                                        lr=kwargs.get('learningrate', learningrate),
                                        amsgrad=True),
                                        MultiDomainClassificationLoss(self, domain=d)
                                    )