""" Base classes for solvers

This module contains abstract base classes for the solvers used in ``salad``.
"""


import os, time
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn

import itertools

from .. import layers, optim

class StructuredInit(object):
    r""" Structured Initialization of Solvers

    Initializes the components of a solver and passes arguments.
    Initialization is done in the following order:

    - ``_init_models``
    - ``_init_losses``
    - ``_init_optims``

    Parameters
    ----------

    kwargs : Keyword arguments
        Pass arguments for **all** initialization functions. Keyword arguments
        are passed through the functions in the order specified above. Unused
        keyword arguments will be printed afterwards.
        In general, solvers should be designed in a way that ensure that all keyword
        argumnets are used.

    .. note:
        Don't instantiate or subclass this class directly.
    """

    def __init__(self, **kwargs):
        # TODO more elegant solution?

        self._init_models(**kwargs) 
        self._init_losses(**kwargs)
        self._init_optims(**kwargs)

        # print('Unused kwargs in Solver: {}'.format(', '.join(kwargs.keys())))

    def _init_models(self, **kwargs):
        return kwargs

    def _init_losses(self, **kwargs):
        return kwargs

    def _init_optims(self, **kwargs):
        return kwargs

class EventBasedSolver(object):
    r""" Event handling for solvers

    All solvers derived from the ``EventBasedSolver`` are extended with event handlers, currently
    for the following events:

    - ``start_epoch``
    - ``start_batch``
    - ``finish_batch``
    - ``stop_epoch``

    .. note:
        Don't instantiate or subclass this class directly.

    """

    def __init__(self):

        self.start_epoch_handlers  = []
        self.finish_epoch_handlers = []
        self.start_batch_handlers  = []
        self.finish_batch_handlers = []
    
    def _call_eventhandler(self, handler, *args, **kwargs):
        for h in handler:
            h(*args, **kwargs)
 
    def start_epoch(self, *args, **kwargs):
        self._call_eventhandler(self.start_epoch_handlers, *args, **kwargs)

    def finish_epoch(self, *args, **kwargs):
        self._call_eventhandler(self.start_epoch_handlers, *args, **kwargs)

    def start_batch(self, *args, **kwargs):
        self._call_eventhandler(self.start_batch_handlers, *args, **kwargs)

    def finish_batch(self, *args, **kwargs):
        self._call_eventhandler(self.finish_batch_handlers, *args, **kwargs)

class Solver(EventBasedSolver, StructuredInit):

    """ General gradient descent solver for deep learning experiments

    This is a helper class for training of PyTorch models that makes very little assumptions
    about the structure of a deep learning experiment.
    Solvers are generally constructed to take one or several models (`torch.nn.Module`) and *one*
    DataLoader instance that provides a (possibly nested) tuple of examples for training.

    The ``Solver`` implements the following features:

    - Logging of losses and model checkpoints
    
    While offering these functionality in the background, this class implementation aims at being very
    flexible when it comes to designing any kind of deep learning experiment.

    When defining your own solver class, you should first

    - register models [register_model] 
    - register loss functions [register_loss]
    - register optimizers [register_optimizer]

    The abstraction goes as follows:
    
    - An experiment is fully characterized by its Solver class
    - An experiment can have multiple models
    - Parameters of the models are processed by optimizers
    - Optimizers have a functions to derive losses

    In the optimization process, the following algorithm is used:

    for opt in optimizers:
       losses = L(opt)
       grad_losses = grad(losses)
       opt.step(grad_losses)


    Parameters
    ----------
    dataset : Dataset
        Dataset used for training
    n_epochs : int
        Number of epochs (defined as full passes through the ``DataLoader``)
    savedir  : str
        log directory for saving model checkpoints and the loss history
    gpu      : int
        Number of GPU to be used. If ``None``, use CPU training instead
    dryrun   : bool
        Train only for the first batch. Useful for testing a new solver

    Notes
    -----

    After initializing all internal dictionaries, the constructor makes calls to the
    ``_init_models``, 
    ``_init_optims`` and
    ``_init_losses``
    functions. If these functions should make use of any additional keyword arguments
    you passed in your class, make sure that you initialize them prior to calling
    ``super().__init__`` in your constructor. 

    """

    def __init__(self, dataset, n_epochs=1, savedir="./log",
                 gpu=None, dryrun=False, **kwargs):

        EventBasedSolver.__init__(self)

        self.timestamp      = time.strftime("%Y%m%d-%H%M%S")
        self.savedir        = os.path.join(savedir, str(self))
        self.n_epochs       = n_epochs
        self.dataset        = dataset
        self.dryrun         = dryrun
        self.save_frequency = 1

        os.makedirs(self.savedir, exist_ok=True)

        self.to_cuda = (lambda x: x.cuda(gpu)) if (gpu is not None) else lambda x : x

        self.models       = []
        self.display_loss = []

        # loss functions
        self.loss_weights = {}
        self.loss_funcs   = {}
        self.loss_kwargs  = {}

        # models
        self.savenames    = {}

        # optimizers
        self.optims       = []
        self.retain_graph = {}
        self.agg_loss     = {}
        self.optim_name   = {}
        self.optim_steps  = {}



        StructuredInit.__init__(self, **kwargs)

    def cuda(self, obj):
        """ Move nested iterables between CUDA or CPU
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            obj = [self.cuda(el) for el in obj]
        else:
            obj = self.to_cuda(obj)
        return obj

    def register_loss(self, func, weight = 1.,
                      name=None, display = True, 
                      override = False, **kwargs):
        """ Register a new loss function

        Parameters
        ----------

        func : 
            pass
        weight : float

        """

        assert name is None or isinstance(name, str)

        # TODO refactor, add checks

        if isinstance(weight, int):
            weight = float(weight)

        assert weight is None or isinstance(weight, float)
        
        # TODO assert for torch Function
        # assert isinstance(func, nn.Module)
        if name is None:
            name = 'unnamed_{}'.format(len(self.loss_funcs)+1)

        if isinstance(func, nn.Module):
            self.cuda(func)

        if name in self.loss_funcs:
            if override:
                if name in self.display_loss:
                    self.display_loss.remove(name)

            else:
                raise ValueError('Name {} for loss func {} already taken.'.format(
                    name, self.loss_funcs[name].__class__.__name__
                )
                    + ' Call register_loss with the override=True option if this was intended.'
                )

        self.loss_funcs[name]   = func
        if weight is not None:
            self.loss_weights[name] = weight
        self.loss_kwargs[name]  = kwargs

        if display:
            self.display_loss.append(name)

        print('Registered {} as "{}" with weight {}'.format(
            func.__class__.__name__, name, weight))
    
    def remove_loss(self, name):
        buffers = [
            self.loss_funcs,
            self.loss_weights,
            self.loss_kwargs, 
            self.display_loss
        ]

        for b in buffers:
            if name in b:
                if isinstance(b, list):
                    b.remove(name)
                else:
                    del(b[name])

    def register_model(self, model, name=""):

        """ Add a model to the solver

        This method will also move the model directly to the correct device you specified 
        when constructin the solver.

        Parameters
        ----------
        model     : torch.nn.Module
            The model to be optimized. Should return a non-empty iterable when the
            ``paramters()`` method is called
        name      : str, optional
            Name for the model. Useful for checkpoints when multiple models are optimized.
        """

        assert isinstance(model, nn.Module)

        self.cuda(model)
        self.savenames[model] = name

        print('Registered {} as "{}"'.format(model.__class__.__name__, name))

    def register_optimizer(self, optimizer, loss_func, retain_graph = False, name="", n_steps = 1):
        """ Add an optimizer to the solver

        Parameters
        ----------
        optimizer : Optimizer
            A function used for updating model weights during training
        loss_func : LossFunction
            A function (or callable object) that, given the current batch passed by the Solver's data
            loader, returns a dictionary containing either a dictionary mapping loss function names
            to arguments, or a dictionary mapping loss function names to the loss.
        retain_graph : bool
            ``True`` if the computational graph should be retained after calling the loss function
            associated to the optimizer. This is usually not needed.
        name : str
            Optimizer name. Useful for logging
        n_steps : int
            Number of consecutive steps the optimizer is exectued. Usually set to 1.
        """


        self.optims.append(optimizer)
        self.retain_graph[optimizer] = retain_graph
        self.agg_loss[optimizer]   = loss_func
        self.optim_name[optimizer]   = name
        self.optim_steps[optimizer]   = n_steps

        print('Registered {} as "{}"'.format(optimizer.__class__.__name__, name))


    def format_train_report(self, losses):

        loss_str = ['{:.3f}'.format(float(losses[-1][key])) for key in self.display_loss]
        return '; '.join(loss_str)

    def format_summary_report(self, losses):

        loss_str = ['{:.3f}'.format(losses[key].mean()) for key in self.display_loss]
        return '; '.join(loss_str)

    def compute_loss_dict(self, loss_args):
        loss_dict = {}
        for n, args in loss_args.items():
            try:
                if isinstance(args, tuple) or isinstance(args, list):
                    loss_dict[n] = self.loss_funcs[n](*args)
                elif isinstance(args, torch.Tensor):
                    loss_dict[n] = args
                else:
                    raise ValueError('Loss args for {} have wrong type. Found {}, expected iterable or tensor'.format(
                        n, type(args)
                    ))
            except Exception as e:
                print('Error in resolving: {}'.format(n))
                raise e
        return loss_dict

    def _loss(self, optim, batch):
        """ Compute loss functions for a particular model
        """

        # retrieve the registered loss
        loss_args = self.agg_loss[optim](batch)
        loss_dict = self.compute_loss_dict(loss_args)

        # TODO make normalization optional?
        keys = set(self.loss_weights.keys()).intersection(set(loss_dict.keys()))
        if len(keys) > 0:
            # NOTE losses are normalized by their average absolute weight.
            # TODO make this optional in the future? Might make it harder to match hyperparamters to reference
            # implementations
            norm = sum( abs(self.loss_weights[k]) for k in keys)
            loss = sum(loss_dict[k]*self.loss_weights[k]
                       for k in keys) / norm
        else:
            loss = torch.tensor(0)

        return loss, loss_dict 

    def _step(self, batch):
        total_losses = {}

        # loop through models and optimize
        for n_optim, optimizer in enumerate(self.optims):
            last_pass = (n_optim == len(self.optims) - 1)

            for _ in range(self.optim_steps[optimizer]):
                loss, loss_dict = self._loss(optimizer, batch)

                optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward(retain_graph=(not last_pass) and
                                                self.retain_graph[optimizer])
                optimizer.step()

            total_losses.update(
                {n : float(l.data.cpu().numpy()) for n,l in loss_dict.items()}
            )

        return total_losses

    def _epoch(self, epoch):
        self.start_epoch()

        for model in self.models:
            self.cuda(model)

        losses = []
        n_batches = len(self.dataset)

        if n_batches == 0:
            raise ValueError("No Data in DataLoader!")

        tic = time.time()
        pbar = tqdm(enumerate(self.dataset), total=n_batches)
        for batch_idx, batch in pbar:
            self.start_batch()
            try:
                batch = self.cuda(batch)
                total_losses = self._step(batch)
                total_losses.update({'epoch' : int(epoch), 'batch' : int(1+batch_idx)})
                losses.append(total_losses)

                pbar.set_description(self.format_train_report(losses))

                if self.dryrun:
                    return losses, True

            except KeyboardInterrupt:
                print("Training was interrupted. Finishing training")
                return losses, True

            self.finish_batch()

        self.finish_epoch()

        return losses, False

    def optimize(self):

        """ Start the optimization process

        Notes
        -----
        Prior to the optimization process, all models will be set to training
        mode by a call to ``model.train(True)``

        """

        for model in self.models:
            model.train(True)

        losses = []
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            history, terminate = self._epoch(epoch)

            if epoch % self.save_frequency == 0:
                self._save_models(epoch)

            if history is not None: losses += history
            df_losses = self._save_history(epoch, losses)

            # TODO do for all losses
            epoch_losses = df_losses[df_losses['epoch'] == epoch]
            pbar.set_description('Average Loss: {}'.format(
                self.format_summary_report(epoch_losses)))

            if terminate:
                break

    def _save_history(self, epoch, losses):
        fname = '{}-losshistory-ep{}.csv'.format(self.timestamp, epoch)
        fname = os.path.join(self.savedir, fname)
        df_losses = pd.DataFrame.from_dict({i:l for i, l in enumerate(losses)}).T
        df_losses.to_csv(fname)
        df_losses.to_csv(os.path.join(self.savedir, 'losshistory.csv'))

        return df_losses

    def _save_models(self, epoch):

        fname = '{}-checkpoint-ep{}.pth'.format(self.timestamp, epoch)
        fname = os.path.join(self.savedir, fname)
        print("Save model checkpoint after {} epochs to {}".format(epoch, fname))
        torch.save(self.model, fname)
        torch.save(self.model, os.path.join(self.savedir, "checkpoint.pth"))


    def __str__(self):
        return '{}_{}'.format(self.timestamp, self.__class__.__name__)
