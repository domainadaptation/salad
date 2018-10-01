from torch.optim import Optimizer

class JointOptimizer(object):

    """ Concat multiple optimizers



    Parameters
    ----------
    *optims: list of ``torch.optim.Optimizer``
        Optimizers. The ``step`` and ``zero_grad`` functions will be executed in
        the same order.

    """
    
    def __init__(self, *optims):        
        self.optims = optims
        
    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()
    
    def step(self):
        for optim in self.optims:
            optim.step()

class _DelayedWeight (object):
    def __init__(self, params, src_params):

        self.params = list(params)
        self.src_params = list(src_params)

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        for p, src_p in zip(self.params, self.src_params):
            p.data.set_(src_p.data)

    def zero_grad(self):
        pass

class WeightEMA (object):
    """ Exponential moving average weight optimizer for mean teacher model

    Used for Self-Ensembling, code adapted from [1]_.

    See Also
    --------

    ``salad.solver.SelfEnsemblingSolver``

    Reference
    ---------

    .. [1] https://github.com/Britefury/self-ensemble-visual-domain-adapt
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
