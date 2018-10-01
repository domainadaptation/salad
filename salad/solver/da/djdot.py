from .base import BaseClassSolver

class DJDOTSolver(BaseClassSolver):

    """ Deep Joint Optimal Transport solver

    TODO
    """

    def __init__(self, model, dataset, *args, **kwargs):

        super(DJDOTSolver, self).__init__(model, dataset, *args, **kwargs)

    def derive_losses(self, batch):

        # compute the

        Gamma = None # optimal transport matrix

        pass