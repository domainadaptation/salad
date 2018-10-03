
salad.solver Demo
=================

In this tutorial we will step through the use of solvers within the
``salad`` package. Solvers are located in the ``salad.solver`` package
and form a hierarchy for different application purposes. All solvers are
subclasses of ``salad.solver.Solver``, which contains the basic training
mechanisms, functions for logging etc.

.. code:: ipython3

    import numpy
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'

Solvers are instantiated in the following way:
