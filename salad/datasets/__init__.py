""" Datasets for Domain Adaptation Experiments.

This package contains datasets and tools for handling datasets.
Similar as in ``torchvision.datasets``, data is accessed through
subclasses of ``torch.utils.data.DataLoader`` and 
``torch.utils.data.Dataset``.

As one very established Domain Adaptation benchmarks, the ``digits``
package focusses on the small digit benchmark consisting of

- MNIST
- USPS
- SVHN
- SYNTH

Principally two main methods for loading data are currently implemented.
In general, multiple datasets are loaded.

In **cat** mode, the dataset returns values of the form

>>> for x,y,d in data_loader:
>>>     print(x.size(), y.size(), d.size())

In **stack** mode, the dataset returns tuples (of possible different sizes):

>>> for (xs,ys), (xt, yt) in data_loader:
>>>     pass

"""

from .da import *
from .visda import *
from .digits import *
from .utils import *
# from .instance import *