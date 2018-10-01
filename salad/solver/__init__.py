#__import__('pkg_resources').declare_namespace(__name__)
""" Model optimization by stochastic gradient descent and variants 
The general structure:

- Each experiment configuration is a subclass of Solver or some derivative
  loss functions
- Solvers only specify how data and models are used to generate the losses
- Similarities between deep learning experiments (checkpointing, logging, ...)
  are implemented in the Solver class.

In general, for many experiments, it makes sense to set up a solver as a
subclass of a specific other solver; i.e. when the general problem is concerned
with classifcation, a `CrossEntropySolver` would be a natural choice.

Classes where designed with the possibility of re-use in mind. The goal is to
exploit the particular structure most deep learning experiments share.

"""

from .base import *
from .classification import *
from .gan import *
from .da import *