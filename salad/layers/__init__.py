""" Domain adaptation network layers

This module is comprised of network layers required and introduced in various publications.
"""


from .base import KLDivWithLogits, AccuracyScore, MeanAccuracyScore, WeightedCE
from .association import *
from .vat import *
from .coral import *

from .funcs import *
from .noise import *