r"""
LTNN (Latent time neuron network)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import models,plot,datasets
from .models import lsi
from .models import scLTNN

from ._pseudotimekernel import PseudotimeKernel

name = "scltnn"
__version__ = version(name)
