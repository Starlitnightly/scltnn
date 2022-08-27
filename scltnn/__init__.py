r"""
LTNN (Latent time neuron network)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import models,plot,utils,datasets

name = "scltnn"
__version__ = version(name)

__all__ = [
    "Sequential",
    "Dense",
    "r2_score",
    "plt",
    "distfit",
    "norm",
    "dweibull",
    "load_model",
    "pd",
    "np",
    "Dense",
    "Conv1D",
    "Convolution1D",
    "Activation",
    "MaxPooling1D",
    "Flatten",
    "Dropout",
    "TensorBoard",
    "regularizers",
]