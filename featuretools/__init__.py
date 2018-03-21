from __future__ import absolute_import
# flake8: noqa
from . import config
from .core import *
from . import variable_types
from .entityset.api import *
from . import primitives
from .synthesis.api import *
from .primitives import Feature
from .computational_backends.api import *
from . import tests
from .utils.pickle_utils import *
import featuretools.demo

__version__ = '0.1.19'
