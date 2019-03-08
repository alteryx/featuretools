from __future__ import absolute_import
# flake8: noqa
from .config_init import config
from . import variable_types
from .entityset.api import *
from . import primitives
from .synthesis.api import *
from .primitives import list_primitives, install_primitives
from .computational_backends.api import *
from . import tests
from .utils.pickle_utils import *
from .utils.time_utils import *
import featuretools.demo
import featuretools.wrappers
from . import feature_base
from .feature_base import AggregationFeature, DirectFeature, Feature, FeatureBase, IdentityFeature, TransformFeature

__version__ = '0.6.1'

import pkg_resources
for entry_point in pkg_resources.iter_entry_points('featuretools_initialize'):
    try:
        module = entry_point.load()
        if hasattr(module, 'initialize'):
            module.initialize()
    except:
        pass