# flake8: noqa
from .version import __version__
from .config_init import config
from .entityset.api import *
from . import primitives
from .synthesis.api import *
from .primitives import list_primitives
from .computational_backends.api import *
from . import tests
from .utils.time_utils import *
from .utils.cli_utils import show_info
import featuretools.demo
from . import feature_base
from . import selection
from .feature_base import (
    AggregationFeature,
    DirectFeature,
    Feature,
    FeatureBase,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    graph_feature,
    describe_feature,
    save_features,
    load_features,
)

import logging
import pkg_resources
import sys
import traceback
import warnings
from woodwork import list_logical_types, list_semantic_tags

logger = logging.getLogger("featuretools")

# Call functions registered by other libraries when featuretools is imported
for entry_point in pkg_resources.iter_entry_points("featuretools_initialize"):
    try:
        method = entry_point.load()
        if callable(method):
            method()
    except Exception:
        pass
for entry_point in pkg_resources.iter_entry_points("alteryx_open_src_initialize"):
    try:
        method = entry_point.load()
        if callable(method):
            method("featuretools")
    except Exception:
        pass

# Load in submodules registered by other libraries into Featuretools namespace
for entry_point in pkg_resources.iter_entry_points("featuretools_plugin"):
    try:
        sys.modules["featuretools." + entry_point.name] = entry_point.load()
    except Exception:
        message = "Featuretools failed to load plugin {} from library {}. "
        message += "For a full stack trace, set logging to debug."
        logger.warning(message.format(entry_point.name, entry_point.module_name))
        logger.debug(traceback.format_exc())

if sys.version_info.major == 3 and sys.version_info.minor == 7:
    warnings.warn(
        "Featuretools may not support Python 3.7 in next non-bugfix release.",
        FutureWarning,
    )
