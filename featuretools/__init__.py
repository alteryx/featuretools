# flake8: noqa
import logging
import sys
import traceback

import pkg_resources

import featuretools.demo
from featuretools import feature_base
from featuretools import primitives
from featuretools import selection
from featuretools import tests
from featuretools.computational_backends.api import *
from featuretools.config_init import config
from featuretools.entityset.api import *
from featuretools.feature_base import (
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
from featuretools.primitives import list_primitives, summarize_primitives
from featuretools.synthesis.api import *
from featuretools.utils.time_utils import *
from featuretools.utils.utils_info import show_info
from featuretools.version import (
    __version__,
    ENTITYSET_SCHEMA_VERSION,
    FEATURES_SCHEMA_VERSION,
)

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
