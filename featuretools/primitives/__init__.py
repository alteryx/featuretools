# flake8: noqa
from .api import *

import pkg_resources
# Load in primitives registered by other libraries into Featuretools namespace
for entry_point in pkg_resources.iter_entry_points('featuretools_primitives'):
    try:
        loaded = entry_point.load()
        if hasattr(loaded, 'primitives'):
            for name, obj in loaded.primitives.items():
                globals()[name] = obj
    except Exception:
        pass
