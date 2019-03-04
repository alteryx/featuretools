# flake8: noqa
from .api import *

import pkg_resources
for entry_point in pkg_resources.iter_entry_points('featuretools_primitives'):
    for name, obj in entry_point.load().primitives.items():
        globals()[name] = obj
