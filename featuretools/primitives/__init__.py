# flake8: noqa
from .api import *

import pkg_resources
# Load in a list of primitives registered by other libraries into Featuretools
# Example entry_points definition for a library using this entry point:
#    entry_points={
#        "featuretools_primitives": [
#            other_library = other_library:LIST_OF_PRIMITIVES
#        ]
#    }
for entry_point in pkg_resources.iter_entry_points('featuretools_primitives'):
    try:
        loaded = entry_point.load()
        for primitive in loaded:
            if issubclass(primitive, (AggregationPrimitive, TransformPrimitive)):
                globals()[primitive.__name__] = primitive
    except Exception:
        pass
