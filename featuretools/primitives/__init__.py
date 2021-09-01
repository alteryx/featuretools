import inspect
import logging

import pkg_resources

from .api import *  # noqa: F403

# Load in a list of primitives registered by other libraries into Featuretools.
#
# Example entry_points definition for a library using this entry point:
#
#    entry_points={
#        "featuretools_primitives": [
#            other_library = other_library:LIST_OF_PRIMITIVES
#        ]
#    }
#
# where `LIST_OF_PRIMITIVES` is an iterable of primitive class objects defined
# in module `other_library`.

for entry_point in pkg_resources.iter_entry_points('featuretools_primitives'):  # pragma: no cover
    try:
        loaded = entry_point.load()
    except Exception as e:
        logging.warning(
            "entry point \"%s\" in package \"%s\" threw exception while loading: %s",
            entry_point.name,
            entry_point.dist.project_name,
            repr(e),
        )
        continue

    for primitive in loaded:
        if primitive.__name__ in globals():
            raise RuntimeError(
                f"primitive with name \"{primitive.__name__}\" already exists"
            )

        if (
            inspect.isclass(primitive) and
            issubclass(primitive, (AggregationPrimitive, TransformPrimitive)) and  # noqa: F405
            primitive is not AggregationPrimitive and  # noqa: F405
            primitive is not TransformPrimitive  # noqa: F405
        ):
            globals()[primitive.__name__] = primitive
