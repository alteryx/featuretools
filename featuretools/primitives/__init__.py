import inspect
import logging

import pkg_resources

from .api import *  # noqa: F403


def _load_primitives():
    """Load in a list of primitives registered by other libraries into Featuretools.

        Example entry_points definition for a library using this entry point:
        entry_points={
            "featuretools_primitives": [
                other_library = other_library:LIST_OF_PRIMITIVES
            ]
        }
        where `LIST_OF_PRIMITIVES` is an iterable of primitive class objects defined
        in module `other_library`.
    """
    base_primitives = (AggregationPrimitive, TransformPrimitive)  # noqa: F405
    msg = "entry point \"%s\" in package \"%s\" threw exception while loading: %s",
    for entry_point in pkg_resources.iter_entry_points('featuretools_primitives'):  # pragma: no cover
        try:
            loaded = entry_point.load()
        except Exception as e:
            logging.warning(
                msg,
                entry_point.name,
                entry_point.dist.project_name,
                repr(e),
            )
            continue

        for attr in dir(loaded):
            primitive = getattr(loaded, attr)

            if (
                inspect.isclass(primitive) and
                issubclass(primitive, base_primitives) and  # noqa: F405
                primitive not in base_primitives  # noqa: F405
            ):
                scope = globals()
                if primitive.__name__ in scope:
                    error = f"primitive with name \"{primitive.__name__}\" already exists"
                    raise RuntimeError(error)

                else:
                    scope[primitive.__name__] = primitive


_load_primitives()
