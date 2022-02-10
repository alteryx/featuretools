import inspect
import logging

import pkg_resources

from .api import *  # noqa: F403


def _load_primitives():
    """Load in a list of primitives registered by other libraries into Featuretools.

        Example entry_points definition for a library using this entry point either in:

            - setup.py:

                setup(
                    entry_points={
                        'featuretools_primitives': [
                            'other_library = other_library',
                        ],
                    },
                )

            - setup.cfg:

                [options.entry_points]
                featuretools_primitives =
                    other_library = other_library

        where `other_library` is a top-level module containing all the primitives.
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
                issubclass(primitive, base_primitives) and
                primitive not in base_primitives
            ):
                scope = globals()
                if primitive.__name__ in scope:
                    error = f"primitive with name \"{primitive.__name__}\" already exists"
                    raise RuntimeError(error)
                else:
                    scope[primitive.__name__] = primitive


_load_primitives()
