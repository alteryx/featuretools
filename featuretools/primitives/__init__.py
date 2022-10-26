# flake8: noqa
import inspect
import logging
import traceback

import pkg_resources

from featuretools.primitives.standard import *
from featuretools.primitives.utils import (
    get_aggregation_primitives,
    get_default_aggregation_primitives,
    get_default_transform_primitives,
    get_transform_primitives,
    list_primitives,
    summarize_primitives,
)


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

        - pyproject.toml:

            [project.entry-points."featuretools_primitives"]
            other_library = "other_library"

    where `other_library` is a top-level module containing all the primitives.
    """
    logger = logging.getLogger("featuretools")
    base_primitives = AggregationPrimitive, TransformPrimitive  # noqa: F405

    for entry_point in pkg_resources.iter_entry_points("featuretools_primitives"):
        try:
            loaded = entry_point.load()
        except Exception:
            message = f'Featuretools failed to load "{entry_point.name}" primitives from "{entry_point.module_name}". '
            message += "For a full stack trace, set logging to debug."
            logger.warning(message)
            logger.debug(traceback.format_exc())
            continue

        for key in dir(loaded):
            primitive = getattr(loaded, key, None)

            if (
                inspect.isclass(primitive)
                and issubclass(primitive, base_primitives)
                and primitive not in base_primitives
            ):
                name = primitive.__name__
                scope = globals()

                if name in scope:
                    this_module, that_module = (
                        primitive.__module__,
                        scope[name].__module__,
                    )
                    message = f'While loading primitives via "{entry_point.name}" entry point, '
                    message += (
                        f'ignored primitive "{name}" from "{this_module}" because '
                    )
                    message += (
                        f'a primitive with that name already exists in "{that_module}"'
                    )
                    logger.warning(message)
                else:
                    scope[name] = primitive


_load_primitives()
