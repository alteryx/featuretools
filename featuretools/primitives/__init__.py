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
    logger = logging.getLogger('featuretools')
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

        for key in dir(loaded):
            primitive = getattr(loaded, key, None)

            if (
                inspect.isclass(primitive) and
                issubclass(primitive, base_primitives) and
                primitive not in base_primitives
            ):
                name = primitive.__name__
                scope = globals()

                if name in scope:
                    this_module, that_module = primitive.__module__, scope[name].__module__
                    message = f'Ignoring primitive "{name}" from "{this_module}" '
                    message += f'because it already exists in "{that_module}"'
                    logger.warning(message)
                else:
                    scope[name] = primitive


_load_primitives()
