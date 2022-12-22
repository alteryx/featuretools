import importlib
import logging
import re
import sys
from enum import Enum

from tqdm import tqdm

logger = logging.getLogger("featuretools.utils")


def make_tqdm_iterator(**kwargs):
    options = {"file": sys.stdout, "leave": True}
    options.update(kwargs)
    return tqdm(**options)


def get_relationship_column_id(path):
    _, r = path[0]
    child_link_name = r._child_column_name
    for _, r in path[1:]:
        parent_link_name = child_link_name
        child_link_name = "%s.%s" % (r.parent_name, parent_link_name)
    return child_link_name


def find_descendents(cls):
    """
    A generator which yields all descendent classes of the given class
    (including the given class)

    Args:
        cls (Class): the class to find descendents of
    """
    yield cls
    for sub in cls.__subclasses__():
        for c in find_descendents(sub):
            yield c


def import_or_raise(library, error_msg):
    """
    Attempts to import the requested library.  If the import fails, raises an
    ImportErorr with the supplied

    Args:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        raise ImportError(error_msg)


def import_or_none(library):
    """
    Attemps to import the requested library.

    Args:
        library (str): the name of the library
    Returns: the library if it is installed, else None
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        return None


def is_instance(obj, modules, classnames):
    """
    Check if the given object is an instance of classname in module(s). Module
    can be None (i.e. not installed)

    Args:
        obj (obj): object to test
        modules (module or tuple[module]): module to check, can be also be None (will be ignored)
        classnames (str or tuple[str]): classname from module to check. If multiple values are
                                        provided, they should match with a single module in order.
                                        If a single value is provided, will be used for all modules.
    Returns:
        bool: True if object is an instance of classname from corresponding module, otherwise False.
              Also returns False if the module is None (i.e. module is not installed)
    """
    if type(modules) is not tuple:
        modules = (modules,)
    if type(classnames) is not tuple:
        classnames = (classnames,) * len(modules)
    if len(modules) != len(classnames):
        raise ValueError("Number of modules does not match number of classnames")
    to_check = tuple(
        getattr(mod, classname, mod)
        for mod, classname in zip(modules, classnames)
        if mod
    )
    return isinstance(obj, to_check)


def camel_and_title_to_snake(name):
    name = re.sub(r"([^_\d]+)(\d+)", r"\1_\2", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class Library(str, Enum):
    PANDAS = "pandas"
    DASK = "Dask"
    SPARK = "Spark"
