import importlib


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
