import os
import sys
from inspect import isclass

from .base import AggregationPrimitive, PrimitiveBase, TransformPrimitive

import featuretools

IS_PY2 = (sys.version_info[0] == 2)


if IS_PY2:
    import imp
else:
    import importlib.util


def get_featuretools_root():
    return os.path.dirname(featuretools.__file__)


def list_primitive_files(directory):
    """returns list of files in directory that might contain primitives"""
    files = os.listdir(directory)
    keep = []
    for path in files:
        if not check_valid_primitive_path(path):
            continue
        keep.append(os.path.join(directory, path))
    return keep


def check_valid_primitive_path(path):
    if os.path.isdir(path):
        return False

    filename = os.path.basename(path)

    if filename[:2] == "__" or filename[0] == "." or filename[-3:] != ".py":
        return False

    return True


def load_primitive_from_file(filepath):
    """load primitive objects in a file"""
    module = os.path.basename(filepath)[:-3]
    if IS_PY2:
        # for python 2.7
        module = imp.load_source(module, filepath)
    else:
        # TODO: what is the first argument"?
        # for python >3.5
        spec = importlib.util.spec_from_file_location(module, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    primitives = []
    for primitive_name in vars(module):
        primitive_class = getattr(module, primitive_name)
        if (isclass(primitive_class) and
                issubclass(primitive_class, PrimitiveBase) and
                primitive_class not in (AggregationPrimitive,
                                        TransformPrimitive)):
            primitives.append((primitive_name, primitive_class))

    if len(primitives) == 0:
        raise RuntimeError("No primitive defined in file %s" % filepath)
    elif len(primitives) > 1:
        raise RuntimeError("More than one primitive defined in file %s" % filepath)

    return primitives[0]
