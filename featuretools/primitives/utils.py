import os
from inspect import isclass

import pandas as pd

from .base import AggregationPrimitive, PrimitiveBase, TransformPrimitive

import featuretools
from featuretools.utils import is_python_2

if is_python_2():
    import imp
else:
    import importlib.util


def get_aggregation_primitives():
    aggregation_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.AggregationPrimitive):
                if attribute.name:
                    aggregation_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in aggregation_primitives}


def get_transform_primitives():
    transform_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.TransformPrimitive):
                if attribute.name:
                    transform_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in transform_primitives}


def list_primitives():
    trans_names, trans_primitives = _get_names_primitives(get_transform_primitives)
    transform_df = pd.DataFrame({'name': trans_names,
                                 'description': _get_descriptions(trans_primitives)})
    transform_df['type'] = 'transform'

    agg_names, agg_primitives = _get_names_primitives(get_aggregation_primitives)
    agg_df = pd.DataFrame({'name': agg_names,
                           'description': _get_descriptions(agg_primitives)})
    agg_df['type'] = 'aggregation'

    return pd.concat([agg_df, transform_df], ignore_index=True)[['name', 'type', 'description']]


def _get_descriptions(primitives):
    descriptions = []
    for prim in primitives:
        description = ''
        if prim.__doc__ is not None:
            description = prim.__doc__.split("\n")[0]
        descriptions.append(description)
    return descriptions


def _get_names_primitives(primitive_func):
    names = []
    primitives = []
    for name, primitive in primitive_func().items():
        names.append(name)
        primitives.append(primitive)
    return names, primitives


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
    if is_python_2():
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
