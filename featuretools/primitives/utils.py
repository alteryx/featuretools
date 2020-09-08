import importlib.util
import os
from inspect import isclass

import pandas as pd

import featuretools
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from featuretools.utils.gen_utils import Library, find_descendents


# returns all aggregation primitives, regardless of compatibility
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


# returns all transform primitives, regardless of compatibility
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
    trans_dask = [Library.DASK in primitive.compatibility for primitive in trans_primitives]
    trans_koalas = [Library.KOALAS in primitive.compatibility for primitive in trans_primitives]
    transform_df = pd.DataFrame({'name': trans_names,
                                 'description': _get_descriptions(trans_primitives),
                                 'dask_compatible': trans_dask,
                                 'koalas_compatible': trans_koalas})
    transform_df['type'] = 'transform'

    agg_names, agg_primitives = _get_names_primitives(get_aggregation_primitives)
    agg_dask = [Library.DASK in primitive.compatibility for primitive in agg_primitives]
    agg_koalas = [Library.KOALAS in primitive.compatibility for primitive in agg_primitives]
    agg_df = pd.DataFrame({'name': agg_names,
                           'description': _get_descriptions(agg_primitives),
                           'dask_compatible': agg_dask,
                           'koalas_compatible': agg_koalas})
    agg_df['type'] = 'aggregation'

    return pd.concat([agg_df, transform_df], ignore_index=True)[['name', 'type', 'dask_compatible', 'koalas_compatible', 'description']]


def get_default_aggregation_primitives():
    agg_primitives = [featuretools.primitives.Sum,
                      featuretools.primitives.Std,
                      featuretools.primitives.Max,
                      featuretools.primitives.Skew,
                      featuretools.primitives.Min,
                      featuretools.primitives.Mean,
                      featuretools.primitives.Count,
                      featuretools.primitives.PercentTrue,
                      featuretools.primitives.NumUnique,
                      featuretools.primitives.Mode]
    return agg_primitives


def get_default_transform_primitives():
    # featuretools.primitives.TimeSince
    trans_primitives = [featuretools.primitives.Age,
                        featuretools.primitives.Day,
                        featuretools.primitives.Year,
                        featuretools.primitives.Month,
                        featuretools.primitives.Weekday,
                        featuretools.primitives.Haversine,
                        featuretools.primitives.NumWords,
                        featuretools.primitives.NumCharacters]
    return trans_primitives


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
    # TODO: what is the first argument"?
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


def serialize_primitive(primitive):
    """build a dictionary with the data necessary to construct the given primitive"""
    args_dict = {name: val for name, val in primitive.get_arguments()}
    cls = type(primitive)
    return {
        'type': cls.__name__,
        'module': cls.__module__,
        'arguments': args_dict,
    }


class PrimitivesDeserializer(object):
    """
    This class wraps a cache and a generator which iterates over all primitive
    classes. When deserializing a primitive if it is not in the cache then we
    iterate until it is found, adding every seen class to the cache. When
    deseriazing the next primitive the iteration resumes where it left off. This
    means that we never visit a class more than once.
    """

    def __init__(self):
        self.class_cache = {}  # (class_name, module_name) -> class
        self.primitive_classes = find_descendents(PrimitiveBase)

    def deserialize_primitive(self, primitive_dict):
        """
        Construct a primitive from the given dictionary (output from
        serialize_primitive).
        """
        class_name = primitive_dict['type']
        module_name = primitive_dict['module']
        cache_key = (class_name, module_name)

        if cache_key in self.class_cache:
            cls = self.class_cache[cache_key]
        else:
            cls = self._find_class_in_descendants(cache_key)

            if not cls:
                raise RuntimeError('Primitive "%s" in module "%s" not found' %
                                   (class_name, module_name))

        arguments = primitive_dict['arguments']
        return cls(**arguments)

    def _find_class_in_descendants(self, search_key):
        for cls in self.primitive_classes:
            cls_key = (cls.__name__, cls.__module__)
            self.class_cache[cls_key] = cls

            if cls_key == search_key:
                return cls
