import importlib.util
import os
from inspect import isclass

import numpy as np
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
    trans_names, trans_primitives, valid_inputs, return_type = _get_names_primitives(get_transform_primitives)
    trans_dask = [Library.DASK in primitive.compatibility for primitive in trans_primitives]
    trans_koalas = [Library.KOALAS in primitive.compatibility for primitive in trans_primitives]
    transform_df = pd.DataFrame({'name': trans_names,
                                 'description': _get_descriptions(trans_primitives),
                                 'dask_compatible': trans_dask,
                                 'koalas_compatible': trans_koalas,
                                 'valid_inputs': valid_inputs,
                                 'return_type': return_type})
    transform_df['type'] = 'transform'

    agg_names, agg_primitives, valid_inputs, return_type = _get_names_primitives(get_aggregation_primitives)
    agg_dask = [Library.DASK in primitive.compatibility for primitive in agg_primitives]
    agg_koalas = [Library.KOALAS in primitive.compatibility for primitive in agg_primitives]
    agg_df = pd.DataFrame({'name': agg_names,
                           'description': _get_descriptions(agg_primitives),
                           'dask_compatible': agg_dask,
                           'koalas_compatible': agg_koalas,
                           'valid_inputs': valid_inputs,
                           'return_type': return_type})
    agg_df['type'] = 'aggregation'

    columns = ['name', 'type', 'dask_compatible', 'koalas_compatible', 'description', 'valid_inputs', 'return_type']
    return pd.concat([agg_df, transform_df], ignore_index=True)[columns]


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
    valid_inputs = []
    return_type = []
    for name, primitive in primitive_func().items():
        names.append(name)
        primitives.append(primitive)
        input_types = _get_unique_input_types(primitive.input_types)
        valid_inputs.append(', '.join(input_types))
        return_type.append(getattr(primitive.return_type, '__name__', None))
    return names, primitives, valid_inputs, return_type


def _get_unique_input_types(input_types):
    types = set()
    for input_type in input_types:
        if isinstance(input_type, list):
            types |= _get_unique_input_types(input_type)
        else:
            types.add(str(input_type))
    return types


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


def _roll_series_with_gap(series, window_size, gap=0, min_periods=1):
    """Provide rolling window calculations with window specified with both a window length
        and a gap that indicates the amount of time between each instance and its window.

    Args:
        series (Series): The series over which the windows will be created. Must be numeric in nature
            and have a datetime64[ns] index.
        window_length (int): The number of rows to be included in each window. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows backward from the target instance before the
            window of usable data begins. Defaults to 0, which will include the target instance
            in the window.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length. Defaults to 1.

    Returns:
        pandas.core.window.rolling.Rolling: The Rolling object for the series passed in.

    Note:
        Certain operations, like `pandas.core.window.rolling.Rolling.count` that can be performed
        on the Rolling object returned here may treat NaNs as periods to include in window calculations.
        So a window [NaN, 1, 3]  when `min_periods=3` will proceed with count, saying there are three periods
        but only two values and would return count=2. The calculation `max` on the other hand,
        would say that there are not three periods in that window and would return max=NaN.
        Most rolling calculations act this way. The implication of that here is that in order to
        achieve the gap, we insert NaNs at the beinning of the series, which would cause `count` to calculate
        on windows that technically should not have the correct number of periods. In the RollingCount primitive,
        we handle this case manually, replacing those values with NaNs. Any primitive that uses this function
        should determine whether this kind of handling is also necessary.

    """
    # Workaround for pandas' bug: https://github.com/pandas-dev/pandas/issues/43016
    # Can remove when upgraded to pandas 1.4.0
    if str(series.dtype) == 'Int64':
        series = series.astype('float64')

    gap_applied = series
    if gap > 0:
        gap_applied = series.shift(gap)

    return gap_applied.rolling(window_size, min_periods)


def _deconstrct_latlongs(latlongs):
    lats = np.array([x[0] if isinstance(x, tuple) else np.nan for x in latlongs])
    longs = np.array([x[1] if isinstance(x, tuple) else np.nan for x in latlongs])
    return lats, longs


def _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, unit):
    # https://stackoverflow.com/a/29546836/2512385
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0)**2
    radius_earth = 3958.7613
    if unit == 'kilometers':
        radius_earth = 6371.0088
    distances = radius_earth * 2 * np.arcsin(np.sqrt(a))
    return distances
