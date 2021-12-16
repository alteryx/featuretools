import importlib.util
import os
from inspect import isclass

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

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
    """Provide rolling window calculations where the windows are determined using both a gap parameter
    that indicates the amount of time between each instance and its window and a window length parameter
    that determines the amount of data in each window.

    Args:
        series (Series): The series over which rolling windows will be created. Must be numeric in nature
            and have a DatetimeIndex.
        window_size (int, string): Specifies the amount of data included in each window.
            If an integer is provided, will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases, can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 0, which will include the target instance in the window.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

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

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_lengt.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months are different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    """
    # Workaround for pandas' bug: https://github.com/pandas-dev/pandas/issues/43016
    # Can remove when upgraded to pandas 1.4.0
    if str(series.dtype) == 'Int64':
        series = series.astype('float64')

    # --> add catching of non offset string except Exception:
    # raise ValueError('must be a valid time frame in seconds, minutes, hours, or days (e.g. 1s, 5min, 4h, 7d, etc.)')
    # --> handle situation where no datetime is present? maybe not necessary bc only used internally! just add a note in the docstring about needinst a datetime if you want to allow offset strings
    # --> add note that the gap is most predictable when it's a fixed frequency (like hour or day) rather than one that can be a variable nubmer of days (like year or month)
    # The gap will just use the offset of that string, so if it's variable,
    # --> window length must be a fixed freq

    # If gap is an offset string, it'll get applied at the primitive call
    functional_window_length = window_size
    # --> gap and window length must both be fixed tobe added to one another
    # --> and they're assumed to be the same type
    if isinstance(gap, str):
        if not isinstance(window_size, str):
            raise TypeError(f"Cannot roll series with offset gap, {gap}, and numeric window length, {window_size}."
                            "If an offset alias is used for gap, the window length must also be defined as an offset alias."
                            "Please either change gap to be numeric or change window length to be an offset alias.")
        functional_window_length = to_offset(window_size) + to_offset(gap)
    elif gap > 0:
        series = series.shift(gap)

    return series.rolling(functional_window_length, min_periods)


def _get_rolled_series_without_gap(series, offset_string):
    """Determines how many rows of the series
    """
    if not len(series):
        return series

    window_start_date = series.index[0]
    window_end_date = series.index[-1]

    gap_bound = window_end_date - to_offset(offset_string)

    # If the gap is larger than the series, no rows are left in the wndow
    if gap_bound < window_start_date:
        return pd.Series()

    # Only return the rows that are within the offset's bounds
    # Assumes series has a datetime index and is sorted by that index
    return series[series.index <= gap_bound]


def _apply_roll_with_offset_gap(rolled_sub_series, offset_gap, reducer_fn, min_periods):
    """Takes in a series to which an offset gap will be applied, removing however many
    rows fall under the gap before applying the reducing function.
    """
    # Gets the sub series without the gap component
    rolled_sub_series = _get_rolled_series_without_gap(rolled_sub_series, offset_gap)

    if min_periods is None:
        min_periods = 1

    if len(rolled_sub_series) < min_periods or not len(rolled_sub_series):
        return np.nan

    return reducer_fn(rolled_sub_series)
